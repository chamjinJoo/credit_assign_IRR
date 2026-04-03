from __future__ import annotations

import math
from functools import partial
from typing import Any, Dict, Optional

import flax.serialization as serialization
import jax
import jax.numpy as jnp

from .common import (
    CausalBridgeNetwork,
    FeedForwardActor,
    FeedForwardCritic,
    RecurrentActor,
    RecurrentCritic,
    action_one_hot,
    add_agent_id_features,
    categorical_stats,
    compute_gae,
    gather_other_agent_obs,
    global_norm,
    make_train_state,
    stack_metrics,
    tree_to_host,
)


class JAXRPCIAlgorithm:
    requires_jax_trainer = True

    def __init__(
        self,
        env: Any,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        recurrent_N: int = 1,
        lr_actor: float = 3.0e-4,
        lr_critic: float = 3.0e-4,
        lr_bridge: float = 3.0e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 512,
        num_batches: int = 8,
        device: str = "cpu",
        global_reward: bool = True,
        add_agent_id: bool = True,
        use_centralized_critic: bool = False,
        clip_delta: float = 0.2,
        causal_lambda: float = 0.0,
        causal_lambda_schedule: bool = False,
        causal_lambda_start: float = 0.0,
        causal_lambda_final: float = 0.0,
        causal_lambda_warmup_steps: int = 10000,
        causal_lambda_startingpoint: int = 0,
        bridge_epoch: int = 4,
        causal_mini_batch_size: int = 256,
        tikhonov_gamma: float = 1.0e-4,
        stabilizer_lambda: float = 1.0,
        normalize_returns: bool = True,
        use_recurrent_policy: bool = True,
        seed: int = 42,
        **_: Any,
    ):
        if use_recurrent_policy and recurrent_N != 1:
            raise ValueError("JAX recurrent path currently supports recurrent_N=1 only.")

        self.env = env
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.recurrent_N = recurrent_N
        self.device = device
        self.global_reward = global_reward
        self.add_agent_id = add_agent_id
        self.use_centralized_critic = use_centralized_critic
        self.clip_delta = clip_delta
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.num_batches = max(1, num_batches)
        self.normalize_returns = normalize_returns
        self.use_recurrent_policy = use_recurrent_policy
        self.attack_action_threshold = self._infer_attack_action_threshold(action_dim)

        self.causal_lambda = causal_lambda
        self.causal_lambda_schedule = causal_lambda_schedule
        self.causal_lambda_start = causal_lambda_start
        self.causal_lambda_final = causal_lambda_final
        self.causal_lambda_warmup_steps = causal_lambda_warmup_steps
        self.causal_lambda_startingpoint = causal_lambda_startingpoint
        self.bridge_epoch = bridge_epoch
        self.causal_mini_batch_size = causal_mini_batch_size
        self.tikhonov_gamma = tikhonov_gamma
        self.stabilizer_lambda = stabilizer_lambda
        self.training_steps = 0
        self.latest_rollout: Optional[Dict[str, Any]] = None
        self._rollout_fns: Dict[int, Any] = {}

        self.policy_obs_dim = obs_dim + num_agents if add_agent_id else obs_dim
        self.actor_state_dim = hidden_dim if use_recurrent_policy else 0
        self.critic_state_dim = hidden_dim if use_recurrent_policy else 0
        self.bridge_hidden_dim = (num_agents - 1) * self.actor_state_dim
        self.bridge_output_dim = 1 if global_reward else num_agents
        self.critic_output_dim = num_agents if (use_centralized_critic and not global_reward) else 1
        self.rng = jax.random.PRNGKey(seed)

        self.actor_model = (
            RecurrentActor(hidden_dim=hidden_dim, action_dim=action_dim)
            if use_recurrent_policy
            else FeedForwardActor(hidden_dim=hidden_dim, action_dim=action_dim)
        )
        self.critic_model = (
            RecurrentCritic(hidden_dim=hidden_dim, output_dim=self.critic_output_dim)
            if use_recurrent_policy
            else FeedForwardCritic(hidden_dim=hidden_dim, output_dim=self.critic_output_dim)
        )
        self.bridge_model = (
            CausalBridgeNetwork(
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                num_agents=self.bridge_output_dim,
            )
            if causal_lambda > 0.0
            else None
        )

        self.rng, actor_key, critic_key, bridge_key = jax.random.split(self.rng, 4)

        self.actor_state = make_train_state(
            self.actor_model,
            actor_key,
            (
                jnp.zeros((1, self.policy_obs_dim), dtype=jnp.float32),
                jnp.zeros((1, action_dim), dtype=jnp.float32),
                jnp.zeros((1, self.actor_state_dim), dtype=jnp.float32),
                jnp.ones((1,), dtype=jnp.float32),
            ),
            lr_actor,
            max_grad_norm,
        )

        critic_obs_dim = self.policy_obs_dim * num_agents if use_centralized_critic else self.policy_obs_dim
        critic_prev_action_dim = action_dim * num_agents if use_centralized_critic else action_dim
        self.critic_state = make_train_state(
            self.critic_model,
            critic_key,
            (
                jnp.zeros((1, critic_obs_dim), dtype=jnp.float32),
                jnp.zeros((1, critic_prev_action_dim), dtype=jnp.float32),
                jnp.zeros((1, self.critic_state_dim), dtype=jnp.float32),
                jnp.ones((1,), dtype=jnp.float32),
            ),
            lr_critic,
            max_grad_norm,
        )

        self.bridge_state = None
        if self.bridge_model is not None:
            self.bridge_state = make_train_state(
                self.bridge_model,
                bridge_key,
                (
                    jnp.zeros((1, self.policy_obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, (num_agents - 1) * self.policy_obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, self.bridge_hidden_dim), dtype=jnp.float32),
                    jnp.zeros((1,), dtype=jnp.int32),
                ),
                lr_bridge,
                max_grad_norm,
            )

    def initialize_runner_state(self, n_envs: int) -> Dict[str, jnp.ndarray]:
        critic_shape = (
            (n_envs, self.critic_state_dim)
            if self.use_centralized_critic
            else (n_envs, self.num_agents, self.critic_state_dim)
        )
        return {
            "actor_hidden": jnp.zeros((n_envs, self.num_agents, self.actor_state_dim), dtype=jnp.float32),
            "critic_hidden": jnp.zeros(critic_shape, dtype=jnp.float32),
            "prev_actions": jnp.zeros((n_envs, self.num_agents, self.action_dim), dtype=jnp.float32),
            "prev_obs": jnp.zeros((n_envs, self.num_agents, self.policy_obs_dim), dtype=jnp.float32),
            "done_prev": jnp.zeros((n_envs,), dtype=jnp.float32),
        }

    def reset_agents(self):
        return None

    def reset_agents_batch(self, n_envs: int):
        return self.initialize_runner_state(n_envs)

    def get_current_causal_lambda(self) -> float:
        if not self.causal_lambda_schedule:
            return float(self.causal_lambda)
        if self.training_steps < self.causal_lambda_startingpoint:
            return float(self.causal_lambda_start)
        effective_steps = self.training_steps - self.causal_lambda_startingpoint
        if effective_steps >= self.causal_lambda_warmup_steps:
            return float(self.causal_lambda_final)
        progress = effective_steps / max(1, self.causal_lambda_warmup_steps)
        return float(
            self.causal_lambda_start
            + progress * (self.causal_lambda_final - self.causal_lambda_start)
        )

    def _augment_obs(self, obs: jnp.ndarray) -> jnp.ndarray:
        return add_agent_id_features(obs) if self.add_agent_id else obs

    def _infer_attack_action_threshold(self, action_dim: int) -> Optional[int]:
        env_name = self.env.__class__.__name__
        if env_name not in {'SMACWrapper', 'SMACv2Wrapper', 'SMAXWrapper', 'SMAXBatchWrapper'}:
            return None

        n_enemies = getattr(self.env, "num_enemies", None)
        if n_enemies is None:
            env_inner = getattr(self.env, "_env", None)
            if env_inner is not None:
                n_enemies = getattr(env_inner, "num_enemies", None)
                if n_enemies is None:
                    nested = getattr(env_inner, "env", None)
                    if nested is not None:
                        n_enemies = getattr(nested, "n_enemies", None)

        threshold = None
        try:
            if n_enemies is not None:
                threshold = int(action_dim) - int(n_enemies)
        except (TypeError, ValueError):
            threshold = None

        if threshold is None:
            n_actions_no_attack = getattr(self.env, "n_actions_no_attack", None)
            try:
                if n_actions_no_attack is not None:
                    threshold = int(n_actions_no_attack)
            except (TypeError, ValueError):
                threshold = None

        if threshold is None:
            threshold = 6 if action_dim > 6 else None

        if threshold is None:
            return None
        threshold = max(0, min(int(threshold), int(action_dim) - 1))
        return threshold

    def _normalize_advantage_signal(self, signal: jnp.ndarray) -> jnp.ndarray:
        if signal.size <= 1:
            return jnp.zeros_like(signal)
        std = jnp.clip(signal.std(), a_min=1.0e-8)
        return (signal - signal.mean()) / std

    def _orthogonalize_advantage_signal(
        self,
        base_signal: jnp.ndarray,
        aux_signal: jnp.ndarray,
    ) -> jnp.ndarray:
        base_centered = base_signal - base_signal.mean()
        aux_centered = aux_signal - aux_signal.mean()
        denom = jnp.dot(base_centered, base_centered)
        projection = jnp.where(
            denom > 1.0e-8,
            jnp.dot(aux_centered, base_centered) / (denom + 1.0e-8),
            0.0,
        )
        return aux_centered - projection * base_centered

    def _combine_env_and_pci_advantages(
        self,
        env_advantages: jnp.ndarray,
        pci_signal: jnp.ndarray,
        causal_lambda: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        env_norm = self._normalize_advantage_signal(env_advantages)
        pci_norm = self._normalize_advantage_signal(pci_signal)
        combined = env_norm + causal_lambda * pci_norm
        return combined, env_norm, pci_norm

    def _env_advantages_for_actor(self, env_advantages: jnp.ndarray) -> jnp.ndarray:
        if env_advantages.ndim == 2:
            return jnp.repeat(env_advantages[..., None], self.num_agents, axis=-1)
        return env_advantages

    def _combine_actor_advantages(
        self,
        env_advantages: jnp.ndarray,
        pci_advantages: jnp.ndarray,
        causal_lambda: float,
    ) -> tuple[jnp.ndarray, Dict[str, float]]:
        env_for_actor = self._env_advantages_for_actor(env_advantages)
        combined_agents = []
        env_norm_agents = []
        pci_res_agents = []

        for agent_id in range(self.num_agents):
            combined, env_norm, pci_res = self._combine_env_and_pci_advantages(
                env_for_actor[..., agent_id].reshape(-1),
                pci_advantages[..., agent_id].reshape(-1),
                causal_lambda,
            )
            combined_agents.append(combined.reshape(env_for_actor.shape[:2]))
            env_norm_agents.append(env_norm.reshape(env_for_actor.shape[:2]))
            pci_res_agents.append(pci_res.reshape(env_for_actor.shape[:2]))

        combined_adv = jnp.stack(combined_agents, axis=-1)
        env_norm = jnp.stack(env_norm_agents, axis=-1)
        pci_res = jnp.stack(pci_res_agents, axis=-1)
        info = {
            "env_adv_std": float(jnp.std(env_norm)),
            "pci_adv_res_std": float(jnp.std(pci_res)),
            "pci_bonus_std": float(jnp.std(causal_lambda * pci_res)),
            "combined_adv_std": float(jnp.std(combined_adv)),
        }
        return combined_adv, info

    def _pci_residual_tensor(
        self,
        env_advantages: jnp.ndarray,
        pci_advantages: jnp.ndarray,
        causal_lambda: float,
    ) -> jnp.ndarray:
        env_for_actor = self._env_advantages_for_actor(env_advantages)
        pci_res_agents = []
        for agent_id in range(self.num_agents):
            _, _, pci_res = self._combine_env_and_pci_advantages(
                env_for_actor[..., agent_id].reshape(-1),
                pci_advantages[..., agent_id].reshape(-1),
                causal_lambda,
            )
            pci_res_agents.append(pci_res.reshape(env_for_actor.shape[:2]))
        return jnp.stack(pci_res_agents, axis=-1)

    def _smac_attack_reward_event_metrics(
        self,
        rollout: Dict[str, jnp.ndarray],
        delta: jnp.ndarray,
        pci_advantages: jnp.ndarray,
        pci_residual: jnp.ndarray,
        causal_lambda: float,
    ) -> Dict[str, float]:
        if self.attack_action_threshold is None:
            return {}

        actions = rollout["actions"]  # (T, E, N)
        team_reward = rollout["rewards"].sum(axis=-1) / self.num_agents  # (T, E)
        reward_hit = team_reward > 1.0e-8

        if self.action_dim > 1:
            alive = (rollout["avail_actions"][..., 1:].sum(axis=-1) > 0)
        else:
            alive = jnp.ones_like(actions, dtype=bool)
        attack_intent = actions >= self.attack_action_threshold
        event_mask = attack_intent & alive & reward_hit[..., None]

        metrics: Dict[str, float] = {}
        bonus_abs = jnp.abs(causal_lambda * pci_residual)
        for agent_id in range(self.num_agents):
            mask_i = event_mask[..., agent_id].astype(jnp.float32)
            count = float(jnp.sum(mask_i))
            metrics[f"delta_agent_{agent_id}_attack_reward_count"] = count
            if count <= 0.0:
                metrics[f"delta_agent_{agent_id}_attack_reward_mean"] = 0.0
                metrics[f"pci_raw_adv_agent_{agent_id}_attack_reward_mean"] = 0.0
                metrics[f"pci_res_adv_agent_{agent_id}_attack_reward_mean"] = 0.0
                metrics[f"pci_bonus_abs_agent_{agent_id}_attack_reward_mean"] = 0.0
                continue
            denom = count + 1.0e-8
            metrics[f"delta_agent_{agent_id}_attack_reward_mean"] = float(
                jnp.sum(delta[..., agent_id] * mask_i) / denom
            )
            metrics[f"pci_raw_adv_agent_{agent_id}_attack_reward_mean"] = float(
                jnp.sum(pci_advantages[..., agent_id] * mask_i) / denom
            )
            metrics[f"pci_res_adv_agent_{agent_id}_attack_reward_mean"] = float(
                jnp.sum(pci_residual[..., agent_id] * mask_i) / denom
            )
            metrics[f"pci_bonus_abs_agent_{agent_id}_attack_reward_mean"] = float(
                jnp.sum(bonus_abs[..., agent_id] * mask_i) / denom
            )
        return metrics

    @partial(jax.jit, static_argnums=0)
    def _policy_step(
        self,
        actor_params: Dict[str, Any],
        obs: jnp.ndarray,
        avail_actions: jnp.ndarray,
        actor_hidden: jnp.ndarray,
        prev_actions: jnp.ndarray,
        done_prev: jnp.ndarray,
        rng: jax.Array,
        deterministic: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        n_envs = obs.shape[0]
        masks = jnp.broadcast_to((1.0 - done_prev)[:, None], (n_envs, self.num_agents))
        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_prev_actions = prev_actions.reshape(-1, prev_actions.shape[-1])
        flat_hidden = actor_hidden.reshape((n_envs * self.num_agents, actor_hidden.shape[-1]))
        flat_masks = masks.reshape(-1)
        flat_avail = avail_actions.reshape(-1, avail_actions.shape[-1])

        next_hidden_flat, logits_flat = self.actor_state.apply_fn(
            {"params": actor_params},
            flat_obs,
            flat_prev_actions,
            flat_hidden,
            flat_masks,
        )

        action_keys = jax.random.split(rng, flat_obs.shape[0])
        flat_actions, flat_log_probs, flat_probs = jax.vmap(
            categorical_stats,
            in_axes=(0, 0, 0, None),
        )(logits_flat, flat_avail, action_keys, deterministic)

        next_hidden = next_hidden_flat.reshape(n_envs, self.num_agents, actor_hidden.shape[-1])
        actions = flat_actions.reshape(n_envs, self.num_agents)
        log_probs = flat_log_probs.reshape(n_envs, self.num_agents)
        probs = flat_probs.reshape(n_envs, self.num_agents, self.action_dim)
        return next_hidden, actions, log_probs, probs

    @partial(jax.jit, static_argnums=0)
    def _value_step(
        self,
        critic_params: Dict[str, Any],
        obs: jnp.ndarray,
        critic_hidden: jnp.ndarray,
        prev_actions: jnp.ndarray,
        done_prev: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.use_centralized_critic:
            masks = 1.0 - done_prev
            critic_obs = obs.reshape(obs.shape[0], -1)
            critic_prev_actions = prev_actions.reshape(prev_actions.shape[0], -1)
            next_hidden, values = self.critic_state.apply_fn(
                {"params": critic_params},
                critic_obs,
                critic_prev_actions,
                critic_hidden,
                masks,
            )
            return next_hidden, values

        n_envs = obs.shape[0]
        masks = jnp.broadcast_to((1.0 - done_prev)[:, None], (n_envs, self.num_agents))
        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_prev_actions = prev_actions.reshape(-1, prev_actions.shape[-1])
        flat_hidden = critic_hidden.reshape((n_envs * self.num_agents, critic_hidden.shape[-1]))
        flat_masks = masks.reshape(-1)
        next_hidden_flat, values_flat = self.critic_state.apply_fn(
            {"params": critic_params},
            flat_obs,
            flat_prev_actions,
            flat_hidden,
            flat_masks,
        )
        return (
            next_hidden_flat.reshape(n_envs, self.num_agents, critic_hidden.shape[-1]),
            values_flat.reshape(n_envs, self.num_agents),
        )

    def _get_rollout_fn(self, env: Any, horizon: int):
        if horizon in self._rollout_fns:
            return self._rollout_fns[horizon]

        def rollout_fn(
            env_rng: jax.Array,
            batch_states: Any,
            obs: jnp.ndarray,
            avail_actions: jnp.ndarray,
            runner_state: Dict[str, jnp.ndarray],
            actor_params: Dict[str, Any],
            critic_params: Dict[str, Any],
        ):
            def body_fn(carry, _):
                rng, states, current_obs, current_avail, current_runner = carry
                rng, action_rng = jax.random.split(rng)
                obs_aug = self._augment_obs(current_obs)

                actor_hidden_prev = current_runner["actor_hidden"]
                critic_hidden_prev = current_runner["critic_hidden"]
                prev_actions_prev = current_runner["prev_actions"]
                prev_obs_prev = current_runner["prev_obs"]
                done_prev = current_runner["done_prev"]

                actor_hidden_next, actions, log_probs, probs = self._policy_step(
                    actor_params,
                    obs_aug,
                    current_avail,
                    actor_hidden_prev,
                    prev_actions_prev,
                    done_prev,
                    action_rng,
                    False,
                )
                critic_hidden_next, values = self._value_step(
                    critic_params,
                    obs_aug,
                    critic_hidden_prev,
                    prev_actions_prev,
                    done_prev,
                )

                (
                    rng,
                    next_states,
                    next_obs,
                    next_avail,
                    rewards,
                    _,
                    dones_all,
                    battle_won,
                ) = env.step_jax_pure(rng, states, actions)

                next_prev_actions = action_one_hot(actions, self.action_dim)
                next_prev_actions = next_prev_actions * (1.0 - dones_all)[:, None, None]

                next_runner = {
                    "actor_hidden": actor_hidden_next,
                    "critic_hidden": critic_hidden_next,
                    "prev_actions": next_prev_actions,
                    "prev_obs": obs_aug,
                    "done_prev": dones_all.astype(jnp.float32),
                }

                transition = {
                    "prev_obs": prev_obs_prev.astype(jnp.float32),
                    "obs": obs_aug,
                    "avail_actions": current_avail.astype(jnp.float32),
                    "actions": actions,
                    "log_probs": log_probs,
                    "probs": probs,
                    "rewards": rewards.astype(jnp.float32),
                    "dones": dones_all.astype(jnp.float32),
                    "masks": (1.0 - done_prev).astype(jnp.float32),
                    "values": values.astype(jnp.float32),
                    "actor_hidden": actor_hidden_prev.astype(jnp.float32),
                    "critic_hidden": critic_hidden_prev.astype(jnp.float32),
                    "prev_actions": prev_actions_prev.astype(jnp.float32),
                    "battle_won": battle_won.astype(jnp.float32),
                }
                next_carry = (
                    rng,
                    next_states,
                    next_obs.astype(jnp.float32),
                    next_avail.astype(jnp.float32),
                    next_runner,
                )
                return next_carry, transition

            final_carry, rollout = jax.lax.scan(body_fn, (env_rng, batch_states, obs, avail_actions, runner_state), None, length=horizon)
            rng, states, next_obs, next_avail, next_runner = final_carry
            next_obs_aug = self._augment_obs(next_obs)
            _, last_value = self._value_step(
                critic_params,
                next_obs_aug,
                next_runner["critic_hidden"],
                next_runner["prev_actions"],
                next_runner["done_prev"],
            )
            return rng, states, next_obs, next_avail, next_runner, rollout, last_value

        self._rollout_fns[horizon] = jax.jit(rollout_fn)
        return self._rollout_fns[horizon]

    def collect_rollout(
        self,
        env: Any,
        env_rng: jax.Array,
        batch_states: Any,
        obs: jnp.ndarray,
        avail_actions: jnp.ndarray,
        runner_state: Dict[str, jnp.ndarray],
        horizon: int,
    ):
        rollout_fn = self._get_rollout_fn(env, horizon)
        return rollout_fn(
            env_rng,
            batch_states,
            obs.astype(jnp.float32),
            avail_actions.astype(jnp.float32),
            runner_state,
            self.actor_state.params,
            self.critic_state.params,
        )

    def _flatten_actor_batch(self, rollout: Dict[str, jnp.ndarray], advantages: jnp.ndarray):
        total_actor_steps = rollout["obs"].shape[0] * rollout["obs"].shape[1] * rollout["obs"].shape[2]
        obs = rollout["obs"].reshape(-1, self.policy_obs_dim)
        avail_actions = rollout["avail_actions"].reshape(-1, self.action_dim)
        actions = rollout["actions"].reshape(-1)
        old_log_probs = rollout["log_probs"].reshape(-1)
        prev_actions = rollout["prev_actions"].reshape(-1, self.action_dim)
        hidden = rollout["actor_hidden"].reshape(total_actor_steps, self.actor_state_dim)
        masks = jnp.broadcast_to(
            rollout["masks"].reshape(rollout["masks"].shape + (1,)),
            rollout["actions"].shape,
        ).reshape(-1)
        if self.action_dim > 1:
            alive_masks = (rollout["avail_actions"][..., 1:].sum(axis=-1) > 0).astype(jnp.float32).reshape(-1)
        else:
            alive_masks = jnp.ones_like(actions, dtype=jnp.float32)
        loss_masks = masks * alive_masks

        if advantages.ndim == 2:
            adv = jnp.repeat(advantages[..., None], self.num_agents, axis=-1)
        else:
            adv = advantages
        adv = adv.reshape(-1)
        if adv.size > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1.0e-8)
        return {
            "obs": obs,
            "avail_actions": avail_actions,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "prev_actions": prev_actions,
            "hidden": hidden,
            "masks": masks,
            "loss_masks": loss_masks,
            "advantages": adv,
        }

    def _flatten_critic_batch(self, rollout: Dict[str, jnp.ndarray], returns: jnp.ndarray):
        if self.use_centralized_critic:
            total_steps = rollout["obs"].shape[0] * rollout["obs"].shape[1]
            base_masks = rollout["masks"].reshape(-1)
            if returns.ndim == 2:
                returns_flat = returns.reshape(-1)
                loss_masks = base_masks
            else:
                returns_flat = returns.reshape(-1, self.num_agents)
                if self.action_dim > 1:
                    alive_masks = (rollout["avail_actions"][..., 1:].sum(axis=-1) > 0).astype(jnp.float32)
                else:
                    alive_masks = jnp.ones(rollout["actions"].shape, dtype=jnp.float32)
                loss_masks = base_masks[:, None] * alive_masks.reshape(-1, self.num_agents)
            return {
                "obs": rollout["obs"].reshape(rollout["obs"].shape[0], rollout["obs"].shape[1], -1).reshape(-1, self.policy_obs_dim * self.num_agents),
                "prev_actions": rollout["prev_actions"].reshape(rollout["prev_actions"].shape[0], rollout["prev_actions"].shape[1], -1).reshape(-1, self.action_dim * self.num_agents),
                "hidden": rollout["critic_hidden"].reshape(total_steps, self.critic_state_dim),
                "masks": base_masks,
                "loss_masks": loss_masks,
                "returns": returns_flat,
            }

        total_steps = rollout["obs"].shape[0] * rollout["obs"].shape[1] * rollout["obs"].shape[2]
        per_agent_masks = jnp.broadcast_to(
            rollout["masks"].reshape(rollout["masks"].shape + (1,)),
            rollout["actions"].shape,
        ).reshape(-1)
        if self.action_dim > 1:
            alive_masks = (rollout["avail_actions"][..., 1:].sum(axis=-1) > 0).astype(jnp.float32).reshape(-1)
        else:
            alive_masks = jnp.ones((total_steps,), dtype=jnp.float32)
        return {
            "obs": rollout["obs"].reshape(-1, self.policy_obs_dim),
            "prev_actions": rollout["prev_actions"].reshape(-1, self.action_dim),
            "hidden": rollout["critic_hidden"].reshape(total_steps, self.critic_state_dim),
            "masks": per_agent_masks,
            "loss_masks": per_agent_masks * alive_masks,
            "returns": returns.reshape(-1),
        }

    def _causal_batch(self, rollout: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        time_steps, n_envs = rollout["actions"].shape[:2]
        flat_batch = time_steps * n_envs

        prev_joint_obs = rollout["prev_obs"].reshape(flat_batch, -1)
        z = jnp.repeat(prev_joint_obs, self.num_agents, axis=0)

        obs_flat = rollout["obs"].reshape(flat_batch, self.num_agents, self.policy_obs_dim)
        local_obs = obs_flat.reshape(-1, self.policy_obs_dim)
        others_obs = gather_other_agent_obs(obs_flat).reshape(-1, (self.num_agents - 1) * self.policy_obs_dim)
        if self.use_recurrent_policy:
            hidden_all = rollout["actor_hidden"].reshape(flat_batch, self.num_agents, self.actor_state_dim)
            hidden_others = gather_other_agent_obs(hidden_all).reshape(-1, self.bridge_hidden_dim)
        else:
            hidden_others = jnp.zeros((flat_batch * self.num_agents, 0), dtype=jnp.float32)

        actions = rollout["actions"].reshape(-1)
        if self.global_reward:
            team_rewards = rollout["rewards"].sum(axis=-1) / self.num_agents
            targets = jnp.repeat(team_rewards.reshape(-1), self.num_agents, axis=0)[:, None]
        else:
            targets = jnp.repeat(
                rollout["rewards"][:, :, None, :],
                self.num_agents,
                axis=2,
            ).reshape(-1, self.num_agents)

        return {
            "z": z.astype(jnp.float32),
            "w": others_obs.astype(jnp.float32),
            "x_obs": local_obs.astype(jnp.float32),
            "x_hid": hidden_others.astype(jnp.float32),
            "actions": actions.astype(jnp.int32),
            "targets": targets.astype(jnp.float32),
        }

    def _bridge_inputs(self, rollout: Dict[str, jnp.ndarray]):
        obs = rollout["obs"]
        local_obs = obs.reshape(-1, self.policy_obs_dim)
        others_obs = gather_other_agent_obs(obs).reshape(-1, (self.num_agents - 1) * self.policy_obs_dim)
        if self.use_recurrent_policy:
            hidden_others = gather_other_agent_obs(rollout["actor_hidden"]).reshape(-1, self.bridge_hidden_dim)
        else:
            hidden_others = jnp.zeros((local_obs.shape[0], 0), dtype=jnp.float32)
        actions = rollout["actions"].reshape(-1)
        if self.global_reward:
            team_rewards = rollout["rewards"].sum(axis=-1) / self.num_agents
            reward_targets = jnp.repeat(team_rewards.reshape(-1), self.num_agents, axis=0)[:, None]
        else:
            reward_targets = jnp.repeat(
                rollout["rewards"][:, :, None, :],
                self.num_agents,
                axis=2,
            ).reshape(-1, self.num_agents)
        probs = rollout["probs"].reshape(-1, self.action_dim)
        avail_actions = rollout["avail_actions"].reshape(-1, self.action_dim)
        return local_obs, others_obs, hidden_others, actions, reward_targets, probs, avail_actions

    @partial(jax.jit, static_argnums=0)
    def _bridge_predictions(
        self,
        bridge_params: Dict[str, Any],
        local_obs: jnp.ndarray,
        others_obs: jnp.ndarray,
        hidden_others: jnp.ndarray,
        probs: jnp.ndarray,
        actions: jnp.ndarray,
        avail_actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        actual_pred = self.bridge_state.apply_fn(
            {"params": bridge_params},
            local_obs,
            others_obs,
            hidden_others,
            actions,
        )

        all_actions = jnp.arange(self.action_dim, dtype=jnp.int32)

        def predict_one_action(action_id: jnp.ndarray):
            tiled = jnp.full((local_obs.shape[0],), action_id, dtype=jnp.int32)
            return self.bridge_state.apply_fn(
                {"params": bridge_params},
                local_obs,
                others_obs,
                hidden_others,
                tiled,
            )

        all_preds = jax.vmap(predict_one_action)(all_actions).transpose(1, 0, 2)
        factual_mask = 1.0 - jax.nn.one_hot(actions.astype(jnp.int32), self.action_dim, dtype=jnp.float32)
        probs_counterfactual = probs * factual_mask * avail_actions
        renorm = probs_counterfactual.sum(axis=-1, keepdims=True)
        renormalized_probs = probs_counterfactual / jnp.clip(renorm, a_min=1.0e-8)
        baseline = jnp.sum(renormalized_probs[..., None] * all_preds, axis=1)

        no_valid_cf = renorm.squeeze(-1) <= 1.0e-8
        baseline = jnp.where(no_valid_cf[:, None], actual_pred, baseline)
        return actual_pred, baseline, all_preds

    @partial(jax.jit, static_argnums=0)
    def _actor_update_step(
        self,
        actor_state: Any,
        batch: Dict[str, jnp.ndarray],
    ):
        def loss_fn(params):
            _, logits = actor_state.apply_fn(
                {"params": params},
                batch["obs"],
                batch["prev_actions"],
                batch["hidden"],
                batch["masks"],
            )
            _, new_log_probs, _ = categorical_stats(
                logits,
                batch["avail_actions"],
                jax.random.PRNGKey(0),
                True,
            )
            log_probs = new_log_probs
            masked_logits = jnp.where(batch["avail_actions"] > 0, logits, -1.0e9)
            probs = jax.nn.softmax(masked_logits, axis=-1)
            selected = jnp.take_along_axis(probs, batch["actions"][:, None], axis=-1).squeeze(-1)
            log_probs = jnp.log(jnp.clip(selected, a_min=1.0e-8))
            entropy = -jnp.sum(probs * jnp.log(jnp.clip(probs, a_min=1.0e-8)), axis=-1)

            ratio = jnp.exp(log_probs - batch["old_log_probs"])
            unclipped = ratio * batch["advantages"]
            clipped = jnp.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch["advantages"]
            weights = batch["loss_masks"]
            denom = jnp.sum(weights) + 1.0e-8
            policy_loss = -jnp.sum(jnp.minimum(unclipped, clipped) * weights) / denom
            entropy_bonus = jnp.sum(entropy * weights) / denom
            total_loss = policy_loss - self.entropy_coef * entropy_bonus
            approx_kl = jnp.sum((batch["old_log_probs"] - log_probs) * weights) / denom
            return total_loss, {
                "policy_loss": policy_loss,
                "entropy": entropy_bonus,
                "approx_kl": approx_kl,
            }

        grads, metrics = jax.grad(loss_fn, has_aux=True)(actor_state.params)
        new_actor_state = actor_state.apply_gradients(grads=grads)
        metrics["actor_grad_norm"] = global_norm(grads)
        return new_actor_state, metrics

    @partial(jax.jit, static_argnums=0)
    def _critic_update_step(
        self,
        critic_state: Any,
        batch: Dict[str, jnp.ndarray],
    ):
        def loss_fn(params):
            _, values = critic_state.apply_fn(
                {"params": params},
                batch["obs"],
                batch["prev_actions"],
                batch["hidden"],
                batch["masks"],
            )
            weights = batch["loss_masks"]
            if values.ndim == 1:
                denom = jnp.sum(weights) + 1.0e-8
                value_loss = jnp.sum(jnp.square(values - batch["returns"]) * weights) / denom
            else:
                denom = jnp.sum(weights) + 1.0e-8
                value_loss = jnp.sum(jnp.square(values - batch["returns"]) * weights) / denom
            return self.value_coef * value_loss, {"value_loss": value_loss}

        grads, metrics = jax.grad(loss_fn, has_aux=True)(critic_state.params)
        new_critic_state = critic_state.apply_gradients(grads=grads)
        metrics["critic_grad_norm"] = global_norm(grads)
        return new_critic_state, metrics

    @partial(jax.jit, static_argnums=0)
    def _bridge_update_step(
        self,
        bridge_state: Any,
        batch: Dict[str, jnp.ndarray],
    ):
        def _standardize(x: jnp.ndarray) -> jnp.ndarray:
            mean = jnp.mean(x, axis=0, keepdims=True)
            std = jnp.std(x, axis=0, keepdims=True)
            return (x - mean) / jnp.clip(std, a_min=1.0e-6)

        def _rbf_gram(x: jnp.ndarray, gamma: float) -> jnp.ndarray:
            x_norm = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
            dists_sq = x_norm - 2.0 * (x @ x.T) + x_norm.T
            dists_sq = jnp.clip(dists_sq, a_min=0.0)
            return jnp.exp(-gamma * dists_sq)

        def loss_fn(params):
            preds = bridge_state.apply_fn(
                {"params": params},
                batch["x_obs"],
                batch["w"],
                batch["x_hid"],
                batch["actions"],
            )
            residuals = batch["targets"] - preds

            z_norm = _standardize(batch["z"])
            x_obs_norm = _standardize(batch["x_obs"])
            action_onehot = jax.nn.one_hot(batch["actions"], self.action_dim, dtype=jnp.float32)

            gamma_z = 1.0 / max(1, batch["z"].shape[-1])
            gamma_a = 1.0 / max(1, self.action_dim)
            gamma_xobs = 1.0 / max(1, batch["x_obs"].shape[-1])

            gram = (
                _rbf_gram(z_norm, gamma_z)
                * _rbf_gram(action_onehot, gamma_a)
                * _rbf_gram(x_obs_norm, gamma_xobs)
            )
            if batch["x_hid"].shape[-1] > 0:
                x_hid_norm = _standardize(batch["x_hid"])
                gamma_xhid = 1.0 / max(1, batch["x_hid"].shape[-1])
                gram = gram * _rbf_gram(x_hid_norm, gamma_xhid)
            eye = jnp.eye(gram.shape[0], dtype=gram.dtype)
            reg_gram = (self.tikhonov_gamma + 1.0e-6) * eye + self.stabilizer_lambda * gram
            solved = jnp.linalg.solve(reg_gram, residuals)
            per_sample_loss = jnp.sum(residuals * (gram @ solved), axis=-1)
            outcome_bridge_loss = jnp.mean(per_sample_loss)
            action_bridge_loss = jnp.array(0.0, dtype=jnp.float32)
            diag = jnp.diag(gram)
            total_sum = jnp.sum(gram) - jnp.sum(diag)
            off_count = gram.shape[0] * max(gram.shape[0] - 1, 1)
            kz_offdiag_mean = total_sum / off_count
            centered = gram - kz_offdiag_mean
            sq_sum = jnp.sum(jnp.square(centered)) - jnp.sum(jnp.square(diag - kz_offdiag_mean))
            kz_offdiag_std = jnp.sqrt(jnp.clip(sq_sum / off_count, a_min=0.0))
            return outcome_bridge_loss + action_bridge_loss, {
                "outcome_bridge_loss": outcome_bridge_loss,
                "action_bridge_loss": action_bridge_loss,
                "Kz_offdiag_mean": kz_offdiag_mean,
                "Kz_offdiag_std": kz_offdiag_std,
            }

        grads, metrics = jax.grad(loss_fn, has_aux=True)(bridge_state.params)
        new_bridge_state = bridge_state.apply_gradients(grads=grads)
        metrics["bridge_grad_norm"] = global_norm(grads)
        return new_bridge_state, metrics

    def _iterate_minibatches(self, size: int, rng: jax.Array):
        perm = jax.random.permutation(rng, size)
        effective_batch = self.batch_size if self.batch_size > 0 else math.ceil(size / self.num_batches)
        for start in range(0, size, effective_batch):
            yield perm[start:start + effective_batch]

    def update_causal_networks(self, rollout: Optional[Dict[str, jnp.ndarray]] = None) -> Dict[str, float]:
        if rollout is None:
            rollout = self.latest_rollout
        if rollout is None:
            return {}
        if self.get_current_causal_lambda() == 0.0:
            return {"outcome_bridge_loss": 0.0, "action_bridge_loss": 0.0}
        if self.bridge_state is None:
            return {"outcome_bridge_loss": 0.0, "action_bridge_loss": 0.0}

        causal_batch = self._causal_batch(rollout)
        metrics = []
        for epoch in range(self.bridge_epoch):
            self.rng, perm_key = jax.random.split(self.rng)
            perm = jax.random.permutation(perm_key, causal_batch["actions"].shape[0])
            for start in range(0, causal_batch["actions"].shape[0], self.causal_mini_batch_size):
                indices = perm[start:start + self.causal_mini_batch_size]
                batch = {
                    "z": causal_batch["z"][indices],
                    "w": causal_batch["w"][indices],
                    "x_obs": causal_batch["x_obs"][indices],
                    "x_hid": causal_batch["x_hid"][indices],
                    "actions": causal_batch["actions"][indices],
                    "targets": causal_batch["targets"][indices],
                }
                self.bridge_state, batch_metrics = self._bridge_update_step(self.bridge_state, batch)
                metrics.append({key: float(value) for key, value in tree_to_host(batch_metrics).items()})
        stacked = stack_metrics(metrics)
        stacked.setdefault("outcome_bridge_loss", 0.0)
        stacked.setdefault("action_bridge_loss", 0.0)
        stacked.setdefault("Kz_offdiag_mean", 0.0)
        stacked.setdefault("Kz_offdiag_std", 0.0)
        return stacked

    def compute_causal_rewards(
        self,
        rollout: Dict[str, jnp.ndarray],
        baseline_type: str = "weighted",
    ) -> tuple[jnp.ndarray, Dict[str, float]]:
        if baseline_type != "weighted":
            raise ValueError(f"Unsupported baseline_type for JAXRPCIAlgorithm: {baseline_type}")
        if self.bridge_state is None or self.get_current_causal_lambda() == 0.0:
            zeros = jnp.zeros(rollout["actions"].shape + (self.num_agents,), dtype=jnp.float32)
            return zeros, {
                "causal_lambda": self.get_current_causal_lambda(),
                "delta_mean": 0.0,
                "delta_std": 0.0,
            }

        local_obs, others_obs, hidden_others, actions, reward_targets, probs, avail_actions = self._bridge_inputs(rollout)
        actual_pred, baseline, _ = self._bridge_predictions(
            self.bridge_state.params,
            local_obs,
            others_obs,
            hidden_others,
            probs,
            actions,
            avail_actions,
        )
        actual_team_reward = reward_targets
        # raw_effects = actual_pred - baseline
        raw_effects = actual_team_reward - baseline

        if self.action_dim > 1:
            alive = (rollout["avail_actions"][..., 1:].sum(axis=-1) > 0).astype(jnp.float32)
        else:
            alive = jnp.ones_like(rollout["actions"], dtype=jnp.float32)
        if self.global_reward:
            actor_team_effects = raw_effects.reshape(rollout["actions"].shape + (1,))
            alive_counts = jnp.clip(alive.sum(axis=-1, keepdims=True), a_min=1.0)
            recipient_weights = alive[..., None, :] / alive_counts[..., None, :]
            c_matrix = actor_team_effects * recipient_weights
            c_matrix = c_matrix * alive[..., :, None]
        else:
            c_matrix = raw_effects.reshape(rollout["actions"].shape + (self.num_agents,))
            c_matrix = c_matrix * alive[..., :, None] * alive[..., None, :]

        delta = c_matrix.sum(axis=-1)
        info = {
            "causal_lambda": self.get_current_causal_lambda(),
            "delta_mean": float(jnp.mean(delta)),
            "delta_std": float(jnp.std(delta)),
        }
        return c_matrix, info

    def _update_actor_critic(
        self,
        rollout: Dict[str, jnp.ndarray],
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
    ) -> Dict[str, float]:
        actor_batch = self._flatten_actor_batch(rollout, advantages)
        critic_batch = self._flatten_critic_batch(rollout, returns)

        metrics = []
        for epoch in range(self.ppo_epochs):
            self.rng, actor_key, critic_key = jax.random.split(self.rng, 3)

            for indices in self._iterate_minibatches(actor_batch["actions"].shape[0], actor_key):
                actor_mb = {key: value[indices] for key, value in actor_batch.items()}
                self.actor_state, actor_metrics = self._actor_update_step(self.actor_state, actor_mb)
                metrics.append({key: float(value) for key, value in tree_to_host(actor_metrics).items()})

            for indices in self._iterate_minibatches(critic_batch["returns"].shape[0], critic_key):
                critic_mb = {key: value[indices] for key, value in critic_batch.items()}
                self.critic_state, critic_metrics = self._critic_update_step(self.critic_state, critic_mb)
                metrics.append({key: float(value) for key, value in tree_to_host(critic_metrics).items()})

        return stack_metrics(metrics)

    def update(self, **_: Any) -> Dict[str, float]:
        if self.latest_rollout is None:
            return {}

        rollout = self.latest_rollout
        metrics = {}
        causal_lambda = self.get_current_causal_lambda()

        bridge_metrics = self.update_causal_networks(rollout)
        metrics.update(bridge_metrics)

        c_matrix, pci_metrics = self.compute_causal_rewards(rollout, baseline_type="weighted")
        delta = c_matrix.sum(axis=-1)
        delta = jnp.where(jnp.abs(delta) < self.clip_delta, 0.0, delta)
        metrics.update(pci_metrics)

        raw_team_rewards = rollout["rewards"].sum(axis=-1) / self.num_agents
        values = rollout["values"]
        last_value = rollout["last_value"]

        if self.global_reward:
            if self.use_centralized_critic:
                env_rewards = raw_team_rewards
            else:
                env_rewards = jnp.repeat(raw_team_rewards[..., None], self.num_agents, axis=-1)
        else:
            env_rewards = rollout["rewards"]

        env_advantages, returns = compute_gae(
            env_rewards,
            values,
            last_value,
            rollout["dones"],
            self.gamma,
            self.gae_lambda,
        )

        zero_values = jnp.zeros_like(delta)
        zero_last_value = jnp.zeros_like(delta[0])
        pci_advantages, _ = compute_gae(
            delta,
            zero_values,
            zero_last_value,
            rollout["dones"],
            self.gamma,
            self.gae_lambda,
        )

        combined_advantages, combine_metrics = self._combine_actor_advantages(
            env_advantages,
            pci_advantages,
            causal_lambda,
        )
        metrics.update(combine_metrics)
        pci_residual = self._pci_residual_tensor(env_advantages, pci_advantages, causal_lambda)
        metrics.update(
            self._smac_attack_reward_event_metrics(
                rollout,
                delta,
                pci_advantages,
                pci_residual,
                causal_lambda,
            )
        )

        metrics.update(self._update_actor_critic(rollout, combined_advantages, returns))
        self.training_steps += int(rollout["actions"].shape[0] * rollout["actions"].shape[1])
        self.latest_rollout = None
        return metrics

    def greedy_actions(
        self,
        obs: jnp.ndarray,
        avail_actions: jnp.ndarray,
        runner_state: Dict[str, jnp.ndarray],
    ) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        obs_aug = self._augment_obs(obs)
        self.rng, action_rng = jax.random.split(self.rng)
        actor_hidden_next, actions, _, _ = self._policy_step(
            self.actor_state.params,
            obs_aug,
            avail_actions,
            runner_state["actor_hidden"],
            runner_state["prev_actions"],
            runner_state["done_prev"],
            action_rng,
            True,
        )
        next_runner = dict(runner_state)
        next_runner["actor_hidden"] = actor_hidden_next
        return actions, next_runner

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor_state": serialization.to_state_dict(self.actor_state),
            "critic_state": serialization.to_state_dict(self.critic_state),
            "bridge_state": serialization.to_state_dict(self.bridge_state) if self.bridge_state is not None else None,
            "rng": tree_to_host(self.rng),
            "training_steps": self.training_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor_state = serialization.from_state_dict(self.actor_state, state_dict["actor_state"])
        self.critic_state = serialization.from_state_dict(self.critic_state, state_dict["critic_state"])
        if self.bridge_state is not None and state_dict.get("bridge_state") is not None:
            self.bridge_state = serialization.from_state_dict(self.bridge_state, state_dict["bridge_state"])
        self.rng = jnp.asarray(state_dict["rng"])
        self.training_steps = int(state_dict.get("training_steps", 0))
