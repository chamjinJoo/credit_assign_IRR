from __future__ import annotations

import math
from functools import partial
from typing import Any, Dict, Optional

import flax.serialization as serialization
import jax
import jax.numpy as jnp

from .common import (
    CausalBridgeNetwork,
    DeltaCritic,
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
        causal_mini_batch_size: int = 10000,
        tikhonov_gamma: float = 1.0e-4,
        stabilizer_lambda: float = 1.0,
        normalize_returns: bool = True,
        use_recurrent_policy: bool = True,
        print_bridge_predictions: bool = False,
        bridge_prediction_print_samples: int = 1,
        bridge_prediction_print_interval: int = 50,
        normalize_pci_signal: bool = False,
        use_delta_critic: bool = False,
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
        self.print_bridge_predictions = bool(print_bridge_predictions)
        self.bridge_prediction_print_samples = max(1, int(bridge_prediction_print_samples))
        self.bridge_prediction_print_interval = max(1, int(bridge_prediction_print_interval))
        self.attack_action_threshold = self._infer_attack_action_threshold(action_dim)
        # SMAX: stop (attack_threshold-1) is always available even for dead agents.
        # SMAC: action 0 (no-op) is the dead-only action.
        env_name = self.env.__class__.__name__
        print(f"Environment name: {env_name}")
        if env_name in ('SMAXBatchWrapper', 'SMAXWrapper') and self.attack_action_threshold is not None:
            self._dead_only_action_idx: Optional[int] = self.attack_action_threshold - 1
        else:
            self._dead_only_action_idx = None  # use avail[1:].sum() fallback
        allowed_target_matrix = getattr(self.env, "_allowed_target_matrix_jax", None)
        if allowed_target_matrix is None:
            allowed_target_matrix = getattr(self.env, "_allowed_target_matrix_np", None)
        self._allowed_target_matrix = (
            None if allowed_target_matrix is None else jnp.asarray(allowed_target_matrix, dtype=jnp.float32)
        )

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
        self._bridge_prediction_debug_lines: list[str] = []
        self._rollout_fns: Dict[int, Any] = {}

        self.policy_obs_dim = obs_dim + num_agents if add_agent_id else obs_dim
        self.actor_state_dim = hidden_dim if use_recurrent_policy else 0
        self.critic_state_dim = hidden_dim if use_recurrent_policy else 0
        self.normalize_pci_signal = normalize_pci_signal
        self.bridge_hidden_dim = (num_agents - 1) * self.actor_state_dim
        self.use_delta_critic = use_delta_critic
        print("normalize_pci_signal:", normalize_pci_signal)
        print("use_delta_critic:", use_delta_critic)
        self.bridge_output_dim = 1 if global_reward else num_agents
        self.critic_output_dim = num_agents if (use_centralized_critic and not global_reward) else 1
        self.rng = jax.random.PRNGKey(seed)
        self.rng, self.bridge_rng = jax.random.split(self.rng)

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
            # others_obs dim: mean-pooled → policy_obs_dim; concat → (N-1)*policy_obs_dim
            bridge_others_obs_dim = (num_agents - 1) * self.policy_obs_dim
            self.bridge_state = make_train_state(
                self.bridge_model,
                bridge_key,
                (
                    jnp.zeros((1, self.policy_obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, bridge_others_obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, self.bridge_hidden_dim), dtype=jnp.float32),
                    jnp.zeros((1,), dtype=jnp.int32),
                ),
                lr_bridge,
                max_grad_norm,
            )

        self.delta_critic_state = None
        if use_delta_critic:
            delta_critic_input_dim = self.policy_obs_dim + self.actor_state_dim
            self.delta_critic_model = DeltaCritic(
                hidden_dim=hidden_dim,
                input_dim=delta_critic_input_dim,
            )
            self.rng, delta_critic_key = jax.random.split(self.rng)
            self.delta_critic_state = make_train_state(
                self.delta_critic_model,
                delta_critic_key,
                (
                    jnp.zeros((1, self.policy_obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, self.actor_state_dim), dtype=jnp.float32),
                ),
                lr_critic,
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

    def _alive_from_avail(self, avail_actions: jnp.ndarray) -> jnp.ndarray:
        """Return boolean alive mask.

        SMAX: stop action (index = attack_action_threshold - 1) is always
        available even for dead agents, so exclude it from the alive check.
        SMAC / default: action 0 is the dead-only no-op, so avail[1:].sum()>0 works.
        """
        if self._dead_only_action_idx is not None:
            stop = avail_actions[..., self._dead_only_action_idx]
            return (avail_actions.sum(axis=-1) - stop) > 0
        return avail_actions[..., 1:].sum(axis=-1) > 0

    def _mask_obs_keep_agent_id(
        self,
        obs: jnp.ndarray,
        alive_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        alive_f = alive_mask.astype(obs.dtype)[..., None]
        if not self.add_agent_id:
            return obs * alive_f
        obs_cont = obs[..., :self.obs_dim] * alive_f
        obs_id = obs[..., self.obs_dim:]
        return jnp.concatenate([obs_cont, obs_id], axis=-1)

    def _other_obs_for_bridge(
        self,
        obs: jnp.ndarray,
        alive_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        masked_obs = self._mask_obs_keep_agent_id(obs, alive_mask)

        return gather_other_agent_obs(masked_obs).reshape(
            obs.shape[:-1] + ((self.num_agents - 1) * self.policy_obs_dim,)
        )

    def _other_hidden_for_bridge(
        self,
        hidden: jnp.ndarray,
        alive_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        if hidden.shape[-1] == 0:
            return hidden
        alive_f = alive_mask.astype(hidden.dtype)[..., None]
        masked_hidden = hidden * alive_f

        return gather_other_agent_obs(masked_hidden).reshape(
            hidden.shape[:-1] + ((self.num_agents - 1) * hidden.shape[-1],)
        )

    def _normalize_advantage_signal(self, signal: jnp.ndarray) -> jnp.ndarray:
        if signal.size <= 1:
            return jnp.zeros_like(signal)
        std = jnp.clip(signal.std(), a_min=1.0e-8)
        return (signal - signal.mean()) / std

    def _normalize_actor_advantages(
        self,
        advantages: jnp.ndarray,
        alive_masks: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if advantages.shape[0] <= 1:
            return jnp.zeros_like(advantages)
        if alive_masks is not None:
            # Compute mean/std only over alive agent steps so dead agents'
            # near-zero advantages do not distort the normalization statistics.
            w = alive_masks.astype(jnp.float32)          # (T*E, N)
            counts = w.sum(axis=0, keepdims=True)         # (1, N)
            safe_counts = jnp.maximum(counts, 1.0)
            mean = (advantages * w).sum(axis=0, keepdims=True) / safe_counts
            centered = advantages - mean
            sq_sum = (jnp.square(centered) * w).sum(axis=0, keepdims=True)
            denom = jnp.maximum(counts - 1.0, 1.0)       # Bessel correction
        else:
            centered = advantages - advantages.mean(axis=0, keepdims=True)
            sq_sum = jnp.sum(jnp.square(centered), axis=0, keepdims=True)
            denom = jnp.maximum(advantages.shape[0] - 1, 1)
        std = jnp.sqrt(jnp.clip(sq_sum / denom, a_min=1.0e-8))
        return centered / std

    def _env_advantages_for_actor(self, env_advantages: jnp.ndarray) -> jnp.ndarray:
        if env_advantages.ndim == 2:
            return jnp.repeat(env_advantages[..., None], self.num_agents, axis=-1)
        return env_advantages

    def _normalize_columns(self, x: jnp.ndarray) -> jnp.ndarray:
        """Normalize each column (agent) of x:(T*E, N) independently — same as
        calling _normalize_advantage_signal per agent then stacking."""
        mean = x.mean(axis=0, keepdims=True)
        std = jnp.clip(x.std(axis=0, keepdims=True), a_min=1.0e-8)
        return (x - mean) / std

    def _combine_actor_advantages(
        self,
        env_advantages: jnp.ndarray,
        pci_advantages: jnp.ndarray,
        causal_lambda: float,
    ) -> tuple[jnp.ndarray, Dict[str, float]]:
        env_for_actor = self._env_advantages_for_actor(env_advantages)
        orig_shape = env_for_actor.shape  # (T, E, N)
        env_flat = env_for_actor.reshape(-1, self.num_agents)   # (T*E, N)
        pci_flat = pci_advantages.reshape(-1, self.num_agents)

        # normalize_pci_signal=True : per-agent normalize before combining so that
        #   λ is a scale-invariant coefficient (N-robust, bridge-convergence-robust).
        # normalize_pci_signal=False: raw pci_flat added directly  [original]
        pci_for_combine = (
            self._normalize_columns(pci_flat) if self.normalize_pci_signal else pci_flat
        )
        combined = (env_flat + causal_lambda * pci_for_combine).reshape(orig_shape)

        info = {
            "env_adv_std": float(jnp.std(env_flat)),
            "pci_adv_res_std": float(jnp.std(pci_for_combine)),
            "pci_bonus_std": float(jnp.std(causal_lambda * pci_for_combine)),
            "combined_adv_std": float(jnp.std(combined)),
        }
        return combined, info

    def _pci_residual_tensor(
        self,
        env_advantages: jnp.ndarray,
        pci_advantages: jnp.ndarray,
        causal_lambda: float,
    ) -> jnp.ndarray:
        env_for_actor = self._env_advantages_for_actor(env_advantages)
        orig_shape = env_for_actor.shape  # (T, E, N)
        pci_flat = pci_advantages.reshape(-1, self.num_agents)
        pci_res  = self._normalize_columns(pci_flat)
        return pci_res.reshape(orig_shape)

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
            alive = (self._alive_from_avail(rollout["avail_actions"]))
        else:
            alive = jnp.ones_like(actions, dtype=bool)
        attack_intent = actions >= self.attack_action_threshold
        event_mask = attack_intent & alive & reward_hit[..., None]

        bonus_abs = jnp.abs(causal_lambda * pci_residual)
        # Vectorize over agent dim: sum over (T, E) axes.
        mask_f = event_mask.astype(jnp.float32)                        # (T, E, N)
        counts   = mask_f.sum(axis=(0, 1))                             # (N,)
        denoms   = counts + 1.0e-8
        delta_means   = (delta        * mask_f).sum(axis=(0, 1)) / denoms
        pci_means     = (pci_advantages * mask_f).sum(axis=(0, 1)) / denoms
        res_means     = (pci_residual   * mask_f).sum(axis=(0, 1)) / denoms
        bonus_means   = (bonus_abs      * mask_f).sum(axis=(0, 1)) / denoms
        # Zero out agents with no events (count == 0 ⟹ numerator == 0 already, but be explicit).
        has_events = counts > 0
        delta_means = jnp.where(has_events, delta_means, 0.0)
        pci_means   = jnp.where(has_events, pci_means,   0.0)
        res_means   = jnp.where(has_events, res_means,   0.0)
        bonus_means = jnp.where(has_events, bonus_means, 0.0)
        # Move to host once, then build the dict.
        counts_h      = [float(v) for v in counts]
        delta_means_h = [float(v) for v in delta_means]
        pci_means_h   = [float(v) for v in pci_means]
        res_means_h   = [float(v) for v in res_means]
        bonus_means_h = [float(v) for v in bonus_means]
        metrics: Dict[str, float] = {}
        for agent_id in range(self.num_agents):
            metrics[f"delta_agent_{agent_id}_attack_reward_count"] = counts_h[agent_id]
            metrics[f"delta_agent_{agent_id}_attack_reward_mean"]  = delta_means_h[agent_id]
            metrics[f"pci_raw_adv_agent_{agent_id}_attack_reward_mean"] = pci_means_h[agent_id]
            metrics[f"pci_res_adv_agent_{agent_id}_attack_reward_mean"] = res_means_h[agent_id]
            metrics[f"pci_bonus_abs_agent_{agent_id}_attack_reward_mean"] = bonus_means_h[agent_id]
        return metrics

    def _per_action_delta_metrics(
        self,
        rollout: Dict[str, jnp.ndarray],
        delta: jnp.ndarray,
    ) -> Dict[str, float]:
        """Log mean delta per (agent, action) for all alive steps."""
        actions = rollout["actions"]  # (T, E, N)
        if self.action_dim > 1:
            alive = self._alive_from_avail(rollout["avail_actions"])  # (T, E, N)
        else:
            alive = jnp.ones_like(actions, dtype=bool)

        # one_hot: (T, E, N, A); alive_f: (T, E, N, 1)
        one_hot = jax.nn.one_hot(actions, self.action_dim, dtype=jnp.float32)
        alive_f = alive.astype(jnp.float32)[..., None]
        # mask[t,e,n,a] = 1 iff agent n was alive and took action a
        mask = one_hot * alive_f                                       # (T, E, N, A)
        counts     = mask.sum(axis=(0, 1))                             # (N, A)
        delta_sums = (delta[..., None] * mask).sum(axis=(0, 1))       # (N, A)
        means = jnp.where(counts > 0, delta_sums / jnp.maximum(counts, 1.0e-8), 0.0)
        # Move to host once.
        counts_h = [[float(counts[n, a]) for a in range(self.action_dim)] for n in range(self.num_agents)]
        means_h  = [[float(means[n, a])  for a in range(self.action_dim)] for n in range(self.num_agents)]
        metrics: Dict[str, float] = {}
        for agent_id in range(self.num_agents):
            for a in range(self.action_dim):
                key_prefix = f"delta_agent{agent_id}_action{a}"
                metrics[f"{key_prefix}_count"] = counts_h[agent_id][a]
                metrics[f"{key_prefix}_mean"]  = means_h[agent_id][a]
        return metrics

    def _categorized_action_delta_metrics(
        self,
        rollout: Dict[str, jnp.ndarray],
        delta: jnp.ndarray,
    ) -> Dict[str, float]:
        """Per-agent delta split into 3 action categories for SMAX target allocation.

        target_attack    : action >= attack_threshold AND chosen enemy is within
                           the agent's allowed target set.
        non_target_attack: action >= attack_threshold AND chosen enemy is outside
                           the agent's allowed target set.
        other_action     : action < attack_threshold
                           (move / stop)

        If the environment does not expose an allowed-target matrix, fall back to
        the older avail-action based categorisation.
        """
        if self.attack_action_threshold is None:
            return {}

        actions       = rollout["actions"]        # (T, E, N)
        avail_actions = rollout["avail_actions"]  # (T, E, N, action_dim)
        thr = self.attack_action_threshold

        if self.action_dim > 1:
            alive_f = self._alive_from_avail(avail_actions).astype(jnp.float32)  # (T, E, N)
        else:
            alive_f = jnp.ones_like(actions, dtype=jnp.float32)

        is_attack = (actions >= thr).astype(jnp.float32)               # (T, E, N)

        if self._allowed_target_matrix is not None:
            num_targets = self._allowed_target_matrix.shape[-1]
            attack_targets = jnp.clip(actions - thr, 0, num_targets - 1)
            chosen_target_one_hot = jax.nn.one_hot(
                attack_targets.astype(jnp.int32),
                num_targets,
                dtype=jnp.float32,
            )
            chosen_target_allowed = jnp.sum(
                chosen_target_one_hot * self._allowed_target_matrix[None, None, :, :],
                axis=-1,
            )
            target_mask    = alive_f * is_attack * (chosen_target_allowed > 0).astype(jnp.float32)
            nontarget_mask = alive_f * is_attack * (chosen_target_allowed <= 0).astype(jnp.float32)
        else:
            # Fallback for environments without explicit target-allocation metadata.
            action_avail = jnp.take_along_axis(
                avail_actions, actions[..., None].astype(jnp.int32), axis=-1
            ).squeeze(-1)                                               # (T, E, N)
            target_mask    = alive_f * is_attack * (action_avail > 0).astype(jnp.float32)
            nontarget_mask = alive_f * is_attack * (action_avail == 0).astype(jnp.float32)

        other_mask      = alive_f * (1.0 - is_attack)

        categories = [
            ("target_attack",     target_mask),
            ("non_target_attack", nontarget_mask),
            ("other_action",      other_mask),
        ]

        # Compute per-agent counts and means vectorised over (T, E), then move to host once.
        all_counts = {}
        all_means  = {}
        for cat_name, cat_mask in categories:
            counts    = cat_mask.sum(axis=(0, 1))                         # (N,)
            delta_sum = (delta * cat_mask).sum(axis=(0, 1))               # (N,)
            means     = jnp.where(counts > 0, delta_sum / jnp.maximum(counts, 1.0e-8), 0.0)
            all_counts[cat_name] = [float(v) for v in tree_to_host(counts)]
            all_means[cat_name]  = [float(v) for v in tree_to_host(means)]

        metrics: Dict[str, float] = {}
        for agent_id in range(self.num_agents):
            for cat_name, _ in categories:
                metrics[f"delta_agent{agent_id}_{cat_name}_count"] = all_counts[cat_name][agent_id]
                metrics[f"delta_agent{agent_id}_{cat_name}_mean"]  = all_means[cat_name][agent_id]
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

        sample_ids = jnp.arange(flat_obs.shape[0], dtype=jnp.uint32)
        action_keys = jax.vmap(lambda idx: jax.random.fold_in(rng, idx))(sample_ids)
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

    @partial(jax.jit, static_argnums=0)
    def _delta_value_step(
        self,
        delta_critic_params: Dict[str, Any],
        obs: jnp.ndarray,           # (..., policy_obs_dim)
        actor_hidden: jnp.ndarray,  # (..., actor_state_dim)
    ) -> jnp.ndarray:               # (...)
        orig_shape = obs.shape[:-1]
        flat_obs = obs.reshape(-1, self.policy_obs_dim)
        if self.actor_state_dim > 0:
            flat_hidden = actor_hidden.reshape(-1, self.actor_state_dim)
        else:
            flat_hidden = jnp.zeros((flat_obs.shape[0], 0), dtype=jnp.float32)
        values_flat = self.delta_critic_state.apply_fn(
            {"params": delta_critic_params}, flat_obs, flat_hidden
        )
        return values_flat.reshape(orig_shape)

    @partial(jax.jit, static_argnums=0)
    def _compute_delta_values(
        self,
        delta_critic_params: Dict[str, Any],
        obs: jnp.ndarray,           # (T, E, N, policy_obs_dim)
        actor_hidden: jnp.ndarray,  # (T, E, N, actor_state_dim)
    ) -> jnp.ndarray:               # (T, E, N)
        T, E, N = obs.shape[:3]
        flat_obs = obs.reshape(T * E * N, self.policy_obs_dim)
        if self.actor_state_dim > 0:
            flat_hidden = actor_hidden.reshape(T * E * N, self.actor_state_dim)
        else:
            flat_hidden = jnp.zeros((T * E * N, 0), dtype=jnp.float32)
        values_flat = self.delta_critic_state.apply_fn(
            {"params": delta_critic_params}, flat_obs, flat_hidden
        )
        return values_flat.reshape(T, E, N)

    @partial(jax.jit, static_argnums=0)
    def _delta_critic_update_step(
        self,
        delta_critic_state: Any,
        obs: jnp.ndarray,           # (T*E*N, policy_obs_dim)
        actor_hidden: jnp.ndarray,  # (T*E*N, actor_state_dim)
        delta_returns: jnp.ndarray, # (T*E*N,)
        loss_masks: jnp.ndarray,    # (T*E*N,)
    ):
        def loss_fn(params):
            values = delta_critic_state.apply_fn(
                {"params": params}, obs, actor_hidden
            )
            denom = jnp.sum(loss_masks) + 1.0e-8
            value_loss = jnp.sum(jnp.square(values - delta_returns) * loss_masks) / denom
            return self.value_coef * value_loss, {"delta_critic_loss": value_loss}

        grads, metrics = jax.grad(loss_fn, has_aux=True)(delta_critic_state.params)
        new_state = delta_critic_state.apply_gradients(grads=grads)
        metrics["delta_critic_grad_norm"] = global_norm(grads)
        return new_state, metrics

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
                raw_prev_actions_prev = current_runner["prev_actions"]
                actor_prev_actions_prev = raw_prev_actions_prev
                critic_prev_actions_prev = raw_prev_actions_prev
                prev_obs_prev = current_runner["prev_obs"]
                done_prev = current_runner["done_prev"]
                if self.action_dim > 1:
                    current_alive = self._alive_from_avail(current_avail).astype(jnp.float32)
                    actor_hidden_prev = actor_hidden_prev * current_alive[..., None]
                    actor_prev_actions_prev = actor_prev_actions_prev * current_alive[..., None]
                    # For global-reward decentralized critics, keep dead agents'
                    # critic history intact so value bootstrap remains coherent.
                    if not self.use_centralized_critic and not self.global_reward:
                        critic_hidden_prev = critic_hidden_prev * current_alive[..., None]
                        critic_prev_actions_prev = critic_prev_actions_prev * current_alive[..., None]

                actor_hidden_next, actions, log_probs, probs = self._policy_step(
                    actor_params,
                    obs_aug,
                    current_avail,
                    actor_hidden_prev,
                    actor_prev_actions_prev,
                    done_prev,
                    action_rng,
                    False,
                )
                critic_hidden_next, values = self._value_step(
                    critic_params,
                    obs_aug,
                    critic_hidden_prev,
                    critic_prev_actions_prev,
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
                if self.action_dim > 1:
                    next_alive = self._alive_from_avail(next_avail).astype(jnp.float32)
                    actor_hidden_next = actor_hidden_next * next_alive[..., None]
                    if not self.use_centralized_critic and not self.global_reward:
                        critic_hidden_next = critic_hidden_next * next_alive[..., None]

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
                    "prev_actions": raw_prev_actions_prev.astype(jnp.float32),
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
            rollout_out = dict(rollout)
            rollout_out["last_next_obs_aug"] = next_obs_aug                      # (E, N, obs_dim)
            rollout_out["last_next_actor_hidden"] = next_runner["actor_hidden"]  # (E, N, hidden_dim)
            rollout_out["last_next_avail"] = next_avail                          # (E, N, action_dim)
            return rng, states, next_obs, next_avail, next_runner, rollout_out, last_value

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

    def _flatten_actor_batch(
        self,
        rollout: Dict[str, jnp.ndarray],
        advantages: jnp.ndarray,
        pci_advantages: Optional[jnp.ndarray] = None,
        causal_lambda: float = 0.0,
    ):
        del pci_advantages, causal_lambda
        total_steps = rollout["obs"].shape[0] * rollout["obs"].shape[1]
        obs = rollout["obs"].reshape(total_steps, self.num_agents, self.policy_obs_dim)
        avail_actions = rollout["avail_actions"].reshape(total_steps, self.num_agents, self.action_dim)
        actions = rollout["actions"].reshape(total_steps, self.num_agents)
        old_log_probs = rollout["log_probs"].reshape(total_steps, self.num_agents)
        prev_actions = rollout["prev_actions"].reshape(total_steps, self.num_agents, self.action_dim)
        hidden = rollout["actor_hidden"].reshape(total_steps, self.num_agents, self.actor_state_dim)
        masks = jnp.broadcast_to(
            rollout["masks"].reshape(rollout["masks"].shape + (1,)),
            rollout["actions"].shape,
        ).reshape(total_steps, self.num_agents)
        if self.action_dim > 1:
            alive_masks = (
                self._alive_from_avail(rollout["avail_actions"])
            ).astype(jnp.float32).reshape(total_steps, self.num_agents)
        else:
            alive_masks = jnp.ones_like(actions, dtype=jnp.float32)
        loss_masks = masks * alive_masks

        if advantages.ndim == 2:
            adv = jnp.repeat(advantages[..., None], self.num_agents, axis=-1)
        else:
            adv = advantages
        adv = adv.reshape(total_steps, self.num_agents)
        adv = self._normalize_actor_advantages(adv, alive_masks)

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
                    alive_masks = (self._alive_from_avail(rollout["avail_actions"])).astype(jnp.float32)
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
            alive_masks = (self._alive_from_avail(rollout["avail_actions"])).astype(jnp.float32).reshape(-1)
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

        obs_flat = rollout["obs"].reshape(flat_batch, self.num_agents, self.policy_obs_dim)
        local_obs = obs_flat.reshape(-1, self.policy_obs_dim)
        if self.action_dim > 1:
            alive_by_agent = self._alive_from_avail(rollout["avail_actions"]).reshape(
                flat_batch, self.num_agents
            )
        else:
            alive_by_agent = jnp.ones((flat_batch, self.num_agents), dtype=bool)
        others_obs = self._other_obs_for_bridge(obs_flat, alive_by_agent).reshape(-1, (self.num_agents-1)*self.policy_obs_dim)
        if self.use_recurrent_policy:
            hidden_all = rollout["actor_hidden"].reshape(flat_batch, self.num_agents, self.actor_state_dim)
            z = hidden_all.reshape(-1, self.actor_state_dim)
            hidden_others = self._other_hidden_for_bridge(hidden_all, alive_by_agent).reshape(
                -1, self.bridge_hidden_dim
            )
        else:
            z = jnp.zeros((flat_batch * self.num_agents, 0), dtype=jnp.float32)
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

        # final dead agent masking to exclude dataset when the focal agent is dead.
        if self.action_dim > 1:
            alive_mask = (
                self._alive_from_avail(rollout["avail_actions"])
            ).reshape(-1)
        else:
            alive_mask = jnp.ones((flat_batch * self.num_agents,), dtype=bool)

        return {
            "z": z[alive_mask].astype(jnp.float32),
            "w": others_obs[alive_mask].astype(jnp.float32),
            "x_obs": local_obs[alive_mask].astype(jnp.float32),
            "x_hid": hidden_others[alive_mask].astype(jnp.float32),
            "actions": actions[alive_mask].astype(jnp.int32),
            "targets": targets[alive_mask].astype(jnp.float32),
        }

    def _bridge_inputs(self, rollout: Dict[str, jnp.ndarray]):
        obs = rollout["obs"]
        time_steps, n_envs = obs.shape[:2]
        flat_batch = time_steps * n_envs
        local_obs = obs.reshape(-1, self.policy_obs_dim)
        obs_4d = obs.reshape(flat_batch, self.num_agents, self.policy_obs_dim)
        if self.action_dim > 1:
            alive_by_agent = self._alive_from_avail(rollout["avail_actions"]).reshape(
                flat_batch, self.num_agents
            )
        else:
            alive_by_agent = jnp.ones((flat_batch, self.num_agents), dtype=bool)
        others_obs = self._other_obs_for_bridge(obs_4d, alive_by_agent).reshape(-1, (self.num_agents-1)*self.policy_obs_dim)
        if self.use_recurrent_policy:
            hidden_4d = rollout["actor_hidden"].reshape(flat_batch, self.num_agents, self.actor_state_dim)
            hidden_others = self._other_hidden_for_bridge(hidden_4d, alive_by_agent).reshape(
                -1, self.bridge_hidden_dim
            )
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
        # Match Torch: clip with 1e-10 so that when no valid CF exists,
        # renormalized_probs ≈ 0 → baseline ≈ 0 → delta = r_factual (same as Torch).
        renormalized_probs = probs_counterfactual / jnp.clip(renorm, a_min=1.0e-10)
        baseline = jnp.sum(renormalized_probs[..., None] * all_preds, axis=1)

        no_valid_cf = renorm.squeeze(-1) <= 1.0e-8
        baseline = jnp.where(no_valid_cf[:, None], jnp.zeros_like(baseline), baseline)
        no_valid_cf_rate = jnp.mean(no_valid_cf.astype(jnp.float32))
        return actual_pred, baseline, all_preds, no_valid_cf_rate, no_valid_cf

    def _update_bridge_prediction_debug_lines(
        self,
        rollout: Dict[str, jnp.ndarray],
        actions: jnp.ndarray,
        reward_targets: jnp.ndarray,
        probs: jnp.ndarray,
        avail_actions: jnp.ndarray,
        actual_pred: jnp.ndarray,
        baseline: jnp.ndarray,
        all_preds: jnp.ndarray,
    ) -> None:
        if not self.print_bridge_predictions:
            self._bridge_prediction_debug_lines = []
            return

        actions_np = tree_to_host(actions)
        reward_targets_np = tree_to_host(reward_targets)
        probs_np = tree_to_host(probs)
        avail_np = tree_to_host(avail_actions)
        actual_np = tree_to_host(actual_pred)
        baseline_np = tree_to_host(baseline)
        all_preds_np = tree_to_host(all_preds)

        rollout_actions = rollout["actions"]
        time_steps, num_envs, _ = rollout_actions.shape
        flat_states = time_steps * num_envs

        if self.action_dim > 1:
            alive_flat = tree_to_host(self._alive_from_avail(rollout["avail_actions"]).reshape(-1))
            state_alive = alive_flat.reshape(flat_states, self.num_agents).any(axis=-1)
        else:
            state_alive = jnp.ones((flat_states,), dtype=bool)
            state_alive = tree_to_host(state_alive)

        selected_states = []
        for state_idx in range(flat_states):
            if bool(state_alive[state_idx]):
                selected_states.append(state_idx)
            if len(selected_states) >= self.bridge_prediction_print_samples:
                break

        if not selected_states:
            self._bridge_prediction_debug_lines = ["Bridge predictions: no alive rollout states available."]
            return

        sampled_coords = [(state_idx // num_envs, state_idx % num_envs) for state_idx in selected_states]
        lines = [
            f"Bridge predictions averaged over {len(selected_states)} rollout states: {sampled_coords}"
        ]
        for agent_id in range(self.num_agents):
            flat_indices = [state_idx * self.num_agents + agent_id for state_idx in selected_states]
            factual_preds = []
            baseline_preds = []
            target_preds = []
            action_parts = []

            for flat_idx in flat_indices:
                factual_pred = actual_np[flat_idx]
                baseline_pred = baseline_np[flat_idx]
                target_pred = reward_targets_np[flat_idx]
                factual_preds.append(float(factual_pred[0]) if getattr(factual_pred, "shape", ()) else float(factual_pred))
                baseline_preds.append(float(baseline_pred[0]) if getattr(baseline_pred, "shape", ()) else float(baseline_pred))
                target_preds.append(float(target_pred[0]) if getattr(target_pred, "shape", ()) else float(target_pred))

            for action_id in range(self.action_dim):
                pred_values = []
                prob_values = []
                factual_hits = 0
                avail_hits = 0
                for flat_idx in flat_indices:
                    pred = all_preds_np[flat_idx, action_id]
                    pred_values.append(float(pred[0]) if getattr(pred, "shape", ()) else float(pred))
                    prob_values.append(float(probs_np[flat_idx, action_id]))
                    factual_hits += int(int(actions_np[flat_idx]) == action_id)
                    avail_hits += int(avail_np[flat_idx, action_id] > 0)
                action_parts.append(
                    f"{action_id}:{sum(pred_values)/len(pred_values):.4f}"
                    + f"(p={sum(prob_values)/len(prob_values):.3f},"
                    + f"f={factual_hits/len(flat_indices):.2f},"
                    + f"avail={avail_hits/len(flat_indices):.2f})"
                )

            lines.append(
                "  "
                + f"agent {agent_id}: target_mean={sum(target_preds)/len(target_preds):.4f}, "
                + f"factual_mean={sum(factual_preds)/len(factual_preds):.4f}, "
                + f"baseline_mean={sum(baseline_preds)/len(baseline_preds):.4f}, "
                + f"delta_mean={(sum(target_preds)/len(target_preds)) - (sum(baseline_preds)/len(baseline_preds)):.4f}, "
                + f"actions=[{' | '.join(action_parts)}]"
            )

        self._bridge_prediction_debug_lines = lines

    def consume_bridge_prediction_debug_lines(self) -> list[str]:
        lines = list(self._bridge_prediction_debug_lines)
        self._bridge_prediction_debug_lines = []
        return lines

    @partial(jax.jit, static_argnums=0)
    def _actor_update_step(
        self,
        actor_state: Any,
        batch: Dict[str, jnp.ndarray],
    ):
        def loss_fn(params):
            batch_size = batch["actions"].shape[0]
            flat_obs = batch["obs"].reshape(batch_size * self.num_agents, self.policy_obs_dim)
            flat_prev_actions = batch["prev_actions"].reshape(
                batch_size * self.num_agents, self.action_dim
            )
            flat_hidden = batch["hidden"].reshape(batch_size * self.num_agents, self.actor_state_dim)
            flat_masks = batch["masks"].reshape(batch_size * self.num_agents)
            _, logits = actor_state.apply_fn(
                {"params": params},
                flat_obs,
                flat_prev_actions,
                flat_hidden,
                flat_masks,
            )
            masked_logits = jnp.where(
                batch["avail_actions"].reshape(batch_size * self.num_agents, self.action_dim) > 0,
                logits,
                -1.0e9,
            )
            probs = jax.nn.softmax(masked_logits, axis=-1)
            selected = jnp.take_along_axis(
                probs,
                batch["actions"].reshape(batch_size * self.num_agents, 1),
                axis=-1,
            ).squeeze(-1)
            log_probs = jnp.log(jnp.clip(selected, a_min=1.0e-8)).reshape(batch_size, self.num_agents)
            entropy = -jnp.sum(probs * jnp.log(jnp.clip(probs, a_min=1.0e-8)), axis=-1)
            entropy = entropy.reshape(batch_size, self.num_agents)

            ratio = jnp.exp(log_probs - batch["old_log_probs"])
            clipped_ratio = jnp.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            surrogate = jnp.minimum(
                ratio * batch["advantages"],
                clipped_ratio * batch["advantages"],
            )
            weights = batch["loss_masks"]
            denom = jnp.sum(weights, axis=0) + 1.0e-8

            policy_loss_per_agent = -jnp.sum(surrogate * weights, axis=0) / denom
            entropy_per_agent = jnp.sum(entropy * weights, axis=0) / denom
            total_loss_per_agent = policy_loss_per_agent - self.entropy_coef * entropy_per_agent

            valid_agents = (jnp.sum(weights, axis=0) > 0).astype(jnp.float32)
            valid_count = jnp.maximum(valid_agents.sum(), 1.0)
            total_loss = jnp.sum(total_loss_per_agent * valid_agents) / valid_count
            approx_kl = jnp.sum(
                (batch["old_log_probs"] - log_probs) * weights,
                axis=0,
            ) / denom

            metrics = {
                "policy_loss": jnp.sum(policy_loss_per_agent * valid_agents) / valid_count,
                "entropy": jnp.sum(entropy_per_agent * valid_agents) / valid_count,
                "approx_kl": jnp.sum(approx_kl * valid_agents) / valid_count,
            }
            for agent_id in range(self.num_agents):
                metrics[f"agent_{agent_id}_policy_loss"] = policy_loss_per_agent[agent_id]
                metrics[f"agent_{agent_id}_entropy"] = entropy_per_agent[agent_id]
                metrics[f"agent_{agent_id}_approx_kl"] = approx_kl[agent_id]
            return total_loss, metrics

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
    def _compute_gram(
        self,
        batch: Dict[str, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute Gram matrix and regularized Gram once per mini-batch (reused across bridge epochs)."""
        def _standardize(x: jnp.ndarray) -> jnp.ndarray:
            mean = jnp.mean(x, axis=0, keepdims=True)
            std = jnp.std(x, axis=0, keepdims=True)
            return (x - mean) / jnp.clip(std, a_min=1.0e-6)

        def _rbf_gram(x: jnp.ndarray, gamma: float) -> jnp.ndarray:
            x_norm = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
            dists_sq = x_norm - 2.0 * (x @ x.T) + x_norm.T
            dists_sq = jnp.clip(dists_sq, a_min=0.0)
            return jnp.exp(-gamma * dists_sq)

        z_norm = _standardize(batch["z"])
        # Standardize only the continuous obs features; keep agent_id one-hot (raw 0/1) as-is
        # so it acts as a categorical selector in the kernel rather than being inflated by
        # feature-wise normalization (one-hot std ≈ sqrt(1/N*(1-1/N)) is small → large scale-up).
        x_obs_cont = batch["x_obs"][:, :self.obs_dim]
        x_obs_id   = batch["x_obs"][:, self.obs_dim:]   # shape (B, num_agents); empty if add_agent_id=False
        x_obs_norm = jnp.concatenate([_standardize(x_obs_cont), x_obs_id], axis=-1)
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
        # 1.0e-6 is for torch.linalg.solve fallback in Torch rpci.
        reg_gram = (self.tikhonov_gamma + 1.0e-6) * eye + self.stabilizer_lambda * gram

        diag = jnp.diag(gram)
        total_sum = jnp.sum(gram) - jnp.sum(diag)
        off_count = gram.shape[0] * max(gram.shape[0] - 1, 1)
        kz_offdiag_mean = total_sum / off_count
        centered = gram - kz_offdiag_mean
        sq_sum = jnp.sum(jnp.square(centered)) - jnp.sum(jnp.square(diag - kz_offdiag_mean))
        kz_offdiag_std = jnp.sqrt(jnp.clip(sq_sum / off_count, a_min=0.0))

        gram_metrics = {
            "Kz_offdiag_mean": kz_offdiag_mean,
            "Kz_offdiag_std": kz_offdiag_std,
        }
        return gram, reg_gram, gram_metrics

    @partial(jax.jit, static_argnums=0)
    def _bridge_update_step(
        self,
        bridge_state: Any,
        batch: Dict[str, jnp.ndarray],
        gram: jnp.ndarray,
        reg_gram: jnp.ndarray,
    ):
        """Update bridge parameters using a precomputed Gram matrix (matches Torch: gram computed once per mini-batch)."""
        def loss_fn(params):
            preds = bridge_state.apply_fn(
                {"params": params},
                batch["x_obs"],
                batch["w"],
                batch["x_hid"],
                batch["actions"],
            )
            residuals = batch["targets"] - preds
            solved = jnp.linalg.solve(reg_gram, residuals)
            per_sample_loss = jnp.sum(residuals * (gram @ solved), axis=-1)
            outcome_bridge_loss = jnp.mean(per_sample_loss)
            action_bridge_loss = jnp.array(0.0, dtype=jnp.float32)
            return outcome_bridge_loss + action_bridge_loss, {
                "outcome_bridge_loss": outcome_bridge_loss,
                "action_bridge_loss": action_bridge_loss,
            }

        grads, metrics = jax.grad(loss_fn, has_aux=True)(bridge_state.params)
        new_bridge_state = bridge_state.apply_gradients(grads=grads)
        metrics["bridge_grad_norm"] = global_norm(grads)
        return new_bridge_state, metrics

    def _iterate_minibatches(self, size: int):
        effective_batch = self.batch_size if self.batch_size > 0 else math.ceil(size / self.num_batches)
        for start in range(0, size, effective_batch):
            yield jnp.arange(start, min(start + effective_batch, size))

    def update_causal_networks(self, rollout: Optional[Dict[str, jnp.ndarray]] = None) -> Dict[str, float]:
        if rollout is None:
            rollout = self.latest_rollout
        if rollout is None:
            return {}
        if self.bridge_state is None:
            return {"outcome_bridge_loss": 0.0, "action_bridge_loss": 0.0}

        causal_batch = self._causal_batch(rollout)
        metrics = []
        total = causal_batch["actions"].shape[0]
        for start in range(0, total, self.causal_mini_batch_size):
            indices = jnp.arange(
                start,
                min(start + self.causal_mini_batch_size, total),
            )
            batch = {
                "z": causal_batch["z"][indices],
                "w": causal_batch["w"][indices],
                "x_obs": causal_batch["x_obs"][indices],
                "x_hid": causal_batch["x_hid"][indices],
                "actions": causal_batch["actions"][indices],
                "targets": causal_batch["targets"][indices],
            }
            # Compute Gram once per mini-batch and reuse across bridge epochs (matches Torch).
            gram, reg_gram, gram_metrics = self._compute_gram(batch)
            gram_metrics_host = {key: float(value) for key, value in tree_to_host(gram_metrics).items()}
            for _ in range(self.bridge_epoch):
                self.bridge_state, batch_metrics = self._bridge_update_step(self.bridge_state, batch, gram, reg_gram)
                combined = {key: float(value) for key, value in tree_to_host(batch_metrics).items()}
                combined.update(gram_metrics_host)
                metrics.append(combined)
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
        if baseline_type not in ("weighted", "averaged", "max"):
            raise ValueError(f"Unsupported baseline_type for JAXRPCIAlgorithm: {baseline_type}")
        if self.bridge_state is None or self.get_current_causal_lambda() == 0.0:
            self._bridge_prediction_debug_lines = []
            zeros = jnp.zeros(rollout["actions"].shape + (self.num_agents,), dtype=jnp.float32)
            return zeros, {
                "causal_lambda": self.get_current_causal_lambda(),
                "delta_mean": 0.0,
                "delta_std": 0.0,
            }

        local_obs, others_obs, hidden_others, actions, reward_targets, probs, avail_actions = self._bridge_inputs(rollout)
        actual_pred, baseline_weighted, all_preds, no_valid_cf_rate, no_valid_cf = self._bridge_predictions(
            self.bridge_state.params,
            local_obs,
            others_obs,
            hidden_others,
            probs,
            actions,
            avail_actions,
        )

        if baseline_type == "weighted":
            baseline = baseline_weighted
            # no_valid_cf: policy concentrates prob on factual action → renorm ≈ 0
            effective_no_valid_cf = no_valid_cf
        elif baseline_type == "averaged":
            factual_mask = 1.0 - jax.nn.one_hot(actions.astype(jnp.int32), self.action_dim, dtype=jnp.float32)
            valid_non_factual = factual_mask * avail_actions  # (B*N, action_dim)
            denom = jnp.clip(valid_non_factual.sum(axis=-1, keepdims=True), a_min=1.0)
            baseline = (all_preds * valid_non_factual[..., None]).sum(axis=1) / denom  # (B*N, reward_dim)
            effective_no_valid_cf = valid_non_factual.sum(axis=-1) <= 0  # (B*N,)
        else:  # max
            factual_mask = 1.0 - jax.nn.one_hot(
                actions.astype(jnp.int32), self.action_dim, dtype=jnp.float32
            )
            valid_non_factual = (factual_mask > 0) & (avail_actions > 0)
            masked_preds = jnp.where(valid_non_factual[..., None], all_preds, -jnp.inf)
            baseline = jnp.max(masked_preds, axis=1)
            effective_no_valid_cf = ~jnp.any(valid_non_factual, axis=1)
            baseline = jnp.where(effective_no_valid_cf[:, None], actual_pred, baseline)

        self._update_bridge_prediction_debug_lines(
            rollout,
            actions,
            reward_targets,
            probs,
            avail_actions,
            actual_pred,
            baseline,
            all_preds,
        )

        actual_team_reward = reward_targets
        raw_effects = actual_team_reward - baseline
        raw_effects = jnp.where(effective_no_valid_cf[:, None], jnp.zeros_like(raw_effects), raw_effects)

        # raw_effects = actual_pred - baseline
        # the final r^i = r - lambda_c*(h_factual - E[h_cf])
        
        if self.action_dim > 1:
            alive = (self._alive_from_avail(rollout["avail_actions"])).astype(jnp.float32)
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

        alive_flat = alive.reshape(-1).astype(jnp.bool_)   # (T*E*N,)
        dead_fraction = float(jnp.mean(~alive_flat))
        no_valid_cf_flat = no_valid_cf                       # (T*E*N,)
        alive_count = jnp.sum(alive_flat).astype(jnp.float32)
        dead_count  = jnp.sum(~alive_flat).astype(jnp.float32)
        no_valid_cf_rate_alive = float(
            jnp.sum(no_valid_cf_flat & alive_flat) / jnp.maximum(alive_count, 1.0)
        )
        no_valid_cf_rate_dead = float(
            jnp.sum(no_valid_cf_flat & ~alive_flat) / jnp.maximum(dead_count, 1.0)
        )

        info = {
            "causal_lambda": self.get_current_causal_lambda(),
            "delta_mean": float(jnp.mean(delta)),
            "delta_std": float(jnp.std(delta)),
            "no_valid_cf_rate": float(no_valid_cf_rate),
            "no_valid_cf_rate_alive": no_valid_cf_rate_alive,   # policy entropy 지표
            "no_valid_cf_rate_dead": no_valid_cf_rate_dead,     # 항상 ≈1.0 (sanity check)
            "dead_agent_fraction": dead_fraction,               # 팀 생존율 지표
        }
        return c_matrix, info

    def _update_actor_critic(
        self,
        rollout: Dict[str, jnp.ndarray],
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        pci_advantages: Optional[jnp.ndarray] = None,
        causal_lambda: float = 0.0,
    ) -> Dict[str, float]:
        del pci_advantages, causal_lambda
        actor_batch = self._flatten_actor_batch(rollout, advantages)
        critic_batch = self._flatten_critic_batch(rollout, returns)

        metrics = []
        for epoch in range(self.ppo_epochs):
            self.actor_state, actor_metrics = self._actor_update_step(self.actor_state, actor_batch)
            metrics.append({key: float(value) for key, value in tree_to_host(actor_metrics).items()})

            self.critic_state, critic_metrics = self._critic_update_step(self.critic_state, critic_batch)
            metrics.append({key: float(value) for key, value in tree_to_host(critic_metrics).items()})

        return stack_metrics(metrics)

    def update(self, **_: Any) -> Dict[str, float]:
        if self.latest_rollout is None:
            return {}

        rollout = self.latest_rollout # training data for one iteration. handled in runner_jax/trainer.py
        metrics = {}
        causal_lambda = self.get_current_causal_lambda()

        bridge_metrics = self.update_causal_networks(rollout)
        metrics.update(bridge_metrics)

        c_matrix, pci_metrics = self.compute_causal_rewards(rollout, baseline_type="weighted")
        delta = c_matrix.sum(axis=-1)
        delta = jnp.where(jnp.abs(delta) < self.clip_delta, 0.0, delta)
        # delta = jnp.maximum(delta, 0.0) # ReLU
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

        '''
        mask dead agents' values to 0 before GAE — only for global_reward=False.
        With global_reward=True, team rewards continue after an agent's death, so
        the IPPO critic V is trained to predict non-zero future team value for dead states.
        Forcing V_{t+1}=0 would introduce a negative bias at the last alive step and
        degrade performance.  With individual rewards (global_reward=False), a dead
        agent earns nothing, so V=0 is semantically correct.
        '''
        if not self.use_centralized_critic and self.action_dim > 1 and not self.global_reward:
            alive_ten = self._alive_from_avail(rollout["avail_actions"]).astype(jnp.float32)  # (T, E, N)
            values = values * alive_ten
            last_alive = self._alive_from_avail(rollout["last_next_avail"]).astype(jnp.float32)  # (E, N)
            last_value = last_value * last_alive

        env_advantages, returns = compute_gae(
            env_rewards,
            values,
            last_value,
            rollout["dones"],
            self.gamma,
            self.gae_lambda,
        )

        if self.use_delta_critic and self.delta_critic_state is not None:
            delta_values = self._compute_delta_values(
                self.delta_critic_state.params,
                rollout["obs"],           # (T, E, N, obs_dim)
                rollout["actor_hidden"],  # (T, E, N, hidden_dim)
            )
            last_delta_value = self._delta_value_step(
                self.delta_critic_state.params,
                rollout["last_next_obs_aug"],       # (E, N, obs_dim)
                rollout["last_next_actor_hidden"],  # (E, N, hidden_dim)
            )
            # Mask dead agents' delta_values to 0 before GAE.
            # dones is episode-level (T, E) so compute_gae alone cannot distinguish
            # mid-episode agent death: a dead agent's delta_values would otherwise
            # bootstrap back into earlier alive timesteps via the GAE recursion.
            if self.action_dim > 1:
                alive_ten = self._alive_from_avail(rollout["avail_actions"]).astype(jnp.float32)  # (T, E, N)
                delta_values = delta_values * alive_ten
                last_alive = self._alive_from_avail(rollout["last_next_avail"]).astype(jnp.float32)  # (E, N)
                last_delta_value = last_delta_value * last_alive
            pci_advantages, delta_returns = compute_gae(
                delta,
                delta_values,
                last_delta_value,
                rollout["dones"],
                self.gamma,
                self.gae_lambda,
            )
            # Train delta critic for ppo_epochs on the full batch
            T, E, N = delta.shape
            total = T * E * N
            flat_obs = rollout["obs"].reshape(total, self.policy_obs_dim)
            if self.actor_state_dim > 0:
                flat_hidden = rollout["actor_hidden"].reshape(total, self.actor_state_dim)
            else:
                flat_hidden = jnp.zeros((total, 0), dtype=jnp.float32)
            flat_delta_returns = delta_returns.reshape(total)
            flat_masks = jnp.broadcast_to(
                rollout["masks"].reshape(rollout["masks"].shape + (1,)),
                rollout["actions"].shape,
            ).reshape(total)
            if self.action_dim > 1:
                flat_alive = self._alive_from_avail(rollout["avail_actions"]).astype(jnp.float32).reshape(total)
            else:
                flat_alive = jnp.ones((total,), dtype=jnp.float32)
            flat_loss_masks = flat_masks * flat_alive
            raw_dc: list = []
            for _ in range(self.ppo_epochs):
                self.delta_critic_state, dc_metrics = self._delta_critic_update_step(
                    self.delta_critic_state,
                    flat_obs,
                    flat_hidden,
                    flat_delta_returns,
                    flat_loss_masks,
                )
                raw_dc.append(dc_metrics)
            for bm in raw_dc:
                metrics.update({k: float(v) for k, v in tree_to_host(bm).items()})
        else:
            pci_advantages, _ = compute_gae(
                delta,
                jnp.zeros_like(delta),
                jnp.zeros_like(delta[0]),
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
        metrics.update(self._categorized_action_delta_metrics(rollout, delta))

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
        actor_hidden = runner_state["actor_hidden"]
        prev_actions = runner_state["prev_actions"]
        if self.action_dim > 1:
            current_alive = self._alive_from_avail(avail_actions).astype(jnp.float32)
            actor_hidden = actor_hidden * current_alive[..., None]
            prev_actions = prev_actions * current_alive[..., None]
        actor_hidden_next, actions, _, _ = self._policy_step(
            self.actor_state.params,
            obs_aug,
            avail_actions,
            actor_hidden,
            prev_actions,
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
            "delta_critic_state": serialization.to_state_dict(self.delta_critic_state) if self.delta_critic_state is not None else None,
            "rng": tree_to_host(self.rng),
            "bridge_rng": tree_to_host(self.bridge_rng),
            "training_steps": self.training_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor_state = serialization.from_state_dict(self.actor_state, state_dict["actor_state"])
        self.critic_state = serialization.from_state_dict(self.critic_state, state_dict["critic_state"])
        if self.bridge_state is not None and state_dict.get("bridge_state") is not None:
            self.bridge_state = serialization.from_state_dict(self.bridge_state, state_dict["bridge_state"])
        if self.delta_critic_state is not None and state_dict.get("delta_critic_state") is not None:
            self.delta_critic_state = serialization.from_state_dict(self.delta_critic_state, state_dict["delta_critic_state"])
        self.rng = jnp.asarray(state_dict["rng"])
        if "bridge_rng" in state_dict:
            self.bridge_rng = jnp.asarray(state_dict["bridge_rng"])
        self.training_steps = int(state_dict.get("training_steps", 0))
