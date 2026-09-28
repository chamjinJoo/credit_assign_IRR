from __future__ import annotations

import math
from functools import partial
from typing import Any, Dict, Optional

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import optax

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
    global_norm,
    make_dynamic_lr_train_state,
    make_train_state,
    stack_metrics,
    tree_to_host,
)
from .preprocessing import (
    bridge_inputs_from_rollout,
    causal_batch_from_rollout,
    causal_minibatch_indices,
    flatten_actor_batch,
    flatten_critic_batch,
    flatten_delta_critic_batch,
    minibatch_indices,
    preprocess_causal_rollout,
    slice_batch,
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
        causal_lambda: float = 0.0,
        bridge_epoch: int = 4,
        causal_mini_batch_size: int = 1000,
        tikhonov_gamma: float = 1.0e-4,
        stabilizer_lambda: float = 1.0,
        use_recurrent_policy: bool = True,
        bridge_prediction_batch_size: int = 64,
        train_naive_predictor: Optional[bool] = None,
        train_naivepredictor: Optional[bool] = None,
        use_naive_predictor_for_delta: bool = False,
        anneal_lr: bool = True,
        total_iterations: int = 0,
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
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.num_batches = max(1, num_batches)
        self.use_recurrent_policy = use_recurrent_policy
        self.bridge_prediction_batch_size = max(1, int(bridge_prediction_batch_size))
        if train_naive_predictor is None:
            train_naive_predictor = train_naivepredictor
        self.train_naive_predictor = bool(train_naive_predictor) if train_naive_predictor is not None else False
        self.use_naive_predictor_for_delta = bool(use_naive_predictor_for_delta)
        self.attack_action_threshold = self._infer_attack_action_threshold(action_dim)
 
        allowed_target_matrix = getattr(self.env, "_allowed_target_matrix_jax", None)
        if allowed_target_matrix is None:
            allowed_target_matrix = getattr(self.env, "_allowed_target_matrix_np", None)
        self._allowed_target_matrix = (
            None if allowed_target_matrix is None else jnp.asarray(allowed_target_matrix, dtype=jnp.float32)
        )

        self.causal_lambda = causal_lambda
        self.bridge_epoch = bridge_epoch
        self.causal_mini_batch_size = causal_mini_batch_size
        self.tikhonov_gamma = tikhonov_gamma
        self.stabilizer_lambda = stabilizer_lambda
        self.training_steps = 0
        self.current_iteration = 0
        self.latest_rollout: Optional[Dict[str, Any]] = None
        self._rollout_fns: Dict[int, Any] = {}

        self.policy_obs_dim = obs_dim + num_agents if add_agent_id else obs_dim
        self.actor_state_dim = hidden_dim if use_recurrent_policy else 0
        self.critic_state_dim = hidden_dim if use_recurrent_policy else 0
        self.delta_critic_state_dim = hidden_dim if causal_lambda > 0.0 else 0
        self.bridge_hidden_dim = (num_agents - 1) * self.actor_state_dim
        self.bridge_prev_action_dim = (num_agents - 1) * action_dim
        self.bridge_context_dim = self.bridge_hidden_dim + self.bridge_prev_action_dim
        self.bridge_output_dim = 1 if global_reward else num_agents
        self.critic_output_dim = num_agents if (use_centralized_critic and not global_reward) else 1
        self.delta_critic_output_dim = 1
        base_rng = jax.random.PRNGKey(int(seed))
        init_rng = jax.random.fold_in(base_rng, 0)
        self.rng = jax.random.fold_in(base_rng, 1)
        self.bridge_rng = jax.random.fold_in(base_rng, 2)

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
        self.naive_predictor_model = (
            CausalBridgeNetwork(
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                num_agents=self.bridge_output_dim,
            )
            if self.train_naive_predictor and causal_lambda > 0.0
            else None
        )


        self.anneal_lr = bool(anneal_lr)
        self.total_iterations = int(total_iterations)

        def _maybe_schedule(base_lr: float):
            if not self.anneal_lr or self.total_iterations <= 0:
                return base_lr
            steps_per_iter = max(1, self.ppo_epochs * self.num_batches)
            total_grad_steps = self.total_iterations * steps_per_iter
            return optax.linear_schedule(
                init_value=base_lr,
                end_value=0.0,
                transition_steps=total_grad_steps,
            )

        lr_actor_arg = _maybe_schedule(lr_actor)
        lr_critic_arg = _maybe_schedule(lr_critic)
        # Kept around to log the *current* actor/critic LR (see _current_lr).
        self._lr_actor_schedule = lr_actor_arg
        self._lr_critic_schedule = lr_critic_arg
        self._lr_steps_per_iter = max(1, self.ppo_epochs * self.num_batches)
        # Bridge LR is annealed separately (see update()): bridge_state uses
        # DynamicLearningRateTrainState, whose `learning_rate` field is set
        # directly from current_iteration / total_iterations each update().
        # This avoids relying on optax.linear_schedule's internal step count,
        # which for the bridge varies with len(alive_idx) per iteration and
        # would otherwise reach end_value=0 (and clamp) before iteration
        # total_iterations.
        self.lr_bridge = float(lr_bridge)

        init_rng, actor_key, critic_key, bridge_key = jax.random.split(init_rng, 4)
        naive_key = jax.random.fold_in(base_rng, 3)

        self.actor_state = make_train_state(
            self.actor_model,
            actor_key,
            (
                jnp.zeros((1, self.policy_obs_dim), dtype=jnp.float32),
                jnp.zeros((1, action_dim), dtype=jnp.float32),
                jnp.zeros((1, self.actor_state_dim), dtype=jnp.float32),
                jnp.ones((1,), dtype=jnp.float32),
            ),
            lr_actor_arg,
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
            lr_critic_arg,
            max_grad_norm,
        )

        self.bridge_state = None
        if self.bridge_model is not None:
            # others_obs dim: mean-pooled → policy_obs_dim; concat → (N-1)*policy_obs_dim
            bridge_others_obs_dim = (num_agents - 1) * self.policy_obs_dim
            self.bridge_state = make_dynamic_lr_train_state(
                self.bridge_model,
                bridge_key,
                (
                    jnp.zeros((1, self.policy_obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, bridge_others_obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, self.bridge_context_dim), dtype=jnp.float32),
                    jnp.zeros((1,), dtype=jnp.int32),
                ),
                lr_bridge,
                max_grad_norm,
            )

        self.naive_predictor_state = None
        if self.naive_predictor_model is not None:
            bridge_others_obs_dim = (num_agents - 1) * self.policy_obs_dim
            self.naive_predictor_state = make_dynamic_lr_train_state(
                self.naive_predictor_model,
                naive_key,
                (
                    jnp.zeros((1, self.policy_obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, bridge_others_obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, self.bridge_context_dim), dtype=jnp.float32),
                    jnp.zeros((1,), dtype=jnp.int32),
                ),
                lr_bridge,
                max_grad_norm,
            )


        self.delta_critic_state = None
        if self.bridge_model is not None:
            self.delta_critic_model = RecurrentCritic(
                hidden_dim=hidden_dim,
                output_dim=self.delta_critic_output_dim,
            )
            init_rng, delta_critic_key = jax.random.split(init_rng)
            self.delta_critic_state = make_train_state(
                self.delta_critic_model,
                delta_critic_key,
                (
                    jnp.zeros((1, self.policy_obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, action_dim), dtype=jnp.float32),
                    jnp.zeros((1, self.delta_critic_state_dim), dtype=jnp.float32),
                    jnp.ones((1,), dtype=jnp.float32),
                ),
                lr_critic_arg,
                max_grad_norm,
            )

    def initialize_runner_state(self, n_envs: int) -> Dict[str, jnp.ndarray]:
        critic_shape = (
            (n_envs, self.critic_state_dim)
            if self.use_centralized_critic
            else (n_envs, self.num_agents, self.critic_state_dim)
        )
        delta_critic_shape = (n_envs, self.num_agents, self.delta_critic_state_dim)
        return {
            "actor_hidden": jnp.zeros((n_envs, self.num_agents, self.actor_state_dim), dtype=jnp.float32),
            "critic_hidden": jnp.zeros(critic_shape, dtype=jnp.float32),
            "delta_critic_hidden": jnp.zeros(delta_critic_shape, dtype=jnp.float32),
            "prev_actions": jnp.zeros((n_envs, self.num_agents, self.action_dim), dtype=jnp.float32),
            "prev_obs": jnp.zeros((n_envs, self.num_agents, self.policy_obs_dim), dtype=jnp.float32),
            "done_prev": jnp.zeros((n_envs,), dtype=jnp.float32),
            "agent_done_prev": jnp.zeros((n_envs, self.num_agents), dtype=jnp.float32),
        }

    def reset_agents(self):
        return None

    def reset_agents_batch(self, n_envs: int):
        return self.initialize_runner_state(n_envs)

    def get_current_causal_lambda(self) -> float:
        return float(self.causal_lambda)

    @staticmethod
    def _current_lr(schedule: Any, step: int) -> float:
        return float(schedule(step)) if callable(schedule) else float(schedule)

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

    def _alive_from_agent_dones(self, agent_dones: jnp.ndarray) -> jnp.ndarray:
        return jnp.logical_not(agent_dones.astype(jnp.bool_))

    def _normalize_actor_advantages(
        self,
        advantages: jnp.ndarray,
        alive_masks: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Normalize advantages per agent (axis=0 = T*E direction)."""
        if advantages.shape[0] <= 1:
            return jnp.zeros_like(advantages)
        if alive_masks is not None:
            w = alive_masks.astype(jnp.float32)
            counts = jnp.sum(w, axis=0, keepdims=True)
            mean = jnp.sum(advantages * w, axis=0, keepdims=True) / jnp.maximum(counts, 1.0)
            centered = advantages - mean
            sq_sum = jnp.sum(jnp.square(centered) * w, axis=0, keepdims=True)
            denom = jnp.maximum(counts - 1.0, 1.0)
        else:
            centered = advantages - advantages.mean(axis=0, keepdims=True)
            sq_sum = jnp.sum(jnp.square(centered), axis=0, keepdims=True)
            denom = jnp.maximum(advantages.shape[0] - 1, 1)
        var = sq_sum / denom
        std = jnp.sqrt(jnp.clip(var, a_min=1.0e-8))
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

        # NOTE: 1st adv norm: normalize both adv before combining. 
        env_flat = self._normalize_actor_advantages(env_flat)
        pci_flat = self._normalize_actor_advantages(pci_flat)

        combined = (env_flat + causal_lambda * pci_flat).reshape(orig_shape)

        info = {
            "env_adv_std": float(jnp.std(env_flat)),
            "pci_adv_res_std": float(jnp.std(pci_flat)),
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

    @partial(jax.jit, static_argnums=0)
    def _categorized_action_delta_compute(
        self,
        actions: jnp.ndarray,        # (T, E, N)
        agent_dones: jnp.ndarray,    # (T, E, N)
        delta: jnp.ndarray,          # (T, E, N)
        avail_actions: jnp.ndarray,  # (T, E, N, action_dim)
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        thr = self.attack_action_threshold
        alive_f = self._alive_from_agent_dones(agent_dones).astype(jnp.float32)
        is_attack = (actions >= thr).astype(jnp.float32)

        if self._allowed_target_matrix is not None:
            num_targets = self._allowed_target_matrix.shape[-1]
            attack_targets = jnp.clip(actions - thr, 0, num_targets - 1)
            chosen_target_one_hot = jax.nn.one_hot(
                attack_targets.astype(jnp.int32), num_targets, dtype=jnp.float32,
            )
            chosen_target_allowed = jnp.sum(
                chosen_target_one_hot * self._allowed_target_matrix[None, None, :, :],
                axis=-1,
            )
            target_mask    = alive_f * is_attack * (chosen_target_allowed > 0).astype(jnp.float32)
            nontarget_mask = alive_f * is_attack * (chosen_target_allowed <= 0).astype(jnp.float32)
        else:
            action_avail = jnp.take_along_axis(
                avail_actions, actions[..., None].astype(jnp.int32), axis=-1
            ).squeeze(-1)
            target_mask    = alive_f * is_attack * (action_avail > 0).astype(jnp.float32)
            nontarget_mask = alive_f * is_attack * (action_avail == 0).astype(jnp.float32)

        other_mask = alive_f * (1.0 - is_attack)

        def _stats(mask):
            counts    = mask.sum(axis=(0, 1))
            delta_sum = (delta * mask).sum(axis=(0, 1))
            means     = jnp.where(counts > 0, delta_sum / jnp.maximum(counts, 1.0e-8), 0.0)
            return counts, means

        tc, tm = _stats(target_mask)
        nc, nm = _stats(nontarget_mask)
        oc, om = _stats(other_mask)
        return tc, tm, nc, nm, oc, om

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

        tc, tm, nc, nm, oc, om = self._categorized_action_delta_compute(
            rollout["actions"],
            rollout["agent_dones"],
            delta,
            rollout["avail_actions"],
        )

        cat_data = [
            ("target_attack",     tree_to_host(tc), tree_to_host(tm)),
            ("non_target_attack", tree_to_host(nc), tree_to_host(nm)),
            ("other_action",      tree_to_host(oc), tree_to_host(om)),
        ]

        metrics: Dict[str, float] = {}
        for agent_id in range(self.num_agents):
            for cat_name, counts_h, means_h in cat_data:
                metrics[f"delta_agent{agent_id}_{cat_name}_count"] = float(counts_h[agent_id])
                metrics[f"delta_agent{agent_id}_{cat_name}_mean"]  = float(means_h[agent_id])
        return metrics

    @partial(jax.jit, static_argnums=0)
    def _policy_step(
        self,
        actor_params: Dict[str, Any],
        obs: jnp.ndarray,
        avail_actions: jnp.ndarray,
        actor_hidden: jnp.ndarray,
        prev_actions: jnp.ndarray,
        reset_masks: jnp.ndarray,
        rng: jax.Array,
        deterministic: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        n_envs = obs.shape[0]
        if reset_masks.ndim == 1:
            masks = jnp.broadcast_to((1.0 - reset_masks)[:, None], (n_envs, self.num_agents))
        else:
            masks = 1.0 - reset_masks
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
        reset_masks: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.use_centralized_critic:
            masks = 1.0 - reset_masks
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
        if reset_masks.ndim == 1:
            masks = jnp.broadcast_to((1.0 - reset_masks)[:, None], (n_envs, self.num_agents))
        else:
            masks = 1.0 - reset_masks
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
        obs: jnp.ndarray,
        delta_critic_hidden: jnp.ndarray,
        prev_actions: jnp.ndarray,
        reset_masks: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        n_envs = obs.shape[0]
        if reset_masks.ndim == 1:
            masks = jnp.broadcast_to((1.0 - reset_masks)[:, None], (n_envs, self.num_agents))
        else:
            masks = 1.0 - reset_masks
        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_prev_actions = prev_actions.reshape(-1, prev_actions.shape[-1])
        flat_hidden = delta_critic_hidden.reshape((n_envs * self.num_agents, delta_critic_hidden.shape[-1]))
        flat_masks = masks.reshape(-1)
        next_hidden_flat, values_flat = self.delta_critic_state.apply_fn(
            {"params": delta_critic_params},
            flat_obs,
            flat_prev_actions,
            flat_hidden,
            flat_masks,
        )
        return (
            next_hidden_flat.reshape(n_envs, self.num_agents, delta_critic_hidden.shape[-1]),
            values_flat.reshape(n_envs, self.num_agents),
        )

    @partial(jax.jit, static_argnums=0)
    def _delta_critic_update_step(
        self,
        delta_critic_state: Any,
        batch: Dict[str, jnp.ndarray],
    ):
        def loss_fn(params):
            _, values = delta_critic_state.apply_fn(
                {"params": params},
                batch["obs"],
                batch["prev_actions"],
                batch["hidden"],
                batch["masks"],
            )
            denom = jnp.sum(batch["loss_masks"]) + 1.0e-8
            value_loss = jnp.sum(jnp.square(values - batch["returns"]) * batch["loss_masks"]) / denom
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
            action_rng: jax.Array,
            batch_states: Any,
            obs: jnp.ndarray,
            avail_actions: jnp.ndarray,
            runner_state: Dict[str, jnp.ndarray],
            actor_params: Dict[str, Any],
            critic_params: Dict[str, Any],
            delta_critic_params: Dict[str, Any],
        ):
            def body_fn(carry, _):
                rng, policy_rng, states, current_obs, current_avail, current_runner = carry
                policy_rng, step_action_rng = jax.random.split(policy_rng)
                obs_aug = self._augment_obs(current_obs)

                actor_hidden_prev = current_runner["actor_hidden"]
                critic_hidden_prev = current_runner["critic_hidden"]
                delta_critic_hidden_prev = current_runner["delta_critic_hidden"]
                raw_prev_actions_prev = current_runner["prev_actions"]
                prev_obs_prev = current_runner["prev_obs"]
                done_prev = current_runner["done_prev"]
                agent_done_prev = current_runner["agent_done_prev"]
                actor_reset_masks = agent_done_prev
                critic_reset_masks = done_prev if self.use_centralized_critic else agent_done_prev

                actor_hidden_next, actions, log_probs, probs = self._policy_step(
                    actor_params,
                    obs_aug,
                    current_avail,
                    actor_hidden_prev,
                    raw_prev_actions_prev,
                    actor_reset_masks,
                    step_action_rng,
                    False,
                )
                critic_hidden_next, values = self._value_step(
                    critic_params,
                    obs_aug,
                    critic_hidden_prev,
                    raw_prev_actions_prev,
                    critic_reset_masks,
                )
                if self.delta_critic_state is not None:
                    delta_critic_hidden_next, delta_values = self._delta_value_step(
                        delta_critic_params,
                        obs_aug,
                        delta_critic_hidden_prev,
                        raw_prev_actions_prev,
                        critic_reset_masks,
                    )
                else:
                    delta_critic_hidden_next = delta_critic_hidden_prev
                    delta_values = jnp.zeros((obs_aug.shape[0], self.num_agents), dtype=jnp.float32)

                (
                    rng,
                    next_states,
                    next_obs,
                    next_avail,
                    rewards,
                    agent_dones,
                    dones_all,
                    battle_won,
                ) = env.step_jax_pure(rng, states, actions)

                agent_done_next = jnp.maximum(
                    agent_dones.astype(jnp.float32),
                    dones_all.astype(jnp.float32)[:, None],
                )
                next_prev_actions = action_one_hot(actions, self.action_dim)
                next_prev_actions = next_prev_actions * (1.0 - dones_all)[:, None, None]

                next_runner = {
                    "actor_hidden": actor_hidden_next,
                    "critic_hidden": critic_hidden_next,
                    "delta_critic_hidden": delta_critic_hidden_next,
                    "prev_actions": next_prev_actions,
                    "prev_obs": obs_aug,
                    "done_prev": dones_all.astype(jnp.float32),
                    "agent_done_prev": agent_done_next,
                }

                transition = {
                    "prev_obs": prev_obs_prev.astype(jnp.float32),
                    "obs": obs_aug,
                    "avail_actions": current_avail.astype(jnp.float32),
                    "agent_dones": agent_done_prev.astype(jnp.float32),
                    "actions": actions,
                    "log_probs": log_probs,
                    "probs": probs,
                    "rewards": rewards.astype(jnp.float32),
                    "dones": dones_all.astype(jnp.float32),
                    "masks": (1.0 - done_prev).astype(jnp.float32),
                    "values": values.astype(jnp.float32),
                    "delta_values": delta_values.astype(jnp.float32),
                    "actor_hidden": actor_hidden_prev.astype(jnp.float32),
                    "critic_hidden": critic_hidden_prev.astype(jnp.float32),
                    "delta_critic_hidden": delta_critic_hidden_prev.astype(jnp.float32),
                    "prev_actions": raw_prev_actions_prev.astype(jnp.float32),
                    "battle_won": battle_won.astype(jnp.float32),
                }
                next_carry = (
                    rng,
                    policy_rng,
                    next_states,
                    next_obs.astype(jnp.float32),
                    next_avail.astype(jnp.float32),
                    next_runner,
                )
                return next_carry, transition

            final_carry, rollout = jax.lax.scan(
                body_fn,
                (env_rng, action_rng, batch_states, obs, avail_actions, runner_state),
                None,
                length=horizon,
            )
            rng, action_rng, states, next_obs, next_avail, next_runner = final_carry
            next_obs_aug = self._augment_obs(next_obs)
            last_value_reset_masks = next_runner["done_prev"]
            _, last_value = self._value_step(
                critic_params,
                next_obs_aug,
                next_runner["critic_hidden"],
                next_runner["prev_actions"],
                last_value_reset_masks,
            )
            rollout_out = dict(rollout)
            if self.delta_critic_state is not None:
                _, last_delta_value = self._delta_value_step(
                    delta_critic_params,
                    next_obs_aug,
                    next_runner["delta_critic_hidden"],
                    next_runner["prev_actions"],
                    last_value_reset_masks,
                )
                rollout_out["last_delta_value"] = last_delta_value.astype(jnp.float32)
            return rng, action_rng, states, next_obs, next_avail, next_runner, rollout_out, last_value

        self._rollout_fns[horizon] = jax.jit(rollout_fn)
        return self._rollout_fns[horizon]

    def collect_rollout(
        self,
        env: Any,
        env_rng: jax.Array,
        action_rng: Optional[jax.Array],
        batch_states: Any,
        obs: jnp.ndarray,
        avail_actions: jnp.ndarray,
        runner_state: Dict[str, jnp.ndarray],
        horizon: int,
    ):
        rollout_fn = self._get_rollout_fn(env, horizon)
        if action_rng is None:
            env_rng, _, states, next_obs, next_avail, next_runner, rollout, last_value = rollout_fn(
                env_rng,
                env_rng,
                batch_states,
                obs.astype(jnp.float32),
                avail_actions.astype(jnp.float32),
                runner_state,
                self.actor_state.params,
                self.critic_state.params,
                self.delta_critic_state.params if self.delta_critic_state is not None else self.critic_state.params,
            )
            return env_rng, states, next_obs, next_avail, next_runner, rollout, last_value
        return rollout_fn(
            env_rng,
            action_rng,
            batch_states,
            obs.astype(jnp.float32),
            avail_actions.astype(jnp.float32),
            runner_state,
            self.actor_state.params,
            self.critic_state.params,
            self.delta_critic_state.params if self.delta_critic_state is not None else self.critic_state.params,
        )

    def _flatten_actor_batch(
        self,
        rollout: Dict[str, jnp.ndarray],
        advantages: jnp.ndarray,
        pci_advantages: Optional[jnp.ndarray] = None,
        causal_lambda: float = 0.0,
    ):
        del pci_advantages, causal_lambda
        return flatten_actor_batch(
            rollout,
            advantages,
            num_agents=self.num_agents,
            policy_obs_dim=self.policy_obs_dim,
            action_dim=self.action_dim,
            actor_state_dim=self.actor_state_dim,
            normalize_advantages=self._normalize_actor_advantages,
        )

    def _flatten_critic_batch(self, rollout: Dict[str, jnp.ndarray], returns: jnp.ndarray):
        return flatten_critic_batch(
            rollout,
            returns,
            num_agents=self.num_agents,
            policy_obs_dim=self.policy_obs_dim,
            action_dim=self.action_dim,
            critic_state_dim=self.critic_state_dim,
            use_centralized_critic=self.use_centralized_critic,
        )

    def _flatten_delta_critic_batch(
        self,
        rollout: Dict[str, jnp.ndarray],
        delta_returns: jnp.ndarray,
    ):
        return flatten_delta_critic_batch(
            rollout,
            delta_returns,
            policy_obs_dim=self.policy_obs_dim,
            action_dim=self.action_dim,
            delta_critic_state_dim=self.delta_critic_state_dim,
        )

    def _preprocess_rollout(self, rollout: Dict[str, jnp.ndarray]):
        return preprocess_causal_rollout(
            rollout,
            num_agents=self.num_agents,
            policy_obs_dim=self.policy_obs_dim,
            action_dim=self.action_dim,
            actor_state_dim=self.actor_state_dim,
            bridge_hidden_dim=self.bridge_hidden_dim,
            bridge_prev_action_dim=self.bridge_prev_action_dim,
            use_recurrent_policy=self.use_recurrent_policy,
            global_reward=self.global_reward,
        )

    def _causal_batch(self, rollout: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        return causal_batch_from_rollout(
            rollout,
            num_agents=self.num_agents,
            policy_obs_dim=self.policy_obs_dim,
            action_dim=self.action_dim,
            actor_state_dim=self.actor_state_dim,
            bridge_hidden_dim=self.bridge_hidden_dim,
            bridge_prev_action_dim=self.bridge_prev_action_dim,
            use_recurrent_policy=self.use_recurrent_policy,
            global_reward=self.global_reward,
        )

    def _bridge_inputs(self, rollout: Dict[str, jnp.ndarray]):
        return bridge_inputs_from_rollout(
            rollout,
            num_agents=self.num_agents,
            policy_obs_dim=self.policy_obs_dim,
            action_dim=self.action_dim,
            actor_state_dim=self.actor_state_dim,
            bridge_hidden_dim=self.bridge_hidden_dim,
            bridge_prev_action_dim=self.bridge_prev_action_dim,
            use_recurrent_policy=self.use_recurrent_policy,
            global_reward=self.global_reward,
        )

    @partial(jax.jit, static_argnums=(0, 1))
    def _bridge_predictions_chunk(
        self,
        use_naive_predictor: bool,
        bridge_params: Dict[str, Any],
        local_obs: jnp.ndarray,
        others_obs: jnp.ndarray,
        hidden_others: jnp.ndarray,
        probs: jnp.ndarray,
        actions: jnp.ndarray,
        avail_actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        apply_fn = self.naive_predictor_state.apply_fn if use_naive_predictor else self.bridge_state.apply_fn
        actual_pred = apply_fn(
            {"params": bridge_params},
            local_obs,
            others_obs,
            hidden_others,
            actions,
        )

        all_actions = jnp.arange(self.action_dim, dtype=jnp.int32)

        def predict_one_action(action_id: jnp.ndarray):
            tiled = jnp.full((local_obs.shape[0],), action_id, dtype=jnp.int32)
            return apply_fn(
                {"params": bridge_params},
                local_obs,
                others_obs,
                hidden_others,
                tiled,
            )

        all_preds = jax.vmap(predict_one_action)(all_actions).transpose(1, 0, 2)
        factual_mask = 1.0 - jax.nn.one_hot(actions.astype(jnp.int32), self.action_dim, dtype=jnp.float32)
        probs_counterfactual = probs * factual_mask * avail_actions # consider avail actions
   
        renorm = probs_counterfactual.sum(axis=-1, keepdims=True)
        # Match Torch: clip with 1e-10 so that when no valid CF exists,
        # renormalized_probs ≈ 0 → baseline ≈ 0 → delta = r_factual (same as Torch).
        renormalized_probs = probs_counterfactual / jnp.clip(renorm, a_min=1.0e-10)
        baseline = jnp.sum(renormalized_probs[..., None] * all_preds, axis=1)

        no_valid_cf = renorm.squeeze(-1) <= 1.0e-8
        baseline = jnp.where(no_valid_cf[:, None], jnp.zeros_like(baseline), baseline)
        return actual_pred, baseline, all_preds, no_valid_cf

    def _bridge_predictions(
        self,
        use_naive_predictor: bool,
        bridge_params: Dict[str, Any],
        local_obs: jnp.ndarray,
        others_obs: jnp.ndarray,
        hidden_others: jnp.ndarray,
        probs: jnp.ndarray,
        actions: jnp.ndarray,
        avail_actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        chunk_size = self.bridge_prediction_batch_size
        actual_chunks = []
        baseline_chunks = []
        all_pred_chunks = []
        no_valid_chunks = []

        for start in range(0, local_obs.shape[0], chunk_size):
            end = min(start + chunk_size, local_obs.shape[0])
            actual_pred, baseline, all_preds, no_valid_cf = self._bridge_predictions_chunk(
                use_naive_predictor,
                bridge_params,
                local_obs[start:end],
                others_obs[start:end],
                hidden_others[start:end],
                probs[start:end],
                actions[start:end],
                avail_actions[start:end],
            )
            actual_chunks.append(actual_pred)
            baseline_chunks.append(baseline)
            all_pred_chunks.append(all_preds)
            no_valid_chunks.append(no_valid_cf)

        actual_pred = jnp.concatenate(actual_chunks, axis=0)
        baseline = jnp.concatenate(baseline_chunks, axis=0)
        all_preds = jnp.concatenate(all_pred_chunks, axis=0)
        no_valid_cf = jnp.concatenate(no_valid_chunks, axis=0)
        no_valid_cf_rate = jnp.mean(no_valid_cf.astype(jnp.float32))
        return actual_pred, baseline, all_preds, no_valid_cf_rate, no_valid_cf

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

    @partial(jax.jit, static_argnums=(0, 2))
    def _compute_gram(
        self,
        batch: Dict[str, jnp.ndarray],
        compute_spectrum_metrics: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute per-agent Gram matrices for one mini-batch chunk.

        Exploits the block-diagonal structure of the agent-blocked kernel: instead of
        one dense (B, B) Gram (where N²-1/N² of entries are structural cross-agent zeros),
        computes N independent (B_n, B_n) Grams via jax.vmap. Requires B % N == 0
        (guaranteed by __init__ rounding of causal_mini_batch_size).

        Returns:
            gram_pa:     (N, B_n, B_n) — per-agent Gram matrices.
            reg_gram_pa: (N, B_n, B_n) — Tikhonov-regularized per-agent Grams.
            gram_metrics: dict with off-diagonal and eigen-spectrum summaries
                averaged across agents.
        """
        B   = batch["agent_ids"].shape[0]
        N   = self.num_agents
        B_n = B // N

        loss_mask = batch["loss_mask"].astype(jnp.float32)  # (B,)

        # Kernel bandwidths — gamma_z/gamma_a/gamma_xobs are static (dim-based).
        gamma_z    = 1.0 / max(1, batch["z"].shape[-1])
        gamma_a    = 1.0
        gamma_xobs = 1.0 / max(1, batch["x_obs"].shape[-1])

        # gamma_xhid requires a data-dependent effective-dim estimate; compute globally
        # from the full B samples before the per-agent split so it stays a single scalar.
        gamma_xhid = jnp.array(1.0, dtype=jnp.float32)
        if self.bridge_hidden_dim > 0:
            xhid_hidden_all = batch["x_hid"][:, :self.bridge_hidden_dim]  # (B, bridge_hidden_dim)
            count_g = jnp.maximum(jnp.sum(loss_mask), 1.0)
            mean_g  = (loss_mask[:, None] * xhid_hidden_all).sum(0) / count_g
            cx_g    = xhid_hidden_all - mean_g
            var_g   = (loss_mask[:, None] * jnp.square(cx_g)).sum(0) / count_g
            xhid_norm_g = cx_g / jnp.sqrt(jnp.clip(var_g, a_min=1.0e-6))
            eff_dim = jnp.mean(jnp.sum(jnp.abs(xhid_norm_g) > 1.0e-6, axis=-1).astype(jnp.float32))
            gamma_xhid = 1.0 / jnp.maximum(eff_dim, 1.0)

        def _rbf_gram(x: jnp.ndarray, gamma) -> jnp.ndarray:
            x_sq     = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
            dists_sq = jnp.clip(x_sq - 2.0 * (x @ x.T) + x_sq.T, a_min=0.0)
            return jnp.exp(-gamma * dists_sq)

        # Split (B, *) tensors into (N, B_n, *) by reshaping to (B_n, N, *) then transposing.
        # Valid because agent_ids within any chunk always follows [0,1,...,N-1] repeating
        # (guaranteed since chunk start is always a multiple of N when B is a multiple of N).
        def _split(x: jnp.ndarray) -> jnp.ndarray:
            if x.ndim == 1:
                return x.reshape(B_n, N).T                            # (N, B_n)
            return x.reshape(B_n, N, x.shape[-1]).transpose(1, 0, 2)  # (N, B_n, d)

        z_pa    = _split(batch["z"])                                               # (N, B_n, z_dim)
        aoh_pa  = _split(jax.nn.one_hot(batch["actions"], self.action_dim, dtype=jnp.float32))
        xobs_pa = _split(batch["x_obs"])                                           # (N, B_n, policy_obs_dim)
        xhid_pa = _split(batch["x_hid"])                                           # (N, B_n, xhid_dim)
        mask_pa = _split(loss_mask)                                                # (N, B_n)

        def _standardize_n(x_n: jnp.ndarray, mask_n: jnp.ndarray) -> jnp.ndarray:
            """Masked per-agent standardization: (B_n, d), (B_n,) → (B_n, d)."""
            if x_n.shape[-1] == 0:
                return x_n
            count = jnp.maximum(jnp.sum(mask_n), 1.0)
            mean  = (mask_n[:, None] * x_n).sum(0) / count
            cx    = x_n - mean
            var   = (mask_n[:, None] * jnp.square(cx)).sum(0) / count
            return cx / jnp.sqrt(jnp.clip(var, a_min=1.0e-6))

        def _compute_one_agent(z_n, aoh_n, xobs_n, xhid_n, mask_n):
            """Compute (K_n, reg_K_n) each (B_n, B_n) for agent n's sample block."""
            valid_n = mask_n[:, None] * mask_n[None, :]           # (B_n, B_n)
            eye_n   = jnp.eye(B_n, dtype=jnp.float32)

            z_norm_n    = _standardize_n(z_n, mask_n)
            xobs_cont_n = _standardize_n(xobs_n[:, :self.obs_dim], mask_n)
            # Agent ID one-hot is constant within each agent block → zero inter-sample distance
            # contribution; keep for API consistency with the non-split formulation.
            xobs_norm_n = jnp.concatenate([xobs_cont_n, xobs_n[:, self.obs_dim:]], axis=-1)

            component_kernels = [
                _rbf_gram(z_norm_n, gamma_z),
                _rbf_gram(aoh_n, gamma_a),
                _rbf_gram(xobs_norm_n, gamma_xobs),
            ]

            if self.bridge_hidden_dim > 0:
                xhid_h_n = _standardize_n(xhid_n[:, :self.bridge_hidden_dim], mask_n)
                component_kernels.append(_rbf_gram(xhid_h_n, gamma_xhid))

            if self.bridge_prev_action_dim > 0:
                # Other agents' prev-actions: already float one-hot, no standardization needed.
                component_kernels.append(_rbf_gram(xhid_n[:, self.bridge_hidden_dim:], 1.0))

            K_n = component_kernels[0]
            for component_K in component_kernels[1:]:
                K_n = K_n * component_K

            K_n     = K_n * valid_n
            reg_K_n = (self.tikhonov_gamma + 1.0e-6) * eye_n + self.stabilizer_lambda * K_n
            return K_n, reg_K_n

        gram_pa, reg_gram_pa = jax.vmap(_compute_one_agent)(z_pa, aoh_pa, xobs_pa, xhid_pa, mask_pa)
        # gram_pa: (N, B_n, B_n),  reg_gram_pa: (N, B_n, B_n)

        # Metrics: aggregate off-diagonal statistics across all N agent blocks.
        eye_n         = jnp.eye(B_n, dtype=jnp.float32)
        offdiag_pa    = mask_pa[:, :, None] * mask_pa[:, None, :] * (1.0 - eye_n[None])  # (N, B_n, B_n)
        off_count     = jnp.clip(jnp.sum(offdiag_pa), a_min=1.0)
        offdiag_vals  = gram_pa * offdiag_pa
        kz_off_mean   = jnp.sum(offdiag_vals) / off_count
        kz_off_std    = jnp.sqrt(jnp.sum(jnp.square((gram_pa - kz_off_mean) * offdiag_pa)) / off_count)

        gram_metrics = {
            "Kz_offdiag_mean": kz_off_mean,
            "Kz_offdiag_std":  kz_off_std,
        }

        if compute_spectrum_metrics:
            # Eigen-spectrum diagnostics are O(B_n^3) per agent block, so keep them
            # out of the training update path and compute them only for eval snapshots.
            gram_sym = 0.5 * (gram_pa + jnp.swapaxes(gram_pa, -1, -2))
            eigvals = jnp.clip(jnp.linalg.eigvalsh(gram_sym), a_min=0.0)  # (N, B_n), ascending
            eig_sum = jnp.sum(eigvals, axis=-1)
            eig_prob = eigvals / jnp.clip(eig_sum[:, None], a_min=1.0e-8)
            eig_entropy = -jnp.sum(
                jnp.where(
                    eig_prob > 0.0,
                    eig_prob * jnp.log(jnp.clip(eig_prob, a_min=1.0e-12)),
                    0.0,
                ),
                axis=-1,
            )
            has_spectrum = eig_sum > 1.0e-8
            effective_rank = jnp.where(has_spectrum, jnp.exp(eig_entropy), 0.0)
            valid_counts = jnp.maximum(jnp.sum(mask_pa, axis=-1), 1.0)
            effective_rank_norm = effective_rank / valid_counts

            eig_pos = eigvals > 1.0e-8
            eig_min_pos = jnp.min(jnp.where(eig_pos, eigvals, jnp.inf), axis=-1)
            eig_min_pos = jnp.where(jnp.isfinite(eig_min_pos), eig_min_pos, 0.0)
            eig_max = eigvals[:, -1]
            eig_condition = eig_max / jnp.clip(eig_min_pos, a_min=1.0e-8)

            eig_top1 = eigvals[:, -1] if B_n >= 1 else jnp.zeros((N,), dtype=jnp.float32)
            eig_top2 = eigvals[:, -2] if B_n >= 2 else jnp.zeros((N,), dtype=jnp.float32)
            eig_top3 = eigvals[:, -3] if B_n >= 3 else jnp.zeros((N,), dtype=jnp.float32)
            eig_top4 = eigvals[:, -4] if B_n >= 4 else jnp.zeros((N,), dtype=jnp.float32)
            eig_top5 = eigvals[:, -5] if B_n >= 5 else jnp.zeros((N,), dtype=jnp.float32)
            eig_sum_safe = jnp.clip(eig_sum, a_min=1.0e-8)

            gram_metrics.update({
                "K_effective_rank_mean": jnp.mean(effective_rank),
                "K_effective_rank_std": jnp.std(effective_rank),
                "K_effective_rank_norm_mean": jnp.mean(effective_rank_norm),
                "K_effective_rank_norm_std": jnp.std(effective_rank_norm),
                "K_eig_trace_mean": jnp.mean(eig_sum),
                "K_eig_max_mean": jnp.mean(eig_max),
                "K_eig_min_pos_mean": jnp.mean(eig_min_pos),
                "K_eig_condition_mean": jnp.mean(eig_condition),
                "K_eig_log_condition_mean": jnp.mean(jnp.log1p(eig_condition)),
                "K_eig_top1_mean": jnp.mean(eig_top1),
                "K_eig_top2_mean": jnp.mean(eig_top2),
                "K_eig_top3_mean": jnp.mean(eig_top3),
                "K_eig_top4_mean": jnp.mean(eig_top4),
                "K_eig_top5_mean": jnp.mean(eig_top5),
                "K_eig_top1_frac_mean": jnp.mean(eig_top1 / eig_sum_safe),
                "K_eig_top2_frac_mean": jnp.mean(eig_top2 / eig_sum_safe),
                "K_eig_top3_frac_mean": jnp.mean(eig_top3 / eig_sum_safe),
                "K_eig_top4_frac_mean": jnp.mean(eig_top4 / eig_sum_safe),
                "K_eig_top5_frac_mean": jnp.mean(eig_top5 / eig_sum_safe),
            })
        return gram_pa, reg_gram_pa, gram_metrics

    @partial(jax.jit, static_argnums=0)
    def _bridge_grad_step(
        self,
        params: Any,
        batch: Dict[str, jnp.ndarray],
        gram_pa: jnp.ndarray,      # (N, B_n, B_n)
        reg_gram_pa: jnp.ndarray,  # (N, B_n, B_n)
    ):
        """Compute bridge gradients for one chunk using per-agent precomputed Gram matrices.

        The bridge forward pass runs on the full (B,) batch. Residuals are then split per
        agent — matching the (N, B_n, B_n) block structure of gram_pa — and each agent's
        KRR loss is solved independently via jax.vmap. This eliminates the O(B³) dense
        solve and replaces it with N independent O(B_n³) solves, where B_n = B // N.
        """
        def loss_fn(params):
            B   = batch["x_obs"].shape[0]
            N   = self.num_agents
            B_n = B // N

            preds = self.bridge_state.apply_fn(
                {"params": params},
                batch["x_obs"], batch["w"], batch["x_hid"], batch["actions"],
            )
            residuals = batch["targets"] - preds  # (B, d_y)

            # Split to per-agent blocks: (B, d_y) → (N, B_n, d_y)
            res_pa  = residuals.reshape(B_n, N, -1).transpose(1, 0, 2)  # (N, B_n, d_y)
            mask_pa = batch["loss_mask"].reshape(B_n, N).T               # (N, B_n)

            def _agent_loss(gram_n, reg_gram_n, res_n, mask_n):
                """Per-agent KRR loss: scalar mean over active samples in agent n's block."""
                solved_n = jnp.linalg.solve(reg_gram_n, res_n)               # (B_n, d_y)
                # gram_pa masking already zeros loss for masked-out samples; jnp.where guards NaN/Inf.
                per_sample_n = jnp.where(
                    mask_n > 0,
                    jnp.sum(res_n * (gram_n @ solved_n), axis=-1),
                    0.0,
                )                                                              # (B_n,)
                return jnp.sum(per_sample_n) / jnp.maximum(jnp.sum(mask_n), 1.0)

            per_agent_losses = jax.vmap(_agent_loss)(gram_pa, reg_gram_pa, res_pa, mask_pa)  # (N,)
            outcome_bridge_loss = jnp.mean(per_agent_losses)
            action_bridge_loss  = jnp.array(0.0, dtype=jnp.float32)

            metrics = {
                "outcome_bridge_loss": outcome_bridge_loss,
                "action_bridge_loss":  action_bridge_loss,
            }
            for agent_id in range(self.num_agents):
                metrics[f"agent_{agent_id}_outcome_bridge_loss"] = per_agent_losses[agent_id]
            return outcome_bridge_loss + action_bridge_loss, metrics

        grads, metrics = jax.grad(loss_fn, has_aux=True)(params)
        return grads, metrics

    @partial(jax.jit, static_argnums=0)
    def _naive_predictor_grad_step(
        self,
        params: Any,
        batch: Dict[str, jnp.ndarray],
    ):
        def loss_fn(params):
            preds = self.naive_predictor_state.apply_fn(
                {"params": params},
                batch["x_obs"], batch["w"], batch["x_hid"], batch["actions"],
            )
            per_sample = jnp.mean(jnp.square(batch["targets"] - preds), axis=-1)
            mask = batch["loss_mask"]
            naive_loss = jnp.sum(per_sample * mask) / jnp.maximum(jnp.sum(mask), 1.0)
            return naive_loss, {"naive_loss": naive_loss}

        grads, metrics = jax.grad(loss_fn, has_aux=True)(params)
        return grads, metrics

    def _slice_batch(self, batch: Dict[str, jnp.ndarray], idx: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        return slice_batch(batch, idx)

    def _iterate_minibatches(self, size: int):
        yield from minibatch_indices(
            size,
            batch_size=self.batch_size,
            num_batches=self.num_batches,
        )

    def _iterate_causal_minibatches(self, te_total: int):
        yield from causal_minibatch_indices(
            te_total,
            num_agents=self.num_agents,
            causal_mini_batch_size=self.causal_mini_batch_size,
        )

    def update_causal_networks(self, rollout: Optional[Dict[str, jnp.ndarray]] = None) -> Dict[str, float]:
        if rollout is None:
            rollout = self.latest_rollout
        if rollout is None:
            return {}
        if self.bridge_state is None:
            return {"outcome_bridge_loss": 0.0, "action_bridge_loss": 0.0}

        causal_batch = self._causal_batch(rollout)
        total_flat = causal_batch["x_obs"].shape[0]   # T*E*N
        te_total   = total_flat // self.num_agents     # T*E — iteration unit is (t,e) pairs
        _KEYS = ("z", "w", "x_obs", "x_hid", "actions", "targets", "agent_ids", "loss_mask")

        # Gram/reg_gram depend only on rollout data, not on bridge params, so compute
        # them once per chunk and reuse across all bridge_epoch gradient accumulations.
        chunks = []
        gram_metric_entries = []
        for idx in self._iterate_causal_minibatches(te_total):
            batch = {k: causal_batch[k][idx] for k in _KEYS}
            gram, reg_gram, gram_metrics = self._compute_gram(batch, False)
            chunks.append((batch, gram, reg_gram))
            gram_metric_entries.append({key: float(value) for key, value in tree_to_host(gram_metrics).items()})

        loss_metric_entries = []
        grad_norms = []
        naive_metric_entries = []
        naive_grad_norms = []
        for _ in range(self.bridge_epoch):
            base_params = self.bridge_state.params
            grads_sum = None
            for batch, gram, reg_gram in chunks:
                grads, batch_metrics = self._bridge_grad_step(base_params, batch, gram, reg_gram)
                grads_sum = grads if grads_sum is None else jax.tree_util.tree_map(jnp.add, grads_sum, grads)
                loss_metric_entries.append({key: float(value) for key, value in tree_to_host(batch_metrics).items()})

            grads_avg = jax.tree_util.tree_map(lambda g: g / len(chunks), grads_sum)
            self.bridge_state = self.bridge_state.apply_gradients(grads=grads_avg)
            grad_norms.append(float(global_norm(grads_avg)))

            if self.naive_predictor_state is not None:
                naive_params = self.naive_predictor_state.params
                naive_grads_sum = None
                for batch, _, _ in chunks:
                    naive_grads, naive_metrics = self._naive_predictor_grad_step(naive_params, batch)
                    naive_grads_sum = (
                        naive_grads
                        if naive_grads_sum is None
                        else jax.tree_util.tree_map(jnp.add, naive_grads_sum, naive_grads)
                    )
                    naive_metric_entries.append({key: float(value) for key, value in tree_to_host(naive_metrics).items()})

                naive_grads_avg = jax.tree_util.tree_map(lambda g: g / len(chunks), naive_grads_sum)
                self.naive_predictor_state = self.naive_predictor_state.apply_gradients(grads=naive_grads_avg)
                naive_grad_norms.append(float(global_norm(naive_grads_avg)))

        stacked = stack_metrics(loss_metric_entries)
        stacked.update(stack_metrics(gram_metric_entries))
        if naive_metric_entries:
            stacked.update(stack_metrics(naive_metric_entries))
        stacked["bridge_grad_norm"] = float(sum(grad_norms) / len(grad_norms))
        if naive_grad_norms:
            stacked["naive_grad_norm"] = float(sum(naive_grad_norms) / len(naive_grad_norms))
        stacked.setdefault("outcome_bridge_loss", 0.0)
        stacked.setdefault("action_bridge_loss", 0.0)
        stacked.setdefault("Kz_offdiag_mean", 0.0)
        stacked.setdefault("Kz_offdiag_std", 0.0)
        return stacked

    def compute_gram_spectrum_metrics(
        self,
        rollout: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Dict[str, float]:
        """Compute expensive Gram spectrum diagnostics for eval-time logging."""
        if rollout is None:
            rollout = self.latest_rollout
        if rollout is None or self.bridge_state is None:
            return {}

        causal_batch = self._causal_batch(rollout)
        total_flat = causal_batch["x_obs"].shape[0]
        te_total = total_flat // self.num_agents
        keys = ("z", "w", "x_obs", "x_hid", "actions", "targets", "agent_ids", "loss_mask")
        metric_entries = []
        for idx in self._iterate_causal_minibatches(te_total):
            batch = {key: causal_batch[key][idx] for key in keys}
            _, _, gram_metrics = self._compute_gram(batch, True)
            metric_entries.append({key: float(value) for key, value in tree_to_host(gram_metrics).items()})

        metrics = stack_metrics(metric_entries)
        important_keys = (
            "K_effective_rank_norm_mean",
            "K_eig_top1_frac_mean",
            "Kz_offdiag_mean",
            "K_eig_log_condition_mean",
        )
        return {key: metrics[key] for key in important_keys if key in metrics}

    def compute_causal_rewards(
        self,
        rollout: Dict[str, jnp.ndarray],
        baseline_type: str = "weighted",
    ) -> tuple[jnp.ndarray, Dict[str, float]]:
        if baseline_type not in ("weighted", "full_weighted", "averaged", "max"):
            raise ValueError(f"Unsupported baseline_type for JAXRPCIAlgorithm: {baseline_type}")
        if self.bridge_state is None or self.get_current_causal_lambda() == 0.0:
            zeros = jnp.zeros(rollout["actions"].shape + (self.num_agents,), dtype=jnp.float32)
            return zeros, {
                "causal_lambda": self.get_current_causal_lambda(),
                "delta_mean": 0.0,
                "delta_std": 0.0,
            }

        local_obs, others_obs, hidden_others, actions, reward_targets, probs, avail_actions = self._bridge_inputs(rollout)
        use_naive_predictor = self.use_naive_predictor_for_delta
        if use_naive_predictor and self.naive_predictor_state is None:
            raise RuntimeError(
                "The observational-regressor delta requires train_naivepredictor=True "
                "and causal_lambda > 0.0."
            )
        prediction_state = self.naive_predictor_state if use_naive_predictor else self.bridge_state
        actual_pred, baseline_weighted, all_preds, no_valid_cf_rate, no_valid_cf = self._bridge_predictions(
            use_naive_predictor,
            prediction_state.params,
            local_obs,
            others_obs,
            hidden_others,
            probs,
            actions,
            avail_actions,
        )

        if baseline_type == "weighted":
            baseline = baseline_weighted
            effective_no_valid_cf = no_valid_cf # no_valid_cf: policy concentrates prob on factual action → renorm ≈ 0
        elif baseline_type == "full_weighted":
            baseline = jnp.sum(probs[..., None] * all_preds, axis=1)
            effective_no_valid_cf = jnp.zeros(actions.shape, dtype=jnp.bool_)
        elif baseline_type == "averaged":
            factual_mask = 1.0 - jax.nn.one_hot(actions.astype(jnp.int32), self.action_dim, dtype=jnp.float32)
            
            valid_non_factual = factual_mask * avail_actions  # (B*N, action_dim)
            # valid_non_factual = factual_mask # consider all actions (including unavailable ones) as counterfactuals.
            
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

        actual_team_reward = reward_targets
        raw_effects = actual_team_reward - baseline
        raw_effects = jnp.where(effective_no_valid_cf[:, None], jnp.zeros_like(raw_effects), raw_effects)
        
        alive = self._alive_from_agent_dones(rollout["agent_dones"]).astype(jnp.float32)
            
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
            "use_naive_predictor_for_delta": float(use_naive_predictor),
            "no_valid_cf_rate": float(no_valid_cf_rate),
            "no_valid_cf_rate_alive": no_valid_cf_rate_alive,   # Policy entropy metric.
            "no_valid_cf_rate_dead": no_valid_cf_rate_dead,     # Usually ~= 1.0 as a sanity check.
            "dead_agent_fraction": dead_fraction,               # Team survival metric.
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
        actor_size = int(actor_batch["obs"].shape[0])
        critic_size = int(critic_batch["obs"].shape[0])
        for _ in range(self.ppo_epochs):
            self.rng, actor_perm_rng = jax.random.split(self.rng)
            actor_perm = jax.random.permutation(actor_perm_rng, actor_size)

            # shuffle
            for idx in self._iterate_minibatches(actor_size):
                idx = actor_perm[idx]
                actor_minibatch = self._slice_batch(actor_batch, idx)
                self.actor_state, actor_metrics = self._actor_update_step(self.actor_state, actor_minibatch)
                metrics.append({key: float(value) for key, value in tree_to_host(actor_metrics).items()})

            self.rng, critic_perm_rng = jax.random.split(self.rng)
            critic_perm = jax.random.permutation(critic_perm_rng, critic_size)

            # shuffle
            for idx in self._iterate_minibatches(critic_size):
                idx = critic_perm[idx]
                critic_minibatch = self._slice_batch(critic_batch, idx)
                self.critic_state, critic_metrics = self._critic_update_step(self.critic_state, critic_minibatch)
                metrics.append({key: float(value) for key, value in tree_to_host(critic_metrics).items()})

        return stack_metrics(metrics)

    def update(self, **_: Any) -> Dict[str, float]:
        if self.latest_rollout is None:
            return {}

        rollout = self.latest_rollout # training data for one iteration. handled in runner/jax/trainer.py
        metrics = {}
        causal_lambda = self.get_current_causal_lambda()

        if self.bridge_state is not None:
            if self.anneal_lr and self.total_iterations > 0:
                frac = max(0.0, 1.0 - self.current_iteration / self.total_iterations)
            else:
                frac = 1.0
            bridge_lr = self.lr_bridge * frac
            self.bridge_state = self.bridge_state.replace(
                learning_rate=jnp.asarray(bridge_lr, dtype=jnp.float32)
            )
            if self.naive_predictor_state is not None:
                self.naive_predictor_state = self.naive_predictor_state.replace(
                    learning_rate=jnp.asarray(bridge_lr, dtype=jnp.float32)
                )
            metrics["lr_bridge"] = bridge_lr

        # Current actor/critic LR, for monitoring the anneal_lr schedule in wandb.
        lr_step = self.current_iteration * self._lr_steps_per_iter
        metrics["lr_actor"] = self._current_lr(self._lr_actor_schedule, lr_step)
        metrics["lr_critic"] = self._current_lr(self._lr_critic_schedule, lr_step)
        if self.delta_critic_state is not None:
            metrics["lr_delta_critic"] = metrics["lr_critic"]

        bridge_metrics = self.update_causal_networks(rollout)
        metrics.update(bridge_metrics)

        c_matrix, pci_metrics = self.compute_causal_rewards(rollout, baseline_type="weighted")
        delta = c_matrix.sum(axis=-1)
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

        if self.delta_critic_state is not None:
            delta_values = rollout["delta_values"]
            last_delta_value = rollout["last_delta_value"]
            pci_advantages, delta_returns = compute_gae(
                delta,
                delta_values,
                last_delta_value,
                rollout["dones"],
                self.gamma,
                self.gae_lambda,
            )
            delta_critic_batch = self._flatten_delta_critic_batch(rollout, delta_returns)
            raw_dc: list = []
            delta_critic_size = int(delta_critic_batch["obs"].shape[0])
            for _ in range(self.ppo_epochs):
                self.rng, delta_critic_perm_rng = jax.random.split(self.rng)
                delta_critic_perm = jax.random.permutation(delta_critic_perm_rng, delta_critic_size)
                for idx in self._iterate_minibatches(delta_critic_size):
                    idx = delta_critic_perm[idx]
                    delta_critic_minibatch = self._slice_batch(delta_critic_batch, idx)
                    self.delta_critic_state, dc_metrics = self._delta_critic_update_step(
                        self.delta_critic_state,
                        delta_critic_minibatch,
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
        metrics.update(self._categorized_action_delta_metrics(rollout, delta))

        metrics.update(self._update_actor_critic(rollout, combined_advantages, returns))
        self.training_steps += int(rollout["actions"].shape[0] * rollout["actions"].shape[1])
        self.current_iteration += 1
        self.latest_rollout = None
        return metrics

    def greedy_actions(
        self,
        obs: jnp.ndarray,
        avail_actions: jnp.ndarray,
        runner_state: Dict[str, jnp.ndarray],
    ) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        obs_aug = self._augment_obs(obs)
        actor_hidden_next, actions, _, _ = self._policy_step(
            self.actor_state.params,
            obs_aug,
            avail_actions,
            runner_state["actor_hidden"],
            runner_state["prev_actions"],
            runner_state["done_prev"],
            jax.random.PRNGKey(0),
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
            "naive_predictor_state": serialization.to_state_dict(self.naive_predictor_state) if self.naive_predictor_state is not None else None,
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
        if self.naive_predictor_state is not None and state_dict.get("naive_predictor_state") is not None:
            self.naive_predictor_state = serialization.from_state_dict(
                self.naive_predictor_state,
                state_dict["naive_predictor_state"],
            )
        if self.delta_critic_state is not None and state_dict.get("delta_critic_state") is not None:
            self.delta_critic_state = serialization.from_state_dict(self.delta_critic_state, state_dict["delta_critic_state"])
        self.rng = jnp.asarray(state_dict["rng"])
        if "bridge_rng" in state_dict:
            self.bridge_rng = jnp.asarray(state_dict["bridge_rng"])
        self.training_steps = int(state_dict.get("training_steps", 0))

    def load_actor_state_dict(self, state_dict: Dict[str, Any]):
        actor_state_dict = state_dict["actor_state"]
        try:
            self.actor_state = serialization.from_state_dict(self.actor_state, actor_state_dict)
        except (ValueError, KeyError, TypeError):
            params_state = actor_state_dict.get("params", actor_state_dict)
            params = serialization.from_state_dict(self.actor_state.params, params_state)
            updated_state = self.actor_state.replace(params=params)
            if isinstance(actor_state_dict, dict) and "step" in actor_state_dict:
                updated_state = updated_state.replace(step=jnp.asarray(actor_state_dict["step"]))
            self.actor_state = updated_state
        self.training_steps = int(state_dict.get("training_steps", 0))
