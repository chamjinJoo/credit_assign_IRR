from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np

from .common import (
    CausalBridgeNetwork,
    gather_other_agent_obs,
    global_norm,
    make_dynamic_lr_train_state,
    tree_to_host,
)
from .preprocessing import masked_delta_metrics, qmix_batch_to_bridge_rollout
from .qmix import JAXQMIXAlgorithm
from .r_pci import JAXRPCIAlgorithm


class JAXPCIQMIXAlgorithm(JAXQMIXAlgorithm):
    """JAX PCI-QMIX using the JAX R-PCI outcome bridge machinery."""

    requires_jax_trainer = True
    use_episode_buffer = True

    # Reuse the R-PCI outcome bridge/KRR implementation verbatim. PCI-QMIX
    # adapts sampled QMIX replay batches to the rollout schema these methods use.
    _alive_from_agent_dones = JAXRPCIAlgorithm._alive_from_agent_dones
    _categorized_action_delta_compute = JAXRPCIAlgorithm._categorized_action_delta_compute
    _preprocess_rollout = JAXRPCIAlgorithm._preprocess_rollout
    _causal_batch = JAXRPCIAlgorithm._causal_batch
    _bridge_inputs = JAXRPCIAlgorithm._bridge_inputs
    _bridge_predictions_chunk = JAXRPCIAlgorithm._bridge_predictions_chunk
    _bridge_predictions = JAXRPCIAlgorithm._bridge_predictions
    _compute_gram = JAXRPCIAlgorithm._compute_gram
    _bridge_grad_step = JAXRPCIAlgorithm._bridge_grad_step
    _slice_batch = JAXRPCIAlgorithm._slice_batch
    _iterate_causal_minibatches = JAXRPCIAlgorithm._iterate_causal_minibatches
    update_causal_networks = JAXRPCIAlgorithm.update_causal_networks
    compute_gram_spectrum_metrics = JAXRPCIAlgorithm.compute_gram_spectrum_metrics
    compute_causal_rewards = JAXRPCIAlgorithm.compute_causal_rewards

    def __init__(
        self,
        env: Any,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 5.0e-4,
        lr_bridge: float = 2.0e-3,
        gamma: float = 0.99,
        max_grad_norm: float = 10.0,
        target_update_interval: int = 200,
        epsilon_start: float = 1.0,
        epsilon_finish: float = 0.05,
        epsilon_anneal_time: int = 50000,
        double_q: bool = True,
        mixing_embed_dim: int = 32,
        hypernet_layers: int = 2,
        hypernet_embed: int = 64,
        replay_buffer_size: int = 5000,
        train_batch_size: int = 32,
        qmix_update_epochs: Optional[int] = None,
        learning_starts: int = 0,
        device: str = "cpu",
        add_agent_id: bool = True,
        global_reward: bool = True,
        causal_lambda: float = 1.0,
        bridge_epoch: int = 2,
        causal_mini_batch_size: int = 3200,
        tikhonov_gamma: float = 1.0e-4,
        stabilizer_lambda: float = 1.0,
        clip_delta: float = 0.0,
        standardize_delta: bool = False,
        agent_delta_td_coef: float = 1.0,
        bridge_prediction_batch_size: int = 2048,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(
            env=env,
            num_agents=num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            max_grad_norm=max_grad_norm,
            target_update_interval=target_update_interval,
            epsilon_start=epsilon_start,
            epsilon_finish=epsilon_finish,
            epsilon_anneal_time=epsilon_anneal_time,
            double_q=double_q,
            mixing_embed_dim=mixing_embed_dim,
            hypernet_layers=hypernet_layers,
            hypernet_embed=hypernet_embed,
            replay_buffer_size=replay_buffer_size,
            train_batch_size=train_batch_size,
            qmix_update_epochs=qmix_update_epochs,
            learning_starts=learning_starts,
            device=device,
            add_agent_id=add_agent_id,
            global_reward=global_reward,
            seed=seed,
            **kwargs,
        )

        self.causal_lambda = float(causal_lambda)
        self.bridge_epoch = max(1, int(bridge_epoch))
        self.causal_mini_batch_size = max(1, int(causal_mini_batch_size))
        self.tikhonov_gamma = float(tikhonov_gamma)
        self.stabilizer_lambda = float(stabilizer_lambda)
        self.clip_delta = float(clip_delta)
        self.standardize_delta = bool(standardize_delta)
        self.agent_delta_td_coef = float(agent_delta_td_coef)
        self.bridge_prediction_batch_size = max(1, int(bridge_prediction_batch_size))
        self.lr_bridge = float(lr_bridge)
        self.current_iteration = 0
        self.use_recurrent_policy = True
        self.actor_state_dim = self.hidden_dim
        self.bridge_hidden_dim = (self.num_agents - 1) * self.actor_state_dim
        self.bridge_prev_action_dim = (self.num_agents - 1) * self.action_dim
        self.bridge_context_dim = self.bridge_hidden_dim + self.bridge_prev_action_dim
        self.bridge_output_dim = 1 if self.global_reward else self.num_agents

        self.attack_action_threshold = self._infer_attack_action_threshold(self.action_dim)
        allowed_target_matrix = getattr(self.env, "_allowed_target_matrix_jax", None)
        if allowed_target_matrix is None:
            allowed_target_matrix = getattr(self.env, "_allowed_target_matrix_np", None)
        self._allowed_target_matrix = (
            None
            if allowed_target_matrix is None
            else jnp.asarray(allowed_target_matrix, dtype=jnp.float32)
        )

        self.bridge_model = None
        self.bridge_state = None
        if self.causal_lambda > 0.0:
            bridge_rng = jax.random.fold_in(jax.random.PRNGKey(int(seed)), 23)
            self.bridge_model = CausalBridgeNetwork(
                hidden_dim=self.hidden_dim,
                action_dim=self.action_dim,
                num_agents=self.bridge_output_dim,
            )
            bridge_others_obs_dim = (self.num_agents - 1) * self.policy_obs_dim
            self.bridge_state = make_dynamic_lr_train_state(
                self.bridge_model,
                bridge_rng,
                (
                    jnp.zeros((1, self.policy_obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, bridge_others_obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, self.bridge_context_dim), dtype=jnp.float32),
                    jnp.zeros((1,), dtype=jnp.int32),
                ),
                self.lr_bridge,
                self.max_grad_norm,
            )

    def get_current_causal_lambda(self) -> float:
        return float(self.causal_lambda)

    def _infer_attack_action_threshold(self, action_dim: int) -> Optional[int]:
        if self.env is None:
            return 6 if action_dim > 6 else None
        env_name = self.env.__class__.__name__
        if env_name not in {"SMACWrapper", "SMACv2Wrapper", "SMAXWrapper", "SMAXBatchWrapper"}:
            return None
        threshold = getattr(self.env, "n_actions_no_attack", None)
        try:
            if threshold is not None:
                return max(0, min(int(threshold), int(action_dim) - 1))
        except (TypeError, ValueError):
            pass
        return 6 if action_dim > 6 else None

    def _categorized_action_delta_metrics(
        self,
        rollout: Dict[str, jnp.ndarray],
        delta: jnp.ndarray,
    ) -> Dict[str, float]:
        if self.attack_action_threshold is None:
            return {}
        return JAXRPCIAlgorithm._categorized_action_delta_metrics(self, rollout, delta)

    def _batch_to_bridge_rollout(
        self,
        batch: Dict[str, jnp.ndarray],
        mac_out: Optional[jnp.ndarray] = None,
        agent_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, jnp.ndarray]:
        actor_hidden = None
        if agent_params is not None:
            _, actor_hidden = self._forward_agent_q_values_and_hidden(
                agent_params,
                batch["agent_obs"],
                batch["actions"],
            )
            actor_hidden = jax.lax.stop_gradient(actor_hidden[:, :-1])
        return qmix_batch_to_bridge_rollout(
            batch,
            action_dim=self.action_dim,
            masked_q_values=self._masked_q_values,
            mac_out=mac_out,
            actor_hidden=actor_hidden,
        )

    def _masked_delta_metrics(
        self,
        delta_agent: jnp.ndarray,
        masks: jnp.ndarray,
    ) -> Dict[str, float]:
        return masked_delta_metrics(delta_agent, masks)

    @partial(jax.jit, static_argnums=0)
    def _weighted_delta_from_batch(
        self,
        bridge_params: Dict[str, Any],
        agent_params: Dict[str, Any],
        batch: Dict[str, jnp.ndarray],
        mac_out: jnp.ndarray,
    ):
        batch_size = batch["agent_rewards"].shape[0]
        time_steps = batch["agent_rewards"].shape[1]
        flat_batch = batch_size * time_steps

        obs_tm = jnp.swapaxes(batch["agent_obs"][:, :-1], 0, 1)
        obs_4d = obs_tm.reshape(flat_batch, self.num_agents, self.policy_obs_dim)
        local_obs = obs_4d.reshape(-1, self.policy_obs_dim)
        others_obs = gather_other_agent_obs(obs_4d).reshape(
            -1,
            (self.num_agents - 1) * self.policy_obs_dim,
        )

        _, actor_hidden = self._forward_agent_q_values_and_hidden(
            agent_params,
            batch["agent_obs"],
            batch["actions"],
        )
        actor_hidden = jax.lax.stop_gradient(actor_hidden[:, :-1])
        hidden_tm = jnp.swapaxes(actor_hidden, 0, 1)
        hidden_4d = hidden_tm.reshape(flat_batch, self.num_agents, self.actor_state_dim)
        hidden_others = gather_other_agent_obs(hidden_4d).reshape(-1, self.bridge_hidden_dim)

        prev_actions_tm = jnp.swapaxes(batch["prev_actions"], 0, 1)
        prev_actions_4d = prev_actions_tm.reshape(flat_batch, self.num_agents, self.action_dim)
        prev_actions_others = gather_other_agent_obs(prev_actions_4d).reshape(
            -1,
            self.bridge_prev_action_dim,
        )
        bridge_context = jnp.concatenate([hidden_others, prev_actions_others], axis=-1)

        actions_tm = jnp.swapaxes(batch["actions"][:, :-1], 0, 1)
        actions_flat = actions_tm.reshape(-1)

        rewards_tm = jnp.swapaxes(batch["agent_rewards"], 0, 1)
        if self.global_reward:
            team_rewards = rewards_tm.sum(axis=-1) / self.num_agents
            reward_targets = jnp.repeat(team_rewards.reshape(-1), self.num_agents, axis=0)[:, None]
        else:
            reward_targets = jnp.repeat(
                rewards_tm[:, :, None, :],
                self.num_agents,
                axis=2,
            ).reshape(-1, self.num_agents)

        avail_tm = jnp.swapaxes(batch["available_actions"][:, :-1], 0, 1)
        q_tm = jnp.swapaxes(jax.lax.stop_gradient(mac_out[:, :-1]), 0, 1)
        masked_q = self._masked_q_values(q_tm, avail_tm)
        probs = jax.nn.softmax(masked_q, axis=-1).reshape(-1, self.action_dim)
        avail_flat = avail_tm.reshape(-1, self.action_dim)

        flat_rows = local_obs.shape[0]
        chunk_size = min(self.bridge_prediction_batch_size, flat_rows)
        num_chunks = (flat_rows + chunk_size - 1) // chunk_size
        padded_rows = num_chunks * chunk_size
        pad_rows = padded_rows - flat_rows

        def pad_rows_to_chunk(x):
            if pad_rows == 0:
                return x
            pad_width = [(0, 0)] * x.ndim
            pad_width[0] = (0, pad_rows)
            return jnp.pad(x, pad_width)

        local_obs_chunks = pad_rows_to_chunk(local_obs).reshape(
            num_chunks,
            chunk_size,
            self.policy_obs_dim,
        )
        others_obs_chunks = pad_rows_to_chunk(others_obs).reshape(
            num_chunks,
            chunk_size,
            (self.num_agents - 1) * self.policy_obs_dim,
        )
        bridge_context_chunks = pad_rows_to_chunk(bridge_context).reshape(
            num_chunks,
            chunk_size,
            self.bridge_context_dim,
        )
        all_actions = jnp.arange(self.action_dim, dtype=jnp.int32)

        def predict_chunk(chunk):
            local_obs_c, others_obs_c, bridge_context_c = chunk

            def predict_one_action(action_id):
                tiled = jnp.full((chunk_size,), action_id, dtype=jnp.int32)
                return self.bridge_state.apply_fn(
                    {"params": bridge_params},
                    local_obs_c,
                    others_obs_c,
                    bridge_context_c,
                    tiled,
                )

            return jax.vmap(predict_one_action)(all_actions).transpose(1, 0, 2)

        all_preds = jax.lax.map(
            predict_chunk,
            (local_obs_chunks, others_obs_chunks, bridge_context_chunks),
        ).reshape(padded_rows, self.action_dim, self.bridge_output_dim)[:flat_rows]
        factual_mask = 1.0 - jax.nn.one_hot(
            actions_flat.astype(jnp.int32),
            self.action_dim,
            dtype=jnp.float32,
        )
        probs_counterfactual = probs * factual_mask * avail_flat
        renorm = probs_counterfactual.sum(axis=-1, keepdims=True)
        renormalized_probs = probs_counterfactual / jnp.clip(renorm, a_min=1.0e-10)
        baseline = jnp.sum(renormalized_probs[..., None] * all_preds, axis=1)
        no_valid_cf = renorm.squeeze(-1) <= 1.0e-8
        baseline = jnp.where(no_valid_cf[:, None], jnp.zeros_like(baseline), baseline)

        raw_effects = reward_targets - baseline
        raw_effects = jnp.where(no_valid_cf[:, None], jnp.zeros_like(raw_effects), raw_effects)

        agent_dones_tm = jnp.swapaxes(batch["agent_dones"], 0, 1)
        alive = self._alive_from_agent_dones(agent_dones_tm).astype(jnp.float32)
        if self.global_reward:
            actor_team_effects = raw_effects.reshape(time_steps, batch_size, self.num_agents, 1)
            alive_counts = jnp.clip(alive.sum(axis=-1, keepdims=True), a_min=1.0)
            recipient_weights = alive[..., None, :] / alive_counts[..., None, :]
            c_matrix = actor_team_effects * recipient_weights
            c_matrix = c_matrix * alive[..., :, None]
        else:
            c_matrix = raw_effects.reshape(time_steps, batch_size, self.num_agents, self.num_agents)
            c_matrix = c_matrix * alive[..., :, None] * alive[..., None, :]

        delta_time_major = c_matrix.sum(axis=-1)
        delta_agent = jnp.swapaxes(delta_time_major, 0, 1)
        if self.clip_delta > 0.0:
            delta_agent = jnp.where(jnp.abs(delta_agent) < self.clip_delta, 0.0, delta_agent)

        masks = batch["masks"]
        agent_masks = jnp.broadcast_to(masks[..., None], delta_agent.shape)
        denom = jnp.maximum(jnp.sum(agent_masks), 1.0)
        delta_mean = jnp.sum(delta_agent * agent_masks) / denom
        centered = (delta_agent - delta_mean) * agent_masks
        delta_std = jnp.sqrt(jnp.sum(jnp.square(centered)) / denom)
        delta_abs_mean = jnp.sum(jnp.abs(delta_agent) * agent_masks) / denom
        team_delta = jnp.sum(delta_agent, axis=-1)
        team_denom = jnp.maximum(jnp.sum(masks), 1.0)
        delta_team_mean = jnp.sum(team_delta * masks) / team_denom

        alive_flat = alive.reshape(-1).astype(jnp.bool_)
        alive_count = jnp.sum(alive_flat).astype(jnp.float32)
        dead_count = jnp.sum(~alive_flat).astype(jnp.float32)
        metrics = {
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "delta_abs_mean": delta_abs_mean,
            "delta_team_mean": delta_team_mean,
            "no_valid_cf_rate": jnp.mean(no_valid_cf.astype(jnp.float32)),
            "no_valid_cf_rate_alive": jnp.sum(no_valid_cf & alive_flat) / jnp.maximum(alive_count, 1.0),
            "no_valid_cf_rate_dead": jnp.sum(no_valid_cf & ~alive_flat) / jnp.maximum(dead_count, 1.0),
            "dead_agent_fraction": jnp.mean((~alive_flat).astype(jnp.float32)),
        }
        return delta_agent.astype(jnp.float32), delta_time_major.astype(jnp.float32), metrics

    @partial(jax.jit, static_argnums=0)
    def _train_step_pci(
        self,
        train_state_value: Any,
        target_params: Dict[str, Any],
        batch: Dict[str, jnp.ndarray],
    ):
        def loss_fn(params):
            mac_out = self._forward_agent_q_values(
                params["agent"],
                batch["agent_obs"],
                batch["actions"],
            )
            chosen_q = jnp.take_along_axis(
                mac_out[:, :-1],
                batch["actions"][:, :-1, :, None],
                axis=-1,
            ).squeeze(-1)

            target_mac_out = self._forward_agent_q_values(
                target_params["agent"],
                batch["agent_obs"],
                batch["actions"],
            )
            target_future = self._masked_q_values(
                target_mac_out[:, 1:],
                batch["available_actions"][:, 1:],
            )

            if self.double_q:
                online_future = self._masked_q_values(
                    mac_out[:, 1:],
                    batch["available_actions"][:, 1:],
                )
                max_actions = jnp.argmax(online_future, axis=-1)
                target_max_q = jnp.take_along_axis(
                    target_future,
                    max_actions[..., None],
                    axis=-1,
                ).squeeze(-1)
            else:
                target_max_q = jnp.max(target_future, axis=-1)

            chosen_tot = self.mixer_model.apply(
                {"params": params["mixer"]},
                chosen_q,
                batch["joint_obs"][:, :-1],
            )
            target_tot = self.mixer_model.apply(
                {"params": target_params["mixer"]},
                target_max_q,
                batch["joint_obs"][:, 1:],
            )

            env_rewards = batch["rewards"][..., None]
            delta_rewards = batch["pci_rewards"][..., None]
            causal_lambda = jnp.asarray(self.get_current_causal_lambda(), dtype=jnp.float32)
            shaped_rewards = env_rewards + causal_lambda * delta_rewards
            terminated = batch["terminated"][..., None]
            masks = batch["masks"][..., None]
            targets = shaped_rewards + self.gamma * (1.0 - terminated) * target_tot
            td_error = chosen_tot - jax.lax.stop_gradient(targets)
            masked_td_error = td_error * masks
            denom = jnp.maximum(jnp.sum(masks), 1.0)
            qmix_loss = jnp.sum(jnp.square(masked_td_error)) / denom

            if self.agent_delta_td_coef > 0.0:
                agent_targets = (
                    batch["pci_agent_rewards"]
                    + self.gamma * (1.0 - batch["terminated"][..., None]) * target_max_q
                )
                agent_masks = jnp.broadcast_to(batch["masks"][..., None], chosen_q.shape)
                agent_td_error = chosen_q - jax.lax.stop_gradient(agent_targets)
                masked_agent_td_error = agent_td_error * agent_masks
                agent_denom = jnp.maximum(jnp.sum(agent_masks), 1.0)
                agent_delta_td_loss = jnp.sum(jnp.square(masked_agent_td_error)) / agent_denom
                agent_td_error_abs = jnp.sum(jnp.abs(masked_agent_td_error)) / agent_denom
            else:
                agent_delta_td_loss = jnp.array(0.0, dtype=jnp.float32)
                agent_td_error_abs = jnp.array(0.0, dtype=jnp.float32)

            loss = qmix_loss + self.agent_delta_td_coef * agent_delta_td_loss
            metrics = {
                "loss": loss,
                "qmix_loss": qmix_loss,
                "agent_delta_td_loss": agent_delta_td_loss,
                "causal_lambda": causal_lambda,
                "env_reward_mean": jnp.sum(env_rewards * masks) / denom,
                "delta_team_mean": jnp.sum(delta_rewards * masks) / denom,
                "shaped_reward_mean": jnp.sum(shaped_rewards * masks) / denom,
                "td_error_abs": jnp.sum(jnp.abs(masked_td_error)) / denom,
                "agent_td_error_abs": agent_td_error_abs,
                "q_taken_mean": jnp.sum(chosen_tot * masks) / (denom * self.num_agents),
                "target_mean": jnp.sum(targets * masks) / (denom * self.num_agents),
            }
            return loss, metrics

        grads, metrics = jax.grad(loss_fn, has_aux=True)(train_state_value.params)
        new_train_state = train_state_value.apply_gradients(grads=grads)
        metrics["grad_norm"] = global_norm(grads)
        return new_train_state, metrics

    @partial(jax.jit, static_argnums=0)
    def _train_many_pci_steps(
        self,
        train_state_value: Any,
        target_params: Dict[str, Any],
        bridge_params: Dict[str, Any],
        training_steps: jnp.ndarray,
        batches: Dict[str, jnp.ndarray],
    ):
        def train_one(state, tgt_params, bridge_params_value, batch):
            def loss_fn(params):
                mac_out = self._forward_agent_q_values(
                    params["agent"],
                    batch["agent_obs"],
                    batch["actions"],
                )
                delta_agent, _, delta_metrics = self._weighted_delta_from_batch(
                    bridge_params_value,
                    params["agent"],
                    batch,
                    mac_out,
                )
                chosen_q = jnp.take_along_axis(
                    mac_out[:, :-1],
                    batch["actions"][:, :-1, :, None],
                    axis=-1,
                ).squeeze(-1)

                target_mac_out = self._forward_agent_q_values(
                    tgt_params["agent"],
                    batch["agent_obs"],
                    batch["actions"],
                )
                target_future = self._masked_q_values(
                    target_mac_out[:, 1:],
                    batch["available_actions"][:, 1:],
                )

                if self.double_q:
                    online_future = self._masked_q_values(
                        mac_out[:, 1:],
                        batch["available_actions"][:, 1:],
                    )
                    max_actions = jnp.argmax(online_future, axis=-1)
                    target_max_q = jnp.take_along_axis(
                        target_future,
                        max_actions[..., None],
                        axis=-1,
                    ).squeeze(-1)
                else:
                    target_max_q = jnp.max(target_future, axis=-1)

                chosen_tot = self.mixer_model.apply(
                    {"params": params["mixer"]},
                    chosen_q,
                    batch["joint_obs"][:, :-1],
                )
                target_tot = self.mixer_model.apply(
                    {"params": tgt_params["mixer"]},
                    target_max_q,
                    batch["joint_obs"][:, 1:],
                )

                env_rewards = batch["rewards"][..., None]
                causal_lambda = jnp.asarray(self.get_current_causal_lambda(), dtype=jnp.float32)
                terminated = batch["terminated"][..., None]
                masks = batch["masks"][..., None]
                agent_masks = jnp.broadcast_to(batch["masks"][..., None], chosen_q.shape)

                delta_team_raw = jnp.sum(delta_agent, axis=-1)
                team_denom_stats = jnp.maximum(jnp.sum(batch["masks"]), 1.0)
                delta_team_mean = jnp.sum(delta_team_raw * batch["masks"]) / team_denom_stats
                delta_team_var = (
                    jnp.sum(jnp.square(delta_team_raw - delta_team_mean) * batch["masks"])
                    / team_denom_stats
                )
                delta_team_std = jnp.clip(jnp.sqrt(delta_team_var), a_min=1e-8)
                if self.standardize_delta:
                    delta_team = jnp.where(
                        batch["masks"] > 0,
                        (delta_team_raw - delta_team_mean) / delta_team_std,
                        jnp.zeros_like(delta_team_raw),
                    )
                else:
                    delta_team = delta_team_raw
                delta_rewards = delta_team[..., None]
                shaped_rewards = env_rewards + causal_lambda * delta_rewards

                agent_denom_stats = jnp.maximum(jnp.sum(agent_masks), 1.0)
                delta_agent_mean = jnp.sum(delta_agent * agent_masks) / agent_denom_stats
                delta_agent_var = (
                    jnp.sum(jnp.square(delta_agent - delta_agent_mean) * agent_masks)
                    / agent_denom_stats
                )
                delta_agent_std = jnp.clip(jnp.sqrt(delta_agent_var), a_min=1e-8)
                if self.standardize_delta:
                    delta_agent_for_target = jnp.where(
                        agent_masks > 0,
                        (delta_agent - delta_agent_mean) / delta_agent_std,
                        jnp.zeros_like(delta_agent),
                    )
                else:
                    delta_agent_for_target = delta_agent
                agent_shaped_reward = env_rewards + causal_lambda * delta_agent_for_target

                targets = shaped_rewards + self.gamma * (1.0 - terminated) * target_tot
                td_error = chosen_tot - jax.lax.stop_gradient(targets)
                masked_td_error = td_error * masks
                denom = jnp.maximum(jnp.sum(masks), 1.0)
                qmix_loss = jnp.sum(jnp.square(masked_td_error)) / denom

                effective_agent_delta_td_coef = self.agent_delta_td_coef * causal_lambda
                if self.agent_delta_td_coef > 0.0:
                    agent_targets = (
                        agent_shaped_reward
                        + self.gamma * (1.0 - batch["terminated"][..., None]) * target_max_q
                    )
                    agent_td_error = chosen_q - jax.lax.stop_gradient(agent_targets)
                    masked_agent_td_error = agent_td_error * agent_masks
                    agent_delta_td_loss = jnp.sum(jnp.square(masked_agent_td_error)) / agent_denom_stats
                    agent_td_error_abs = jnp.sum(jnp.abs(masked_agent_td_error)) / agent_denom_stats
                else:
                    agent_delta_td_loss = jnp.array(0.0, dtype=jnp.float32)
                    agent_td_error_abs = jnp.array(0.0, dtype=jnp.float32)

                loss = qmix_loss + effective_agent_delta_td_coef * agent_delta_td_loss
                metrics = {
                    "loss": loss,
                    "qmix_loss": qmix_loss,
                    "agent_delta_td_loss": agent_delta_td_loss,
                    "causal_lambda": causal_lambda,
                    "effective_agent_delta_td_coef": effective_agent_delta_td_coef,
                    "env_reward_mean": jnp.sum(env_rewards * masks) / denom,
                    "delta_team_mean": jnp.sum(delta_rewards * masks) / denom,
                    "shaped_reward_mean": jnp.sum(shaped_rewards * masks) / denom,
                    "td_error_abs": jnp.sum(jnp.abs(masked_td_error)) / denom,
                    "agent_td_error_abs": agent_td_error_abs,
                    "q_taken_mean": jnp.sum(chosen_tot * masks) / (denom * self.num_agents),
                    "target_mean": jnp.sum(targets * masks) / (denom * self.num_agents),
                    "standardize_delta": jnp.asarray(float(self.standardize_delta), dtype=jnp.float32),
                    "delta_standardization_mean": delta_team_mean,
                    "delta_standardization_std": delta_team_std,
                    "agent_delta_standardization_mean": delta_agent_mean,
                    "agent_delta_standardization_std": delta_agent_std,
                    "agent_shaped_reward_mean": jnp.sum(agent_shaped_reward * agent_masks)
                    / agent_denom_stats,
                }
                metrics.update(delta_metrics)
                return loss, metrics

            grads, metrics = jax.grad(loss_fn, has_aux=True)(state.params)
            new_state = state.apply_gradients(grads=grads)
            metrics["grad_norm"] = global_norm(grads)
            return new_state, metrics

        def scan_fn(carry, batch):
            state, tgt_params, step_count = carry
            state, metrics = train_one(state, tgt_params, bridge_params, batch)
            step_count = step_count + jnp.asarray(1, dtype=jnp.int32)
            should_update_target = (step_count % self.target_update_interval) == 0
            tgt_params = jax.lax.cond(
                should_update_target,
                lambda _: state.params,
                lambda _: tgt_params,
                operand=None,
            )
            return (state, tgt_params, step_count), metrics

        (new_state, new_target_params, new_training_steps), metrics_seq = jax.lax.scan(
            scan_fn,
            (train_state_value, target_params, training_steps),
            batches,
        )
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics_seq)
        return new_state, new_target_params, new_training_steps, metrics

    def update(self, **kwargs: Any) -> Dict[str, float]:
        n_available = len(self._completed_episodes)
        num_episodes_collected = int(kwargs.get("num_episodes_collected", 1))
        metadata = {
            "epsilon": float(self.epsilon),
            "total_env_steps": float(self.total_env_steps),
            "replay_buffer_episodes": float(n_available),
            "qmix_update_epochs": float(self.qmix_update_epochs or 0),
            "num_episodes_collected": float(num_episodes_collected),
            "learning_starts": float(self.learning_starts),
            "learning_started": float(self.total_env_steps >= self.learning_starts),
            "reward_scale": float(self.reward_scale),
            "mixer_init_scale": float(self.mixer_init_scale),
        }
        if n_available < self.train_batch_size:
            return metadata

        if self.bridge_state is not None:
            self.bridge_state = self.bridge_state.replace(
                learning_rate=jnp.asarray(self.lr_bridge, dtype=jnp.float32)
            )
            metadata["lr_bridge"] = float(self.lr_bridge)

        num_updates = (
            self.qmix_update_epochs
            if self.qmix_update_epochs is not None
            else max(1, num_episodes_collected)
        )

        bridge_batch = self._sample_batch(n_available)
        bridge_rollout = self._batch_to_bridge_rollout(
            bridge_batch,
            agent_params=self.train_state.params["agent"],
        )
        bridge_metrics = self.update_causal_networks(bridge_rollout)

        if self.total_env_steps < self.learning_starts:
            self.current_iteration += 1
            skipped = dict(bridge_metrics)
            skipped.update(metadata)
            skipped["num_updates"] = 0.0
            return skipped

        batches = self._sample_update_batches(n_available, num_updates)

        if self.bridge_state is None:
            self.train_state, self.target_params, training_steps, metrics = self._train_many_steps(
                self.train_state,
                self.target_params,
                jnp.asarray(self.training_steps, dtype=jnp.int32),
                batches,
            )
        else:
            self.train_state, self.target_params, training_steps, metrics = self._train_many_pci_steps(
                self.train_state,
                self.target_params,
                self.bridge_state.params,
                jnp.asarray(self.training_steps, dtype=jnp.int32),
                batches,
            )
        self.training_steps = int(training_steps)

        self.current_iteration += 1
        averaged = {key: float(value) for key, value in tree_to_host(metrics).items()}
        averaged.update(bridge_metrics)
        averaged.update(metadata)
        averaged["num_updates"] = float(num_updates)
        return averaged

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        if self.bridge_state is not None:
            state["bridge_state"] = serialization.to_state_dict(self.bridge_state)
        state["current_iteration"] = int(self.current_iteration)
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)
        if self.bridge_state is not None and "bridge_state" in state_dict:
            self.bridge_state = serialization.from_state_dict(
                self.bridge_state,
                state_dict["bridge_state"],
            )
        self.current_iteration = int(state_dict.get("current_iteration", 0))
