from __future__ import annotations

import copy
from collections import defaultdict
from functools import partial
from typing import Any, Dict, Optional

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np

from .common import (
    CausalBridgeNetwork,
    VERY_NEGATIVE,
    global_norm,
    make_dynamic_lr_train_state,
    stack_metrics,
    tree_to_host,
)
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
        device: str = "cpu",
        add_agent_id: bool = True,
        global_reward: bool = True,
        causal_lambda: float = 1.0,
        bridge_epoch: int = 2,
        causal_mini_batch_size: int = 3200,
        tikhonov_gamma: float = 1.0e-4,
        stabilizer_lambda: float = 1.0,
        clip_delta: float = 0.0,
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
    ) -> Dict[str, jnp.ndarray]:
        obs = jnp.swapaxes(batch["agent_obs"][:, :-1], 0, 1)
        actions = jnp.swapaxes(batch["actions"][:, :-1], 0, 1)
        rewards = jnp.swapaxes(batch["agent_rewards"], 0, 1)
        masks = jnp.swapaxes(batch["masks"], 0, 1)
        dones = jnp.swapaxes(batch["terminated"], 0, 1)
        avail_actions = jnp.swapaxes(batch["available_actions"][:, :-1], 0, 1)
        actor_hidden = jnp.swapaxes(batch["actor_hidden"], 0, 1)
        prev_actions = jnp.swapaxes(batch["prev_actions"], 0, 1)
        agent_dones = jnp.swapaxes(batch["agent_dones"], 0, 1)

        if mac_out is None:
            probs = jnp.ones_like(avail_actions) / float(self.action_dim)
        else:
            q_values = jnp.swapaxes(mac_out[:, :-1], 0, 1)
            masked_q = self._masked_q_values(q_values, avail_actions)
            probs = jax.nn.softmax(masked_q, axis=-1)

        return {
            "obs": obs.astype(jnp.float32),
            "avail_actions": avail_actions.astype(jnp.float32),
            "actor_hidden": actor_hidden.astype(jnp.float32),
            "prev_actions": prev_actions.astype(jnp.float32),
            "actions": actions.astype(jnp.int32),
            "rewards": rewards.astype(jnp.float32),
            "dones": dones.astype(jnp.float32),
            "masks": masks.astype(jnp.float32),
            "agent_dones": agent_dones.astype(jnp.float32),
            "probs": probs.astype(jnp.float32),
        }

    def _masked_delta_metrics(
        self,
        delta_agent: jnp.ndarray,
        masks: jnp.ndarray,
    ) -> Dict[str, float]:
        agent_masks = jnp.broadcast_to(masks[..., None], delta_agent.shape)
        denom = jnp.maximum(jnp.sum(agent_masks), 1.0)
        mean = jnp.sum(delta_agent * agent_masks) / denom
        centered = (delta_agent - mean) * agent_masks
        std = jnp.sqrt(jnp.sum(jnp.square(centered)) / denom)
        abs_mean = jnp.sum(jnp.abs(delta_agent) * agent_masks) / denom
        team_delta = jnp.sum(delta_agent, axis=-1)
        team_mean = jnp.sum(team_delta * masks) / jnp.maximum(jnp.sum(masks), 1.0)
        return {
            "delta_mean": float(mean),
            "delta_std": float(std),
            "delta_abs_mean": float(abs_mean),
            "delta_team_mean": float(team_mean),
        }

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

            agent_targets = (
                batch["pci_agent_rewards"]
                + self.gamma * (1.0 - batch["terminated"][..., None]) * target_max_q
            )
            agent_masks = jnp.broadcast_to(batch["masks"][..., None], chosen_q.shape)
            agent_td_error = chosen_q - jax.lax.stop_gradient(agent_targets)
            masked_agent_td_error = agent_td_error * agent_masks
            agent_denom = jnp.maximum(jnp.sum(agent_masks), 1.0)
            agent_delta_td_loss = jnp.sum(jnp.square(masked_agent_td_error)) / agent_denom

            effective_agent_delta_td_coef = self.agent_delta_td_coef * causal_lambda
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
                "agent_td_error_abs": jnp.sum(jnp.abs(masked_agent_td_error)) / agent_denom,
                "q_taken_mean": jnp.sum(chosen_tot * masks) / (denom * self.num_agents),
                "target_mean": jnp.sum(targets * masks) / (denom * self.num_agents),
            }
            return loss, metrics

        grads, metrics = jax.grad(loss_fn, has_aux=True)(train_state_value.params)
        new_train_state = train_state_value.apply_gradients(grads=grads)
        metrics["grad_norm"] = global_norm(grads)
        return new_train_state, metrics

    def update(self, **kwargs: Any) -> Dict[str, float]:
        n_available = len(self._completed_episodes)
        num_episodes_collected = int(kwargs.get("num_episodes_collected", 1))
        metadata = {
            "epsilon": float(self.epsilon),
            "total_env_steps": float(self.total_env_steps),
            "replay_buffer_episodes": float(n_available),
            "qmix_update_epochs": float(self.qmix_update_epochs or 0),
            "num_episodes_collected": float(num_episodes_collected),
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

        bridge_indices = self._rng_np.choice(n_available, self.train_batch_size, replace=False)
        bridge_sampled = [self._completed_episodes[int(idx)] for idx in bridge_indices]
        bridge_batch = self._pad_and_stack(bridge_sampled)
        bridge_rollout = self._batch_to_bridge_rollout(bridge_batch)
        bridge_metrics = self.update_causal_networks(bridge_rollout)

        accumulated = defaultdict(float)
        for _ in range(num_updates):
            indices = self._rng_np.choice(n_available, self.train_batch_size, replace=False)
            sampled = [self._completed_episodes[int(idx)] for idx in indices]
            batch = self._pad_and_stack(sampled)

            mac_out = self._forward_agent_q_values(
                self.train_state.params["agent"],
                batch["agent_obs"],
                batch["actions"],
            )
            delta_rollout = self._batch_to_bridge_rollout(batch, mac_out=mac_out)
            c_matrix, pci_metrics = self.compute_causal_rewards(delta_rollout, baseline_type="weighted")
            delta_time_major = c_matrix.sum(axis=-1)
            delta_agent = jnp.swapaxes(delta_time_major, 0, 1)
            if self.clip_delta > 0.0:
                delta_agent = jnp.where(jnp.abs(delta_agent) < self.clip_delta, 0.0, delta_agent)

            train_batch = dict(batch)
            train_batch["pci_agent_rewards"] = delta_agent.astype(jnp.float32)
            train_batch["pci_rewards"] = jnp.sum(delta_agent, axis=-1).astype(jnp.float32)
            self.train_state, train_metrics = self._train_step_pci(
                self.train_state,
                self.target_params,
                train_batch,
            )

            metrics_host = {}
            metrics_host.update({key: float(value) for key, value in tree_to_host(train_metrics).items()})
            metrics_host.update(pci_metrics)
            metrics_host.update(self._masked_delta_metrics(delta_agent, batch["masks"]))
            metrics_host.update(self._categorized_action_delta_metrics(delta_rollout, delta_time_major))
            for key, value in metrics_host.items():
                accumulated[key] += float(value)

            self.training_steps += 1
            if self.training_steps % self.target_update_interval == 0:
                self.target_params = copy.deepcopy(self.train_state.params)

        self.current_iteration += 1
        averaged = {key: value / num_updates for key, value in accumulated.items()}
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
