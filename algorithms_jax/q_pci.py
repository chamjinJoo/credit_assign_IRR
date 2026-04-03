from __future__ import annotations

from functools import partial
from typing import Any, Dict

import jax
import jax.numpy as jnp

from .common import (
    CausalBridgeNetwork,
    compute_gae,
    gather_other_agent_obs,
    global_norm,
    make_train_state,
    stack_metrics,
    tree_to_host,
)
from .r_pci import JAXRPCIAlgorithm


class JAXQPCIAlgorithm(JAXRPCIAlgorithm):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # q_pci uses a team-value bridge when global_reward=True, and a vector bridge otherwise.
        self.bridge_model = (
            CausalBridgeNetwork(
                hidden_dim=self.hidden_dim,
                action_dim=self.action_dim,
                num_agents=self.bridge_output_dim,
            )
            if self.causal_lambda > 0.0
            else None
        )
        self.bridge_state = None
        if self.bridge_model is not None:
            self.rng, bridge_key = jax.random.split(self.rng)
            self.bridge_state = make_train_state(
                self.bridge_model,
                bridge_key,
                (
                    jnp.zeros((1, self.policy_obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, (self.num_agents - 1) * self.policy_obs_dim), dtype=jnp.float32),
                    jnp.zeros((1, self.bridge_hidden_dim), dtype=jnp.float32),
                    jnp.zeros((1,), dtype=jnp.int32),
                ),
                kwargs.get("lr_bridge", 3.0e-4),
                self.max_grad_norm,
            )

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

    def _team_rewards(self, rollout: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        return rollout["rewards"].sum(axis=-1) / self.num_agents

    def _reward_signal(self, rollout: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        if self.global_reward:
            return self._team_rewards(rollout)
        return rollout["rewards"]

    def _env_advantages_for_actor(self, env_advantages: jnp.ndarray) -> jnp.ndarray:
        if env_advantages.ndim == 2:
            return jnp.repeat(env_advantages[..., None], self.num_agents, axis=-1)
        return env_advantages

    def _team_bootstrap_values(self, rollout: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        values = rollout["values"]
        last_value = rollout["last_value"]

        if self.use_centralized_critic:
            return jnp.concatenate([values[1:], last_value[None, ...]], axis=0)

        per_agent_next = jnp.concatenate([values[1:], last_value[None, ...]], axis=0)
        if self.global_reward:
            return per_agent_next.mean(axis=-1)
        return per_agent_next

    def _team_bootstrap_targets(self, rollout: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        rewards = self._reward_signal(rollout)
        dones = rollout["dones"].astype(jnp.float32)
        next_values = self._team_bootstrap_values(rollout)
        return rewards + self.gamma * next_values * (1.0 - dones)

    def _q_bridge_batch(self, rollout: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
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
        bootstrap_targets = self._team_bootstrap_targets(rollout)
        if self.global_reward:
            team_targets = bootstrap_targets.reshape(-1, 1)
            if self.normalize_returns:
                mean = jnp.mean(team_targets, axis=0, keepdims=True)
                std = jnp.clip(jnp.std(team_targets, axis=0, keepdims=True), a_min=1.0e-6)
                team_targets = (team_targets - mean) / std
            targets = jnp.repeat(team_targets, self.num_agents, axis=0)
        else:
            team_targets = bootstrap_targets.reshape(-1, self.num_agents)
            if self.normalize_returns:
                mean = jnp.mean(team_targets, axis=0, keepdims=True)
                std = jnp.clip(jnp.std(team_targets, axis=0, keepdims=True), a_min=1.0e-6)
                team_targets = (team_targets - mean) / std
            targets = jnp.repeat(team_targets, self.num_agents, axis=0)

        return {
            "z": z.astype(jnp.float32),
            "w": others_obs.astype(jnp.float32),
            "x_obs": local_obs.astype(jnp.float32),
            "x_hid": hidden_others.astype(jnp.float32),
            "actions": actions.astype(jnp.int32),
            "targets": targets.astype(jnp.float32),
        }

    @staticmethod
    def _rbf_gram(x: jnp.ndarray, gamma: float) -> jnp.ndarray:
        x_norm = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
        dists_sq = x_norm - 2.0 * (x @ x.T) + x_norm.T
        dists_sq = jnp.clip(dists_sq, a_min=0.0)
        return jnp.exp(-gamma * dists_sq)

    @staticmethod
    def _standardize(x: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(x, axis=0, keepdims=True)
        std = jnp.std(x, axis=0, keepdims=True)
        return (x - mean) / jnp.clip(std, a_min=1.0e-6)

    @partial(jax.jit, static_argnums=0)
    def _q_bridge_predictions(
        self,
        bridge_params: Dict[str, Any],
        local_obs: jnp.ndarray,
        others_obs: jnp.ndarray,
        hidden_others: jnp.ndarray,
        probs: jnp.ndarray,
        actions: jnp.ndarray,
        avail_actions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

        q_values = jax.vmap(predict_one_action)(all_actions).transpose(1, 0, 2)
        v_bridge = jnp.sum(probs[..., None] * q_values, axis=1).sum(axis=-1)

        factual_mask = 1.0 - jax.nn.one_hot(actions.astype(jnp.int32), self.action_dim, dtype=jnp.float32)
        probs_counterfactual = probs * factual_mask * avail_actions
        renorm = probs_counterfactual.sum(axis=-1, keepdims=True)
        renormalized_probs = probs_counterfactual / jnp.clip(renorm, a_min=1.0e-8)
        baseline = jnp.sum(renormalized_probs[..., None] * q_values, axis=1)
        no_valid_cf = renorm.squeeze(-1) <= 1.0e-8
        baseline = jnp.where(no_valid_cf[:, None], actual_pred, baseline)

        causal_advantages = jnp.sum(actual_pred - baseline, axis=-1)
        return actual_pred, q_values, v_bridge, causal_advantages

    @partial(jax.jit, static_argnums=0)
    def _q_bridge_update_step(
        self,
        bridge_state: Any,
        batch: Dict[str, jnp.ndarray],
    ):
        def loss_fn(params):
            preds = bridge_state.apply_fn(
                {"params": params},
                batch["x_obs"],
                batch["w"],
                batch["x_hid"],
                batch["actions"],
            )
            residuals = batch["targets"] - preds

            z_norm = self._standardize(batch["z"])
            x_obs_norm = self._standardize(batch["x_obs"])
            action_onehot = jax.nn.one_hot(batch["actions"], self.action_dim, dtype=jnp.float32)

            gamma_z = 1.0 / max(1, batch["z"].shape[-1])
            gamma_a = 1.0 / max(1, self.action_dim)
            gamma_xobs = 1.0 / max(1, batch["x_obs"].shape[-1])

            gram = (
                self._rbf_gram(z_norm, gamma_z)
                * self._rbf_gram(action_onehot, gamma_a)
                * self._rbf_gram(x_obs_norm, gamma_xobs)
            )
            if batch["x_hid"].shape[-1] > 0:
                x_hid_norm = self._standardize(batch["x_hid"])
                gamma_xhid = 1.0 / max(1, batch["x_hid"].shape[-1])
                gram = gram * self._rbf_gram(x_hid_norm, gamma_xhid)

            eye = jnp.eye(gram.shape[0], dtype=gram.dtype)
            reg_gram = (self.tikhonov_gamma + 1.0e-6) * eye + self.stabilizer_lambda * gram
            solved = jnp.linalg.solve(reg_gram, residuals)
            per_sample_loss = jnp.sum(residuals * (gram @ solved), axis=-1)
            q_bridge_loss = jnp.mean(per_sample_loss)

            diag = jnp.diag(gram)
            total_sum = jnp.sum(gram) - jnp.sum(diag)
            off_count = gram.shape[0] * max(gram.shape[0] - 1, 1)
            kz_offdiag_mean = total_sum / off_count
            centered = gram - kz_offdiag_mean
            sq_sum = jnp.sum(jnp.square(centered)) - jnp.sum(jnp.square(diag - kz_offdiag_mean))
            kz_offdiag_std = jnp.sqrt(jnp.clip(sq_sum / off_count, a_min=0.0))

            return q_bridge_loss, {
                "q_bridge_loss": q_bridge_loss,
                "Kz_offdiag_mean": kz_offdiag_mean,
                "Kz_offdiag_std": kz_offdiag_std,
            }

        grads, metrics = jax.grad(loss_fn, has_aux=True)(bridge_state.params)
        new_bridge_state = bridge_state.apply_gradients(grads=grads)
        metrics["bridge_grad_norm"] = global_norm(grads)
        return new_bridge_state, metrics

    def update_causal_networks(self, rollout: Dict[str, jnp.ndarray] | None = None) -> Dict[str, float]:
        if rollout is None:
            rollout = self.latest_rollout
        if rollout is None or self.bridge_state is None or self.get_current_causal_lambda() == 0.0:
            return {"q_bridge_loss": 0.0}

        causal_batch = self._q_bridge_batch(rollout)
        metrics = []
        for _ in range(self.bridge_epoch):
            self.rng, perm_key = jax.random.split(self.rng)
            perm = jax.random.permutation(perm_key, causal_batch["actions"].shape[0])
            for start in range(0, causal_batch["actions"].shape[0], self.causal_mini_batch_size):
                indices = perm[start:start + self.causal_mini_batch_size]
                batch = {key: value[indices] for key, value in causal_batch.items()}
                self.bridge_state, batch_metrics = self._q_bridge_update_step(self.bridge_state, batch)
                metrics.append({key: float(value) for key, value in tree_to_host(batch_metrics).items()})

        stacked = stack_metrics(metrics)
        stacked.setdefault("q_bridge_loss", 0.0)
        stacked.setdefault("Kz_offdiag_mean", 0.0)
        stacked.setdefault("Kz_offdiag_std", 0.0)
        return stacked

    def compute_bridge_advantages(self, rollout: Dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, Dict[str, float]]:
        if self.bridge_state is None or self.get_current_causal_lambda() == 0.0:
            zeros = jnp.zeros(rollout["actions"].shape, dtype=jnp.float32)
            return zeros, {
                "causal_lambda": self.get_current_causal_lambda(),
                "bridge_adv_mean": 0.0,
                "bridge_adv_std": 0.0,
                "v_bridge_mean": 0.0,
            }

        obs = rollout["obs"]
        local_obs = obs.reshape(-1, self.policy_obs_dim)
        others_obs = gather_other_agent_obs(obs).reshape(-1, (self.num_agents - 1) * self.policy_obs_dim)
        if self.use_recurrent_policy:
            hidden_others = gather_other_agent_obs(rollout["actor_hidden"]).reshape(-1, self.bridge_hidden_dim)
        else:
            hidden_others = jnp.zeros((local_obs.shape[0], 0), dtype=jnp.float32)

        probs = rollout["probs"].reshape(-1, self.action_dim)
        avail_actions = rollout["avail_actions"].reshape(-1, self.action_dim)
        actions = rollout["actions"].reshape(-1)

        _, _, v_bridge, causal_advantages = self._q_bridge_predictions(
            self.bridge_state.params,
            local_obs,
            others_obs,
            hidden_others,
            probs,
            actions,
            avail_actions,
        )

        causal_advantages = causal_advantages.reshape(rollout["actions"].shape)
        if self.action_dim > 1:
            alive = (rollout["avail_actions"][..., 1:].sum(axis=-1) > 0).astype(jnp.float32)
            causal_advantages = causal_advantages * alive

        info = {
            "causal_lambda": self.get_current_causal_lambda(),
            "bridge_adv_mean": float(jnp.mean(causal_advantages)),
            "bridge_adv_std": float(jnp.std(causal_advantages)),
            "v_bridge_mean": float(jnp.mean(v_bridge)),
        }
        return causal_advantages, info

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

    def update(self, **_: Any) -> Dict[str, float]:
        if self.latest_rollout is None:
            return {}

        rollout = self.latest_rollout
        metrics: Dict[str, float] = {}

        bridge_metrics = self.update_causal_networks(rollout)
        metrics.update(bridge_metrics)

        rewards = self._reward_signal(rollout)
        values = rollout["values"]
        last_value = rollout["last_value"]

        env_advantages, returns = compute_gae(
            rewards,
            values,
            last_value,
            rollout["dones"],
            self.gamma,
            self.gae_lambda,
        )

        pci_advantages, pci_metrics = self.compute_bridge_advantages(rollout)
        metrics.update(pci_metrics)

        combined_advantages, combine_metrics = self._combine_actor_advantages(
            env_advantages,
            pci_advantages,
            self.get_current_causal_lambda(),
        )
        metrics.update(combine_metrics)

        metrics.update(self._update_actor_critic(rollout, combined_advantages, returns))
        self.training_steps += int(rollout["actions"].shape[0] * rollout["actions"].shape[1])
        self.latest_rollout = None
        return metrics
