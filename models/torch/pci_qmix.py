import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .networks import NaivePredictor, OutcomeBridgeNetwork
from .qmix import QMIXRNNAgent, rQMIXAgent, rQMIXAlgorithm
from .qmix_preprocessing import qmix_extract_policy_hidden
from .r_pci import RBFKernel


class PCIQMIXAgent(rQMIXAgent):
    """QMIX agent wrapper that exposes the decision-time hidden state."""

    def reset_rnn_state(self):
        super().reset_rnn_state()
        self.last_pre_rnn_state = self.rnn_state.clone()

    def reset_rnn_states_batch(self, n_envs: int):
        super().reset_rnn_states_batch(n_envs)
        self.last_pre_rnn_states_batch = self.rnn_states_batch.clone()

    def get_action(
        self,
        obs: torch.Tensor,
        epsilon: float = 0.0,
        available_actions: torch.Tensor = None,
    ) -> Tuple[int, float]:
        self.last_pre_rnn_state = self.rnn_state.clone()
        return super().get_action(obs, epsilon=epsilon, available_actions=available_actions)

    def get_action_batch(
        self,
        obs_batch: torch.Tensor,
        epsilon: float = 0.0,
        available_actions_batch: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.last_pre_rnn_states_batch = self.rnn_states_batch.clone()
        return super().get_action_batch(
            obs_batch,
            epsilon=epsilon,
            available_actions_batch=available_actions_batch,
        )


class rPCIQMIXAlgorithm(rQMIXAlgorithm):
    """
    PCI-QMIX with an outcome bridge reward replacement.

    The value-decomposition learner stays PyMARL-style QMIX with a shared
    recurrent agent Q-network. The raw team reward in the mixed TD target is
    replaced by the sum of agent-specific PCI deltas, and an auxiliary
    per-agent TD loss trains each agent Q head against its own delta.
    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 5e-4,
        lr_bridge: float = 3e-4,
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
        mixer_init_scale: float = 1.0e-3,
        reward_scale: float = 1.0,
        standardize_rewards: bool = False,
        replay_buffer_size: int = 5000,
        train_batch_size: int = 32,
        qmix_update_epochs: Optional[int] = None,
        learning_starts: int = 0,
        device: str = "cpu",
        use_episode_buffer: bool = True,
        global_reward: bool = True,
        add_agent_id: bool = False,
        causal_lambda: float = 1.0,
        causal_lambda_schedule: bool = False,
        causal_lambda_start: float = 0.0,
        causal_lambda_final: float = 1.0,
        causal_lambda_warmup_steps: int = 10000,
        causal_lambda_startingpoint: int = 0,
        bridge_epoch: int = 4,
        bridge_update_interval: int = 1,
        causal_mini_batch_size: Optional[int] = 10000,
        tikhonov_gamma: float = 1e-4,
        stabilizer_lambda: float = 1.0,
        clip_delta: float = 0.0,
        train_naivepredictor: bool = False,
        agent_delta_td_coef: float = 1.0,
        # Compatibility args accepted by the config system.
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 32,
        num_batches: int = 1,
        env=None,
    ):
        super().__init__(
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
            mixer_init_scale=mixer_init_scale,
            reward_scale=reward_scale,
            standardize_rewards=standardize_rewards,
            replay_buffer_size=replay_buffer_size,
            train_batch_size=train_batch_size,
            qmix_update_epochs=qmix_update_epochs,
            learning_starts=learning_starts,
            device=device,
            use_episode_buffer=use_episode_buffer,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            ppo_epochs=ppo_epochs,
            batch_size=batch_size,
            num_batches=num_batches,
        )

        self.env = env
        self.global_reward = global_reward
        self.add_agent_id = add_agent_id
        self.policy_obs_dim = obs_dim
        self.base_obs_dim = obs_dim - num_agents if add_agent_id else obs_dim
        if self.base_obs_dim <= 0:
            raise ValueError(
                f"Invalid observation dimensions for add_agent_id={add_agent_id}: "
                f"obs_dim={obs_dim}, num_agents={num_agents}"
            )

        # QMIX uses a single GRUCell, so the bridge recurrent context has one layer.
        self.recurrent_N = 1
        self.use_recurrent_policy = True
        self.bridge_hidden_dim = hidden_dim * (num_agents - 1)
        self.bridge_prev_action_dim = action_dim * (num_agents - 1)
        self.bridge_context_dim = self.bridge_hidden_dim + self.bridge_prev_action_dim
        self.w_dim = self.policy_obs_dim * (num_agents - 1)

        self.causal_lambda = causal_lambda
        self.causal_lambda_schedule = causal_lambda_schedule
        self.causal_lambda_start = causal_lambda_start
        self.causal_lambda_final = causal_lambda_final
        self.causal_lambda_warmup_steps = causal_lambda_warmup_steps
        self.causal_lambda_startingpoint = causal_lambda_startingpoint
        self.bridge_epoch = bridge_epoch
        self.bridge_update_interval = max(1, int(bridge_update_interval))
        self.causal_mini_batch_size = causal_mini_batch_size
        self.tikhonov_gamma = tikhonov_gamma
        self.stabilizer_lambda = stabilizer_lambda
        self.clip_delta = clip_delta
        self.train_naivepredictor = train_naivepredictor
        self.agent_delta_td_coef = agent_delta_td_coef
        self._other_agent_mask_cache = {}

        # Replace wrappers so rollout storage keeps h_{t-1}, not h_t.
        self.agents = {
            i: PCIQMIXAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_agents=num_agents,
                q_network=self.shared_q_network,
                device=self.device,
            )
            for i in range(num_agents)
        }

        x_dim = self.policy_obs_dim + self.bridge_context_dim
        self.outcome_bridge_net = OutcomeBridgeNetwork(
            action_dim=action_dim,
            num_agents=num_agents,
            w_dim=self.w_dim,
            outcome_dim=1 if global_reward else num_agents,
            hidden_dim=hidden_dim,
            action_embedding_dim=8,
            x_dim=x_dim,
        ).to(self.device)
        self.outcome_bridge_optimizer = optim.Adam(
            self.outcome_bridge_net.parameters(),
            lr=lr_bridge,
        )

        self.critic_kernel_w = RBFKernel(gamma=1.0).to(self.device)
        self.kernel_z_block = RBFKernel(gamma=1.0).to(self.device)
        self.kernel_a_block = RBFKernel(gamma=1.0).to(self.device)
        self.kernel_xobs_block = RBFKernel(gamma=1.0).to(self.device)
        self.kernel_xhid_block = RBFKernel(gamma=1.0).to(self.device)
        self.kernel_xact_block = RBFKernel(gamma=1.0).to(self.device)

        if self.train_naivepredictor:
            self.naive_predictor = NaivePredictor(
                action_dim=action_dim,
                num_agents=num_agents,
                w_dim=self.w_dim,
                x_dim=x_dim,
                output_dim=1 if global_reward else num_agents,
                hidden_dim=hidden_dim,
            ).to(self.device)
            self.naive_optimizer = optim.Adam(
                self.naive_predictor.parameters(),
                lr=lr_bridge,
            )

    def get_current_causal_lambda(self) -> float:
        if not self.causal_lambda_schedule:
            return self.causal_lambda
        if self.training_steps < self.causal_lambda_startingpoint:
            return self.causal_lambda_start
        effective_steps = self.training_steps - self.causal_lambda_startingpoint
        if effective_steps >= self.causal_lambda_warmup_steps:
            return self.causal_lambda_final
        progress = effective_steps / max(1, self.causal_lambda_warmup_steps)
        return self.causal_lambda_start + progress * (
            self.causal_lambda_final - self.causal_lambda_start
        )

    def get_actions(
        self,
        observations: Dict[int, torch.Tensor],
        available_actions: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[int, Tuple[int, float]]:
        actions = {}
        self.temp_hidden_states = {}
        self.temp_masks = {}

        epsilon = self._get_epsilon()
        for agent_id, obs in observations.items():
            avail_actions_tensor = None
            if available_actions is not None and agent_id in available_actions:
                avail_actions_tensor = torch.tensor(
                    available_actions[agent_id],
                    dtype=torch.float32,
                    device=self.device,
                )

            action, log_prob = self.agents[agent_id].get_action(
                obs,
                epsilon=epsilon,
                available_actions=avail_actions_tensor,
            )
            actions[agent_id] = (action, log_prob)
            self.temp_hidden_states[agent_id] = (
                self.agents[agent_id].last_pre_rnn_state.clone()
            )
            self.temp_masks[agent_id] = (
                1.0 if not self.last_dones.get(agent_id, False) else 0.0
            )

        self.total_env_steps += 1
        return actions

    def get_actions_batch(
        self,
        observations_batch: List[Dict[int, torch.Tensor]],
        available_actions_batch: Optional[List[Dict[int, List[int]]]] = None,
    ) -> List[Dict[int, Tuple[int, float]]]:
        n_envs = len(observations_batch)
        actions_batch = [{} for _ in range(n_envs)]
        self.temp_hidden_states_batch = [{} for _ in range(n_envs)]
        self.temp_masks_batch = [{} for _ in range(n_envs)]

        epsilon = self._get_epsilon()
        for agent_id in range(self.num_agents):
            agent_obs_list = []
            agent_avail_list = []
            has_avail = available_actions_batch is not None

            for env_idx in range(n_envs):
                agent_obs_list.append(observations_batch[env_idx][agent_id])
                self.temp_masks_batch[env_idx][agent_id] = (
                    1.0
                    if not self.last_dones_batch[env_idx].get(agent_id, False)
                    else 0.0
                )
                if (
                    has_avail
                    and available_actions_batch[env_idx] is not None
                    and agent_id in available_actions_batch[env_idx]
                ):
                    agent_avail_list.append(available_actions_batch[env_idx][agent_id])
                else:
                    agent_avail_list.append(None)

            agent_obs_batch = torch.stack(agent_obs_list).to(self.device)
            agent_avail_tensor = None
            if any(x is not None for x in agent_avail_list):
                agent_avail_tensor = torch.tensor(
                    agent_avail_list,
                    dtype=torch.float32,
                    device=self.device,
                )

            actions, log_probs = self.agents[agent_id].get_action_batch(
                agent_obs_batch,
                epsilon=epsilon,
                available_actions_batch=agent_avail_tensor,
            )

            for env_idx in range(n_envs):
                act = (
                    actions[env_idx].item()
                    if isinstance(actions[env_idx], torch.Tensor)
                    else actions[env_idx]
                )
                lp = (
                    log_probs[env_idx].item()
                    if isinstance(log_probs[env_idx], torch.Tensor)
                    else log_probs[env_idx]
                )
                actions_batch[env_idx][agent_id] = (act, lp)
                self.temp_hidden_states_batch[env_idx][agent_id] = (
                    self.agents[agent_id]
                    .last_pre_rnn_states_batch[env_idx]
                    .clone()
                )

        self.total_env_steps += n_envs
        return actions_batch

    def _extract_policy_hidden(self, hidden_entry) -> torch.Tensor:
        return qmix_extract_policy_hidden(hidden_entry, self.hidden_dim)

    def _pad_and_stack(self, episodes: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        batch = super()._pad_and_stack(episodes)
        if all(key in batch for key in ("raw_agent_rewards", "prev_actions", "hidden_states")):
            return batch

        batch_size, max_t = batch["rewards"].shape

        raw_agent_rewards = torch.zeros(
            batch_size,
            max_t,
            self.num_agents,
            device=self.device,
        )
        prev_actions = torch.zeros(
            batch_size,
            max_t,
            self.num_agents,
            dtype=torch.long,
            device=self.device,
        )
        hidden_states = torch.zeros(
            batch_size,
            max_t,
            self.num_agents,
            self.hidden_dim,
            device=self.device,
        )

        for b, episode in enumerate(episodes):
            for t, trans in enumerate(episode):
                rewards = trans.get("bold_r_t", {})
                prev_action_dict = trans.get("bold_a_t_minus_1", {})
                hidden_dict = trans.get("hidden_states", {}) or {}

                raw_agent_rewards[b, t] = torch.tensor(
                    [float(rewards.get(i, 0.0)) for i in range(self.num_agents)],
                    dtype=torch.float32,
                    device=self.device,
                )
                prev_actions[b, t] = torch.tensor(
                    [int(prev_action_dict.get(i, 0)) for i in range(self.num_agents)],
                    dtype=torch.long,
                    device=self.device,
                )
                for agent_id in range(self.num_agents):
                    hidden_states[b, t, agent_id] = self._extract_policy_hidden(
                        hidden_dict.get(agent_id)
                    )

        batch["raw_agent_rewards"] = raw_agent_rewards
        batch["prev_actions"] = prev_actions
        batch["hidden_states"] = hidden_states
        return batch

    @staticmethod
    def _standardize_kernel_block(
        x: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if x.size(1) == 0:
            return x
        return (x - x.mean(0, keepdim=True)) / x.std(
            0,
            keepdim=True,
            unbiased=False,
        ).clamp_min(eps)

    def _normalize_z_kernel_block(
        self,
        z: torch.Tensor,
        joint_obs_dim: int,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if z.size(1) == joint_obs_dim + self.num_agents:
            z_obs_n = self._standardize_kernel_block(z[:, :joint_obs_dim], eps=eps)
            z_rew_n = self._standardize_kernel_block(z[:, joint_obs_dim:], eps=eps)
            return torch.cat([z_obs_n, z_rew_n], dim=1)
        return self._standardize_kernel_block(z, eps=eps)

    def _iterate_causal_minibatches(self, te_total: int, device):
        chunk_size = (
            min(self.causal_mini_batch_size, te_total)
            if (self.causal_mini_batch_size and self.causal_mini_batch_size > 0)
            else te_total
        )
        chunk_size = max(1, int(chunk_size))
        agent_offsets = torch.arange(self.num_agents, device=device)
        for te_start in range(0, te_total, chunk_size):
            te_end = min(te_start + chunk_size, te_total)
            te_idx = torch.arange(te_start, te_end, device=device)
            yield (te_idx.unsqueeze(1) * self.num_agents + agent_offsets.unsqueeze(0)).reshape(-1)

    def _other_agent_mask(self, sample_count: int, feature_dim: int, device) -> torch.Tensor:
        key = (sample_count, feature_dim, str(device))
        cached = self._other_agent_mask_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        mask = ~torch.eye(self.num_agents, dtype=torch.bool, device=device)
        mask = mask.unsqueeze(0).unsqueeze(-1).expand(
            sample_count,
            -1,
            -1,
            feature_dim,
        )
        self._other_agent_mask_cache[key] = mask
        return mask

    def _valid_transition_tensors(
        self,
        batch: Dict[str, torch.Tensor],
        mac_out: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        B, Tp1, N = batch["actions"].shape
        T = Tp1 - 1
        flat_valid = batch["masks"].reshape(B * T) > 0
        valid_idx = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)

        if valid_idx.numel() == 0:
            return {"valid_idx": valid_idx}

        def select_time(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(B * T, *tensor.shape[2:])[valid_idx]

        data = {
            "valid_idx": valid_idx,
            "obs": select_time(batch["agent_obs"][:, :-1]),
            "actions": select_time(batch["actions"][:, :-1]),
            "prev_actions": select_time(batch["prev_actions"]),
            "hidden_states": select_time(batch["hidden_states"]),
            "terminated": batch["terminated"].reshape(B * T)[valid_idx],
        }
        if self.global_reward:
            data["rewards"] = batch["rewards"].reshape(B * T)[valid_idx].unsqueeze(-1)
        else:
            data["rewards"] = select_time(batch["raw_agent_rewards"])

        if "available_actions" in batch:
            data["available_actions"] = select_time(batch["available_actions"][:, :-1])
        else:
            data["available_actions"] = None

        if mac_out is not None:
            data["q_values"] = mac_out[:, :-1].detach().reshape(
                B * T,
                N,
                self.action_dim,
            )[valid_idx]
        else:
            data["q_values"] = None
        return data

    def _build_bridge_super_tensors(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        prev_actions: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        sample_count = obs.size(0)
        num_agents = self.num_agents
        device = self.device

        A_super = actions.reshape(-1)
        Y_super = rewards.repeat_interleave(num_agents, dim=0)

        expanded_obs = obs.unsqueeze(1).expand(-1, num_agents, -1, -1)
        obs_mask = self._other_agent_mask(sample_count, self.policy_obs_dim, device)
        W_super = expanded_obs[obs_mask].view(sample_count * num_agents, -1)

        prev_actions_onehot = F.one_hot(
            prev_actions.long(),
            self.action_dim,
        ).float()
        expanded_prev_actions = prev_actions_onehot.unsqueeze(1).expand(
            -1,
            num_agents,
            -1,
            -1,
        )
        prev_action_mask = self._other_agent_mask(sample_count, self.action_dim, device)
        prev_action_others = expanded_prev_actions[prev_action_mask].view(
            sample_count * num_agents,
            -1,
        )

        H_all = hidden_states.reshape(sample_count, num_agents, self.hidden_dim)
        H_all_expanded = H_all.unsqueeze(1).expand(-1, num_agents, -1, -1)
        hid_mask = self._other_agent_mask(sample_count, self.hidden_dim, device)
        H_others = H_all_expanded[hid_mask].view(sample_count * num_agents, -1)

        obs_super = obs.reshape(sample_count * num_agents, self.policy_obs_dim)
        X_super = torch.cat([obs_super, H_others, prev_action_others], dim=1)
        X_kernel_super = X_super
        Z_super = H_all.reshape(sample_count * num_agents, -1)
        return A_super, W_super, Y_super, X_super, X_kernel_super, Z_super

    def update_causal_networks_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        if self.get_current_causal_lambda() == 0.0:
            return {"outcome_bridge_loss": 0.0, "Kz_offdiag_mean": 0.0, "Kz_offdiag_std": 0.0}

        data = self._valid_transition_tensors(batch)
        valid_idx = data["valid_idx"]
        if valid_idx.numel() == 0:
            return {"outcome_bridge_loss": 0.0, "Kz_offdiag_mean": 0.0, "Kz_offdiag_std": 0.0}

        A_super, W_super, Y_super, X_super, X_kernel_super, Z_super = (
            self._build_bridge_super_tensors(
                data["obs"],
                data["actions"],
                data["rewards"],
                data["prev_actions"],
                data["hidden_states"],
            )
        )
        loss_mask_super = (1.0 - data["terminated"]).unsqueeze(1).expand(
            -1,
            self.num_agents,
        ).reshape(-1)

        sample_count = data["obs"].size(0)
        num_agents = self.num_agents
        device = self.device

        avg_outcome_loss = 0.0
        avg_naive_loss = 0.0
        sum_off_mean = 0.0
        sum_off_std = 0.0
        num_updates = 0

        for idx in self._iterate_causal_minibatches(sample_count, device):
            B_n = idx.numel() // num_agents
            if B_n == 0:
                continue

            a_mb = A_super[idx]
            w_mb = W_super[idx]
            y_mb = Y_super[idx]
            z_mb = Z_super[idx]
            x_mb = X_super[idx]
            x_kernel_mb = X_kernel_super[idx]
            loss_mask_mb = loss_mask_super[idx]

            a_by_agent = a_mb.reshape(B_n, num_agents)
            z_by_agent = z_mb.reshape(B_n, num_agents, -1)
            y_by_agent = y_mb.reshape(B_n, num_agents, -1)
            x_by_agent = x_mb.reshape(B_n, num_agents, -1)
            xk_by_agent = x_kernel_mb.reshape(B_n, num_agents, -1)
            mask_by_agent = loss_mask_mb.reshape(B_n, num_agents)

            eps = 1e-6
            joint_obs_dim = self.policy_obs_dim * self.num_agents
            z_dim_sample = z_by_agent[:, 0, :].shape[1]
            x_act_dim = xk_by_agent.shape[-1] - self.policy_obs_dim - self.bridge_hidden_dim
            gamma_z = 1.0 / max(z_dim_sample, 1)
            gamma_a = 1.0
            gamma_xobs = 1.0 / max(self.policy_obs_dim, 1)
            gamma_xact = 1.0

            with torch.no_grad():
                self.kernel_z_block.log_gamma.data.fill_(math.log(gamma_z))
                self.kernel_a_block.log_gamma.data.fill_(math.log(gamma_a))
                self.kernel_xobs_block.log_gamma.data.fill_(math.log(gamma_xobs))
                if x_act_dim > 0:
                    self.kernel_xact_block.log_gamma.data.fill_(math.log(gamma_xact))

            K_blocks = []
            reg_K_blocks = []
            off_vals_list = []

            for ag_i in range(num_agents):
                z_i = z_by_agent[:, ag_i, :]
                xk_i = xk_by_agent[:, ag_i, :]
                xobs_cont_i = xk_i[:, :self.base_obs_dim]
                xobs_id_i = xk_i[:, self.base_obs_dim:self.policy_obs_dim]
                xhid_i = xk_i[:, self.policy_obs_dim:self.policy_obs_dim + self.bridge_hidden_dim]
                xact_i = xk_i[:, self.policy_obs_dim + self.bridge_hidden_dim:]
                a_i = a_by_agent[:, ag_i]

                z_norm_i = self._normalize_z_kernel_block(z_i, joint_obs_dim, eps=eps)
                if xobs_cont_i.size(1) > 0:
                    xobs_cont_norm_i = self._standardize_kernel_block(xobs_cont_i, eps=eps)
                else:
                    xobs_cont_norm_i = xobs_cont_i
                xobs_norm_i = torch.cat([xobs_cont_norm_i, xobs_id_i], dim=1)
                if xhid_i.size(1) > 0:
                    xhid_norm_i = self._standardize_kernel_block(xhid_i, eps=eps)
                else:
                    xhid_norm_i = xhid_i

                a_oh_i = F.one_hot(a_i.long(), self.action_dim).float()
                K_i = self.kernel_z_block(z_norm_i) * self.kernel_a_block(a_oh_i)
                if xobs_norm_i.size(1) > 0:
                    K_i = K_i * self.kernel_xobs_block(xobs_norm_i)
                if xhid_norm_i.size(1) > 0:
                    effective_dim = torch.mean(
                        torch.sum(torch.abs(xhid_norm_i) > 1e-6, dim=-1).float()
                    ).clamp_min(1.0)
                    with torch.no_grad():
                        self.kernel_xhid_block.log_gamma.data.fill_(
                            math.log(1.0 / max(effective_dim.item(), 1.0))
                        )
                    K_i = K_i * self.kernel_xhid_block(xhid_norm_i)
                if xact_i.size(1) > 0:
                    K_i = K_i * self.kernel_xact_block(xact_i)

                with torch.no_grad():
                    if B_n >= 2:
                        off_mask_i = ~torch.eye(B_n, dtype=torch.bool, device=device)
                        off_vals_list.append(K_i[off_mask_i].detach())

                eye_i = torch.eye(B_n, device=device)
                reg_K_i = self.tikhonov_gamma * eye_i + self.stabilizer_lambda * K_i
                K_blocks.append(K_i)
                reg_K_blocks.append(reg_K_i)

            if off_vals_list:
                all_off = torch.cat(off_vals_list)
                sum_off_mean += all_off.mean().item()
                sum_off_std += all_off.std(unbiased=False).item()

            a_flat = a_by_agent.reshape(-1)
            w_flat = w_mb.reshape(B_n * num_agents, -1)
            x_flat = x_by_agent.reshape(B_n * num_agents, -1)

            epoch_outcome_loss = 0.0
            for _ in range(self.bridge_epoch):
                y_hat = self.outcome_bridge_net(a_flat, w_flat, x=x_flat)
                y_hat_by_agent = y_hat.reshape(B_n, num_agents, -1)

                per_agent_losses = []
                for ag_i in range(num_agents):
                    y_hat_i = y_hat_by_agent[:, ag_i, :]
                    y_i = y_by_agent[:, ag_i, :]
                    mask_i = mask_by_agent[:, ag_i]
                    res_i = y_i - y_hat_i
                    K_i = K_blocks[ag_i]
                    reg_K_i = reg_K_blocks[ag_i]
                    try:
                        solved_i = torch.linalg.solve(reg_K_i, res_i)
                    except RuntimeError:
                        jitter = max(float(self.tikhonov_gamma), 1e-6)
                        solved_i = torch.linalg.solve(
                            reg_K_i + jitter * torch.eye(B_n, device=device),
                            res_i,
                        )
                    weighted_res_i = K_i @ solved_i
                    per_sample_i = (res_i * weighted_res_i).sum(dim=1)
                    denom_i = mask_i.sum().clamp(min=1.0)
                    per_agent_losses.append((per_sample_i * mask_i).sum() / denom_i)

                loss_h = (
                    torch.stack(per_agent_losses).mean()
                    if per_agent_losses
                    else torch.tensor(0.0, device=device)
                )

                self.outcome_bridge_optimizer.zero_grad()
                loss_h.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.outcome_bridge_net.parameters(),
                    self.max_grad_norm,
                )
                self.outcome_bridge_optimizer.step()
                epoch_outcome_loss += loss_h.item()

            avg_outcome_loss += epoch_outcome_loss / max(1, self.bridge_epoch)

            if self.train_naivepredictor:
                naive_output = self.naive_predictor(a_mb, w_mb, x_mb)
                naive_loss = F.mse_loss(naive_output, y_mb)
                self.naive_optimizer.zero_grad()
                naive_loss.backward()
                self.naive_optimizer.step()
                avg_naive_loss += naive_loss.item()

            num_updates += 1

        denom_updates = max(1, num_updates)
        metrics = {
            "outcome_bridge_loss": avg_outcome_loss / denom_updates,
            "Kz_offdiag_mean": sum_off_mean / denom_updates,
            "Kz_offdiag_std": sum_off_std / denom_updates,
        }
        if self.train_naivepredictor:
            metrics["naive_loss"] = avg_naive_loss / denom_updates
        return metrics

    def compute_causal_deltas_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
        mac_out: torch.Tensor,
        baseline_type: str = "weighted",
    ) -> torch.Tensor:
        B, Tp1, N = batch["actions"].shape
        T = Tp1 - 1
        delta_flat_all = torch.zeros(B * T, N, device=self.device)

        if self.get_current_causal_lambda() == 0.0:
            return delta_flat_all.view(B, T, N)

        data = self._valid_transition_tensors(batch, mac_out=mac_out)
        valid_idx = data["valid_idx"]
        if valid_idx.numel() == 0:
            return delta_flat_all.view(B, T, N)

        obs = data["obs"]
        actions = data["actions"]
        rewards = data["rewards"]
        prev_actions = data["prev_actions"]
        hidden_states = data["hidden_states"]
        avail_actions = data["available_actions"]
        q_values = data["q_values"]

        sample_count = obs.size(0)
        delta_samples = torch.zeros(sample_count, N, device=self.device)
        infer_chunk_size = 1024

        with torch.no_grad():
            for start in range(0, sample_count, infer_chunk_size):
                end = min(start + infer_chunk_size, sample_count)
                curr_bs = end - start
                obs_b = obs[start:end]
                act_b = actions[start:end]
                rewards_b = rewards[start:end]
                prev_actions_b = prev_actions[start:end]
                hidden_b = hidden_states[start:end]
                q_b = q_values[start:end]
                avail_b = avail_actions[start:end] if avail_actions is not None else None

                _, W_super, _, X_super, _, _ = self._build_bridge_super_tensors(
                    obs_b,
                    act_b,
                    rewards_b,
                    prev_actions_b,
                    hidden_b,
                )
                actions_flat = act_b.reshape(-1)
                action_range = torch.arange(self.action_dim, device=self.device).repeat(
                    curr_bs * N
                )
                W_expanded = W_super.repeat_interleave(self.action_dim, dim=0)
                X_expanded = X_super.repeat_interleave(self.action_dim, dim=0)
                reward_dim = rewards_b.size(-1)

                r_hat_all = self.outcome_bridge_net(
                    action_range,
                    W_expanded,
                    x=X_expanded,
                ).view(curr_bs * N, self.action_dim, reward_dim)
                r_hat_factual = r_hat_all[
                    torch.arange(curr_bs * N, device=self.device),
                    actions_flat,
                ]

                q_flat = q_b.reshape(curr_bs * N, self.action_dim)
                if avail_b is not None:
                    avail_flat = avail_b.reshape(curr_bs * N, self.action_dim)
                    q_flat = q_flat.masked_fill(avail_flat <= 0, -1e10)
                else:
                    avail_flat = None

                action_probs_flat = torch.softmax(q_flat, dim=-1)
                factual_mask = torch.zeros_like(action_probs_flat).scatter_(
                    1,
                    actions_flat.unsqueeze(1),
                    1.0,
                )
                non_factual_mask = 1.0 - factual_mask

                if baseline_type == "weighted":
                    valid_cf = non_factual_mask
                    if avail_flat is not None:
                        valid_cf = valid_cf * avail_flat
                    probs_counterfactual = action_probs_flat * valid_cf
                    denom = probs_counterfactual.sum(dim=1, keepdim=True)
                    renormalized_probs = probs_counterfactual / denom.clamp_min(1e-10)
                    baseline = (renormalized_probs.unsqueeze(2) * r_hat_all).sum(dim=1)
                    no_valid_cf = denom <= 1e-10
                    baseline = torch.where(no_valid_cf, r_hat_factual, baseline)
                elif baseline_type == "averaged":
                    valid_cf = non_factual_mask
                    if avail_flat is not None:
                        valid_cf = valid_cf * avail_flat
                    denom = valid_cf.sum(dim=1, keepdim=True).clamp_min(1.0)
                    baseline = (r_hat_all * valid_cf.unsqueeze(2)).sum(dim=1) / denom
                elif baseline_type == "max":
                    valid_cf = non_factual_mask.bool()
                    if avail_flat is not None:
                        valid_cf = valid_cf & avail_flat.bool()
                    masked_r_hat = r_hat_all.masked_fill(
                        ~valid_cf.unsqueeze(2),
                        float("-inf"),
                    )
                    baseline = masked_r_hat.max(dim=1).values
                    no_valid_cf = ~valid_cf.any(dim=1, keepdim=True)
                    baseline = torch.where(no_valid_cf, r_hat_factual, baseline)
                else:
                    raise ValueError(f"Invalid baseline type: {baseline_type}")

                if self.global_reward:
                    rewards_flat = rewards_b.unsqueeze(1).expand(-1, N, -1).reshape(
                        curr_bs * N,
                        reward_dim,
                    )
                    raw_effects = rewards_flat - baseline
                    delta_chunk = raw_effects.view(curr_bs, N, 1).squeeze(-1)
                else:
                    rewards_flat = rewards_b.unsqueeze(1).expand(-1, N, -1).reshape(
                        curr_bs * N,
                        reward_dim,
                    )
                    raw_effects = rewards_flat - baseline
                    C_chunk = raw_effects.view(curr_bs, N, N)
                    delta_chunk = C_chunk.sum(dim=2)

                if avail_b is not None:
                    if self.action_dim > 1:
                        alive = (avail_b[:, :, 1:].sum(dim=-1) > 0).float()
                    else:
                        alive = (avail_b.sum(dim=-1) > 0).float()
                    delta_chunk = delta_chunk * alive

                if self.clip_delta > 0.0:
                    delta_chunk = delta_chunk.clone()
                    delta_chunk[torch.abs(delta_chunk) < self.clip_delta] = 0.0

                delta_samples[start:end] = delta_chunk

        delta_flat_all[valid_idx] = delta_samples
        return delta_flat_all.view(B, T, N)

    def _train_step_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
        bridge_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        actions = batch["actions"]
        terminated = batch["terminated"]
        masks = batch["masks"]
        joint_obs = batch["joint_obs"]
        avail_actions = batch.get("available_actions", None)

        B, Tp1, _ = actions.shape
        T = Tp1 - 1

        mac_out = self._forward_agent_q_values(batch, self.shared_q_network)
        chosen_action_qvals = torch.gather(
            mac_out[:, :-1],
            dim=3,
            index=actions[:, :-1].unsqueeze(-1),
        ).squeeze(3)

        if bridge_metrics is None:
            bridge_metrics = {
                "outcome_bridge_loss": 0.0,
                "Kz_offdiag_mean": 0.0,
                "Kz_offdiag_std": 0.0,
            }

        delta_agent = self.compute_causal_deltas_from_batch(batch, mac_out.detach())
        delta_team = delta_agent.sum(dim=-1)
        causal_lambda = float(self.get_current_causal_lambda())
        env_team_reward = batch["rewards"]
        shaped_team_reward = env_team_reward + causal_lambda * delta_team

        with torch.no_grad():
            target_mac_out = self._forward_agent_q_values(batch, self.target_q_network)
            target_mac_out_future = target_mac_out[:, 1:].clone()
            if avail_actions is not None:
                target_mac_out_future[avail_actions[:, 1:] == 0] = -9999999

            if self.double_q:
                mac_out_detach = mac_out.clone().detach()
                if avail_actions is not None:
                    mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_max_qvals = torch.gather(
                    target_mac_out_future,
                    3,
                    cur_max_actions,
                ).squeeze(3)
            else:
                target_max_qvals = target_mac_out_future.max(dim=3)[0]

            target_max_q_tot = self.target_mixer(
                target_max_qvals,
                joint_obs[:, 1:],
            )

        chosen_action_q_tot = self.mixer(chosen_action_qvals, joint_obs[:, :-1])

        terminated_expanded = terminated.unsqueeze(-1)
        masks_expanded = masks.unsqueeze(-1)
        targets = (
            shaped_team_reward.unsqueeze(-1)
            + self.gamma * (1 - terminated_expanded) * target_max_q_tot
        )
        td_error = chosen_action_q_tot - targets.detach()
        masked_td_error = td_error * masks_expanded
        qmix_loss = (masked_td_error ** 2).sum() / masks_expanded.sum().clamp(min=1.0)

        agent_targets = (
            delta_agent
            + self.gamma * (1 - terminated.unsqueeze(-1)) * target_max_qvals
        )
        agent_td_error = chosen_action_qvals - agent_targets.detach()
        agent_masks = masks.unsqueeze(-1).expand(B, T, self.num_agents)
        masked_agent_td_error = agent_td_error * agent_masks
        agent_delta_td_loss = (masked_agent_td_error ** 2).sum() / agent_masks.sum().clamp(min=1.0)

        effective_agent_delta_td_coef = self.agent_delta_td_coef * causal_lambda
        loss = qmix_loss + effective_agent_delta_td_coef * agent_delta_td_loss

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

        self.training_steps += 1
        if (self.training_steps - self.last_target_update_step) >= self.target_update_interval:
            self._update_targets()
            self.last_target_update_step = self.training_steps

        valid_agent_mask = agent_masks > 0
        valid_delta = delta_agent[valid_agent_mask]
        if valid_delta.numel() == 0:
            delta_mean = 0.0
            delta_std = 0.0
            delta_abs_mean = 0.0
        else:
            delta_mean = valid_delta.mean().item()
            delta_std = valid_delta.std(unbiased=False).item()
            delta_abs_mean = valid_delta.abs().mean().item()

        mask_sum = masks_expanded.sum().clamp(min=1.0).item()
        agent_mask_sum = agent_masks.sum().clamp(min=1.0).item()
        metrics = {
            "loss": loss.item(),
            "qmix_loss": qmix_loss.item(),
            "agent_delta_td_loss": agent_delta_td_loss.item(),
            "causal_lambda": causal_lambda,
            "effective_agent_delta_td_coef": effective_agent_delta_td_coef,
            "env_reward_mean": (env_team_reward * masks).sum().item()
            / masks.sum().clamp(min=1.0).item(),
            "delta_team_mean": (delta_team * masks).sum().item() / masks.sum().clamp(min=1.0).item(),
            "shaped_reward_mean": (shaped_team_reward * masks).sum().item()
            / masks.sum().clamp(min=1.0).item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "td_error_abs": masked_td_error.abs().sum().item() / mask_sum,
            "agent_td_error_abs": masked_agent_td_error.abs().sum().item() / agent_mask_sum,
            "q_taken_mean": (chosen_action_q_tot * masks_expanded).sum().item()
            / (mask_sum * self.num_agents),
            "target_mean": (targets * masks_expanded).sum().item()
            / (mask_sum * self.num_agents),
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "delta_abs_mean": delta_abs_mean,
        }
        metrics.update(bridge_metrics)
        return metrics

    def _train_step(self, completed_episodes: List[Dict]) -> Dict[str, float]:
        batch = self._sample_batch_from_replay(completed_episodes)
        bridge_metrics = self.update_causal_networks_from_batch(batch)
        return self._train_step_from_batch(batch, bridge_metrics=bridge_metrics)

    def update(self, **kwargs) -> Dict[str, float]:
        if not self.buffer_manager.has_completed_episodes():
            return {}

        completed_episodes = self.buffer_manager.get_completed_episodes()
        if not completed_episodes:
            return {}

        n_available = len(completed_episodes)
        if n_available < self.train_batch_size:
            return {}

        metadata = {
            "epsilon": self.epsilon,
            "total_env_steps": self.total_env_steps,
            "replay_buffer_episodes": n_available,
            "qmix_update_epochs": float(self.qmix_update_epochs or 0),
            "learning_starts": float(self.learning_starts),
            "learning_started": float(self.total_env_steps >= self.learning_starts),
            "reward_scale": float(self.reward_scale),
            "standardize_rewards": float(self.standardize_rewards),
            "mixer_init_scale": float(self.mixer_init_scale),
        }

        bridge_batch = self._sample_batch_from_replay(completed_episodes)
        bridge_metrics = self.update_causal_networks_from_batch(bridge_batch)

        if self.total_env_steps < self.learning_starts:
            skipped = dict(bridge_metrics)
            skipped.update(metadata)
            skipped["num_updates"] = 0
            return skipped

        num_updates = (
            self.qmix_update_epochs
            if self.qmix_update_epochs is not None
            else max(1, kwargs.get("num_episodes_collected", 1))
        )

        accumulated = defaultdict(float)
        for _ in range(num_updates):
            batch = self._sample_batch_from_replay(completed_episodes)
            step_metrics = self._train_step_from_batch(batch)
            for key, value in step_metrics.items():
                accumulated[key] += value

        losses = {key: value / num_updates for key, value in accumulated.items()}
        losses.update(bridge_metrics)
        losses.update(metadata)
        losses["num_updates"] = num_updates
        return losses
