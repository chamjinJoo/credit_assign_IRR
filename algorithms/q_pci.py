import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from algorithms.agent_common import RecurrentActorCriticAgent
from algorithms.common import (
    build_agent_active_masks,
    build_agent_masks_batch,
    build_alive_mask_matrix,
    build_available_actions_batch,
    build_team_done_sequence,
    build_transition_kwargs,
    chunk_sequence,
    compute_gae,
    concatenate_agent_observations,
    move_modules_to_device,
    normalize_advantages,
    prepare_env_info,
    resolve_step_episode_done,
    split_action_and_log_probs,
)
from algorithms.networks.rnn import RNNLayer, RNNValueNetwork
from algorithms.networks.mlp import MLPBase
from algorithms.networks import OutcomeBridgeNetwork
from utils.buffer import TransitionBuffer, EpisodeBuffer


# =============================================================================
# Kernel
# =============================================================================

class RBFKernel(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma)), requires_grad=False)

    def set_gamma(self, x: torch.Tensor):
        with torch.no_grad():
            if x.size(0) < 2: return
            dists_sq = torch.pdist(x, p=2).pow(2)
            median_dist_sq = dists_sq.median()
            gamma = 1.0 / (2 * median_dist_sq + 1e-8)
            self.log_gamma.data.fill_(torch.log(gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_asymmetric(x, x)

    def forward_asymmetric(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        gamma = torch.exp(self.log_gamma)
        x1_norm_sq = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm_sq = x2.pow(2).sum(dim=-1, keepdim=True)
        dists_sq = x1_norm_sq - 2 * torch.matmul(x1, x2.T) + x2_norm_sq.T
        dists_sq = torch.clamp(dists_sq, min=0)
        return torch.exp(-gamma * dists_sq)

    def forward_elementwise(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        gamma = torch.exp(self.log_gamma)
        dists_sq = torch.sum((x1 - x2)**2, dim=-1)
        return torch.exp(-gamma * dists_sq)


# =============================================================================
# Return Bridge Network (replaces OutcomeBridgeNetwork)
# =============================================================================

class ReturnBridgeNetwork(nn.Module):
    """
    Return(G_t) 추정용 Outcome Bridge Network.
    
    Moment restriction:
        E[G_t - h^Q(W^i_t, A^i_t, X^i_t) | Z^i_t, A^i_t, X^i_t] = 0
    
    학습 완료 후:
        h^Q(w^i_t, a, x^i_t) ≈ E[G_t | S_t, A^i_t = a, X^i_t = x]
    
    Bridge-implied value:
        V̂^{i,bridge}_t = Σ_a π^i(a|h^i_t) · h^Q(w^i_t, a, x^i_t)
    """
    def __init__(
        self,
        action_dim: int,
        num_agents: int,
        w_dim: int,
        hidden_dim: int = 64,
        action_embedding_dim: int = 8,
        x_dim: int = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)
        
        input_dim = action_embedding_dim + w_dim
        if x_dim is not None:
            input_dim += x_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 스칼라: team return
        )
    
    def forward(self, actions: torch.Tensor, w: torch.Tensor,
                x: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            actions: (B,) action indices
            w: (B, w_dim) outcome-side proxy
            x: (B, x_dim) pre-action covariates
        Returns:
            (B,) predicted Q-bridge value (scalar per sample)
        """
        a_emb = self.action_embedding(actions.long())
        if x is not None:
            inp = torch.cat([a_emb, w, x], dim=-1)
        else:
            inp = torch.cat([a_emb, w], dim=-1)
        return self.net(inp).squeeze(-1)  # (B,)


# =============================================================================
# Agent (simplified: no value network)
# =============================================================================

class qPCIAgent(RecurrentActorCriticAgent):
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        recurrent_N: int = 1,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
    ):
        super().__init__(
            agent_id=agent_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            recurrent_N=recurrent_N,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            device=None,
            use_recurrent_policy=True,
        )

# =============================================================================
# Algorithm
# =============================================================================

class qPCIAlgorithm:
    def __init__(
        self,
        env,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        recurrent_N: int = 1,
        lr_actor: float = 3e-4,
        lr_bridge: float = 1e-3,
        lr_critic: float = 3e-4,
        clip_delta: float = 0.2,
        causal_lambda: float = 1.0,
        causal_lambda_schedule: bool = False,
        causal_lambda_start: float = 0.0,
        causal_lambda_final: float = 0.2,
        causal_lambda_warmup_steps: int = 10000,
        causal_lambda_startingpoint: int = 0,
        bridge_epoch: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        num_batches: int = 4,
        max_buffer_size: int = 10000,
        device: str = "cpu",
        rkhs_gamma: float = 1.0,
        env_required_agents: Optional[Dict[int, List[int]]] = None,
        causal_mini_batch_size: int = 10000,
        rnn_chunk_length: int = 16,
        use_episode_buffer: bool = False,
        normalize_returns: bool = False,
        lse_temperature: float = 1.0,
        potential_type: str = 'gap',
        use_value_parameter_sharing: bool = True,
        use_centralized_critic: bool = True,
    ):
        self.env = env
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.w_dim = obs_dim * (num_agents - 1)
        self.hidden_dim = hidden_dim
        self.recurrent_N = recurrent_N
        self.rnn_chunk_length = rnn_chunk_length
        self.env_required_agents = env_required_agents
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.causal_mini_batch_size = causal_mini_batch_size
        self.use_episode_buffer = use_episode_buffer
        self.normalize_returns = normalize_returns
        self.lse_temperature = lse_temperature
        self.potential_type = potential_type
        self.use_value_parameter_sharing = use_value_parameter_sharing
        self.use_centralized_critic = use_centralized_critic
        self.value_coef = value_coef
        # Causal lambda scheduling
        self.clip_delta = clip_delta
        self.causal_lambda = causal_lambda
        self.causal_lambda_schedule = causal_lambda_schedule
        self.causal_lambda_start = causal_lambda_start
        self.causal_lambda_final = causal_lambda_final
        self.causal_lambda_warmup_steps = causal_lambda_warmup_steps
        self.causal_lambda_startingpoint = causal_lambda_startingpoint
        self.bridge_epoch = bridge_epoch
        self.lr_bridge = lr_bridge

        self.training_steps = 0
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.num_batches = num_batches
        
        if self.use_episode_buffer:
            self.buffer_manager = EpisodeBuffer(max_episodes=max_buffer_size)
        else:
            self.buffer_manager = TransitionBuffer(max_buffer_size=max_buffer_size)
        
        # --- Agents (policy + value network) ---
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = qPCIAgent(
                agent_id=i, obs_dim=obs_dim, action_dim=action_dim,
                hidden_dim=hidden_dim, recurrent_N=recurrent_N,
                lr_actor=lr_actor, lr_critic=lr_critic
            )
        
        if self.use_centralized_critic:
            self.value_net = RNNValueNetwork(
                obs_dim=obs_dim * num_agents,
                hidden_dim=hidden_dim,
                recurrent_N=recurrent_N,
                out_dim=num_agents,
                action_dim=action_dim,
                num_agents=num_agents,
            )
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)
            self.rnn_state_value = None
            print(
                f"[Q-PCI] Using MAPPO-style centralized critic "
                f"(input_dim={obs_dim * num_agents}, out_dim={num_agents})"
            )

        # --- Shared Value Network (IPPO-style parameter sharing) ---
        elif self.use_value_parameter_sharing:
            self.shared_value_net = RNNValueNetwork(
                obs_dim=obs_dim, hidden_dim=hidden_dim, recurrent_N=recurrent_N,
                action_dim=action_dim, num_agents=1
            )
            self.shared_value_optimizer = optim.Adam(self.shared_value_net.parameters(), lr=lr_critic)
            for agent_id, agent in self.agents.items():
                agent.value_net = self.shared_value_net
                agent.value_optimizer = self.shared_value_optimizer
        
        # --- Return Bridge Network ---
        x_dim = obs_dim + (num_agents - 1) * hidden_dim * recurrent_N
        
        self.outcome_bridge_net = OutcomeBridgeNetwork(
            action_dim=action_dim,
            num_agents=num_agents,
            w_dim=obs_dim * (num_agents - 1),
            hidden_dim=hidden_dim,
            action_embedding_dim=8,
            x_dim=x_dim,
            outcome_dim=1
        )
        self.outcome_bridge_optimizer = optim.Adam(
            self.outcome_bridge_net.parameters(), lr=lr_bridge
        )
        
        # --- Product Kernel ---
        self.kernel_z_block = RBFKernel(gamma=1.0).to(self.device)
        self.kernel_a_block = RBFKernel(gamma=1.0).to(self.device)
        self.kernel_xobs_block = RBFKernel(gamma=1.0).to(self.device)
        self.kernel_xhid_block = RBFKernel(gamma=1.0).to(self.device)
        
        self.stabilizer_lambda = 1.0
        self.tikhonov_gamma = 1e-4
        
        # --- Return normalization running stats ---
        self.return_running_mean = 0.0
        self.return_running_var = 1.0
        self.return_running_count = 0
        
        # Done status tracking
        self.last_dones = {i: False for i in range(num_agents)}
        
        self._move_networks_to_device()
    
    def _move_networks_to_device(self):
        for agent in self.agents.values():
            move_modules_to_device(self.device, agent.policy_net)
        if self.use_centralized_critic:
            move_modules_to_device(self.device, self.value_net)
        elif self.use_value_parameter_sharing:
            move_modules_to_device(self.device, self.shared_value_net)
        else:
            for agent in self.agents.values():
                move_modules_to_device(self.device, agent.value_net)
        move_modules_to_device(
            self.device,
            self.outcome_bridge_net,
            self.kernel_z_block,
            self.kernel_a_block,
            self.kernel_xobs_block,
            self.kernel_xhid_block,
        )
    
    def get_current_causal_lambda(self) -> float:
        if not self.causal_lambda_schedule:
            return self.causal_lambda
        if self.training_steps < self.causal_lambda_startingpoint:
            return self.causal_lambda_start
        effective_steps = self.training_steps - self.causal_lambda_startingpoint
        if effective_steps >= self.causal_lambda_warmup_steps:
            return self.causal_lambda_final
        else:
            progress = effective_steps / self.causal_lambda_warmup_steps
            return self.causal_lambda_start + progress * (self.causal_lambda_final - self.causal_lambda_start)

    # =================================================================
    # Return computation
    # =================================================================
    
    def _compute_trajectory_returns(self, traj: List[Dict]) -> torch.Tensor:
        """For every timestep, compute team discounted reward-to-go. Returns (T,)."""
        T = len(traj)
        team_dones = build_team_done_sequence(traj)
        team_rewards = torch.tensor(
            [sum(traj[t]['bold_r_t'][i] for i in range(self.num_agents)) for t in range(T)],
            dtype=torch.float32, device=self.device
        )
        returns = torch.zeros(T, device=self.device)
        G = 0.0
        for t in reversed(range(T)):
            G = team_rewards[t] + self.gamma * G * (1.0 - float(team_dones[t]))
            returns[t] = G
        return returns

    def _normalize_returns(self, returns: torch.Tensor) -> torch.Tensor:
        """Running normalization for return targets."""
        with torch.no_grad():
            batch_mean = returns.mean().item()
            batch_var = returns.var().item() if returns.numel() > 1 else 1.0
            batch_count = returns.numel()
            
            delta = batch_mean - self.return_running_mean
            total_count = self.return_running_count + batch_count
            if total_count > 0:
                self.return_running_mean += delta * batch_count / total_count
                m_a = self.return_running_var * self.return_running_count
                m_b = batch_var * batch_count
                M2 = m_a + m_b + delta**2 * self.return_running_count * batch_count / total_count
                self.return_running_var = M2 / total_count
                self.return_running_count = total_count
            
            ret_std = max(self.return_running_var ** 0.5, 1e-6)
            return (returns - self.return_running_mean) / ret_std

    def _normalize_advantage_signal(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.numel() <= 1:
            return torch.zeros_like(signal)
        std = signal.std(unbiased=False).clamp_min(1e-8)
        return (signal - signal.mean()) / std

    def _orthogonalize_advantage_signal(
        self,
        base_signal: torch.Tensor,
        aux_signal: torch.Tensor,
    ) -> torch.Tensor:
        base_centered = base_signal - base_signal.mean()
        aux_centered = aux_signal - aux_signal.mean()
        denom = torch.dot(base_centered, base_centered)
        if denom.abs() < 1e-8:
            return aux_centered
        projection = torch.dot(aux_centered, base_centered) / (denom + 1e-8)
        return aux_centered - projection * base_centered

    def _combine_env_and_pci_advantages(
        self,
        env_advantages: torch.Tensor,
        pci_signal: torch.Tensor,
        causal_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        env_norm = self._normalize_advantage_signal(env_advantages)
        pci_norm = self._normalize_advantage_signal(pci_signal)
        combined = env_norm + causal_lambda * pci_norm
        return combined, env_norm, pci_norm

    # =================================================================
    # Q-Bridge training (replaces update_causal_networks)
    # =================================================================
    
    def update_q_bridge(self, batch_data: List[Dict]) -> Dict[str, float]:
        """
        Q-Bridge 학습. Target Y = r_team_t + gamma * V(next), 출력 스칼라.
        
        Minimax closed-form (scalar residual):
            min_h (1/|B|) r^T K^{1/2} (γI + λK)^{-1} K^{1/2} r
        """
        if self.get_current_causal_lambda() == 0.0:
            return {'q_bridge_loss': 0.0}
        
        batch_size = len(batch_data)
        num_agents = self.num_agents
        device = self.device

        # === Data preparation ===
        joint_o_prev = torch.stack([
            concatenate_agent_observations(
                t['bold_o_t_minus_1'],
                self.num_agents,
                self.obs_dim,
                self.device,
            )
            for t in batch_data
        ]).to(device)
        obs_batch = torch.stack([
            torch.stack(list(t['bold_o_t'].values())) for t in batch_data
        ]).to(device)
        action_batch = torch.tensor(
            [[t['bold_a_t'][i] for i in range(num_agents)] for t in batch_data],
            dtype=torch.long, device=device
        )

        # Bootstrapped team target (pre-computed in update())
        bridge_targets = torch.tensor(
            [
                t.get('team_bootstrap_target', t.get('team_return', 0.0))
                for t in batch_data
            ],
            dtype=torch.float32,
            device=device,
        )
        if self.normalize_returns:
            bridge_targets_norm = self._normalize_returns(bridge_targets)
        else:
            bridge_targets_norm = bridge_targets

        # Hidden states
        hidden_states_batch = {}
        if 'hidden_states' in batch_data[0] and batch_data[0]['hidden_states'] is not None:
            for agent_id in range(num_agents):
                if agent_id in batch_data[0]['hidden_states']:
                    p_states = torch.stack([
                        t['hidden_states'][agent_id][0] for t in batch_data
                    ]).to(device)
                    hidden_states_batch[agent_id] = p_states

        # === Super-batch (agent-wise expansion) ===
        A_super = action_batch.view(-1)
        
        # W: other agents' obs
        expanded_obs = obs_batch.unsqueeze(1).expand(-1, num_agents, -1, -1)
        mask_w = ~torch.eye(num_agents, dtype=torch.bool, device=device)\
            .unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.obs_dim)
        W_super = expanded_obs[mask_w].view(batch_size * num_agents, -1)

        # Z: previous joint observation
        Z_super = joint_o_prev.repeat_interleave(num_agents, dim=0)

        # Y: team return (same for all agents in a timestep)
        Y_super = bridge_targets_norm.repeat_interleave(num_agents)  # (B*N,) scalar

        # X: (o^i_t, h^{-i}_{t-1})
        h_list = []
        for i in range(num_agents):
            if i in hidden_states_batch:
                h_list.append(hidden_states_batch[i].view(batch_size, -1))
            else:
                h_list.append(torch.zeros(batch_size, self.recurrent_N * self.hidden_dim, device=device))
        H_all = torch.stack(h_list, dim=1)  # (B, N, RecN*H)

        expanded_H = H_all.unsqueeze(1).expand(-1, num_agents, -1, -1)
        mask_H = ~torch.eye(num_agents, dtype=torch.bool, device=device)\
            .unsqueeze(0).unsqueeze(-1)\
            .expand(batch_size, -1, -1, self.recurrent_N * self.hidden_dim)
        H_others_super = expanded_H[mask_H].view(batch_size * num_agents, -1)
        obs_super = obs_batch.view(-1, self.obs_dim)
        X_super = torch.cat([obs_super, H_others_super], dim=1)

        # === Mini-batch training loop ===
        total_samples = A_super.size(0)
        indices = torch.arange(total_samples, device=device)
        eps = 1e-6

        avg_loss = 0.0
        sum_off_mean = 0.0
        sum_off_std = 0.0
        num_updates = 0
        chunk_size = self.causal_mini_batch_size

        for ci in range(0, total_samples, chunk_size):
            idx = indices[ci:ci + chunk_size]
            current_batch_size = idx.size(0)

            a_mb = A_super[idx]
            w_mb = W_super[idx]
            y_mb = Y_super[idx]  # (chunk,) scalar
            z_mb = Z_super[idx]
            x_mb = X_super[idx]

            # Block-wise standardization
            z_mean = z_mb.mean(dim=0, keepdim=True)
            z_std = z_mb.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
            z_mb_norm = (z_mb - z_mean) / z_std

            x_obs_mb_raw = x_mb[:, :self.obs_dim]
            x_hid_mb_raw = x_mb[:, self.obs_dim:]

            x_obs_mb = (x_obs_mb_raw - x_obs_mb_raw.mean(0, keepdim=True)) / x_obs_mb_raw.std(0, keepdim=True, unbiased=False).clamp_min(eps)
            x_hid_mb = (x_hid_mb_raw - x_hid_mb_raw.mean(0, keepdim=True)) / x_hid_mb_raw.std(0, keepdim=True, unbiased=False).clamp_min(eps)

            action_onehot_mb = F.one_hot(a_mb.long(), self.action_dim).float()

            # Product kernel gamma
            with torch.no_grad():
                for feat, kernel in [
                    (z_mb_norm, self.kernel_z_block),
                    (action_onehot_mb, self.kernel_a_block),
                    (x_obs_mb, self.kernel_xobs_block),
                    (x_hid_mb, self.kernel_xhid_block),
                ]:
                    n = feat.shape[1]
                    kernel.log_gamma.data.fill_(torch.log(torch.tensor(1.0 / n)))

            K_z = (self.kernel_z_block(z_mb_norm)
                   * self.kernel_a_block(action_onehot_mb)
                   * self.kernel_xobs_block(x_obs_mb)
                   * self.kernel_xhid_block(x_hid_mb))

            with torch.no_grad():
                B = K_z.size(0)
                off_mask = ~torch.eye(B, dtype=torch.bool, device=K_z.device)
                sum_off_mean += K_z[off_mask].mean().item()
                sum_off_std += K_z[off_mask].std(unbiased=False).item()

            reg_K_z = self.tikhonov_gamma * torch.eye(current_batch_size, device=device) \
                      + self.stabilizer_lambda * K_z
            try:
                torch.linalg.solve(reg_K_z, y_mb.unsqueeze(1))
            except RuntimeError:
                reg_K_z = reg_K_z + 1e-3 * torch.eye(current_batch_size, device=device)

            # === Q-Bridge update ===
            epoch_loss = 0.0
            for _ in range(self.bridge_epoch):
                q_hat = self.outcome_bridge_net(a_mb, w_mb, x=x_mb).squeeze(-1)  # (chunk,)
                residuals = (y_mb - q_hat).unsqueeze(1)  # (B, 1) for matrix ops

                solved = torch.linalg.solve(reg_K_z, residuals)
                loss = (residuals * (K_z @ solved)).sum(dim=1).mean()

                self.outcome_bridge_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.outcome_bridge_net.parameters(), self.max_grad_norm)
                self.outcome_bridge_optimizer.step()
                epoch_loss += loss.item()

            avg_loss += epoch_loss / self.bridge_epoch
            num_updates += 1

        return {
            'q_bridge_loss': avg_loss / max(num_updates, 1),
            'Kz_offdiag_mean': sum_off_mean / max(num_updates, 1),
            'Kz_offdiag_std': sum_off_std / max(num_updates, 1),
        }

    # =================================================================
    # Bridge-implied value computation (replaces compute_causal_rewards)
    # =================================================================

    def compute_bridge_values(
        self,
        traj: List[Dict],
        hidden_states_batch: Dict[int, torch.Tensor],
        available_actions_batch: Optional[Dict[int, torch.Tensor]] = None,
        prev_actions_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        각 agent에 대해 bridge-implied value, V^LSE, causal advantage를 계산.
        
        Returns:
            bridge_values: {agent_id: (T,)}
            causal_advantages: {agent_id: (T,)}  
            v_lse: {agent_id: (T,)} for PBRS potential
        """
        T = len(traj)
        N = self.num_agents
        device = self.device
        tau = self.lse_temperature

        full_obs_batch = torch.stack([
            torch.stack(list(t['bold_o_t'].values())) for t in traj
        ]).to(device)

        actions_batch = torch.tensor(
            [[t['bold_a_t'][i] for i in range(N)] for t in traj],
            dtype=torch.long, device=device)

        expanded_obs = full_obs_batch.unsqueeze(1).expand(-1, N, -1, -1)
        mask_w = ~torch.eye(N, dtype=torch.bool, device=device)\
            .unsqueeze(0).unsqueeze(-1).expand(T, -1, -1, self.obs_dim)
        w_all = expanded_obs[mask_w].view(T, N, (N - 1) * self.obs_dim)

        h_list = []
        for i in range(N):
            if hidden_states_batch is not None and i in hidden_states_batch:
                h_list.append(hidden_states_batch[i].view(T, -1))
            else:
                h_list.append(torch.zeros(T, self.recurrent_N * self.hidden_dim, device=device))
        H_all = torch.stack(h_list, dim=1)

        bridge_values = {}
        causal_advantages = {}
        v_lse = {}

        with torch.no_grad():
            for agent_id in range(N):
                w_i = w_all[:, agent_id, :]
                obs_i = full_obs_batch[:, agent_id, :]

                h_others_list = [H_all[:, j, :] for j in range(N) if j != agent_id]
                h_others = torch.cat(h_others_list, dim=-1)
                x_i = torch.cat([obs_i, h_others], dim=-1)

                q_values = torch.zeros(T, self.action_dim, device=device)
                for a_idx in range(self.action_dim):
                    a_tensor = torch.full((T,), a_idx, dtype=torch.long, device=device)
                    q_values[:, a_idx] = self.outcome_bridge_net(a_tensor, w_i, x=x_i).squeeze(-1)

                if hidden_states_batch is not None and agent_id in hidden_states_batch:
                    p_states = hidden_states_batch[agent_id]
                else:
                    p_states = torch.zeros(T, self.recurrent_N, self.hidden_dim, device=device)

                masks = torch.ones(T, 1, device=device)
                prev_act = None
                if prev_actions_batch is not None:
                    prev_act_i = prev_actions_batch[:, agent_id]
                    prev_act = F.one_hot(prev_act_i.long(), self.action_dim).float()

                avail = None
                if available_actions_batch is not None and agent_id in available_actions_batch:
                    avail = available_actions_batch[agent_id]

                logits, _ = self.agents[agent_id].policy_net(
                    obs_i, p_states, masks,
                    available_actions=avail, prev_actions=prev_act
                )
                probs = torch.softmax(logits, dim=-1)
                if avail is not None:
                    probs = probs * avail
                    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

                # V̂^bridge = Σ_a π(a) Q̂(a)
                v_bridge = (probs * q_values).sum(dim=-1)
                bridge_values[agent_id] = v_bridge

                # V^LSE = τ · log Σ_a exp(Q̂(a)/τ)
                q_for_lse = q_values.clone()
                if avail is not None:
                    q_for_lse[avail == 0] = -1e8
                v_lse[agent_id] = tau * torch.logsumexp(q_for_lse / tau, dim=-1)

                # Causal advantage
                factual_actions = actions_batch[:, agent_id]
                q_factual = q_values[torch.arange(T, device=device), factual_actions]
                factual_onehot = F.one_hot(factual_actions, self.action_dim).float()
                probs_cf = probs * (1.0 - factual_onehot)
                if avail is not None:
                    probs_cf = probs_cf * avail
                renorm_probs = probs_cf / (probs_cf.sum(dim=-1, keepdim=True) + 1e-10)
                cf_baseline = (renorm_probs * q_values).sum(dim=-1)
                causal_advantages[agent_id] = q_factual - cf_baseline

        return bridge_values, causal_advantages, v_lse

    # =================================================================
    # Bridge-GAE
    # =================================================================

    def compute_bridge_gae(
        self,
        rewards: List[float],
        bridge_values: List[float],
        next_bridge_value: float,
        dones: List[bool],
    ) -> Tuple[List[float], List[float]]:
        """Bridge-implied value로 GAE 계산. compute_gae와 동일 구조."""
        advantages = []
        returns = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_val = next_bridge_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_val = bridge_values[step + 1]
            
            delta = rewards[step] + self.gamma * next_val * next_non_terminal - bridge_values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + bridge_values[step])
        
        return advantages, returns

    # =================================================================
    # Action interfaces
    # =================================================================

    def reset_agents(self):
        for agent in self.agents.values():
            agent.reset_rnn_state(device=self.device)
        if self.use_centralized_critic:
            self.rnn_state_value = torch.zeros(self.recurrent_N, self.hidden_dim).to(self.device)
        self.last_dones = {i: False for i in range(self.num_agents)}
    
    def reset_agents_batch(self, n_envs: int):
        for agent in self.agents.values():
            agent.reset_rnn_states_batch(n_envs, device=self.device)
        if self.use_centralized_critic:
            self.rnn_states_value_batch = torch.zeros(
                n_envs, self.recurrent_N, self.hidden_dim, device=self.device
            )
        self.last_dones_batch = {i: [False] * n_envs for i in range(self.num_agents)}

    def get_actions(self, observations: Dict[int, torch.Tensor], 
                    available_actions: Optional[Dict[int, List[int]]] = None) -> Dict[int, Tuple[int, float]]:
        actions = {}
        self.temp_hidden_states = {}
        self.temp_masks = {}
        self.temp_log_probs = {}

        if self.use_centralized_critic:
            episode_done = bool(all(self.last_dones.values())) if len(self.last_dones) > 0 else False
            mask0 = 1.0 if not episode_done else 0.0
            mask_tensor0 = torch.tensor([mask0], dtype=torch.float32, device=self.device)
            global_obs = concatenate_agent_observations(
                observations,
                self.num_agents,
                self.obs_dim,
                self.device,
            )
            with torch.no_grad():
                curr_rnn_state = self.rnn_state_value.unsqueeze(0)
                agent_prev_actions = [self.agents[i].prev_action for i in range(self.num_agents)]
                global_prev_action_onehot = torch.cat(agent_prev_actions, dim=-1).unsqueeze(0)
                _, next_rnn_state = self.value_net(
                    global_obs.unsqueeze(0),
                    curr_rnn_state,
                    mask_tensor0.unsqueeze(0),
                    prev_actions=global_prev_action_onehot
                )
                self.rnn_state_value = next_rnn_state.squeeze(0)
                central_v_state = curr_rnn_state.squeeze(0).clone()

        for agent_id, obs in observations.items():
            mask = 1.0 if not self.last_dones.get(agent_id, False) else 0.0
            mask_tensor = torch.tensor([mask], dtype=torch.float32, device=self.device)

            avail_actions_tensor = None
            if available_actions is not None and agent_id in available_actions:
                avail_actions_tensor = torch.tensor(available_actions[agent_id], dtype=torch.float32, device=self.device)

            agent_prev_action_onehot = self.agents[agent_id].prev_action.unsqueeze(0).clone()
            action, log_prob, p_state = self.agents[agent_id].get_action(obs, mask_tensor, available_actions=avail_actions_tensor)

            if self.use_centralized_critic:
                v_state = central_v_state
            else:
                _, v_state = self.agents[agent_id].get_value(obs, mask_tensor, prev_action_onehot=agent_prev_action_onehot)
            
            actions[agent_id] = (action, log_prob)
            self.temp_hidden_states[agent_id] = (p_state, v_state)
            self.temp_masks[agent_id] = mask
            self.temp_log_probs[agent_id] = log_prob

        return actions

    def get_actions_batch(
        self,
        observations_batch: List[Dict[int, torch.Tensor]],
        available_actions_batch: Optional[List[Optional[Dict[int, List[int]]]]] = None
    ) -> List[Dict[int, Tuple[int, float]]]:
        n_envs = len(observations_batch)
        actions_batch = [{} for _ in range(n_envs)]
        self.temp_hidden_states_batch = [{} for _ in range(n_envs)]
        self.temp_masks_batch = [{} for _ in range(n_envs)]

        if self.use_centralized_critic:
            global_obs_list = [
                concatenate_agent_observations(
                    observations_batch[env_idx],
                    self.num_agents,
                    self.obs_dim,
                    self.device,
                )
                for env_idx in range(n_envs)
            ]
            global_obs_batch_tensor = torch.stack(global_obs_list).to(self.device)
            global_prev_actions_list = [self.agents[i].prev_actions_batch for i in range(self.num_agents)]
            global_prev_action_onehot_batch = torch.cat(global_prev_actions_list, dim=-1)
            central_masks = torch.tensor([
                [
                    1.0
                    if not all(self.last_dones_batch[agent_id][env_idx] for agent_id in range(self.num_agents))
                    else 0.0
                ]
                for env_idx in range(n_envs)
            ], dtype=torch.float32, device=self.device)

            with torch.no_grad():
                curr_states_value = self.rnn_states_value_batch.clone()
                _, next_states_value = self.value_net(
                    global_obs_batch_tensor,
                    self.rnn_states_value_batch,
                    central_masks,
                    prev_actions=global_prev_action_onehot_batch
                )
                self.rnn_states_value_batch = next_states_value

        for agent_id in range(self.num_agents):
            obs_stacked = torch.stack([
                observations_batch[env_idx][agent_id] for env_idx in range(n_envs)
            ]).to(self.device)

            masks_stacked = torch.tensor([
                [1.0 if not self.last_dones_batch[agent_id][env_idx] else 0.0]
                for env_idx in range(n_envs)
            ], dtype=torch.float32, device=self.device)

            avail_actions_stacked = None
            if available_actions_batch is not None and available_actions_batch[0] is not None:
                avail_actions_stacked = torch.stack([
                    torch.tensor(available_actions_batch[env_idx][agent_id], dtype=torch.float32, device=self.device)
                    for env_idx in range(n_envs)
                ])

            agent_prev_actions_onehot = self.agents[agent_id].prev_actions_batch.clone()
            actions, log_probs, p_states = self.agents[agent_id].get_action_batch(
                obs_stacked, masks_stacked, avail_actions_stacked
            )

            if not self.use_centralized_critic:
                _, v_states = self.agents[agent_id].get_value_batch(
                    obs_stacked, masks_stacked, prev_actions_onehot=agent_prev_actions_onehot
                )

            for env_idx in range(n_envs):
                actions_batch[env_idx][agent_id] = (actions[env_idx].item(), log_probs[env_idx].item())
                if self.use_centralized_critic:
                    v_state_for_env = curr_states_value[env_idx]
                else:
                    v_state_for_env = v_states[env_idx].clone()
                self.temp_hidden_states_batch[env_idx][agent_id] = (p_states[env_idx].clone(), v_state_for_env)
                self.temp_masks_batch[env_idx][agent_id] = masks_stacked[env_idx, 0].item()

        return actions_batch

    # =================================================================
    # Transition storage
    # =================================================================

    def store_transition(self, prev_observations, prev_rewards, prev_actions,
                         observations, actions, rewards, next_observations, dones,
                         info=None, available_actions=None, env_id=0):
        self.last_dones = dones.copy()
        kwargs = build_transition_kwargs(
            prev_observations=prev_observations,
            prev_rewards=prev_rewards,
            prev_actions=prev_actions,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            masks=self.temp_masks,
            hidden_states=self.temp_hidden_states,
            available_actions=available_actions,
            log_probs=self.temp_log_probs,
            info=info,
            env_id=env_id if self.use_episode_buffer else None,
        )
        self.buffer_manager.store_transition(**kwargs)

    def store_transition_from_env(self, env_idx, prev_observations, prev_rewards, prev_actions,
                                  observations, actions, rewards, next_observations, dones,
                                  info=None, available_actions=None):
        actions_only, log_probs_dict = split_action_and_log_probs(actions)
        info = prepare_env_info(info, env_idx)
        episode_done = resolve_step_episode_done(dones, info)
        kwargs = build_transition_kwargs(
            prev_observations=prev_observations,
            prev_rewards=prev_rewards,
            prev_actions=prev_actions,
            observations=observations,
            actions=actions_only,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            masks=self.temp_masks_batch[env_idx],
            hidden_states=self.temp_hidden_states_batch[env_idx],
            available_actions=available_actions,
            log_probs=log_probs_dict,
            info=info,
            env_id=env_idx if self.use_episode_buffer else None,
        )
        self.buffer_manager.store_transition(**kwargs)

        if self.use_episode_buffer:
            if episode_done:
                self.buffer_manager.finish_episode(env_id=env_idx)

        for agent_id in range(self.num_agents):
            self.last_dones_batch[agent_id][env_idx] = dones[agent_id]
            if dones[agent_id]:
                self.agents[agent_id].rnn_states_policy_batch[env_idx] = torch.zeros(
                    self.recurrent_N, self.hidden_dim, device=self.device)
                if not self.use_centralized_critic:
                    self.agents[agent_id].rnn_states_value_batch[env_idx] = torch.zeros(
                        self.recurrent_N, self.hidden_dim, device=self.device)
                self.agents[agent_id].prev_actions_batch[env_idx] = torch.zeros(
                    self.action_dim, device=self.device)

        if self.use_centralized_critic and episode_done:
            self.rnn_states_value_batch[env_idx] = torch.zeros(
                self.recurrent_N, self.hidden_dim, device=self.device
            )

    def finish_episode(self, env_id: int = 0):
        if self.use_episode_buffer:
            self.buffer_manager.finish_episode(env_id=env_id)

    # =================================================================
    # Utility
    # =================================================================

    # =================================================================
    # PPO policy + value update
    # =================================================================

    def update_agents(self, batch_data: List[Dict]) -> Dict[str, float]:
        agent_losses = defaultdict(float)
        
        full_obs_batch = torch.stack([
            torch.stack(list(t['bold_o_t'].values())) for t in batch_data
        ]).to(self.device, non_blocking=True)
        T, N, D = full_obs_batch.shape
        
        actions_batch = torch.tensor(
            [[t['bold_a_t'][i] for i in range(self.num_agents)] for t in batch_data],
            dtype=torch.long, device=self.device
        )

        available_actions_batch = build_available_actions_batch(
            batch_data,
            self.num_agents,
            self.device,
        )

        masks_batch = build_agent_masks_batch(
            batch_data,
            self.num_agents,
            self.device,
        )

        # PPO Update (policy + value)
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue
            
            agent_obs = full_obs_batch[:, agent_id, :]
            my_actions = actions_batch[:, agent_id]
            
            # [Direction 2] Extract env and pci advantages separately
            env_adv_raw = torch.tensor(
                [t['env_advantages'][agent_id] for t in batch_data],
                dtype=torch.float32, device=self.device)
            pci_adv_raw = torch.tensor(
                [t['pci_advantages'][agent_id] for t in batch_data],
                dtype=torch.float32, device=self.device)
            causal_lambda = self.get_current_causal_lambda()

            # [Direction 2] adv_batch = batch-normalize(env_norm)
            adv = normalize_advantages(env_adv_raw)
            pci_adv = normalize_advantages(pci_adv_raw)
            # [Direction 1] adv_batch = batch-normalize(env_norm + λ*pci_norm); uncomment to revert
            # adv_raw = torch.tensor(
            #     [t['advantages'][agent_id] for t in batch_data],
            #     dtype=torch.float32, device=self.device)
            # adv = normalize_advantages(adv_raw)

            # Returns for value loss
            agent_returns = torch.tensor(
                [t['returns'][agent_id] for t in batch_data],
                dtype=torch.float32, device=self.device)

            p_states_all = torch.stack([
                t['hidden_states'][agent_id][0] for t in batch_data
            ]).to(self.device)
            
            avail_actions_all = available_actions_batch.get(agent_id, None)

            T_total = agent_obs.size(0)
            chunk_len = self.rnn_chunk_length
            num_chunks = T_total // chunk_len
            agent_active_masks = build_agent_active_masks(batch_data, agent_id, self.device)

            if num_chunks > 0:
                rnn_obs_batch = chunk_sequence(agent_obs, chunk_len)
                my_actions_batch = chunk_sequence(my_actions, chunk_len).reshape(-1)
                rnn_masks_batch = chunk_sequence(masks_batch[agent_id], chunk_len)
                
                prev_actions_raw = torch.tensor(
                    [t['bold_a_t_minus_1'][agent_id] for t in batch_data[:num_chunks * chunk_len]],
                    dtype=torch.long, device=self.device)
                prev_actions_onehot = F.one_hot(prev_actions_raw, self.action_dim).float()
                prev_actions_batch = chunk_sequence(prev_actions_onehot, chunk_len)

                active_masks_batch = chunk_sequence(agent_active_masks, chunk_len).reshape(-1)

                avail_actions_batch_merged = None
                if avail_actions_all is not None:
                    avail_actions_batch_merged = chunk_sequence(avail_actions_all, chunk_len)\
                        .reshape(-1, avail_actions_all.size(-1))

                chunk_indices = torch.arange(0, num_chunks * chunk_len, chunk_len)
                p_h0_batch = p_states_all[chunk_indices]
                adv_batch = chunk_sequence(adv, chunk_len).reshape(-1)
                pci_adv_batch = chunk_sequence(pci_adv, chunk_len).reshape(-1)  # [Direction 2]
                ret_batch = chunk_sequence(agent_returns, chunk_len).reshape(-1)
            else:
                rnn_obs_batch = agent_obs.unsqueeze(1)
                rnn_masks_batch = masks_batch[agent_id]
                p_h0_batch = p_states_all[0].unsqueeze(0)
                my_actions_batch = my_actions
                adv_batch = adv
                pci_adv_batch = pci_adv  # [Direction 2]
                ret_batch = agent_returns
                active_masks_batch = agent_active_masks
                avail_actions_batch_merged = avail_actions_all
                prev_actions_raw = torch.tensor(
                    [t['bold_a_t_minus_1'][agent_id] for t in batch_data],
                    dtype=torch.long, device=self.device)
                prev_actions_batch = F.one_hot(prev_actions_raw, self.action_dim).float().unsqueeze(1)

            old_log_probs_raw = torch.tensor(
                [t['log_probs'][agent_id] for t in batch_data],
                dtype=torch.float32, device=self.device)
            if num_chunks > 0:
                old_log_probs = chunk_sequence(old_log_probs_raw, chunk_len).reshape(-1)
            else:
                old_log_probs = old_log_probs_raw

            for epoch in range(self.ppo_epochs):
                # Policy loss
                new_log_probs, entropy = self.agents[agent_id].policy_net.evaluate_actions(
                    rnn_obs_batch, p_h0_batch, my_actions_batch, rnn_masks_batch,
                    available_actions=avail_actions_batch_merged, prev_actions=prev_actions_batch
                )
                ratio = torch.exp(new_log_probs - old_log_probs)

                # ── DIRECTION 2: Separate Actor Loss ─────────────────────────────
                surr1_env = ratio * adv_batch
                surr2_env = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
                surr1_pci = ratio * pci_adv_batch
                surr2_pci = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * pci_adv_batch
                policy_loss = -(torch.min(surr1_env, surr2_env) + causal_lambda * torch.min(surr1_pci, surr2_pci))
                policy_loss = (policy_loss * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)
                # ── DIRECTION 1: Simple Weighted Sum (uncomment to revert) ───────
                # surr1 = ratio * adv_batch
                # surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
                # policy_loss = -torch.min(surr1, surr2)
                # policy_loss = (policy_loss * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)
                entropy_loss = (-entropy * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)

                loss = policy_loss + self.entropy_coef * entropy_loss
                
                # Policy update
                self.agents[agent_id].optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[agent_id].policy_net.parameters(), self.max_grad_norm)
                self.agents[agent_id].optimizer.step()

                if not self.use_centralized_critic:
                    v_states_all = torch.stack([
                        t['hidden_states'][agent_id][1] for t in batch_data
                    ]).to(self.device)
                    if num_chunks > 0:
                        v_h0_batch = v_states_all[chunk_indices]
                    else:
                        v_h0_batch = v_states_all[0].unsqueeze(0)

                    new_vals, _ = self.agents[agent_id].value_net(
                        rnn_obs_batch, v_h0_batch, rnn_masks_batch,
                        prev_actions=prev_actions_batch
                    )
                    val_error = (new_vals.squeeze(-1).reshape(-1) - ret_batch) ** 2
                    val_loss = (val_error * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)

                    if not self.use_value_parameter_sharing:
                        self.agents[agent_id].value_optimizer.zero_grad()
                        (self.value_coef * val_loss).backward()
                        torch.nn.utils.clip_grad_norm_(self.agents[agent_id].value_net.parameters(), self.max_grad_norm)
                        self.agents[agent_id].value_optimizer.step()
                        agent_losses[f'agent_{agent_id}_value_loss'] += val_loss.item()
                    else:
                        if not hasattr(self, '_temp_epoch_value_loss'):
                            self._temp_epoch_value_loss = 0.0
                            self._temp_epoch_mask_sum = 0.0
                        self._temp_epoch_value_loss += val_error.sum()
                        self._temp_epoch_mask_sum += active_masks_batch.sum()
                
                agent_losses[f'agent_{agent_id}_policy_loss'] += policy_loss.item()
                agent_losses[f'agent_{agent_id}_entropy_loss'] += entropy_loss.item()

        # Shared value update (once per batch, after all agents)
        if not self.use_centralized_critic and self.use_value_parameter_sharing:
            if hasattr(self, '_temp_epoch_mask_sum') and self._temp_epoch_mask_sum > 0:
                mean_val_loss = self._temp_epoch_value_loss / self._temp_epoch_mask_sum
                self.shared_value_optimizer.zero_grad()
                (self.value_coef * mean_val_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.shared_value_net.parameters(), self.max_grad_norm)
                self.shared_value_optimizer.step()
                agent_losses['value_loss'] = mean_val_loss.item()
                del self._temp_epoch_value_loss
                del self._temp_epoch_mask_sum

        if self.use_centralized_critic:
            global_obs_seq = torch.stack([
                concatenate_agent_observations(
                    t['bold_o_t'],
                    self.num_agents,
                    self.obs_dim,
                    self.device,
                )
                for t in batch_data
            ]).to(self.device, non_blocking=True)
            global_masks_seq = torch.tensor(
                [t['masks'][0] for t in batch_data],
                device=self.device
            ).unsqueeze(1)
            v_states_all_central = torch.stack([
                t['hidden_states'][0][1] for t in batch_data
            ]).to(self.device, non_blocking=True)
            all_returns = torch.tensor(
                [[t['returns'][i] for i in range(self.num_agents)] for t in batch_data],
                dtype=torch.float32,
                device=self.device,
            )

            T_total = global_obs_seq.size(0)
            chunk_len = self.rnn_chunk_length
            num_chunks = T_total // chunk_len

            if num_chunks > 0:
                global_obs_batch = chunk_sequence(global_obs_seq, chunk_len)
                global_masks_batch = chunk_sequence(global_masks_seq, chunk_len)
                if 'active_mask' in batch_data[0]:
                    active_masks = torch.tensor(
                        [t['active_mask'] for t in batch_data],
                        device=self.device
                    ).unsqueeze(1)
                    active_masks_batch = chunk_sequence(active_masks, chunk_len).reshape(-1)
                else:
                    active_masks_batch = torch.ones(num_chunks * chunk_len, device=self.device)
                chunk_indices = torch.arange(0, num_chunks * chunk_len, chunk_len)
                v_h0_batch = v_states_all_central[chunk_indices]
                global_ret_batch = chunk_sequence(all_returns, chunk_len).reshape(-1, self.num_agents)
                alive_mask_batch = chunk_sequence(
                    build_alive_mask_matrix(batch_data, self.num_agents, self.device),
                    chunk_len
                ).reshape(-1, self.num_agents)
            else:
                global_obs_batch = global_obs_seq.unsqueeze(1)
                global_masks_batch = global_masks_seq
                active_masks_batch = torch.ones(T_total, device=self.device)
                v_h0_batch = v_states_all_central[0].unsqueeze(0)
                global_ret_batch = all_returns
                alive_mask_batch = build_alive_mask_matrix(batch_data, self.num_agents, self.device)

            for _ in range(self.ppo_epochs):
                agent_prev_actions_list = []
                for agent_id in range(self.num_agents):
                    agent_prev_actions = torch.tensor(
                        [t['bold_a_t_minus_1'][agent_id] for t in batch_data],
                        dtype=torch.long,
                        device=self.device,
                    )
                    agent_prev_actions_onehot = F.one_hot(agent_prev_actions, self.action_dim).float()
                    agent_prev_actions_list.append(agent_prev_actions_onehot)

                joint_prev_actions_batch_raw = torch.cat(agent_prev_actions_list, dim=-1)
                if num_chunks > 0:
                    global_prev_actions_batch = chunk_sequence(joint_prev_actions_batch_raw, chunk_len)
                else:
                    global_prev_actions_batch = joint_prev_actions_batch_raw.unsqueeze(1)

                new_vals, _ = self.value_net(
                    global_obs_batch,
                    v_h0_batch,
                    global_masks_batch,
                    prev_actions=global_prev_actions_batch
                )
                val_error = (new_vals.squeeze(1) - global_ret_batch) ** 2
                value_mask = active_masks_batch.unsqueeze(1) * alive_mask_batch
                val_loss = (val_error * value_mask).sum() / (value_mask.sum() + 1e-8)

                self.value_optimizer.zero_grad()
                (self.value_coef * val_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                agent_losses['value_loss'] += val_loss.item()

        for k in list(agent_losses.keys()):
            if 'delta' not in k and 'value_loss' not in k:
                agent_losses[k] /= self.ppo_epochs
            elif k == 'value_loss':
                if self.use_centralized_critic or self.use_value_parameter_sharing:
                    agent_losses['value_loss'] /= self.ppo_epochs
                else:
                    total_value_loss = sum(
                        [agent_losses.get(f'agent_{i}_value_loss', 0.0) for i in range(self.num_agents)]
                    )
                    agent_losses['value_loss'] = total_value_loss / (self.num_agents * self.ppo_epochs)

        return dict(agent_losses)

    # =================================================================
    # Main update loop
    # =================================================================

    def update(self, **kwargs):
        trajectories = []
        
        if self.use_episode_buffer:
            if not self.buffer_manager.has_completed_episodes(): return {}
            trajectories = self.buffer_manager.get_completed_episodes()
        else:
            if self.buffer_manager.is_main_buffer_empty(): return {}
            transitions = self.buffer_manager.get_main_buffer_transitions()
            env_trajectories = defaultdict(list)
            for t in transitions:
                env_id = t['info'].get('env_id', 0)
                env_trajectories[env_id].append(t)
            trajectories = list(env_trajectories.values())

        # === 1. Compute team returns for each trajectory ===
        for traj in trajectories:
            if len(traj) == 0: continue
            traj_returns = self._compute_trajectory_returns(traj)
            for t_idx, t in enumerate(traj):
                t['team_return'] = traj_returns[t_idx].item()

        # === 2. Compute raw-return critic targets and PCI-augmented actor advantages ===
        for traj in trajectories:
            if len(traj) == 0: continue
            
            T = len(traj)
            N = self.num_agents
            team_done_seq = build_team_done_sequence(traj)
            team_done_tensor = torch.tensor(
                team_done_seq,
                dtype=torch.float32,
                device=self.device,
            )

            # Prepare hidden states
            hidden_states_batch = {}
            if 'hidden_states' in traj[0] and traj[0]['hidden_states'] is not None:
                for agent_id in range(N):
                    if agent_id in traj[0]['hidden_states']:
                        p_states = torch.stack([
                            t['hidden_states'][agent_id][0] for t in traj
                        ]).to(self.device, non_blocking=True)
                        hidden_states_batch[agent_id] = p_states

            available_actions_batch = {}
            if 'available_actions' in traj[0] and traj[0]['available_actions'] is not None:
                for agent_id in range(N):
                    if agent_id in traj[0]['available_actions']:
                        available_actions_batch[agent_id] = torch.tensor(
                            [t['available_actions'][agent_id] for t in traj],
                            dtype=torch.float32, device=self.device)

            prev_actions_batch = torch.tensor(
                [[t['bold_a_t_minus_1'][i] for i in range(N)] for t in traj],
                dtype=torch.long, device=self.device)

            # --- Step 2a: Bridge → V^bridge, V^LSE, causal advantages ---
            with torch.no_grad():
                if self.get_current_causal_lambda() > 0.0:
                    bridge_values, causal_advantages, v_lse = self.compute_bridge_values(
                        traj, hidden_states_batch,
                        available_actions_batch=available_actions_batch,
                        prev_actions_batch=prev_actions_batch,
                    )
                else:
                    bridge_values = {i: torch.zeros(T, device=self.device) for i in range(N)}
                    causal_advantages = {i: torch.zeros(T, device=self.device) for i in range(N)}
                    v_lse = {i: torch.zeros(T, device=self.device) for i in range(N)}

            # --- Step 2b: Value network → raw GAE for critic, PCI residual for actor ---
            lambda_c = self.get_current_causal_lambda()

            # Team reward per step (raw, no shaping)
            team_rewards = [
                sum(traj[t_idx]['bold_r_t'][i] for i in range(N)) / N
                for t_idx in range(T)
            ]
            team_rewards_tensor = torch.tensor(team_rewards, dtype=torch.float32, device=self.device)

            central_values = None
            central_next_values = None
            if self.use_centralized_critic:
                global_obs_seq = torch.stack([concatenate_agent_observations(t['bold_o_t'], self.num_agents, self.obs_dim, self.device,)
                        for t in traj]).to(self.device, non_blocking=True)
                
                global_masks_seq = torch.tensor([t['masks'][0] for t in traj], dtype=torch.float32, device=self.device,).unsqueeze(1)
                v_h0 = traj[0]['hidden_states'][0][1].to(self.device, non_blocking=True).unsqueeze(0)

                with torch.no_grad():
                    last_next_obs = concatenate_agent_observations(traj[-1]['next_observations'], self.num_agents, self.obs_dim, self.device,).to(self.device, non_blocking=True)
                    last_mask = 1.0 - (1.0 if team_done_seq[-1] else 0.0)

                    extended_obs = torch.cat([
                        global_obs_seq.unsqueeze(1),
                        last_next_obs.unsqueeze(0).unsqueeze(1)
                    ], dim=0)
                    extended_masks = torch.cat([
                        global_masks_seq,
                        torch.tensor([[last_mask]], dtype=torch.float32, device=self.device)
                    ], dim=0)

                    agent_prev_actions_list = []
                    for agent_id in range(self.num_agents):
                        agent_prev_actions = torch.tensor(
                            [t['bold_a_t_minus_1'][agent_id] for t in traj],
                            dtype=torch.long,
                            device=self.device,
                        )
                        agent_prev_actions_onehot = F.one_hot(agent_prev_actions, self.action_dim).float()
                        last_prev_action_onehot = agent_prev_actions_onehot[-1].unsqueeze(0)
                        extended_agent_prev_actions = torch.cat(
                            [agent_prev_actions_onehot, last_prev_action_onehot],
                            dim=0
                        )
                        agent_prev_actions_list.append(extended_agent_prev_actions)

                    joint_prev_actions = torch.cat(agent_prev_actions_list, dim=-1).unsqueeze(1)
                    all_values, _ = self.value_net(
                        extended_obs,
                        v_h0,
                        extended_masks,
                        prev_actions=joint_prev_actions
                    )
                    central_values = all_values[:-1].view(-1, self.num_agents)
                    central_next_values = all_values[1:].view(-1, self.num_agents)

                team_bootstrap_target = team_rewards_tensor + self.gamma * (
                    central_next_values.mean(dim=-1) * (1.0 - team_done_tensor)
                )
            else:
                team_bootstrap_target_sum = torch.zeros(T, dtype=torch.float32, device=self.device)
                team_bootstrap_target_count = 0

            for agent_id in range(N):
                if agent_id not in traj[0]['bold_o_t']: continue

                agent_obs_list = [t['bold_o_t'][agent_id] for t in traj]
                agent_obs_seq = torch.stack(agent_obs_list).to(self.device, non_blocking=True)
                agent_masks_seq = torch.tensor(
                    [t['masks'][agent_id] for t in traj], device=self.device
                ).unsqueeze(1)
                agent_dones_seq = torch.tensor(
                    [t['dones'][agent_id] for t in traj],
                    dtype=torch.float32, device=self.device
                )

                # Value network hidden state
                _, v_h0 = traj[0]['hidden_states'][agent_id]
                v_h0 = v_h0.to(self.device, non_blocking=True).unsqueeze(0)

                with torch.no_grad():
                    if self.use_centralized_critic:
                        values = central_values[:, agent_id]
                        next_val = central_next_values[-1, agent_id].item() * (
                            1.0 - (1.0 if team_done_seq[-1] else 0.0)
                        )
                    else:
                        last_next_obs = traj[-1]['next_observations'][agent_id].to(self.device, non_blocking=True)
                        last_mask = 1.0 - (1.0 if team_done_seq[-1] else 0.0)

                        extended_obs = torch.cat([
                            agent_obs_seq.unsqueeze(1),
                            last_next_obs.unsqueeze(0).unsqueeze(1)
                        ], dim=0)
                        extended_masks = torch.cat([
                            agent_masks_seq,
                            torch.tensor([[last_mask]], dtype=torch.float32, device=self.device)
                        ], dim=0)

                        agent_prev_actions = torch.tensor(
                            [t['bold_a_t_minus_1'][agent_id] for t in traj],
                            dtype=torch.long, device=self.device
                        )
                        agent_prev_actions_onehot = F.one_hot(agent_prev_actions, self.action_dim).float()
                        last_prev_action_onehot = agent_prev_actions_onehot[-1].unsqueeze(0)
                        extended_prev_actions = torch.cat([
                            agent_prev_actions_onehot, last_prev_action_onehot
                        ], dim=0).unsqueeze(1)

                        all_values, _ = self.agents[agent_id].value_net(
                            extended_obs, v_h0, extended_masks,
                            prev_actions=extended_prev_actions
                        )
                        values = all_values[:-1].view(-1)
                        next_val = all_values[-1].item() * last_mask
                        next_value_seq = all_values[1:].view(-1) * (1.0 - team_done_tensor)

                        team_bootstrap_target_sum += team_rewards_tensor + self.gamma * next_value_seq
                        team_bootstrap_target_count += 1

                    env_advantages, returns_for_vloss = compute_gae(
                        team_rewards,
                        values.tolist(),
                        next_val,
                        team_done_seq,
                        self.gamma,
                        self.gae_lambda,
                    )

                    pci_adv = causal_advantages[agent_id]
                    combined_advantages, env_adv_norm, pci_adv_residual = self._combine_env_and_pci_advantages(
                        torch.tensor(env_advantages, dtype=torch.float32, device=self.device),
                        pci_adv,
                        lambda_c,
                    )

                    for t_idx, t in enumerate(traj):
                        if 'advantages' not in t:
                            t['advantages'] = {}
                            t['returns'] = {}
                            t['env_advantages'] = {}
                            t['pci_advantages'] = {}
                        t['advantages'][agent_id] = combined_advantages[t_idx].item()
                        t['returns'][agent_id] = returns_for_vloss[t_idx]  # value loss uses raw returns
                        t['env_advantages'][agent_id] = env_adv_norm[t_idx].item()
                        t['pci_advantages'][agent_id] = pci_adv_residual[t_idx].item()

            if not self.use_centralized_critic and team_bootstrap_target_count > 0:
                team_bootstrap_target = team_bootstrap_target_sum / team_bootstrap_target_count
            elif not self.use_centralized_critic:
                team_bootstrap_target = team_rewards_tensor

            for t_idx, t in enumerate(traj):
                t['team_bootstrap_target'] = team_bootstrap_target[t_idx].item()

        # === 3. Create chunks with padding ===
        all_chunks = []
        chunk_len = self.rnn_chunk_length

        for traj in trajectories:
            if len(traj) == 0: continue
            for t in traj:
                t['active_mask'] = 1.0

            T_traj = len(traj)
            num_full_chunks = T_traj // chunk_len
            remainder = T_traj % chunk_len

            for i in range(num_full_chunks):
                start = i * chunk_len
                all_chunks.append(traj[start:start + chunk_len])

            if remainder > 0:
                last_chunk = traj[num_full_chunks * chunk_len:]
                last_valid = last_chunk[-1]
                for _ in range(chunk_len - remainder):
                    pad_t = last_valid.copy()
                    pad_t['active_mask'] = 0.0
                    last_chunk.append(pad_t)
                all_chunks.append(last_chunk)

        # === 4. Create batches ===
        n_chunks_per_batch = max(1, len(all_chunks) // self.num_batches)
        chunk_batches = [
            all_chunks[i:i + n_chunks_per_batch]
            for i in range(0, len(all_chunks), n_chunks_per_batch)
        ]
        batches = []
        for b in chunk_batches:
            flat = []
            for chunk in b:
                flat.extend(chunk)
            if flat:
                batches.append(flat)

        # Clear buffers
        if self.use_episode_buffer:
            self.buffer_manager.clear_completed_episodes()
        else:
            self.buffer_manager.clear_buffer()

        total_losses = defaultdict(float)

        # === 5. Train Q-Bridge + Policy ===
        for batch_data in batches:
            if not batch_data: continue
            
            if self.get_current_causal_lambda() != 0.0:
                bridge_losses = self.update_q_bridge(batch_data)
            else:
                bridge_losses = {'q_bridge_loss': 0.0}
            
            agent_losses = self.update_agents(batch_data)

            for k, v in bridge_losses.items():
                total_losses[k] += v
            for k, v in agent_losses.items():
                total_losses[k] += v

        self.buffer_manager.clear_buffer()
        self.training_steps += 1

        num = len([b for b in batches if b])
        if num == 0: return {}
        final_losses = {}
        for k, v in total_losses.items():
            if 'delta_agent' in k and '_sum' in k:
                base_key = k.replace('_sum', '')
                count_key = base_key + '_count'
                total_count = total_losses.get(count_key, 0)
                if total_count > 0:
                    final_losses[base_key + '_mean'] = v / total_count
                    final_losses[count_key] = total_count
            elif 'delta_agent' in k and '_count' in k:
                continue
            else:
                final_losses[k] = v / num if num > 0 else v

        return final_losses

    # =================================================================
    # Buffer info
    # =================================================================

    def clear_buffer(self):
        self.buffer_manager.clear_buffer()
    
    def get_buffer_info(self) -> Dict[str, int]:
        return self.buffer_manager.get_buffer_info()
