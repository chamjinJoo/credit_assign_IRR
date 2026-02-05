import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
import random

from algorithms.networks import PolicyNetwork, OutcomeBridgeNetwork, ActionBridgeNetwork, ValueNetwork
from utils.buffer import TransitionBuffer, EpisodeBuffer


class RBFKernel(nn.Module):
    """
    An improved RBF Kernel implementation.
    """
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
        """
        Computes kernel values element-wise: k(x1[i], x2[i])
        Args:
            x1: [N, D]
            x2: [N, D]
        Returns:
            [N] tensor
        """
        gamma = torch.exp(self.log_gamma) # 단순히 뺄셈 후 제곱합 (행렬곱 불필요)
        dists_sq = torch.sum((x1 - x2)**2, dim=-1) 
        return torch.exp(-gamma * dists_sq)

class LinearKernel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, x.T)

    def forward_asymmetric(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x1, x2.T)

class PCIMAPPOAgent:
    def __init__(self, agent_id: int, obs_dim: int, action_dim: int, hidden_dim: int = 64, lr_actor: float = 3e-4, lr_critic: float = 3e-4):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy_net = PolicyNetwork(obs_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_actor)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)
        
    def get_action(self, obs: torch.Tensor, available_actions: Optional[torch.Tensor] = None) -> Tuple[int, float]:
        with torch.no_grad():
            if available_actions is not None:
                available_actions = available_actions.unsqueeze(0) if available_actions.dim() == 1 else available_actions
            
            action, log_prob = self.policy_net.get_action_and_log_prob(obs.unsqueeze(0), available_actions)
        return action.item(), log_prob.item()

    def get_action_batch(self, obs_batch: torch.Tensor, available_actions_batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get actions for a batch of environments.
        """
        with torch.no_grad():
             if available_actions_batch is not None:
                # Ensure correct dimensions if necessary (should be B x ActionDim)
                pass 
             
             action, log_prob = self.policy_net.get_action_and_log_prob(obs_batch, available_actions_batch)
        
        return action, log_prob


class PCIMAPPOAlgorithm:
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        w_dim: int, 
        hidden_dim: int = 64,
        lr_actor: float = 3e-4,
        lr_bridge: float = 3e-4,
        lr_critic: float = 3e-4,
        clip_delta: float = 0.2,
        causal_lambda: float = 1.0,
        causal_lambda_schedule: bool = False,
        causal_lambda_adaptive_per_agent: bool = False,
        causal_lambda_adaptive_alpha: float = 0.9,
        causal_lambda_adaptive_min: float = 0.1,
        causal_lambda_adaptive_max: float = 1.0,
        causal_lambda_adaptive_error_threshold: float = 0.5,
        causal_lambda_start: float = 0.0,
        causal_lambda_final: float = 0.2,
        causal_lambda_warmup_steps: int = 10000,
        causal_lambda_startingpoint: int = 0,
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
        # Optimization parameter
        causal_mini_batch_size: int = 256,  # [ADDED] Chunk size for RKHS solver
        # Value network parameter sharing
        use_value_parameter_sharing: bool = True,  # If False, use individual value networks for each agent
        use_episode_buffer: bool = False,
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.w_dim = w_dim
        self.env_required_agents = env_required_agents
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.causal_mini_batch_size = causal_mini_batch_size # Critical for speed
        self.use_value_parameter_sharing = use_value_parameter_sharing
        
        self.clip_delta = clip_delta
        self.causal_lambda = causal_lambda
        self.causal_lambda_schedule = causal_lambda_schedule
        self.causal_lambda_adaptive_per_agent = causal_lambda_adaptive_per_agent
        self.causal_lambda_adaptive_alpha = causal_lambda_adaptive_alpha
        self.causal_lambda_adaptive_min = causal_lambda_adaptive_min
        self.causal_lambda_adaptive_max = causal_lambda_adaptive_max
        self.causal_lambda_adaptive_error_threshold = causal_lambda_adaptive_error_threshold
        self.causal_lambda_start = causal_lambda_start
        self.causal_lambda_final = causal_lambda_final
        self.causal_lambda_warmup_steps = causal_lambda_warmup_steps
        self.causal_lambda_startingpoint = causal_lambda_startingpoint
        self.training_steps = 0
        
        # Agent별 adaptive lambda를 위한 error tracking
        if self.causal_lambda_adaptive_per_agent:
            self.adaptive_error_ema_per_agent = {i: None for i in range(num_agents)}
            self.adaptive_lambda_per_agent = {i: causal_lambda for i in range(num_agents)}
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.use_episode_buffer = use_episode_buffer
        
        if use_episode_buffer:
            self.buffer_manager = EpisodeBuffer(max_episodes=100)
        else:
            self.buffer_manager = TransitionBuffer(max_buffer_size=max_buffer_size)
        
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = PCIMAPPOAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                lr_actor=lr_actor,
                lr_critic=lr_critic
            )
        
        # Value network parameter sharing 설정
        if self.use_value_parameter_sharing:
            # 하나의 shared value network 생성
            self.shared_value_net = ValueNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim)
            self.shared_value_optimizer = optim.Adam(self.shared_value_net.parameters(), lr=lr_critic)
            # for checkpointing and loading (호환성을 위해 딕셔너리 유지)
            self.value_nets = {agent_id: self.shared_value_net for agent_id in range(num_agents)}
            self.value_optimizers = {agent_id: self.shared_value_optimizer for agent_id in range(num_agents)}
        else:
            # 각 agent별로 별도의 value network 사용
            self.value_nets = {agent_id: agent.value_net for agent_id, agent in self.agents.items()}
            self.value_optimizers = {agent_id: agent.value_optimizer for agent_id, agent in self.agents.items()}
        
        self.outcome_bridge_net = OutcomeBridgeNetwork( 
            action_dim=action_dim,
            num_agents=num_agents,
            w_dim=obs_dim * (num_agents-1),
            outcome_dim=num_agents-1,
            hidden_dim=hidden_dim,
            action_embedding_dim=8
        )
        
        self.action_bridge_net = ActionBridgeNetwork(
            z_dim=obs_dim * num_agents, # Default Z
            # z_dim=obs_dim * num_agents + num_agents, # [SWITCH] Z = [Joint Obs, Joint Reward]
            # z_dim=obs_dim * num_agents + action_dim * num_agents, # [SWITCH] Z = [Joint Obs, Joint Action]
            # z_dim=num_agents, # [SWITCH] previous reward vector as proxy
            action_dim=action_dim,
            num_agents=num_agents,
            hidden_dim=hidden_dim,
            action_embedding_dim=8
        )
        
        self.critic_kernel_z = RBFKernel(gamma=1.0).to(self.device)
        self.critic_kernel_w = RBFKernel(gamma=1.0).to(self.device)
        
        self.stabilizer_lambda = 1.0
        self.tikhonov_gamma = 1e-4

        self.action_embedding_for_kernel = nn.Embedding(action_dim, 8)
        
        self.outcome_bridge_optimizer = optim.Adam(self.outcome_bridge_net.parameters(), lr=lr_bridge)
        self.action_bridge_optimizer = optim.Adam(self.action_bridge_net.parameters(), lr=lr_critic)
        self.kernel_action_embed_optimizer = optim.Adam(self.action_embedding_for_kernel.parameters(), lr=lr_critic)
        
        self._move_networks_to_device()
    
    def _move_networks_to_device(self):
        for agent in self.agents.values():
            agent.policy_net.to(self.device)
            if not self.use_value_parameter_sharing:
                agent.value_net.to(self.device)
        if self.use_value_parameter_sharing:
            self.shared_value_net.to(self.device)
        self.outcome_bridge_net.to(self.device)
        self.action_bridge_net.to(self.device)
        self.critic_kernel_z.to(self.device)
        self.critic_kernel_w.to(self.device)
        self.action_embedding_for_kernel.to(self.device)
    
    def get_current_causal_lambda(self, agent_id: Optional[int] = None) -> float:
        # Agent별 adaptive lambda 사용
        if self.causal_lambda_adaptive_per_agent and agent_id is not None:
            return self.adaptive_lambda_per_agent[agent_id]
        
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
    
    def update_causal_networks(self, batch_data: List[Dict], mc_samples_k_w2: int = 16) -> Dict[str, float]:
        if self.get_current_causal_lambda() == 0.0:
            return {'outcome_bridge_loss': 0.0, 'action_bridge_loss': 0.0}
        
        batch_size = len(batch_data)
        num_agents = self.num_agents
        device = self.device

        # === 1. Data Preparation (Vectorized) ===
        joint_o_prev = torch.stack([self._concatenate_observations(t['bold_o_t_minus_1']) for t in batch_data]).to(device) # Default Z
        # joint_a_prev = torch.tensor([[t['bold_a_t_minus_1'][i] for i in range(num_agents)] for t in batch_data], dtype=torch.long, device=device)
        # joint_a_prev_onehot = F.one_hot(joint_a_prev, num_classes=self.action_dim).view(batch_size, -1).float()
        # joint_r_prev = torch.tensor([[t['bold_r_t_minus_1'][i] for i in range(num_agents)] for t in batch_data], dtype=torch.float32, device=device) # [SWITCH] prev rewards
        obs_batch = torch.stack([torch.stack(list(t['bold_o_t'].values())) for t in batch_data]).to(device)
        action_batch = torch.tensor([[t['bold_a_t'][i] for i in range(num_agents)] for t in batch_data], dtype=torch.long, device=device)
        reward_batch = torch.tensor([[t['bold_r_t'][i] for i in range(num_agents)] for t in batch_data], dtype=torch.float32, device=device)

        A_super = action_batch.view(-1)
        expanded_obs = obs_batch.unsqueeze(1).expand(-1, num_agents, -1, -1)
        mask = ~torch.eye(num_agents, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.obs_dim)
        W_super = expanded_obs[mask].view(batch_size * num_agents, -1)
        
        expanded_rewards = reward_batch.unsqueeze(1).expand(-1, num_agents, -1)
        mask_rewards = ~torch.eye(num_agents, dtype=torch.bool, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        Y_super = expanded_rewards[mask_rewards].view(batch_size * num_agents, -1)

        Z_super = joint_o_prev.repeat_interleave(num_agents, dim=0) # Default Z
        # Z_super = joint_r_prev.repeat_interleave(num_agents, dim=0) # [SWITCH] prev rewards
        # Z_super = torch.cat([joint_o_prev, joint_r_prev], dim=1).repeat_interleave(num_agents, dim=0)
        # Z_super = torch.cat([joint_o_prev, joint_a_prev_onehot], dim=1).repeat_interleave(num_agents, dim=0)

        with torch.no_grad():
            log_probs_list = []
            for i in range(num_agents):
                obs_i = obs_batch[:, i, :]
                act_i = action_batch[:, i]
                lp, _ = self.agents[i].policy_net.evaluate_actions(obs_i, act_i) # log prob, entropy
                # _, lp = self.agents[i].policy_net.evaluate_actions(obs_i, act_i) -> 이전에 잘못 사용하던 것
                log_probs_list.append(lp)
            policy_log_probs = torch.stack(log_probs_list, dim=1).view(-1) # each agent's log prob
            policy_probs_super = torch.exp(policy_log_probs) 
            
        obs_super = obs_batch.view(-1, self.obs_dim)

        # === 2. Mini-batch Training Loop ===
        total_samples = A_super.size(0) 
        indices = torch.arange(total_samples, device=device)

        avg_outcome_loss = 0.0
        avg_action_loss = 0.0
        agent_residual_errors = {i: [] for i in range(num_agents)}
        num_updates = 0
        chunk_size = self.causal_mini_batch_size 

        for i in range(0, total_samples, chunk_size):
            idx = indices[i:i + chunk_size]
            current_batch_size = idx.size(0)
            
            a_mb = A_super[idx]
            w_mb = W_super[idx]
            y_mb = Y_super[idx]
            z_mb = Z_super[idx]
            pi_mb = policy_probs_super[idx]
            o_mb = obs_super[idx]
            batch_agent_ids = idx % num_agents

            # --- Update Outcome Bridge h_φ ---
            r_hat_mb = self.outcome_bridge_net(a_mb, w_mb)
            residuals_mb = y_mb - r_hat_mb
            
            if self.causal_lambda_adaptive_per_agent:
                for j, agent_id in enumerate(batch_agent_ids):
                    residual_error = torch.norm(residuals_mb[j], p=2).item()
                    agent_residual_errors[agent_id.item()].append(residual_error)
            
            action_embeds_mb = self.action_embedding_for_kernel(a_mb)
            kernel_feat_z = torch.cat([z_mb, action_embeds_mb], dim=1)
            K_z = self.critic_kernel_z(kernel_feat_z)

            reg_K_z = self.tikhonov_gamma * torch.eye(current_batch_size, device=device) + self.stabilizer_lambda * K_z
            solved_residuals = torch.linalg.solve(reg_K_z, residuals_mb)
            
            # Compute per-sample loss
            per_sample_loss_h = (residuals_mb * (K_z @ solved_residuals)).sum(dim=1) # [B]
            
            loss_h = per_sample_loss_h.mean()
            
            # --- Update Action Bridge q_θ ---
            # [DISABLED] Action Bridge Network unused
            # q_val_mb = self.action_bridge_net(z_mb, a_mb)
            # phi_q_mb = pi_mb * q_val_mb.squeeze()
            # phi_q_col = phi_q_mb.unsqueeze(1)

            # kernel_feat_w = torch.cat([w_mb, action_embeds_mb], dim=1)
            # K_w1 = self.critic_kernel_w(kernel_feat_w)
            
            # K_w2_sum_mb = self._compute_kw2_sum_minibatch(w_mb, kernel_feat_w, o_mb, batch_agent_ids, mc_samples_k_w2)
            
            # reg_K_w1 = self.tikhonov_gamma * torch.eye(current_batch_size, device=device) + self.stabilizer_lambda * K_w1
            
            # solved_phi_q = torch.linalg.solve(reg_K_w1, phi_q_col)
            # quadratic_term = phi_q_col.T @ K_w1 @ solved_phi_q
            
            # solved_K_w2_sum = torch.linalg.solve(reg_K_w1, K_w2_sum_mb)
            # linear_term = phi_q_col.T @ solved_K_w2_sum
            
            # Action Bridge Loss is scalar by definition in current RKHS formulation
            # But we can try to balance if we had per-sample components.
            # Here it's a global objective (trace), so balancing is implicit in K.
            # loss_q = (quadratic_term - 2.0 * linear_term).squeeze() / current_batch_size
            loss_q = torch.tensor(0.0, device=device)

            total_loss = loss_h + loss_q
            
            self.outcome_bridge_optimizer.zero_grad()
            self.action_bridge_optimizer.zero_grad()
            self.kernel_action_embed_optimizer.zero_grad()
            total_loss.backward()
            self.outcome_bridge_optimizer.step()
            # self.action_bridge_optimizer.step()
            self.kernel_action_embed_optimizer.step()

            avg_outcome_loss += loss_h.item()
            avg_action_loss += loss_q.item()
            num_updates += 1

        return_dict = {
            'outcome_bridge_loss': avg_outcome_loss / num_updates,
            'action_bridge_loss': avg_action_loss / num_updates,
        }
        # Adaptive lambda update: 
        # calculate the error between the predicted outcome and the actual outcome, 
        # and update the lambda accordingly.
        if self.causal_lambda_adaptive_per_agent:
            for agent_id in range(num_agents):
                if len(agent_residual_errors[agent_id]) > 0:
                    avg_error = np.mean(agent_residual_errors[agent_id])
                    
                    if self.adaptive_error_ema_per_agent[agent_id] is None:
                        self.adaptive_error_ema_per_agent[agent_id] = avg_error
                    else:
                        self.adaptive_error_ema_per_agent[agent_id] = (
                            self.causal_lambda_adaptive_alpha * self.adaptive_error_ema_per_agent[agent_id] +
                            (1 - self.causal_lambda_adaptive_alpha) * avg_error
                        )
                    
                    error_ratio = self.adaptive_error_ema_per_agent[agent_id] / (
                        self.causal_lambda_adaptive_error_threshold + 1e-8
                    )
                    
                    lambda_factor = 1.0 / (1.0 + error_ratio)
                    new_lambda = self.causal_lambda * lambda_factor
                    self.adaptive_lambda_per_agent[agent_id] = np.clip(
                        new_lambda,
                        self.causal_lambda_adaptive_min,
                        self.causal_lambda_adaptive_max
                    )
                    
                    return_dict[f'adaptive_lambda_agent_{agent_id}'] = self.adaptive_lambda_per_agent[agent_id]
                    return_dict[f'adaptive_error_ema_agent_{agent_id}'] = self.adaptive_error_ema_per_agent[agent_id]
        
        return return_dict

    # for q-bridge function training
    def _compute_kw2_sum_minibatch(self, w_mb, kernel_feat_w, o_mb, batch_agent_ids, M):
        batch_size = w_mb.size(0)
        
        w_expanded = w_mb.repeat_interleave(M, dim=0)
        
        sampled_action_embeds = torch.zeros(
            batch_size * M, 
            self.action_embedding_for_kernel.embedding_dim, 
            device=self.device
        )
        
        with torch.no_grad():
            unique_agents = torch.unique(batch_agent_ids)
            for ag_id in unique_agents:
                mask = (batch_agent_ids == ag_id)
                if not mask.any(): continue
                
                obs_ag = o_mb[mask]
                obs_exp = obs_ag.repeat_interleave(M, dim=0)
                
                logits = self.agents[ag_id.item()].policy_net(obs_exp)
                dist = torch.distributions.Categorical(logits=logits)
                actions_sampled = dist.sample()
                embeds = self.action_embedding_for_kernel(actions_sampled)
                
                mask_expanded = mask.unsqueeze(1).expand(-1, M).reshape(-1)
                sampled_action_embeds[mask_expanded] = embeds

        features_w_sampled = torch.cat([w_expanded, sampled_action_embeds], dim=1)
        
        # kernel_feat_w is passed from caller!
        K_w2_raw = self.critic_kernel_w.forward_asymmetric(kernel_feat_w, features_w_sampled)
        
        K_w2_mean = K_w2_raw.view(batch_size, batch_size, M).mean(dim=2)
        K_w2_sum = K_w2_mean.sum(dim=1).unsqueeze(1)
        
        return K_w2_sum

    def _concatenate_observations(self, observations: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Helper function to concatenate observations for all agents.
        """
        sorted_obs = [observations[i] for i in range(self.num_agents) if i in observations]
        if not sorted_obs:
            device = next(self.agents[0].policy_net.parameters()).device
            return torch.zeros(self.num_agents * self.obs_dim, device=device)
        return torch.cat(sorted_obs, dim=-1)
    
    def reset_agents(self):
        pass
        
    def reset_agents_batch(self, n_envs: int):
        """Reset agents for multiple environments (batch mode)."""
        self.last_dones_batch = [{i: False for i in range(self.num_agents)} for _ in range(n_envs)]

    def get_actions_batch(
        self,
        observations_batch: List[Dict[int, torch.Tensor]],
        available_actions_batch: Optional[List[Dict[int, List[int]]]] = None
    ) -> List[Dict[int, Tuple[int, float]]]:
        """
        Get actions for a batch of environments (ParallelEnv execution).
        """
        n_envs = len(observations_batch)
        actions_batch = [{} for _ in range(n_envs)]
        
        has_avail = (available_actions_batch is not None)
        
        for agent_id in range(self.num_agents):
            agent_obs_list = []
            agent_avail_list = []
            
            for i in range(n_envs):
                obs = observations_batch[i][agent_id]
                agent_obs_list.append(obs)
                
                if has_avail and available_actions_batch[i] is not None and agent_id in available_actions_batch[i]:
                    agent_avail_list.append(available_actions_batch[i][agent_id])
                else:
                    agent_avail_list.append(None)
            
            agent_obs_tensor = torch.stack(agent_obs_list).to(self.device)
            
            agent_avail_tensor = None
            if any(x is not None for x in agent_avail_list):
                 # Convert list of lists to tensor? Or pad?
                 # Assuming fixed action dim, simply tensorizing works if all are present.
                 # If mixed None/List, we might have issues. SMAC usually provided consistently.
                 try:
                    agent_avail_tensor = torch.tensor(agent_avail_list, dtype=torch.float32, device=self.device)
                 except:
                    # Fallback if inconsistent
                    pass

            actions, log_probs = self.agents[agent_id].get_action_batch(agent_obs_tensor, available_actions_batch=agent_avail_tensor)
            
            for i in range(n_envs):
                act = actions[i].item()
                lp = log_probs[i].item()
                actions_batch[i][agent_id] = (act, lp)
        
        return actions_batch

    def store_transition_from_env(
        self,
        env_idx: int,
        prev_observations: Dict[int, torch.Tensor],
        prev_rewards: Dict[int, float],
        prev_actions: Dict[int, int],
        observations: Dict[int, torch.Tensor],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_observations: Dict[int, torch.Tensor],
        dones: Dict[int, bool],
        info: Dict = None,
        available_actions: Optional[Dict[int, List[int]]] = None
    ):
        """Store transition from a specific parallel environment."""
        if info is None: info = {}
        info['env_id'] = env_idx
        
        if hasattr(self, 'last_dones_batch'):
            for agent_id in range(self.num_agents):
                self.last_dones_batch[env_idx][agent_id] = dones[agent_id]
        
        self.buffer_manager.store_transition(
            prev_observations=prev_observations,
            prev_rewards=prev_rewards,
            prev_actions=prev_actions,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            hidden_states={i: None for i in range(self.num_agents)},
            info=info,
            available_actions=available_actions
        )
        
        if self.use_episode_buffer:
            if any(dones.values()):
                self.buffer_manager.finish_episode(env_id=env_idx)

    def finish_episode(self, env_id: int = 0):
        """Finish current episode if using EpisodeBuffer."""
        if self.use_episode_buffer:
            self.buffer_manager.finish_episode(env_id=env_id)

    def get_actions(self, observations: Dict[int, torch.Tensor], available_actions: Optional[Dict[int, List[int]]] = None) -> Dict[int, Tuple[int, float]]:
        actions = {}
        for agent_id, obs in observations.items():
            # Prepare available actions tensor
            avail_actions_tensor = None
            if available_actions is not None and agent_id in available_actions:
                avail_actions_tensor = torch.tensor(available_actions[agent_id], dtype=torch.float32, device=self.device)

            action, log_prob = self.agents[agent_id].get_action(obs, available_actions=avail_actions_tensor)
            actions[agent_id] = (action, log_prob)
        return actions
    
    def store_transition(
        self,
        prev_observations: Dict[int, torch.Tensor],
        prev_rewards: Dict[int, float],
        prev_actions: Dict[int, int],
        observations: Dict[int, torch.Tensor],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_observations: Dict[int, torch.Tensor],
        dones: Dict[int, bool],
        info: Dict = None,
        available_actions: Optional[Dict[int, List[int]]] = None # [ADDED]
    ):
        self.buffer_manager.store_transition(
            prev_observations=prev_observations,
            prev_rewards=prev_rewards,
            prev_actions=prev_actions,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            masks={i: 1.0 for i in range(self.num_agents)},
            hidden_states={i: None for i in range(self.num_agents)},
            info=info,
            available_actions=available_actions # [ADDED]
        )
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float,
        dones: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """
        Helper function to compute Generalized Advantage Estimation.
        """
        advantages = []
        returns = []
        gae = 0
        next_value = next_value
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_val = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_val * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        return advantages, returns
    
    def _classify_delta_scenario(self, transition: Dict, agent_id: int) -> str:
        """
        Helper function to classify delta scenario.
        """
        info = transition.get('info', {})
        stag_hunts = info.get('stag_hunts', {})
        
        other_agents_stag_reward = False
        other_agents_forage_reward = False
        
        for other_agent_id in range(self.num_agents):
            if other_agent_id != agent_id:
                if transition['bold_r_t'][other_agent_id] == 5.0:
                    other_agents_stag_reward = True
                elif transition['bold_r_t'][other_agent_id] == 3.0:
                    other_agents_forage_reward = True
        
        if other_agents_stag_reward and self.env_required_agents is not None:
            for stag_id, participants in stag_hunts.items():
                required = self.env_required_agents.get(stag_id, [])
                if agent_id in required:
                    if agent_id in participants:
                        return 'stag_reward'
        
        if other_agents_forage_reward:
            return 'forage_reward'
        
        return 'no_reward'

    def compute_causal_rewards(
        self,
        w_t_batch: torch.Tensor,
        z_t_batch: torch.Tensor,
        full_obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        rewards_batch: torch.Tensor,
        baseline_type: str = 'weighted',
        available_actions_batch: Optional[Dict[int, torch.Tensor]] = None # [ADDED]
    ) -> torch.Tensor:
        """
        Computes the pairwise Causal Effect Matrix C.
        C[b, i, j] = Effect of Agent i on Agent j in batch b.
        
        Returns:
            C_matrix: [B, N, N] tensor
        """
        if self.get_current_causal_lambda() == 0.0:
            return torch.zeros(actions_batch.size(0), self.num_agents, self.num_agents, device=self.device)

        batch_size = actions_batch.size(0)
        num_agents = self.num_agents
        action_dim = self.action_dim
        
        # Prepare Output Matrix
        C_matrix = torch.zeros(batch_size, num_agents, num_agents, device=self.device)
        
        infer_chunk_size = 1024 
        
        with torch.no_grad():
            for i in range(0, batch_size, infer_chunk_size):
                end = min(i + infer_chunk_size, batch_size)
                
                act_b = actions_batch[i:end]
                w_b = w_t_batch[i:end]
                z_b = z_t_batch[i:end]
                r_b = rewards_batch[i:end]
                obs_b = full_obs_batch[i:end]
                curr_bs = act_b.size(0)
                
                # --- Vectorized Logic ---
                actions_flat = act_b.view(-1)
                w_t_flat = w_b.view(-1, w_b.size(-1))
                z_t_repeated = z_b.unsqueeze(1).repeat(1, num_agents, 1).view(-1, z_b.size(-1))

                r_pred_flat = self.outcome_bridge_net(actions_flat, w_t_flat)
                # weight_flat = self.action_bridge_net(z_t_repeated, actions_flat)

                mask_indices = torch.tensor([
                    [j for j in range(num_agents) if j != k] 
                    for k in range(num_agents)
                ], device=self.device)
                
                others_rewards = r_b.unsqueeze(1).expand(-1, num_agents, -1)
                others_rewards = torch.gather(others_rewards, 2, mask_indices.unsqueeze(0).expand(curr_bs, -1, -1))
                others_rewards_flat = others_rewards.view(-1, num_agents - 1)
                
                # error_flat = others_rewards_flat - r_pred_flat
                # DR_correction_flat = weight_flat * error_flat
                actions_range = torch.arange(action_dim, device=self.device).repeat(curr_bs * num_agents)
                w_t_expanded = w_t_flat.repeat_interleave(action_dim, dim=0)
                
                reg_estimates_cf = self.outcome_bridge_net(actions_range, w_t_expanded)
                reg_estimates_cf = reg_estimates_cf.view(curr_bs * num_agents, action_dim, num_agents - 1)
                
                # r_hat_all = reg_estimates_cf + DR_correction_flat.unsqueeze(1)
                r_hat_all = reg_estimates_cf # no q
                
                probs_list = []
                for k in range(num_agents):
                    avail_actions_b = None
                    if available_actions_batch is not None and k in available_actions_batch:
                        avail_actions_b = available_actions_batch[k][i:end]
                    
                    logits = self.agents[k].policy_net(obs_b[:, k, :], available_actions=avail_actions_b)
                    probs = torch.softmax(logits, dim=-1)
                    probs_list.append(probs)
                action_probs_flat = torch.stack(probs_list, dim=1).view(-1, action_dim)

                #-----Original baseline calculation-----#
                # factual_indices = actions_flat
                # r_hat_factual = torch.gather(r_hat_all, 1, factual_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, num_agents-1)).squeeze(1)
                # mask = torch.ones(curr_bs * num_agents, action_dim, dtype=torch.bool, device=self.device)
                # mask.scatter_(1, factual_indices.unsqueeze(1), False)
                
                # if baseline_type == 'weighted':
                #     probs_cf = action_probs_flat.clone()
                #     probs_cf.scatter_(1, factual_indices.unsqueeze(1), 0.0)
                #     sum_probs = probs_cf.sum(dim=1, keepdim=True)
                #     baseline = (probs_cf.unsqueeze(2) * r_hat_all).sum(dim=1) / (sum_probs + 1e-8)
                # elif baseline_type == 'average':
                #     r_hat_masked = r_hat_all.clone()
                #     mask_expanded = mask.unsqueeze(-1).expand_as(r_hat_all)
                #     r_hat_masked[~mask_expanded] = 0.0
                #     count = mask.sum(dim=1, keepdim=True).float()
                #     baseline = r_hat_masked.sum(dim=1) / (count + 1e-8)
                # else:
                #     raise ValueError(f"Invalid baseline type: {baseline_type}")
                # using causal domain information: prediction with factual action
                # raw_effects = r_hat_factual - baseline
                ########################################

                # new baseline calculation: contain factual action
                if baseline_type == 'weighted':
                    # 모든 action의 확률을 사용 (factual action 포함)
                    sum_probs = action_probs_flat.sum(dim=1, keepdim=True)
                    baseline = (action_probs_flat.unsqueeze(2) * r_hat_all).sum(dim=1) / sum_probs
                elif baseline_type == 'average':
                    # 모든 action을 포함하여 평균 계산 (factual action 포함)
                    baseline = r_hat_all.mean(dim=1)
                else:
                    raise ValueError(f"Invalid baseline type: {baseline_type}")

                # 1. using real rewards
                raw_effects = others_rewards_flat - baseline
                # 2. using outcome bridge network
                # raw_effects = r_hat_factual - baseline
                
                # Reshape to [B, N, N-1]
                effects_reshaped = raw_effects.view(curr_bs, num_agents, num_agents - 1)
                
                # Scatter values into N x N matrix (Diagonal is 0)
                # target_indices: [B, N, N-1] -> expand for batch
                batch_target_indices = mask_indices.unsqueeze(0).expand(curr_bs, -1, -1)
                
                C_chunk = torch.zeros(curr_bs, num_agents, num_agents, device=self.device)
                C_chunk.scatter_(2, batch_target_indices, effects_reshaped)
                
                C_matrix[i:end] = C_chunk
            
        return C_matrix

    def update_agents(self, batch_data: List[Dict]) -> Dict[str, float]:
        agent_losses = defaultdict(float)

        full_obs_batch = torch.stack([torch.stack(list(t['bold_o_t'].values())) for t in batch_data]).to(self.device)
        B, N, D = full_obs_batch.shape
        
        actions_batch = torch.tensor([[t['bold_a_t'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.long, device=self.device)
        rewards_batch = torch.tensor([[t['bold_r_t'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.float32, device=self.device)
        
        # joint_o_prev = torch.stack([self._concatenate_observations(t['bold_o_t_minus_1']) for t in batch_data]).to(self.device)
        # joint_a_prev = torch.tensor([[t['bold_a_t_minus_1'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.long, device=self.device)
        # joint_a_prev_onehot = F.one_hot(joint_a_prev, num_classes=self.action_dim).view(B, -1).float()
        # joint_r_prev = torch.tensor([[t['bold_r_t_minus_1'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.float32, device=self.device)
        
        z_t_batch = torch.stack([self._concatenate_observations(t['bold_o_t_minus_1']) for t in batch_data]).to(self.device) # Default Z
        # z_t_batch = torch.cat([joint_o_prev, joint_r_prev], dim=1)
        # z_t_batch = torch.cat([joint_o_prev, joint_a_prev_onehot], dim=1)
        # z_t_batch = torch.tensor([[t['bold_r_t_minus_1'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.float32, device=self.device) # [SWITCH] prev rewards
        # z_t_batch = torch.tensor([[t['bold_r_t_minus_1'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.float32, device=self.device) # [SWITCH] prev rewards
        dones_batch = torch.tensor([[t['dones'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.float32, device=self.device)

        # Extract available actions
        available_actions_batch = {}
        if 'available_actions' in batch_data[0] and batch_data[0]['available_actions'] is not None:
            for agent_id in range(self.num_agents):
                if agent_id in batch_data[0]['available_actions']:
                     avail_list = [t['available_actions'][agent_id] for t in batch_data]
                     available_actions_batch[agent_id] = torch.tensor(avail_list, dtype=torch.float32, device=self.device)

        # Extract pre-computed advantages and returns
        agent_returns = {}
        agent_advantages = {}
        
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue
            
            returns_list = [t['returns'][agent_id] for t in batch_data]
            advantages_list = [t['advantages'][agent_id] for t in batch_data]
            
            agent_returns[agent_id] = torch.tensor(returns_list, dtype=torch.float32, device=self.device)
            agent_advantages[agent_id] = torch.tensor(advantages_list, dtype=torch.float32, device=self.device)

        # Log delta (causal intrinsic rewards) by scenario
        for i in range(self.num_agents):
            for t, transition in enumerate(batch_data):
                if 'delta' in transition and i in transition['delta']:
                    scenario = self._classify_delta_scenario(transition, i)
                    agent_losses[f'delta_agent_{i}_{scenario}_sum'] += transition['delta'][i]
                    agent_losses[f'delta_agent_{i}_{scenario}_count'] += 1

        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue
            
            agent_obs = full_obs_batch[:, agent_id, :]
            my_actions = actions_batch[:, agent_id]
            adv = (agent_advantages[agent_id] - agent_advantages[agent_id].mean()) / (agent_advantages[agent_id].std() + 1e-8)
            
            with torch.no_grad():
                avail_actions_all = None
                if agent_id in available_actions_batch:
                    avail_actions_all = available_actions_batch[agent_id]

                logits = self.agents[agent_id].policy_net(agent_obs, available_actions=avail_actions_all)
                old_log_probs = F.log_softmax(logits, dim=-1).gather(1, my_actions.unsqueeze(1)).squeeze(1)

            for _ in range(self.ppo_epochs):
                new_log_probs, entropy = self.agents[agent_id].policy_net.evaluate_actions(agent_obs, my_actions, available_actions=avail_actions_all)
                ratio = torch.exp(new_log_probs - old_log_probs.detach())
                
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.entropy_coef * entropy_loss
                
                self.agents[agent_id].optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[agent_id].policy_net.parameters(), self.max_grad_norm)
                self.agents[agent_id].optimizer.step()
                
                agent_losses[f'agent_{agent_id}_policy_loss'] += policy_loss.item()
                agent_losses[f'agent_{agent_id}_entropy_loss'] += entropy_loss.item()

        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue
            
            for _ in range(self.ppo_epochs):
                agent_obs = full_obs_batch[:, agent_id, :]
                new_vals = self.agents[agent_id].value_net(agent_obs).squeeze()
                val_loss = F.mse_loss(new_vals, agent_returns[agent_id])
                
                self.agents[agent_id].value_optimizer.zero_grad()
                (self.value_coef * val_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.agents[agent_id].value_net.parameters(), self.max_grad_norm)
                self.agents[agent_id].value_optimizer.step()
                agent_losses[f'agent_{agent_id}_value_loss'] += val_loss.item()

        # Average losses for logging
        for k in list(agent_losses.keys()):
            if 'value_loss' in k:
                # Value loss is already averaged per agent per epoch, but we want total average
                # The original code had a specific way to average value_loss, let's adapt the new logic
                # If the new logic is to average all 'loss' keys by ppo_epochs, then value_loss should also be averaged.
                # The original code was:
                # total_value_loss = sum([agent_losses.get(f'agent_{i}_value_loss', 0.0) for i in range(self.num_agents)])
                # agent_losses['value_loss'] = total_value_loss / (self.num_agents * self.ppo_epochs)
                # Let's assume the new instruction implies averaging each agent's value_loss by ppo_epochs,
                # and then the outer loop will average across batches.
                if agent_losses[k] != 0:
                    agent_losses[k] /= self.ppo_epochs
            elif 'loss' in k:
                # Assuming other losses are summed, we average them
                # But careful: some might be count-based logging
                if agent_losses[k] != 0:
                     agent_losses[k] /= self.ppo_epochs
            elif 'delta' not in k: # Keep delta counts/sums as is for now, they are handled differently
                del agent_losses[k] # Remove other non-loss, non-delta keys
        
        return dict(agent_losses)

    def update(self, **kwargs):
        """
        Main update function that orchestrates the training process for one iteration.
        """
        from collections import defaultdict
        
        # Get trajectories based on buffer type
        if self.use_episode_buffer:
            if not self.buffer_manager.has_completed_episodes():
                return {}
            episode_trajectories = self.buffer_manager.get_completed_episodes()
            # episode_trajectories is already grouped by episodes
            env_trajectories = {i: traj for i, traj in enumerate(episode_trajectories)}
        else:
            if self.buffer_manager.is_main_buffer_empty():
                return {}
            transitions = self.buffer_manager.get_main_buffer_transitions()
            
            # Group by env_id for TransitionBuffer
            env_trajectories = defaultdict(list)
            for t in transitions:
                env_id = t['info'].get('env_id', 0)
                env_trajectories[env_id].append(t)
        
        # [FIX] Compute GAE on full trajectories before batching
        try:
            
            # 2. Compute causal rewards for each trajectory
            for env_id, traj in env_trajectories.items():
                if len(traj) == 0 or self.get_current_causal_lambda() == 0.0:
                    continue
                
                # Prepare data for causal reward computation
                B = len(traj)
                N = self.num_agents
                D = self.obs_dim
                
                full_obs_batch = torch.stack([torch.stack(list(t['bold_o_t'].values())) for t in traj]).to(self.device)
                actions_batch = torch.tensor([[t['bold_a_t'][i] for i in range(N)] for t in traj], dtype=torch.long, device=self.device)
                rewards_batch = torch.tensor([[t['bold_r_t'][i] for i in range(N)] for t in traj], dtype=torch.float32, device=self.device)
                dones_batch = torch.tensor([[t['dones'][i] for i in range(N)] for t in traj], dtype=torch.float32, device=self.device)
                
                # Prepare z_t (proxy variable)
                # z_t_batch = torch.stack([self._concatenate_observations(t['bold_o_t_minus_1']) for t in traj]).to(self.device)
                z_t_batch = torch.stack([t['bold_r_t_minus_1'] for t in traj]).to(self.device) # prev rewards
                # Prepare w_t (other agents' observations)
                expanded_obs = full_obs_batch.unsqueeze(1).expand(-1, N, -1, -1)
                mask = ~torch.eye(N, dtype=torch.bool, device=self.device).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, D)
                w_t_all = expanded_obs[mask].view(B, N, (N-1)*D)
                
                # Extract available actions
                available_actions_batch = {}
                if 'available_actions' in traj[0] and traj[0]['available_actions'] is not None:
                    for agent_id in range(N):
                        if agent_id in traj[0]['available_actions']:
                            avail_list = [t['available_actions'][agent_id] for t in traj]
                            available_actions_batch[agent_id] = torch.tensor(avail_list, dtype=torch.float32, device=self.device)
                
                # Compute causal rewards
                with torch.no_grad():
                    C_matrix = self.compute_causal_rewards(
                        w_t_all, z_t_batch, full_obs_batch, actions_batch, rewards_batch,
                        baseline_type='weighted',
                        available_actions_batch=available_actions_batch
                    )
                    delta = C_matrix.sum(dim=2)
                    delta[torch.abs(delta) < self.clip_delta] = 0.0
                    
                    # Apply lambda
                    if self.causal_lambda_adaptive_per_agent:
                        causal_lambdas = torch.tensor([
                            self.get_current_causal_lambda(agent_id=i) for i in range(N)
                        ], device=self.device).unsqueeze(0)
                        shaped_rewards_all = rewards_batch + causal_lambdas * delta
                    else:
                        shaped_rewards_all = rewards_batch + self.get_current_causal_lambda() * delta
                    
                    # Store shaped rewards and delta in transitions
                    for t_idx, t in enumerate(traj):
                        t['shaped_rewards'] = {agent_id: shaped_rewards_all[t_idx, agent_id].item() 
                                              for agent_id in range(N)}
                        # Store delta for logging
                        if 'delta' not in t:
                            t['delta'] = {}
                        for agent_id in range(N):
                            t['delta'][agent_id] = delta[t_idx, agent_id].item()

            
            # 3. Compute GAE for each trajectory using shaped rewards
            for env_id, traj in env_trajectories.items():
                if len(traj) == 0:
                    continue
                
                # For each agent, compute GAE
                for agent_id in range(self.num_agents):
                    if agent_id not in traj[0]['bold_o_t']:
                        continue
                    
                    # Extract agent data
                    agent_obs_seq = torch.stack([t['bold_o_t'][agent_id] for t in traj]).to(self.device)
                    
                    # Use shaped rewards if available, otherwise use original rewards
                    if 'shaped_rewards' in traj[0]:
                        agent_rewards_seq = torch.tensor([t['shaped_rewards'][agent_id] for t in traj],
                                                        dtype=torch.float32, device=self.device)
                    else:
                        agent_rewards_seq = torch.tensor([t['bold_r_t'][agent_id] for t in traj],
                                                        dtype=torch.float32, device=self.device)
                    
                    agent_dones_seq = torch.tensor([t['dones'][agent_id] for t in traj], dtype=torch.float32, device=self.device)
                    
                    # Compute values
                    with torch.no_grad():
                        agent_values = self.agents[agent_id].value_net(agent_obs_seq).squeeze()
                        last_next_obs = traj[-1]['next_observations'][agent_id]
                        is_terminal = agent_dones_seq[-1].item()
                        next_val = 0.0 if is_terminal else self.agents[agent_id].value_net(last_next_obs.unsqueeze(0)).item()
                    
                    # Compute GAE
                    advantages, returns = self.compute_gae(
                        agent_rewards_seq.tolist(),
                        agent_values.tolist(),
                        next_val,
                        agent_dones_seq.tolist()
                    )
                    
                    # Store in transitions
                    for t_idx, t in enumerate(traj):
                        if 'advantages' not in t:
                            t['advantages'] = {}
                            t['returns'] = {}
                        t['advantages'][agent_id] = advantages[t_idx]
                        t['returns'][agent_id] = returns[t_idx]
            
            # 3. Create batches from sorted transitions
            sorted_transitions = []
            for env_id in sorted(env_trajectories.keys()):
                sorted_transitions.extend(env_trajectories[env_id])
                
            batch_size = max(1, len(sorted_transitions) // self.num_batches)
            batches = [sorted_transitions[i:i + batch_size] for i in range(0, len(sorted_transitions), batch_size)]
            
        except Exception as e:
            print(f"[ERROR PCI_MAPPO] Failed to restructure rollout data: {e}")
            import traceback
            traceback.print_exc()
            batch_size = max(1, len(transitions) // self.num_batches)
            batches = [transitions[i:i + batch_size] for i in range(0, len(transitions), batch_size)]
        
        total_losses = defaultdict(float)
        
        # Loop through each mini-batch to perform updates
        for batch_data in batches:
            if not batch_data: continue
            
            # Causal Update
            if self.get_current_causal_lambda() > 0.0:
                causal_losses = self.update_causal_networks(batch_data, mc_samples_k_w2=16)
            else:
                causal_losses = {'outcome_bridge_loss': 0.0, 'action_bridge_loss': 0.0}
            
            for loss_name, loss_value in causal_losses.items():
                total_losses[loss_name] += loss_value
                
            # Agent Update
            agent_and_value_losses = self.update_agents(batch_data)

            for loss_name, loss_value in agent_and_value_losses.items():
                total_losses[loss_name] += loss_value

        self.buffer_manager.clear_buffer()
        self.training_steps += 1
        
        # Average losses over the number of mini-batches for accurate logging
        num_valid_batches = len([b for b in batches if b])
        if num_valid_batches == 0: return {}

        final_losses = {name: val / num_valid_batches for name, val in total_losses.items()}
        final_losses['causal_lambda'] = self.get_current_causal_lambda()

        return final_losses

    def clear_buffer(self):
        self.buffer_manager.clear_buffer()
    
    def get_buffer_info(self) -> Dict[str, int]:
        return self.buffer_manager.get_buffer_info()