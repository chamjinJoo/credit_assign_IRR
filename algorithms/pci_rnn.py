import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
import random

from torch.distributions import Categorical
from algorithms.networks.rnn import RNNLayer, RNNPolicyNetwork, RNNValueNetwork
from algorithms.networks import PolicyNetwork, OutcomeBridgeNetwork, ActionBridgeNetwork, ValueNetwork
from algorithms.networks.mlp import MLPBase
from utils.buffer import TransitionBuffer


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




class PCIRNNAgent:
    def __init__(self, agent_id: int, obs_dim: int, action_dim: int, hidden_dim: int = 64, recurrent_N: int = 1, lr: float = 3e-4, lr_critic: float = 3e-4):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.recurrent_N = recurrent_N
        
        self.policy_net = RNNPolicyNetwork(obs_dim, action_dim, hidden_dim, recurrent_N)
        self.value_net = RNNValueNetwork(obs_dim, hidden_dim, recurrent_N)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)
        
        # Internal hidden state maintenance
        self.rnn_state_policy = None
        self.rnn_state_value = None
        self.reset_rnn_state()

    def reset_rnn_state(self):
        self.rnn_state_policy = torch.zeros(self.recurrent_N, self.hidden_dim)
        self.rnn_state_value = torch.zeros(self.recurrent_N, self.hidden_dim)

    def get_action(self, obs: torch.Tensor, masks: torch.Tensor = None) -> Tuple[int, float, Optional[torch.Tensor]]:
        with torch.no_grad():
            # forward needs (batch, dim), obs is (dim)
            obs_in = obs.unsqueeze(0) if obs.dim() == 1 else obs
            
            # Handle masks
            if masks is None:
                # Called from trainer (evaluation)
                batch_size = obs_in.size(0)
                masks_in = torch.ones(batch_size, 1, device=self.policy_net.output_layer.weight.device)
                return_rnn_state = False
            else:
                # Called from algorithm (collection)
                masks_in = masks.unsqueeze(0) if masks.dim() == 1 else masks
                return_rnn_state = True

            # rnn_state needs (layers, batch, dim)? No, RNNLayer expects (Batch, Layers, Dim)
            # self.rnn_state_policy is (layers, dim) -> unsqueeze(0) -> (1, layers, dim)
            curr_rnn_state = self.rnn_state_policy.unsqueeze(0)
            
            action, log_prob, next_rnn_state = self.policy_net.get_action_and_log_prob(
                obs_in, 
                curr_rnn_state, 
                masks_in
            )
            
            # Update internal state
            self.rnn_state_policy = next_rnn_state.squeeze(0)
            
        if return_rnn_state:
            return action.item(), log_prob.item(), curr_rnn_state.squeeze(0).clone() # Return PREVIOUS state used for this action
        else:
            return action.item(), log_prob.item()

    def get_value(self, obs: torch.Tensor, masks: torch.Tensor) -> Tuple[float, torch.Tensor]:
        with torch.no_grad():
            curr_rnn_state = self.rnn_state_value.unsqueeze(0)
            value, next_rnn_state = self.value_net(
                obs.unsqueeze(0),
                curr_rnn_state,
                masks.unsqueeze(0)
            )
            self.rnn_state_value = next_rnn_state.squeeze(0)
        return value.item(), curr_rnn_state.squeeze(0).clone()


class PCIRNNAlgorithm:
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        w_dim: int, 
        hidden_dim: int = 64,
        recurrent_N: int = 1, # [ADDED] RNN layers
        lr: float = 3e-4,
        lr_bridge: float = 3e-4,
        lr_critic: float = 3e-4,
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
        causal_mini_batch_size: int = 256,
        # Value network parameter sharing
        use_value_parameter_sharing: bool = True,
        rnn_chunk_length: int = 16, # Chunk length for RNN training
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.w_dim = w_dim
        self.hidden_dim = hidden_dim
        self.recurrent_N = recurrent_N
        self.rnn_chunk_length = rnn_chunk_length
        self.env_required_agents = env_required_agents
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.causal_mini_batch_size = causal_mini_batch_size 
        self.use_value_parameter_sharing = use_value_parameter_sharing
        
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
        
        self.buffer_manager = TransitionBuffer(
            max_buffer_size=max_buffer_size
        )
        
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = PCIRNNAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                recurrent_N=recurrent_N,
                lr=lr,
                lr_critic=lr_critic
            )
        
        # Value network parameter sharing 설정
        if self.use_value_parameter_sharing:
            # 하나의 shared value network 생성
            self.shared_value_net = RNNValueNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim, recurrent_N=recurrent_N)
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
        
        # Keep track of done status for masks
        self.last_dones = {i: False for i in range(num_agents)}
        
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
        dones_batch = torch.tensor([[t['dones'][i] for i in range(num_agents)] for t in batch_data], dtype=torch.float32, device=device)

        # Extract hidden states
        hidden_states_batch = {}
        if 'hidden_states' in batch_data[0] and batch_data[0]['hidden_states'] is not None:
             if any(batch_data[0]['hidden_states'].values()):
                for agent_id in range(self.num_agents):
                    if agent_id in batch_data[0]['hidden_states']:
                        p_states = torch.stack([t['hidden_states'][agent_id][0] for t in batch_data]).to(self.device) # (Batch, recurrent_N, hidden_dim)
                        # v_states = torch.stack([t['hidden_states'][agent_id][1] for t in batch_data]).to(self.device)
                        # hidden_states_batch[agent_id] = (p_states, v_states)
                        hidden_states_batch[agent_id] = p_states

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
                
                # Handling RNN states
                if i in hidden_states_batch:                    
                    p_state = hidden_states_batch[i]
                    # Construct masks
                    masks_i = torch.ones_like(dones_batch[:, i]).unsqueeze(1) # (Batch, 1)
                    lp, _ = self.agents[i].policy_net.evaluate_actions(obs_i, p_state, act_i, masks_i)
                else:
                    # Fallback to zeros if no state
                    bs = obs_i.size(0)
                    dummy_state = torch.zeros(bs, self.recurrent_N, self.hidden_dim, device=device)
                    dummy_mask = torch.ones(bs, 1, device=device)
                    lp, _ = self.agents[i].policy_net.evaluate_actions(obs_i, dummy_state, act_i, dummy_mask)

                log_probs_list.append(lp)
            
            policy_log_probs = torch.stack(log_probs_list, dim=1).view(-1)
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
        
        # We need hidden states here too, but they are not passed to this function in current signature.
        # Since Action Bridge is disabled and this function is likely unused, we provide a basic implementation 
        # that uses ZERO hidden states. 
        # TODO: If Action Bridge is enabled, this needs to receive hidden states corresponding to the batch.
        
        with torch.no_grad():
            unique_agents = torch.unique(batch_agent_ids)
            for ag_id in unique_agents:
                mask = (batch_agent_ids == ag_id)
                if not mask.any(): continue
                
                obs_ag = o_mb[mask] # (B_subset, D)
                curr_subset_size = obs_ag.size(0)
                
                obs_exp = obs_ag.repeat_interleave(M, dim=0) # (B_subset * M, D)
                
                # Zero state for sampling
                dummy_state = torch.zeros(curr_subset_size * M, self.recurrent_N, self.hidden_dim, device=self.device)
                dummy_mask = torch.ones(curr_subset_size * M, 1, device=self.device)
                
                # RNN Policy Forward for sampling
                logits, _ = self.agents[ag_id.item()].policy_net(obs_exp, dummy_state, dummy_mask)
                
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
        """Reset agent hidden states."""
        for agent in self.agents.values():
            agent.reset_rnn_state()
        self.last_dones = {i: False for i in range(self.num_agents)}
    
    def get_actions(self, observations: Dict[int, torch.Tensor]) -> Dict[int, Tuple[int, float]]:
        actions = {}
        self.temp_hidden_states = {} # Store current states for store_transition
        self.temp_masks = {} # Store masks used
        for agent_id, obs in observations.items():
            # Create mask: 1.0 if not done, 0.0 if done.
            # Using self.last_dones which is updated in store_transition
            mask = 1.0 if not self.last_dones.get(agent_id, False) else 0.0
            mask_tensor = torch.tensor([mask], dtype=torch.float32, device=self.device)
            
            # Policy
            action, log_prob, p_state = self.agents[agent_id].get_action(obs, mask_tensor)
            
            # Value (Update state)
            _, v_state = self.agents[agent_id].get_value(obs, mask_tensor)
            
            actions[agent_id] = (action, log_prob)
            self.temp_hidden_states[agent_id] = (p_state, v_state)
            self.temp_masks[agent_id] = mask
            
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
        info: Dict = None
    ):
        # Update last_dones for next step's mask
        self.last_dones = dones.copy()
        
        self.buffer_manager.store_transition(
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
            info=info
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
        hidden_states_batch: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        dones_batch: Optional[torch.Tensor] = None,
        baseline_type: str = 'weighted',
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
                # z_b = z_t_batch[i:end] # Unused now
                r_b = rewards_batch[i:end]
                obs_b = full_obs_batch[i:end]
                curr_bs = act_b.size(0)
                
                # --- Vectorized Logic ---
                actions_flat = act_b.view(-1)
                w_t_flat = w_b.view(-1, w_b.size(-1))
                
                # r_pred_flat = self.outcome_bridge_net(actions_flat, w_t_flat) 
                
                mask_indices = torch.tensor([
                    [j for j in range(num_agents) if j != k] 
                    for k in range(num_agents)
                ], device=self.device)
                
                others_rewards = r_b.unsqueeze(1).expand(-1, num_agents, -1)
                others_rewards = torch.gather(others_rewards, 2, mask_indices.unsqueeze(0).expand(curr_bs, -1, -1))
                others_rewards_flat = others_rewards.view(-1, num_agents - 1)
                
                actions_range = torch.arange(action_dim, device=self.device).repeat(curr_bs * num_agents)
                w_t_expanded = w_t_flat.repeat_interleave(action_dim, dim=0)
                
                reg_estimates_cf = self.outcome_bridge_net(actions_range, w_t_expanded)
                reg_estimates_cf = reg_estimates_cf.view(curr_bs * num_agents, action_dim, num_agents - 1)
                
                r_hat_all = reg_estimates_cf # no q
                
                # [최적화] 모든 agent의 hidden states와 masks를 미리 준비
                probs_list = []
                if hidden_states_batch is not None and dones_batch is not None:
                    # 모든 agent의 hidden states와 masks를 미리 준비하여 불필요한 연산 최소화
                    precomputed_states = {}
                    precomputed_masks = {}
                    for k in range(num_agents):
                        p_state_b = hidden_states_batch[k][0][i:end]  # (curr_bs, recurrent_N, hidden_dim)
                        # 이미 device에 있을 가능성이 높지만, 한 번만 확인
                        if p_state_b.device != self.device:
                            p_state_b = p_state_b.to(self.device)
                        precomputed_states[k] = p_state_b
                        
                        dones_b = dones_batch[i:end, k]
                        masks_b = (1.0 - dones_b).unsqueeze(1)  # (curr_bs, 1)
                        precomputed_masks[k] = masks_b
                
                for k in range(num_agents):
                    if hidden_states_batch is not None and dones_batch is not None:
                        # [최적화] 미리 준비된 states와 masks 사용
                        p_state_b = precomputed_states[k]
                        masks_b = precomputed_masks[k]
                        logits, _ = self.agents[k].policy_net(obs_b[:, k, :], p_state_b, masks_b)
                    else:
                        # Fallback (zeros) if not provided - e.g. for non-recurrent compatibility or testing
                        bs = obs_b.size(0)
                        # RNNLayer expects (Batch, recurrent_N, hidden_dim) for batch processing
                        dummy_state = torch.zeros(bs, self.recurrent_N, self.hidden_dim, device=self.device)
                        dummy_mask = torch.ones(bs, 1, device=self.device)
                        logits, _ = self.agents[k].policy_net(obs_b[:, k, :], dummy_state, dummy_mask)
                        
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

                # using factual rewards instead of predicted rewards
                r_factual = others_rewards_flat
                
                # using real rewards
                raw_effects = r_factual - baseline
                
                # using causal domain information: prediction with factual action
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

    def _chunk_sequence(self, tensor: torch.Tensor, chunk_len: int) -> torch.Tensor:
        """
        Reshape (T, ...) tensor into (T//chunk_len, chunk_len, ...).
        If T is not divisible by chunk_len, drop the remainder.
        Returns: (chunk_len, num_chunks, ...) - Ready for RNN (Seq, Batch, Dim)
        """
        T = tensor.size(0)
        num_chunks = T // chunk_len
        if num_chunks == 0:
            return tensor.unsqueeze(1) # (T, 1, ...)
        
        # Truncate
        tensor = tensor[:num_chunks * chunk_len] # (num_chunks * chunk_len, ...)
        
        # Reshape to (num_chunks, chunk_len, ...)
        shape = (num_chunks, chunk_len) + tensor.shape[1:]
        tensor = tensor.view(shape)
        
        # Permute to (chunk_len, num_chunks, ...)
        # Current: (Batch, Seq, ...) -> (Seq, Batch, ...)
        # indices: 0->1, 1->0, rest same
        permute_dims = (1, 0) + tuple(range(2, tensor.dim()))
        return tensor.permute(permute_dims).contiguous()

    def update_agents(self, batch_data: List[Dict]) -> Dict[str, float]:
        agent_losses = defaultdict(float)

        full_obs_batch = torch.stack([torch.stack(list(t['bold_o_t'].values())) for t in batch_data]).to(self.device)
        T, N, D = full_obs_batch.shape
        
        actions_batch = torch.tensor([[t['bold_a_t'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.long, device=self.device)
        rewards_batch = torch.tensor([[t['bold_r_t'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.float32, device=self.device)
        dones_batch = torch.tensor([[t['dones'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.float32, device=self.device)
        
        z_t_batch = torch.stack([self._concatenate_observations(t['bold_o_t_minus_1']) for t in batch_data]).to(self.device) 
        
        expanded_obs = full_obs_batch.unsqueeze(1).expand(-1, N, -1, -1)
        mask = ~torch.eye(N, dtype=torch.bool, device=self.device).unsqueeze(0).unsqueeze(-1).expand(T, -1, -1, D)
        w_t_all = expanded_obs[mask].view(T, N, (N-1)*D)

        # Extract hidden states for causal reward computation
        hidden_states_batch = {}
        # Pre-process hidden states from list of dicts to dict of stacked tensors
        # Each t['hidden_states'][agent_id] is (p_state, v_state)
        # p_state is (recurrent_N, hidden_dim)
        if 'hidden_states' in batch_data[0] and batch_data[0]['hidden_states'] is not None:
             # Check if any agent has hidden states
             if any(batch_data[0]['hidden_states'].values()):
                for agent_id in range(self.num_agents):
                    if agent_id in batch_data[0]['hidden_states']:
                        p_states = torch.stack([t['hidden_states'][agent_id][0] for t in batch_data]).to(self.device) # (Batch, recurrent_N, hidden_dim)
                        v_states = torch.stack([t['hidden_states'][agent_id][1] for t in batch_data]).to(self.device)
                        hidden_states_batch[agent_id] = (p_states, v_states)

        with torch.no_grad():
            C_matrix = self.compute_causal_rewards(
                w_t_all, z_t_batch, full_obs_batch, actions_batch, rewards_batch,
                hidden_states_batch=hidden_states_batch,
                dones_batch=dones_batch,
                baseline_type='weighted'
            )
            delta = C_matrix.sum(dim=2) # sum merging

            clip_delta = 0.2
            delta[torch.abs(delta) < clip_delta] = 0.0

            # Apply Adaptive Lambda per agent
            if self.causal_lambda_adaptive_per_agent:
                causal_lambdas = torch.tensor([
                    self.get_current_causal_lambda(agent_id=i) for i in range(self.num_agents)
                ], device=self.device).unsqueeze(0)
                shaped_rewards_all = rewards_batch + causal_lambdas * delta
            else:
                shaped_rewards_all = rewards_batch + self.get_current_causal_lambda() * delta
            
            # Logging
            for i in range(N):
                for t, transition in enumerate(batch_data):
                    scenario = self._classify_delta_scenario(transition, i)
                    agent_losses[f'delta_agent_{i}_{scenario}_sum'] += delta[t, i].item()
                    agent_losses[f'delta_agent_{i}_{scenario}_count'] += 1

        agent_returns = {}
        agent_advantages = {}
        
        # Prepare inputs for RNNs
        # We process the whole chunk as one sequence (Batch=1)
        # T: sequence length
        
        # Initial hidden states for the chunk
        # batch_data[0]['hidden_states'] is {agent_id: (p_state, v_state)}
        # init_hidden_states = batch_data[0]['hidden_states']
        if 'hidden_states' in batch_data[0] and batch_data[0]['hidden_states'] is not None:
             init_hidden_states = batch_data[0]['hidden_states']
        else:
             init_hidden_states = {i: (torch.zeros(self.recurrent_N, self.hidden_dim), torch.zeros(self.recurrent_N, self.hidden_dim)) for i in range(self.num_agents)}
             
        # Extract masks
        masks_batch = {}
        for agent_id in range(self.num_agents):
             if 'masks' in batch_data[0] and agent_id in batch_data[0]['masks']:
                 masks_batch[agent_id] = torch.tensor([t['masks'][agent_id] for t in batch_data], device=self.device).unsqueeze(1)
             else:
                 # Fallback if no masks stored (should not happen with new code)
                 masks_batch[agent_id] = torch.ones(len(batch_data), 1, device=self.device)
        
        # Store GAE data
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue
            
            agent_obs = full_obs_batch[:, agent_id, :] # (T, D)
            agent_shaped_rewards = shaped_rewards_all[:, agent_id]
            agent_dones = dones_batch[:, agent_id]
            
            
            # Prepare RNN inputs
            rnn_obs = agent_obs.unsqueeze(1) # (T, 1, D)
            
            # Correct Masking using stored execution masks
            rnn_masks = masks_batch[agent_id] # (T, 1) Is this correct?
            # Yes, 'masks' stored is the mask used AT that step.
            # If t=0, masks[0] is the mask used at start of t=0.
            # If t=0 was start of episode, masks[0]=0.
            # If t=0 was continuation, masks[0]=1.
            # This is EXACTLY what we need.
            
            # Initial Value RNN state
            _, v_h0 = init_hidden_states[agent_id]
            v_h0 = v_h0.to(self.device).unsqueeze(0) # (1, recurrent_N, hidden_dim) - Batch first
            
            with torch.no_grad():
                # [최적화] 한 번의 forward로 전체 sequence + next value 계산
                last_next_obs = batch_data[-1]['next_observations'][agent_id].to(self.device)
                last_mask = 1.0 - batch_data[-1]['dones'][agent_id]
                
                # Extended sequence: (T+1, 1, D)
                extended_obs = torch.cat([
                    rnn_obs,
                    last_next_obs.unsqueeze(0).unsqueeze(1)
                ], dim=0)
                
                # Extended masks: (T+1, 1)
                extended_masks = torch.cat([
                    rnn_masks,
                    torch.tensor([[last_mask]], dtype=torch.float32, device=self.device)
                ], dim=0)
                
                # 한 번의 forward로 모든 값 계산
                all_values, _ = self.agents[agent_id].value_net(extended_obs, v_h0, extended_masks)
                agent_values = all_values[:-1].reshape(-1) # (T)
                next_val = all_values[-1].item() * last_mask

            advantages, returns = self.compute_gae(
                agent_shaped_rewards.tolist(),
                agent_values.tolist(),
                next_val,
                agent_dones.tolist()
            )
            
            agent_returns[agent_id] = torch.tensor(returns, dtype=torch.float32, device=self.device)
            agent_advantages[agent_id] = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # PPO Update
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue
            
            agent_obs = full_obs_batch[:, agent_id, :]
            my_actions = actions_batch[:, agent_id]
            agent_advantages[agent_id] = torch.tensor(agent_advantages[agent_id], device=self.device)
            
            if agent_advantages[agent_id].numel() > 1:
                adv_std = agent_advantages[agent_id].std() + 1e-8
                adv = (agent_advantages[agent_id] - agent_advantages[agent_id].mean()) / adv_std
            else:
                adv = torch.zeros_like(agent_advantages[agent_id]) # Variance undefined -> 0 signal
            
            # Prepare RNN inputs
            # rnn_obs = agent_obs.unsqueeze(1) # (T, 1, D)
            # rnn_masks = masks_batch[agent_id] # (T, 1)
            # p_h0, _ = init_hidden_states[agent_id]
            # p_h0 = p_h0.to(self.device).unsqueeze(0) # Batch first

            # [OPTIMIZATION] Chunking for Parallel Processing
            
            p_states_all = torch.stack([t['hidden_states'][agent_id][0] for t in batch_data]).to(self.device) # (T, recurrent_N, hidden_dim)
            v_states_all = torch.stack([t['hidden_states'][agent_id][1] for t in batch_data]).to(self.device)
            
            T_total = agent_obs.size(0)
            chunk_len = self.rnn_chunk_length
            num_chunks = T_total // chunk_len
            
            if num_chunks > 0:
                # Prepare Chunked Data
                rnn_obs_chunked = self._chunk_sequence(agent_obs, chunk_len) # (L, B, D)
                my_actions_chunked = self._chunk_sequence(my_actions, chunk_len) 
                rnn_masks_main = masks_batch[agent_id] # (T, 1)
                rnn_masks_chunked = self._chunk_sequence(rnn_masks_main, chunk_len) # (L, B, 1)
                
                # Hidden States
                # We need hidden states at indices: 0, chunk_len, 2*chunk_len, ...
                chunk_indices = torch.arange(0, num_chunks * chunk_len, chunk_len)
                
                # p_states_all is (T, layers, hidden). We want (B, layers, hidden) = (NumChunks, layers, hidden)
                p_h0_chunked = p_states_all[chunk_indices] 
                # p_h0_chunked is already (NumChunks, layers, hidden). RNNLayer expects Batch first!
                
                v_h0_chunked = v_states_all[chunk_indices]
                
                adv_chunked = self._chunk_sequence(adv, chunk_len)
                ret_chunked = self._chunk_sequence(agent_returns[agent_id], chunk_len)
                
                p_h0_batch = p_h0_chunked
                v_h0_batch = v_h0_chunked
                rnn_obs_batch = rnn_obs_chunked
                rnn_masks_batch = rnn_masks_chunked
                my_actions_batch = my_actions_chunked.reshape(-1) # Flatten (L*B)
                adv_batch = adv_chunked.reshape(-1)
                ret_batch = ret_chunked.reshape(-1)

            else:
                # Fallback to single batch
                rnn_obs_batch = agent_obs.unsqueeze(1)
                rnn_masks_batch = masks_batch[agent_id]
                p_h0_batch, _ = init_hidden_states[agent_id]
                p_h0_batch = p_h0_batch.to(self.device).unsqueeze(0) # Batch first (1, L, H)
                _, v_h0_batch = init_hidden_states[agent_id]
                v_h0_batch = v_h0_batch.to(self.device).unsqueeze(0)
                my_actions_batch = my_actions
                adv_batch = adv
                ret_batch = agent_returns[agent_id]
            
            with torch.no_grad():
                old_log_probs, _ = self.agents[agent_id].policy_net.evaluate_actions(rnn_obs_batch, p_h0_batch, my_actions_batch, rnn_masks_batch)
                old_log_probs = old_log_probs.detach()
                
            for epoch in range(self.ppo_epochs):
                new_log_probs, entropy = self.agents[agent_id].policy_net.evaluate_actions(rnn_obs_batch, p_h0_batch, my_actions_batch, rnn_masks_batch)
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean() # entropy is (T, 1) usually or (T). check.
                loss = policy_loss + self.entropy_coef * entropy_loss
                
                self.agents[agent_id].optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[agent_id].policy_net.parameters(), self.max_grad_norm)
                self.agents[agent_id].optimizer.step()
                
                agent_losses[f'agent_{agent_id}_policy_loss'] += policy_loss.item()
                agent_losses[f'agent_{agent_id}_entropy_loss'] += entropy_loss.item()
                
                # [최적화] Value loss - v_h0는 이미 준비되어 있으므로 재사용
                # Value forward는 gradient 계산을 위해 각 epoch마다 필요하지만,
                # initial state는 동일하므로 재사용 가능
                new_vals, _ = self.agents[agent_id].value_net(rnn_obs_batch, v_h0_batch, rnn_masks_batch)
                val_loss = F.mse_loss(new_vals.squeeze(1), ret_batch)
                
                self.agents[agent_id].value_optimizer.zero_grad()
                (self.value_coef * val_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.agents[agent_id].value_net.parameters(), self.max_grad_norm)
                self.agents[agent_id].value_optimizer.step()
                
                agent_losses[f'agent_{agent_id}_value_loss'] += val_loss.item()

        for k in list(agent_losses.keys()):
            if 'delta' not in k:
                if k == 'value_loss':
                    total_value_loss = sum([agent_losses.get(f'agent_{i}_value_loss', 0.0) for i in range(self.num_agents)])
                    agent_losses['value_loss'] = total_value_loss / (self.num_agents * self.ppo_epochs)
                else:
                    agent_losses[k] /= self.ppo_epochs
            elif 'sum' in k:
                count_key = k.replace('sum', 'count')
                if agent_losses[count_key] > 0:
                    agent_losses[k.replace('sum', 'mean')] = agent_losses[k] / agent_losses[count_key]
                del agent_losses[k]
        
        return dict(agent_losses)

    def update(self, **kwargs):
        if self.buffer_manager.is_main_buffer_empty(): return {}
        transitions = self.buffer_manager.get_main_buffer_transitions()
        
        # With Chunking, we can process larger batches, but standard PPO usually batches.
        batch_size = max(1, len(transitions) // self.num_batches)
        batches = [transitions[i:i + batch_size] for i in range(0, len(transitions), batch_size)]
        total_losses = defaultdict(float)
        
        for batch_data in batches:
            if not batch_data: continue
            
            if self.get_current_causal_lambda() > 0.0:
                causal_losses = self.update_causal_networks(batch_data, mc_samples_k_w2=16)
            else:
                causal_losses = {'outcome_bridge_loss': 0.0, 'action_bridge_loss': 0.0}
            
            agent_and_value_losses = self.update_agents(batch_data)

            for loss_name, loss_value in causal_losses.items():
                total_losses[loss_name] += loss_value
            for loss_name, loss_value in agent_and_value_losses.items():
                total_losses[loss_name] += loss_value

        self.buffer_manager.clear_buffer()
        self.training_steps += 1
        
        num_valid_batches = len([b for b in batches if b])
        if num_valid_batches == 0: return {}

        final_losses = {name: val / num_valid_batches for name, val in total_losses.items()}
        final_losses['causal_lambda'] = self.get_current_causal_lambda()

        return final_losses

    def clear_buffer(self):
        self.buffer_manager.clear_buffer()
    
    def get_buffer_info(self) -> Dict[str, int]:
        return self.buffer_manager.get_buffer_info()