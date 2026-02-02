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
    def __init__(self, agent_id: int, obs_dim: int, action_dim: int, hidden_dim: int = 64, recurrent_N: int = 1, lr_actor: float = 3e-4, lr_critic: float = 3e-4):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.recurrent_N = recurrent_N
        
        self.policy_net = RNNPolicyNetwork(obs_dim, action_dim, hidden_dim, recurrent_N)
        self.value_net = RNNValueNetwork(obs_dim, hidden_dim, recurrent_N)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_actor)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)
        
        # Internal hidden state maintenance
        self.rnn_state_policy = None
        self.rnn_state_value = None
        self.reset_rnn_state()
        
        # Batch processing support (for parallel environments)
        self.n_envs = 1
        self.rnn_states_policy_batch = None  # (n_envs, recurrent_N, hidden_dim)
        self.rnn_states_value_batch = None

    def reset_rnn_state(self, device=None):
        """Reset RNN state for single environment (backward compatibility)"""
        if device is None: device = self.policy_net.output_layer.weight.device
        self.rnn_state_policy = torch.zeros(self.recurrent_N, self.hidden_dim, device=device)
        self.rnn_state_value = torch.zeros(self.recurrent_N, self.hidden_dim, device=device)
    
    def reset_rnn_states_batch(self, n_envs: int, device=None):
        """Reset RNN states for multiple environments (batch mode)"""
        if device is None: device = self.policy_net.output_layer.weight.device
        self.n_envs = n_envs
        self.rnn_states_policy_batch = torch.zeros(n_envs, self.recurrent_N, self.hidden_dim, device=device)
        self.rnn_states_value_batch = torch.zeros(n_envs, self.recurrent_N, self.hidden_dim, device=device)


    def get_action(self, obs: torch.Tensor, masks: torch.Tensor = None, available_actions: torch.Tensor = None) -> Tuple[int, float, Optional[torch.Tensor]]:
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
            
            # Handle available actions
            if available_actions is not None:
                available_actions = available_actions.unsqueeze(0) if available_actions.dim() == 1 else available_actions

            action, log_prob, next_rnn_state = self.policy_net.get_action_and_log_prob(
                obs_in, 
                curr_rnn_state, 
                masks_in,
                available_actions
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
    
    def get_action_batch(
        self,
        obs_batch: torch.Tensor,  # (n_envs, obs_dim)
        masks_batch: torch.Tensor,  # (n_envs, 1)
        available_actions_batch: Optional[torch.Tensor] = None  # (n_envs, action_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions for a batch of observations (for parallel environments).
        
        Args:
            obs_batch: Batch of observations (n_envs, obs_dim)
            masks_batch: Batch of masks (n_envs, 1)
            available_actions_batch: Batch of available actions (n_envs, action_dim) or None
        
        Returns:
            actions: (n_envs,)
            log_probs: (n_envs,)
            prev_states: (n_envs, recurrent_N, hidden_dim) - states used for action selection
        """
        with torch.no_grad():
            # Save current states (these will be returned as prev_states)
            prev_states = self.rnn_states_policy_batch.clone()
            
            # Get actions
            actions, log_probs, next_states = self.policy_net.get_action_and_log_prob(
                obs_batch,
                self.rnn_states_policy_batch,
                masks_batch,
                available_actions_batch
            )
            
            # Update internal states
            self.rnn_states_policy_batch = next_states
            
        return actions, log_probs, prev_states
    
    def get_value_batch(
        self,
        obs_batch: torch.Tensor,  # (n_envs, obs_dim)
        masks_batch: torch.Tensor  # (n_envs, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get values for a batch of observations (for parallel environments).
        
        Args:
            obs_batch: Batch of observations (n_envs, obs_dim)
            masks_batch: Batch of masks (n_envs, 1)
        
        Returns:
            values: (n_envs,)
            prev_states: (n_envs, recurrent_N, hidden_dim) - states used for value calculation
        """
        with torch.no_grad():
            # Save current states
            prev_states = self.rnn_states_value_batch.clone()
            
            # Get values
            values, next_states = self.value_net(
                obs_batch,
                self.rnn_states_value_batch,
                masks_batch
            )
            
            # Update internal states
            self.rnn_states_value_batch = next_states
            
        return values.squeeze(-1), prev_states



class PCIRNNAlgorithm:
    def __init__(
        self,
        env,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        w_dim: int, 
        hidden_dim: int = 64,
        recurrent_N: int = 1, # [ADDED] RNN layers
        lr_actor: float = 3e-4,
        lr_bridge: float = 3e-4,
        lr_critic: float = 3e-4,
        clip_delta: float = 0.2, # for delta clipping
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
        self.env = env
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

        # for causal intrinsic reward
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

        # training steps
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
                lr_actor=lr_actor,
                lr_critic=lr_critic
            )
        
        # Value network parameter sharing 설정
        if self.use_value_parameter_sharing:
            # 하나의 shared value network 생성
            self.shared_value_net = RNNValueNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim, recurrent_N=recurrent_N)
            self.shared_value_optimizer = optim.Adam(self.shared_value_net.parameters(), lr=lr_critic)
            
            # for checkpointing and loading (호환성을 위해 딕셔너리 유지)
            self.value_nets = {agent_id: self.shared_value_net for agent_id in range(num_agents)}

            # for agent_id, agent in self.agents.items():
            #    agent.value_net = self.shared_value_net
            
            self.value_optimizers = {agent_id: self.shared_value_optimizer for agent_id in range(num_agents)}
        else:
            # 각 agent별로 별도의 value network 사용
            self.value_nets = {agent_id: agent.value_net for agent_id, agent in self.agents.items()}
            self.value_optimizers = {agent_id: agent.value_optimizer for agent_id, agent in self.agents.items()}
        
        self.outcome_bridge_net = OutcomeBridgeNetwork( 
            # [edit] outcome bridge function predicts all reward vector \in \mathbb{R}^N
            action_dim=action_dim,
            num_agents=num_agents,
            w_dim=obs_dim * (num_agents-1),
            outcome_dim=num_agents,
            hidden_dim=hidden_dim,
            action_embedding_dim=8,
            # x_dim=obs_dim # x = o^i_t
            x_dim=obs_dim + (num_agents-1) * hidden_dim * recurrent_N # [SWITCH] x = (o^i_t, \mathbf{h}^{-i}_{t-1})
        )
        
        self.action_bridge_net = ActionBridgeNetwork(
            # z_dim=obs_dim * num_agents, # prev joint observation
            z_dim=num_agents, # [SWITCH] Z=prev rewards
            # z_dim=hidden_dim * recurrent_N, # [SWITCH] Z=own hidden state
            # z_dim=obs_dim * num_agents + num_agents, # [SWITCH] Z = [Joint Obs, Joint Reward]
            # z_dim=obs_dim * num_agents + action_dim * num_agents, # [SWITCH] Z = [Joint Obs, Joint Action]
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
        # joint_o_prev = torch.stack([self._concatenate_observations(t['bold_o_t_minus_1']) for t in batch_data]).to(device) # Default Z
        joint_r_prev = torch.tensor([[t['bold_r_t_minus_1'][i] for i in range(num_agents)] for t in batch_data], dtype=torch.float32, device=device) # [SWITCH] prev rewards
        # joint_a_prev = torch.tensor([[t['bold_a_t_minus_1'][i] for i in range(num_agents)] for t in batch_data], dtype=torch.long, device=device)
        # joint_a_prev_onehot = F.one_hot(joint_a_prev, num_classes=self.action_dim).view(batch_size, -1).float()
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
        
        # [EDIT] Y = r_t
        Y_super = reward_batch.repeat_interleave(num_agents, dim=0)  # (batch_size * num_agents, num_agents)

        # [SWITCH] Constructing X_super = {o^i_t, h^{-i}_{t-1}}
        # 1. Gather all hidden states: (B, N, RecN*H)
        h_list = []
        for i in range(num_agents):
            if i in hidden_states_batch:
                h_list.append(hidden_states_batch[i].view(batch_size, -1))
            else:
                h_list.append(torch.zeros(batch_size, self.recurrent_N * self.hidden_dim, device=device))
        H_all = torch.stack(h_list, dim=1) # (B, N, RecN*H)

        # 2. Get others' hidden states for each agent i
        expanded_H = H_all.unsqueeze(1).expand(-1, num_agents, -1, -1) # (B, N, N, RecN*H)
        mask_H = ~torch.eye(num_agents, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.recurrent_N * self.hidden_dim)
        H_others_super = expanded_H[mask_H].view(batch_size * num_agents, -1) # (B*N, (N-1)*RecN*H)

        # 3. Concatenate with ego observation (obs_super is already B*N, ObsDim)
        obs_super = obs_batch.view(-1, self.obs_dim)
        X_super = torch.cat([obs_super, H_others_super], dim=1)

        # Z_super = joint_o_prev.repeat_interleave(num_agents, dim=0) # Default Z
        Z_super = joint_r_prev.repeat_interleave(num_agents, dim=0) # [SWITCH] prev rewards
        # Z_super = torch.cat([joint_o_prev, joint_r_prev], dim=1).repeat_interleave(num_agents, dim=0)
        # Z_super = torch.cat([joint_o_prev, joint_a_prev_onehot], dim=1).repeat_interleave(num_agents, dim=0)
        
        ''' [SWITCH] Z=own hidden state 
        z_list = [] 
        for i in range(num_agents):
            if i in hidden_states_batch:
                # hidden_states_batch[i] is (Batch, RecN, H)
                z_list.append(hidden_states_batch[i].view(batch_size, -1))
            else:
                z_list.append(torch.zeros(batch_size, self.recurrent_N * self.hidden_dim, device=device))
                # default self.recurrent_N is 1, but for generality.
        Stack (B, N, RecN*H) then flatten -> (B*N, RecN*H)
        Z_super = torch.stack(z_list, dim=1).view(-1, self.recurrent_N * self.hidden_dim)
        '''
        
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
            x_mb = X_super[idx]
            o_mb = obs_super[idx] 
            batch_agent_ids = idx % num_agents

            # --- Update Outcome Bridge h_φ ---
            # r_hat_mb = self.outcome_bridge_net(a_mb, w_mb), x=None)
            # r_hat_mb = self.outcome_bridge_net(a_mb, w_mb), x=None)
            r_hat_mb = self.outcome_bridge_net(a_mb, w_mb, x=x_mb) # [SWITCH] x=(o_mb, h_others)
            residuals_mb = y_mb - r_hat_mb
            
            if self.causal_lambda_adaptive_per_agent:
                for j, agent_id in enumerate(batch_agent_ids):
                    residual_error = torch.norm(residuals_mb[j], p=2).item()
                    agent_residual_errors[agent_id.item()].append(residual_error)
            
            action_embeds_mb = self.action_embedding_for_kernel(a_mb)
            # kernel_feat_z = torch.cat([z_mb, action_embeds_mb], dim=1)
            kernel_feat_z = torch.cat([z_mb, action_embeds_mb, o_mb], dim=1) # [ADD] x=o_mb
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
            # torch.nn.utils.clip_grad_norm_(self.outcome_bridge_net.parameters(), self.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.action_embedding_for_kernel.parameters(), self.max_grad_norm)
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
            agent.reset_rnn_state(device=self.device)
        self.last_dones = {i: False for i in range(self.num_agents)}
    
    def reset_agents_batch(self, n_envs: int):
        """Reset agent hidden states for batch mode (parallel environments)."""
        for agent in self.agents.values():
            agent.reset_rnn_states_batch(n_envs, device=self.device)
        # Track done status for each agent in each environment
        self.last_dones_batch = {i: [False] * n_envs for i in range(self.num_agents)}
   
    def get_actions(self, observations: Dict[int, torch.Tensor], available_actions: Optional[Dict[int, List[int]]] = None) -> Dict[int, Tuple[int, float]]:
        actions = {}
        self.temp_hidden_states = {} # Store current states for store_transition
        self.temp_masks = {} # Store masks used
        for agent_id, obs in observations.items():
            # Create mask: 1.0 if not done, 0.0 if done.
            # Using self.last_dones which is updated in store_transition
            mask = 1.0 if not self.last_dones.get(agent_id, False) else 0.0
            mask_tensor = torch.tensor([mask], dtype=torch.float32, device=self.device)
            
            # Prepare available actions tensor
            avail_actions_tensor = None
            if available_actions is not None and agent_id in available_actions:
                avail_actions_tensor = torch.tensor(available_actions[agent_id], dtype=torch.float32, device=self.device)

            # Policy
            action, log_prob, p_state = self.agents[agent_id].get_action(obs, mask_tensor, available_actions=avail_actions_tensor)
            
            # Value (Update state)
            _, v_state = self.agents[agent_id].get_value(obs, mask_tensor)
            
            actions[agent_id] = (action, log_prob)
            self.temp_hidden_states[agent_id] = (p_state, v_state)
            self.temp_masks[agent_id] = mask
            
        return actions
    
    def get_actions_batch(
        self,
        observations_batch: List[Dict[int, torch.Tensor]],  # List of length n_envs
        available_actions_batch: Optional[List[Optional[Dict[int, List[int]]]]] = None
    ) -> List[Dict[int, Tuple[int, float]]]:
        """
        Get actions for a batch of observations (parallel environments).
        
        Args:
            observations_batch: List of observation dicts, length = n_envs
                               Each dict maps agent_id to observation tensor
            available_actions_batch: List of available actions dicts or None
        
        Returns:
            List of action dicts, length = n_envs
            Each dict maps agent_id to (action, log_prob) tuple
        """
        n_envs = len(observations_batch)
        actions_batch = [{} for _ in range(n_envs)]
        
        # Store hidden states and masks for later use in store_transition
        self.temp_hidden_states_batch = [{} for _ in range(n_envs)]
        self.temp_masks_batch = [{} for _ in range(n_envs)]
        
        # Process each agent
        for agent_id in range(self.num_agents):
            # Stack observations from all environments
            obs_stacked = torch.stack([
                observations_batch[env_idx][agent_id]
                for env_idx in range(n_envs)
            ]).to(self.device)  # (n_envs, obs_dim)
            
            # Create masks
            masks_stacked = torch.tensor([
                [1.0 if not self.last_dones_batch[agent_id][env_idx] else 0.0]
                for env_idx in range(n_envs)
            ], dtype=torch.float32, device=self.device)  # (n_envs, 1)
            
            # Stack available actions if provided
            avail_actions_stacked = None
            if available_actions_batch is not None and available_actions_batch[0] is not None:
                avail_actions_stacked = torch.stack([
                    torch.tensor(available_actions_batch[env_idx][agent_id], dtype=torch.float32, device=self.device)
                    for env_idx in range(n_envs)
                ])  # (n_envs, action_dim)
            
            # Get actions in batch
            actions, log_probs, p_states = self.agents[agent_id].get_action_batch(
                obs_stacked,
                masks_stacked,
                avail_actions_stacked
            )
            
            # Get values in batch
            _, v_states = self.agents[agent_id].get_value_batch(
                obs_stacked,
                masks_stacked
            )
            
            # Distribute results to each environment
            for env_idx in range(n_envs):
                actions_batch[env_idx][agent_id] = (
                    actions[env_idx].item(),
                    log_probs[env_idx].item()
                )
                self.temp_hidden_states_batch[env_idx][agent_id] = (
                    p_states[env_idx].clone(),
                    v_states[env_idx].clone()
                )
                self.temp_masks_batch[env_idx][agent_id] = masks_stacked[env_idx, 0].item()
        
        return actions_batch

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
            available_actions=available_actions, # [ADDED]
            info=info
        )
    
    def store_transition_from_env( # for vectorized env
        self,
        env_idx: int,
        prev_observations: Dict[int, torch.Tensor],
        prev_rewards: Dict[int, float],
        prev_actions: Dict[int, int],
        observations: Dict[int, torch.Tensor],
        actions: Dict[int, Tuple[int, float]],  # (action, log_prob)
        rewards: Dict[int, float],
        next_observations: Dict[int, torch.Tensor],
        dones: Dict[int, bool],
        info: Dict = None,
        available_actions: Optional[Dict[int, List[int]]] = None
    ):
        """
        Store transition from a specific parallel environment.
        
        Args:
            env_idx: Index of the environment (0 to n_envs-1)
            ... (same as store_transition)
        """
        # Extract actions only (remove log_probs)
        actions_only = {agent_id: action for agent_id, (action, _) in actions.items()}
        
        # Extract hidden states for this environment
        hidden_states = self.temp_hidden_states_batch[env_idx]
        
        # Extract masks for this environment
        masks = self.temp_masks_batch[env_idx]
        
        if info is None:
            info = {}
        info['env_id'] = env_idx
        
        # Store transition
        self.buffer_manager.store_transition(
            prev_observations=prev_observations,
            prev_rewards=prev_rewards,
            prev_actions=prev_actions,
            observations=observations,
            actions=actions_only,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            masks=masks,
            hidden_states=hidden_states,
            available_actions=available_actions,
            info=info
        )
        
        # Update done status for this environment
        for agent_id in range(self.num_agents):
            self.last_dones_batch[agent_id][env_idx] = dones[agent_id]
            
            # Reset RNN states if episode is done
            if dones[agent_id]:
                self.agents[agent_id].rnn_states_policy_batch[env_idx] = torch.zeros(
                    self.recurrent_N, self.hidden_dim, device=self.device
                )
                self.agents[agent_id].rnn_states_value_batch[env_idx] = torch.zeros(
                    self.recurrent_N, self.hidden_dim, device=self.device
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
                # z_b = z_t_batch[i:end] # Unused now
                r_b = rewards_batch[i:end]
                obs_b = full_obs_batch[i:end]
                curr_bs = act_b.size(0)
                
                # --- Vectorized Logic ---
                actions_flat = act_b.view(-1)
                w_t_flat = w_b.view(-1, w_b.size(-1))
                
                # r_pred_flat = self.outcome_bridge_net(actions_flat, w_t_flat) 
                
                rewards_flat = r_b.unsqueeze(1).expand(-1, num_agents, -1).reshape(-1, num_agents)
                
                actions_range = torch.arange(action_dim, device=self.device).repeat(curr_bs * num_agents)
                w_t_expanded = w_t_flat.repeat_interleave(action_dim, dim=0)

                # Expand x for all actions
                # X construction for inference _inf: inference
                h_list_inf = []
                for k in range(num_agents):
                    if hidden_states_batch is not None and k in hidden_states_batch:
                         # hidden_states_batch[k][0][i:end] is (curr_bs, RecN, H)
                        h_list_inf.append(hidden_states_batch[k][0][i:end].view(curr_bs, -1))
                    else:
                        h_list_inf.append(torch.zeros(curr_bs, self.recurrent_N * self.hidden_dim, device=self.device))
                H_all_inf = torch.stack(h_list_inf, dim=1) # (curr_bs, N, RecN*H)

                # get others hiddens
                expanded_H_inf = H_all_inf.unsqueeze(1).expand(-1, num_agents, -1, -1)
                mask_H_inf = ~torch.eye(num_agents, dtype=torch.bool, device=self.device).unsqueeze(0).unsqueeze(-1).expand(curr_bs, -1, -1, self.recurrent_N * self.hidden_dim)
                H_others_inf = expanded_H_inf[mask_H_inf].view(curr_bs * num_agents, -1)

                obs_flat = obs_b.view(-1, self.obs_dim)
                x_flat = torch.cat([obs_flat, H_others_inf], dim=1)

                x_expanded = x_flat.repeat_interleave(action_dim, dim=0)
                
                # reg_estimates_cf = self.outcome_bridge_net(actions_range, w_t_expanded)
                reg_estimates_cf = self.outcome_bridge_net(actions_range, w_t_expanded, x=x_expanded) # [SWITCH] x=x_expanded
                
                # [EDIT] outcome bridge now predicts all agents' rewards (num_agents instead of num_agents-1)
                reg_estimates_cf = reg_estimates_cf.view(curr_bs * num_agents, action_dim, num_agents)
                r_hat_all = reg_estimates_cf # no q

                # collect r_hat by factual actions
                r_hat_factual = r_hat_all[torch.arange(r_hat_all.size(0)), actions_flat]

                probs_list = []
                if hidden_states_batch is not None and dones_batch is not None:
                    precomputed_states = {}
                    precomputed_masks = {}
                    for k in range(num_agents):
                        p_state_b = hidden_states_batch[k][0][i:end]  # (curr_bs, recurrent_N, hidden_dim)
                        if p_state_b.device != self.device: p_state_b = p_state_b.to(self.device)
                        precomputed_states[k] = p_state_b
                        
                        dones_b = dones_batch[i:end, k]
                        masks_b = (1.0 - dones_b).unsqueeze(1)  # (curr_bs, 1)
                        precomputed_masks[k] = masks_b
                
                for k in range(num_agents):
                    avail_actions_b = None
                    if available_actions_batch is not None and k in available_actions_batch:
                        avail_actions_b = available_actions_batch[k][i:end]

                    if hidden_states_batch is not None and dones_batch is not None:
                        p_state_b = precomputed_states[k]
                        masks_b = precomputed_masks[k]
                        
                        logits, _ = self.agents[k].policy_net(obs_b[:, k, :], p_state_b, masks_b, available_actions=avail_actions_b)
                    else:
                        # Fallback (zeros) if not provided - e.g. for non-recurrent compatibility or testing
                        bs = obs_b.size(0)
                        # RNNLayer expects (Batch, recurrent_N, hidden_dim) for batch processing
                        dummy_state = torch.zeros(bs, self.recurrent_N, self.hidden_dim, device=self.device)
                        dummy_mask = torch.ones(bs, 1, device=self.device)                            
                        logits, _ = self.agents[k].policy_net(obs_b[:, k, :], dummy_state, dummy_mask, available_actions=avail_actions_b)
                        
                    probs = torch.softmax(logits, dim=-1)
                    probs_list.append(probs)
                action_probs_flat = torch.stack(probs_list, dim=1).view(-1, action_dim)

                # baseline: exclude factual action
                factual_mask = torch.zeros_like(action_probs_flat).scatter_(1, actions_flat.unsqueeze(1), 1.0)
                non_factual_mask = 1.0 - factual_mask
                
                if baseline_type == 'weighted':
                    # Factual action을 제외한 나머지 action들에 대해 조건부 확률(conditional probability)로 가중 평균 계산
                    probs_counterfactual = action_probs_flat * non_factual_mask
                    renormalized_probs = probs_counterfactual / (probs_counterfactual.sum(dim=1, keepdim=True) + 1e-10)
                    baseline = (renormalized_probs.unsqueeze(2) * r_hat_all).sum(dim=1)
                elif baseline_type == 'averaged':
                    # Factual action을 제외한 나머지 action들의 단순 평균 계산
                    baseline = (r_hat_all * non_factual_mask.unsqueeze(2)).sum(dim=1) / (action_dim - 1)
                else:
                    raise ValueError(f"Invalid baseline type: {baseline_type}")

                # Opt 1) using real rewards
                # [EDIT] Now using all agents' rewards (num_agents dimension)
                r_factual = rewards_flat
                raw_effects = r_factual - baseline
                
                # Opt 2) using outcome bridge -- worse performance, do not use
                # raw_effects = r_hat_factual - baseline

                # [EDIT] Reshape to [B, N, N] (now includes all agents)
                effects_reshaped = raw_effects.view(curr_bs, num_agents, num_agents)
                
                # [EDIT] effects_reshaped is already [B, N, N], so directly assign it
                # No need for scatter operation since we have all agents' effects
                C_chunk = effects_reshaped
                
                C_matrix[i:end] = C_chunk
            
        return C_matrix

    def causal_intrinsic_reward(
        self, 
        agent_id: int, 
        w_t: torch.Tensor, 
        z_t: torch.Tensor, 
        o_t: torch.Tensor, 
        current_actions: Dict[int, int], 
        current_rewards: Dict[int, float],
        hidden_states: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[float, float]:
        """
        Wrapper for compute_causal_rewards to be used by evaluation scripts.
        Returns:
            delta: The calculated intrinsic reward (causal effect sum).
            dummy: A placeholder return value.
        """
        with torch.no_grad():
            # Actions to tensor
            actions_list = [current_actions[i] for i in range(self.num_agents)]
            actions_batch = torch.tensor([actions_list], dtype=torch.long, device=self.device) # (1, N)
            
            # Rewards to tensor
            rewards_list = [current_rewards[i] for i in range(self.num_agents)]
            rewards_batch = torch.tensor([rewards_list], dtype=torch.float32, device=self.device) # (1, N)
            
            # Reconstruct full observations batch (1, N, D)
            # Assuming w_t is concatenation of N observations of obs_dim
            full_obs_batch = w_t.view(1, self.num_agents, -1)
            B, N, D = full_obs_batch.shape

            # Prepare w_t_batch (1, N, (N-1)*D) - Confounders (other agents' observations)
            expanded_obs = full_obs_batch.unsqueeze(1).expand(-1, N, -1, -1)
            mask = ~torch.eye(N, dtype=torch.bool, device=self.device).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, D)
            w_t_batch = expanded_obs[mask].view(B, N, (N-1)*D)

            z_t_batch = z_t.unsqueeze(0) # (1, z_dim) - This might need similar processing if used, but currently z_t is just passed through or unused if Z=prev_obs.
            # Actually compute_causal_rewards uses z_t_batch. Line 598: z_t_repeated = z_b.unsqueeze(1).repeat(1, num_agents, 1).view(...)
            # If z_t is prev joint obs, then (1, N*D). 
            # z_t_repeated becomes (1, N, N*D) -> (N, N*D).
            # Wrapper for hidden_states
            hidden_states_batch = None
            if hidden_states is not None:
                hidden_states_batch = {}
                for k, v in hidden_states.items():
                    # v is (h, c) or (h, v_h). h is (RecN, H). Need (1, RecN, H)
                    h_unsqueezed = v[0].unsqueeze(0)
                    c_unsqueezed = v[1].unsqueeze(0)
                    hidden_states_batch[k] = (h_unsqueezed, c_unsqueezed)

            C_matrix = self.compute_causal_rewards(
                w_t_batch, z_t_batch, full_obs_batch, actions_batch, rewards_batch, 
                hidden_states_batch=hidden_states_batch,
                baseline_type='weighted'
            )
            
            # Calculate delta for the specific agent (sum of effects this agent has on others)
            # C_matrix[0, agent_id, :] contains effects of agent_id on others
            delta = C_matrix[0, agent_id].sum().item()
            
            return delta, 0.0

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
        
        # Extract available actions
        available_actions_batch = {}
        if 'available_actions' in batch_data[0] and batch_data[0]['available_actions'] is not None:
            for agent_id in range(self.num_agents):
                if agent_id in batch_data[0]['available_actions']:
                     avail_list = [t['available_actions'][agent_id] for t in batch_data]
                     available_actions_batch[agent_id] = torch.tensor(avail_list, dtype=torch.float32, device=self.device)

        # Prepare inputs for RNNs
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
        
        # Extract pre-computed advantages and returns
        agent_returns = {}
        agent_advantages = {}
        
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue
            
            # Extract pre-computed values
            returns_list = [t['returns'][agent_id] for t in batch_data]
            advantages_list = [t['advantages'][agent_id] for t in batch_data]
            
            agent_returns[agent_id] = torch.tensor(returns_list, dtype=torch.float32, device=self.device)
            agent_advantages[agent_id] = torch.tensor(advantages_list, dtype=torch.float32, device=self.device)

        # Log delta (causal intrinsic rewards) by scenario for StagHunt
        for i in range(self.num_agents):
            for t, transition in enumerate(batch_data):
                if 'delta' in transition and i in transition['delta']:
                    if self.env.__class__.__name__ == 'StagHuntEnv':
                        scenario = self._classify_delta_scenario(transition, i)
                        agent_losses[f'delta_agent_{i}_{scenario}_sum'] += transition['delta'][i]
                        agent_losses[f'delta_agent_{i}_{scenario}_count'] += 1
                    else:
                        # check the accuracy of delta (predicting others' rewards)
                        others_rewards_sum = sum([transition['bold_r_t'][j] for j in range(self.num_agents) if j != i])
                        if others_rewards_sum == 0:
                            # 1st case: no reward
                            agent_losses[f'delta_agent{i}_no_reward_sum'] += transition['delta'][i]
                            agent_losses[f'delta_agent{i}_no_reward_count'] += 1
                        else:
                            # 2nd case: positive reward
                            agent_losses[f'delta_agent{i}_positive_reward_sum'] += transition['delta'][i]
                            agent_losses[f'delta_agent{i}_positive_reward_count'] += 1

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
                adv = torch.zeros_like(agent_advantages[agent_id]) 

            # Chunking for Parallel Processing, actor and critic hidden states
            p_states_all = torch.stack([t['hidden_states'][agent_id][0] for t in batch_data]).to(self.device) # (T, recurrent_N, hidden_dim)
            v_states_all = torch.stack([t['hidden_states'][agent_id][1] for t in batch_data]).to(self.device)
            
            avail_actions_all = None
            if agent_id in available_actions_batch:
                avail_actions_all = available_actions_batch[agent_id] # (T, ActionDim)

            T_total = agent_obs.size(0)
            chunk_len = self.rnn_chunk_length
            num_chunks = T_total // chunk_len
            
            if num_chunks > 0:
                # Prepare Chunked Data
                rnn_obs_chunked = self._chunk_sequence(agent_obs, chunk_len) # (L, B, D)
                my_actions_chunked = self._chunk_sequence(my_actions, chunk_len) 
                rnn_masks_main = masks_batch[agent_id] # (T, 1)
                rnn_masks_chunked = self._chunk_sequence(rnn_masks_main, chunk_len) # (L, B, 1)
                
                # Active Masks (Padding Mask)
                if 'active_mask' in batch_data[0]:
                    active_masks = torch.tensor([t['active_mask'] for t in batch_data], device=self.device).unsqueeze(1) # (T, 1)
                    active_masks_chunked = self._chunk_sequence(active_masks, chunk_len) # (L, B, 1)
                else:
                    active_masks = torch.ones(T_total, 1, device=self.device)
                    active_masks_chunked = self._chunk_sequence(active_masks, chunk_len)

                avail_actions_chunked = None
                if avail_actions_all is not None:
                    avail_actions_chunked = self._chunk_sequence(avail_actions_all, chunk_len) # (L, B, ActionDim)
                
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
                active_masks_batch = active_masks_chunked.reshape(-1) # (L*B)
                
                avail_actions_batch_merged = None
                if avail_actions_chunked is not None:
                     avail_actions_batch_merged = avail_actions_chunked.reshape(-1, avail_actions_chunked.size(-1)) # Flatten (L*B, ActionDim)

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
                active_masks_batch = active_masks.squeeze(1) if 'active_mask' in batch_data[0] else torch.ones(T_total, device=self.device)
                avail_actions_batch_merged = avail_actions_all
            
            with torch.no_grad():
                old_log_probs, _ = self.agents[agent_id].policy_net.evaluate_actions(rnn_obs_batch, p_h0_batch, my_actions_batch, rnn_masks_batch, available_actions=avail_actions_batch_merged)
                old_log_probs = old_log_probs.detach()
                
            for epoch in range(self.ppo_epochs):
                new_log_probs, entropy = self.agents[agent_id].policy_net.evaluate_actions(rnn_obs_batch, p_h0_batch, my_actions_batch, rnn_masks_batch, available_actions=avail_actions_batch_merged)
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
                
                # Masked Policy Loss
                policy_loss = -torch.min(surr1, surr2)
                policy_loss = (policy_loss * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)

                # Masked Entropy Loss
                entropy_loss = -entropy # entropy is (T)
                entropy_loss = (entropy_loss * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)
                
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
                
                # Masked Value Loss
                # val_loss = F.mse_loss(new_vals.squeeze(1), ret_batch) # Old mean
                val_error = (new_vals.squeeze(1) - ret_batch) ** 2
                val_loss = (val_error * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)
                
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
                # [FIX] Delta logging: Keep sum/count for global aggregation
                if 'delta' in k:
                    continue

                count_key = k.replace('sum', 'count')
                if agent_losses[count_key] > 0:
                    agent_losses[k.replace('sum', 'mean')] = agent_losses[k] / agent_losses[count_key]
                del agent_losses[k]
        
        return dict(agent_losses)

    def update(self, **kwargs):
        if self.buffer_manager.is_main_buffer_empty(): return {}
        transitions = self.buffer_manager.get_main_buffer_transitions()
        
        # [FIX] Compute GAE on full trajectories before chunking
        try:
            # 1. Group by env_id
            env_trajectories = defaultdict(list)
            for t in transitions:
                env_id = t['info'].get('env_id', 0)
                env_trajectories[env_id].append(t)
   
            # 2. Compute causal rewards for each trajectory
            for env_id, traj in env_trajectories.items():
                # NOTE: traj is the many episodes of a specific environment, done mask should work
                if len(traj) == 0:
                    continue
                # Prepare data for causal reward computation
                B = len(traj)
                N = self.num_agents
                
                full_obs_batch = torch.stack([torch.stack(list(t['bold_o_t'].values())) for t in traj]).to(self.device)
                actions_batch = torch.tensor([[t['bold_a_t'][i] for i in range(N)] for t in traj], dtype=torch.long, device=self.device)
                rewards_batch = torch.tensor([[t['bold_r_t'][i] for i in range(N)] for t in traj], dtype=torch.float32, device=self.device)
                dones_batch = torch.tensor([[t['dones'][i] for i in range(N)] for t in traj], dtype=torch.float32, device=self.device)
                
                # Prepare z_t (proxy variable)
                # z_t_batch = torch.stack([self._concatenate_observations(t['bold_o_t_minus_1']) for t in traj]).to(self.device)
                z_t_batch = torch.tensor([[t['bold_r_t_minus_1'][i] for i in range(N)] for t in traj], dtype=torch.float32, device=self.device) # prev rewards
                
                # Prepare w_t (other agents' observations)
                expanded_obs = full_obs_batch.unsqueeze(1).expand(-1, N, -1, -1)
                mask = ~torch.eye(N, dtype=torch.bool, device=self.device).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, self.obs_dim)
                w_t_all = expanded_obs[mask].view(B, N, (N-1)*self.obs_dim)
                
                # Prepare hidden states
                hidden_states_batch = {}
                if 'hidden_states' in traj[0] and traj[0]['hidden_states'] is not None:
                    # Check if any agent has hidden states
                    for agent_id in range(N):
                        if agent_id in traj[0]['hidden_states']:
                            p_states = torch.stack([t['hidden_states'][agent_id][0] for t in traj]).to(self.device) # actor hidden state
                            v_states = torch.stack([t['hidden_states'][agent_id][1] for t in traj]).to(self.device) # critic hidden state
                            hidden_states_batch[agent_id] = (p_states, v_states)
                
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
                        hidden_states_batch=hidden_states_batch,
                        dones_batch=dones_batch,
                        baseline_type='weighted',
                        available_actions_batch=available_actions_batch
                    )
                    # merging 1. summation
                    delta = C_matrix.sum(dim=2)
                    # merging 2. max
                    # delta = C_matrix.max(dim=2).values
                
                    # delta = torch.clamp(delta, min=0.0) # same as ReLU
                    delta[torch.abs(delta) < self.clip_delta] = 0.0
                    
                    # compute shaped rewards
                    # (1-causal_lambda) * r^i + causal_lambda * causal_margin( = r - h^{\pi^i}(a', a^{-i}) )
                    if self.causal_lambda_adaptive_per_agent:
                        causal_lambdas = torch.tensor([
                            self.get_current_causal_lambda(agent_id=i) for i in range(N)
                        ], device=self.device).unsqueeze(0)
                        shaped_rewards_all = (1-causal_lambdas) * rewards_batch + causal_lambdas * delta
                    else:
                        shaped_rewards_all = (1-self.get_current_causal_lambda()) * rewards_batch + self.get_current_causal_lambda() * delta
                    shaped_rewards_all = shaped_rewards_all / N # normalized with num_agents

                    # Store shaped rewards and delta in transitions
                    for t_idx, t in enumerate(traj):
                        t['shaped_rewards'] = {agent_id: shaped_rewards_all[t_idx, agent_id].item() for agent_id in range(N)}
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
                    
                    # Extract agent observations
                    agent_obs_seq = torch.stack([t['bold_o_t'][agent_id] for t in traj]).to(self.device)
                    agent_masks_seq = torch.tensor([t['masks'][agent_id] for t in traj], device=self.device).unsqueeze(1)
                    agent_rewards_seq = torch.tensor([t['shaped_rewards'][agent_id] for t in traj], dtype=torch.float32, device=self.device)
                    agent_dones_seq = torch.tensor([t['dones'][agent_id] for t in traj], dtype=torch.float32, device=self.device)
                    
                    # Initial hidden state for value network
                    _, v_h0 = traj[0]['hidden_states'][agent_id]
                    v_h0 = v_h0.to(self.device).unsqueeze(0)
                    
                    # Compute values for entire trajectory
                    with torch.no_grad():
                        last_next_obs = traj[-1]['next_observations'][agent_id].to(self.device)
                        last_mask = 1.0 - traj[-1]['dones'][agent_id]

                        # Extended sequence: (T+1, 1, D)
                        extended_obs = torch.cat([agent_obs_seq.unsqueeze(1),
                                                last_next_obs.unsqueeze(0).unsqueeze(1)], dim=0)
                        extended_masks = torch.cat([agent_masks_seq,
                                                   torch.tensor([[last_mask]], dtype=torch.float32, device=self.device)], dim=0)
                        
                        all_values, _ = self.agents[agent_id].value_net(extended_obs, v_h0, extended_masks)
                        values = all_values[:-1].view(-1) # value of agent_id
                        next_val = all_values[-1].item() * last_mask # next value
                        
                        # Compute GAE
                        advantages, returns = self.compute_gae(
                            agent_rewards_seq.tolist(),
                            values.tolist(),
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
            
            # 4. Create chunks from trajectories with Padding
            all_chunks = []
            chunk_len = self.rnn_chunk_length
            
            for env_id, traj in env_trajectories.items():
                if len(traj) == 0: continue
                
                # Add active_mask to existing transitions
                for t in traj:
                    t['active_mask'] = 1.0
                
                T_traj = len(traj)
                num_full_chunks = T_traj // chunk_len
                remainder = T_traj % chunk_len
                
                # Full chunks
                for i in range(num_full_chunks):
                    start = i * chunk_len
                    end = start + chunk_len
                    chunk = traj[start:end]
                    all_chunks.append(chunk)
                
                # Handling remainder with padding
                if remainder > 0:
                    last_chunk = traj[num_full_chunks * chunk_len:]
                    
                    # Create dummy padding transitions
                    # We copy the last transition but set active_mask to 0
                    padding_len = chunk_len - remainder
                    last_valid_trans = last_chunk[-1]
                    
                    padding_chunk = []
                    for _ in range(padding_len):
                        # Deep copy for safety, though only active_mask matters
                        # Ideally, we just need structure. 
                        # To save memory, we can use reference but must be careful not to mutate shared objects.
                        # Since we only read in update_agents, we can create a shallow copy dict 
                        # and just overwrite active_mask.
                        pad_t = last_valid_trans.copy() 
                        pad_t['active_mask'] = 0.0
                        # Ensure nested dicts (actions, rewards) are safe if we modified them, 
                        # but we only read.
                        padding_chunk.append(pad_t)
                    
                    full_last_chunk = last_chunk + padding_chunk
                    all_chunks.append(full_last_chunk)

            # 5. Shuffle chunks for IID 
            # random.shuffle(all_chunks)
            
            # 5. Create batches
            n_chunks_per_batch = max(1, len(all_chunks) // self.num_batches)
            chunk_batches = [all_chunks[i:i + n_chunks_per_batch] 
                           for i in range(0, len(all_chunks), n_chunks_per_batch)]
            
            # 6. Flatten chunks back to transitions
            batches = []
            for b in chunk_batches:
                flat_batch = []
                for chunk in b:
                    flat_batch.extend(chunk)
                if flat_batch:
                    batches.append(flat_batch)
                    
        except Exception as e:
            print(f"[ERROR] Failed to restructure rollout data: {e}")
            import traceback
            traceback.print_exc()
            # Fallback
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
        
        num = len([b for b in batches if b])
        if num == 0: return {}
        final_losses = {}
        for k, v in total_losses.items():
            if 'delta_agent' in k and '_sum' in k:
                # [FIX] Recompute global mean for delta
                base_key = k.replace('_sum', '')
                count_key = base_key + '_count'
                mean_key = base_key + '_mean'
                
                total_count = total_losses.get(count_key, 0)
                if total_count > 0:
                    final_losses[mean_key] = v / total_count
                    final_losses[count_key] = total_count
                else:
                    final_losses[mean_key] = 0.0
                    final_losses[count_key] = 0.0
            elif 'delta_agent' in k and '_count' in k:
                # Already handled above
                continue
            else:
                 # Other losses: average across batches
                 if num > 0:
                    final_losses[k] = v / num
                 else:
                    final_losses[k] = v
        
        return final_losses


    def clear_buffer(self):
        self.buffer_manager.clear_buffer()
    
    def get_buffer_info(self) -> Dict[str, int]:
        return self.buffer_manager.get_buffer_info()