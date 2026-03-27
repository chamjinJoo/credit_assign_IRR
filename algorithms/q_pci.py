import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
from torch.distributions import Categorical
from algorithms.networks.rnn import RNNLayer, RNNPolicyNetwork, RNNValueNetwork
from algorithms.networks import OutcomeBridgeNetwork, ActionBridgeNetwork, NaivePredictor
from algorithms.networks.mlp import MLPBase
from utils.buffer import TransitionBuffer, EpisodeBuffer

class RBFKernel(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma)), requires_grad=False)

    def set_gamma(self, x: torch.Tensor):
        # median based gamma
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

class ConditionalMeanNet(nn.Module):
    def __init__(self, z_dim, x_dim, action_dim, emb_dim=8, hidden=64):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + x_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, z, a, x):
        ea = F.one_hot(a.long(), self.action_dim).float()  # (B, action_dim)
        inp = torch.cat([z, x, ea], dim=-1)
        return self.net(inp).squeeze(-1)  # (B,)

class qPCIAgent:
    """Q-PCI Agent: Same as rPCIAgent, but the bridge network predicts values instead of rewards."""
    def __init__(self, agent_id: int, obs_dim: int, action_dim: int, hidden_dim: int = 64, recurrent_N: int = 1, lr_actor: float = 3e-4, lr_critic: float = 3e-4):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.recurrent_N = recurrent_N
        
        self.policy_net = RNNPolicyNetwork(obs_dim, action_dim, hidden_dim, recurrent_N)
        self.value_net = RNNValueNetwork(obs_dim, hidden_dim, recurrent_N, action_dim=action_dim, num_agents=1)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_actor)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)
        
        # Internal hidden state maintenance
        self.rnn_state_policy = None
        self.rnn_state_value = None
        
        # Previous action tracking for (obs, action) history encoding
        self.prev_action = None  # single env: (action_dim,) one-hot
        self.prev_actions_batch = None  # batch: (n_envs, action_dim) one-hot
        
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
        self.prev_action = torch.zeros(self.action_dim, device=device)
    
    def reset_rnn_states_batch(self, n_envs: int, device=None):
        """Reset RNN states for multiple environments (batch mode)"""
        if device is None: device = self.policy_net.output_layer.weight.device
        self.n_envs = n_envs
        self.rnn_states_policy_batch = torch.zeros(n_envs, self.recurrent_N, self.hidden_dim, device=device)
        self.rnn_states_value_batch = torch.zeros(n_envs, self.recurrent_N, self.hidden_dim, device=device)
        self.prev_actions_batch = torch.zeros(n_envs, self.action_dim, device=device)

    def get_action(self, obs: torch.Tensor, masks: torch.Tensor = None, available_actions: torch.Tensor = None) -> Tuple[int, float, Optional[torch.Tensor]]:
        with torch.no_grad():
            obs_in = obs.unsqueeze(0) if obs.dim() == 1 else obs
            
            if masks is None:
                batch_size = obs_in.size(0)
                masks_in = torch.ones(batch_size, 1, device=self.policy_net.output_layer.weight.device)
                return_rnn_state = False
            else:
                masks_in = masks.unsqueeze(0) if masks.dim() == 1 else masks
                return_rnn_state = True

            curr_rnn_state = self.rnn_state_policy.unsqueeze(0)
            
            if available_actions is not None:
                available_actions = available_actions.unsqueeze(0) if available_actions.dim() == 1 else available_actions

            prev_action_onehot = self.prev_action.unsqueeze(0)

            action, log_prob, next_rnn_state = self.policy_net.get_action_and_log_prob(
                obs_in, 
                curr_rnn_state, 
                masks_in,
                available_actions,
                prev_actions=prev_action_onehot
            )
            
            self.rnn_state_policy = next_rnn_state.squeeze(0)
            self.prev_action = F.one_hot(action, self.action_dim).float().squeeze(0)
            
        if return_rnn_state:
            return action.item(), log_prob.item(), curr_rnn_state.squeeze(0).clone()
        else:
            return action.item(), log_prob.item()

    def get_value(self, obs: torch.Tensor, masks: torch.Tensor, prev_action_onehot: torch.Tensor = None) -> Tuple[float, torch.Tensor]:
        with torch.no_grad():
            curr_rnn_state = self.rnn_state_value.unsqueeze(0)
            if prev_action_onehot is None:
                prev_action_onehot = self.prev_action.unsqueeze(0)
            value, next_rnn_state = self.value_net(
                obs.unsqueeze(0),
                curr_rnn_state,
                masks.unsqueeze(0),
                prev_actions=prev_action_onehot
            )
            self.rnn_state_value = next_rnn_state.squeeze(0)
        return value.item(), curr_rnn_state.squeeze(0).clone()
    
    def get_action_batch(
        self,
        obs_batch: torch.Tensor,
        masks_batch: torch.Tensor,
        available_actions_batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get actions for a batch of observations (for parallel environments)."""
        with torch.no_grad():
            prev_states = self.rnn_states_policy_batch
            prev_actions_onehot = self.prev_actions_batch

            actions, log_probs, next_states = self.policy_net.get_action_and_log_prob(
                obs_batch,
                prev_states,
                masks_batch,
                available_actions_batch,
                prev_actions=prev_actions_onehot
            )

            prev_states_copy = prev_states.clone()
            self.rnn_states_policy_batch = next_states
            self.prev_actions_batch = F.one_hot(actions, self.action_dim).float()

        return actions, log_probs, prev_states_copy
    
    def get_value_batch(
        self,
        obs_batch: torch.Tensor,
        masks_batch: torch.Tensor,
        prev_actions_onehot: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get values for a batch of observations (for parallel environments)."""
        with torch.no_grad():
            prev_states = self.rnn_states_value_batch

            if prev_actions_onehot is None:
                prev_actions_onehot = self.prev_actions_batch

            values, next_states = self.value_net(
                obs_batch,
                prev_states,
                masks_batch,
                prev_actions=prev_actions_onehot
            )

            prev_states_copy = prev_states.clone()
            self.rnn_states_value_batch = next_states

        return values.squeeze(-1), prev_states_copy

class qPCIAlgorithm:
    """
    Q-PCI Algorithm: Uses ValueBridgeNetwork to predict action values (Q-values)
    based on current action, confounder (other agents' obs), and covariate (own obs, hidden state).
    
    Key difference from r_pci:
    - OutcomeBridgeNetwork predicts one-step rewards
    - ValueBridgeNetwork predicts returns (value targets)
    - Individual policy updates use value predictions instead of value network
    """
    def __init__(
        self,
        env,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        recurrent_N: int = 1,
        lr_actor: float = 3e-4,
        lr_bridge: float = 3e-4,
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
        train_naivepredictor: bool = False,
        use_value_parameter_sharing: bool = True,
        rnn_chunk_length: int = 16,
        use_episode_buffer: bool = False,
        use_centralized_critic: bool = False,
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
        self.train_naivepredictor = train_naivepredictor
        self.use_value_parameter_sharing = use_value_parameter_sharing
        self.use_episode_buffer = use_episode_buffer
        self.use_centralized_critic = use_centralized_critic

        self.clip_delta = clip_delta
        self.causal_lambda = causal_lambda
        self.causal_lambda_schedule = causal_lambda_schedule
        self.causal_lambda_start = causal_lambda_start
        self.causal_lambda_final = causal_lambda_final
        self.causal_lambda_warmup_steps = causal_lambda_warmup_steps
        self.causal_lambda_startingpoint = causal_lambda_startingpoint
        self.bridge_epoch = bridge_epoch

        self.training_steps = 0
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.num_batches = num_batches
        
        if self.use_episode_buffer:
            self.buffer_manager = EpisodeBuffer(max_episodes=max_buffer_size)
        else:
            self.buffer_manager = TransitionBuffer(max_buffer_size=max_buffer_size)
        
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = qPCIAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                recurrent_N=recurrent_N,
                lr_actor=lr_actor,
                lr_critic=lr_critic
            )
        
        # Value network 설정
        if self.use_centralized_critic:
            self.value_net = RNNValueNetwork(
                obs_dim=obs_dim * num_agents,
                hidden_dim=hidden_dim,
                recurrent_N=recurrent_N,
                out_dim=num_agents,
                action_dim=action_dim,
                num_agents=num_agents
            )
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)
            self.rnn_state_value = None
            self.value_nets = {agent_id: self.value_net for agent_id in range(num_agents)}
            self.value_optimizers = {agent_id: self.value_optimizer for agent_id in range(num_agents)}
            print(f'[Q-PCI] Using MAPPO-style centralized critic (input_dim={obs_dim * num_agents}, out_dim={num_agents})')

        elif self.use_value_parameter_sharing:
            self.shared_value_net = RNNValueNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim, recurrent_N=recurrent_N, action_dim=action_dim, num_agents=1)
            self.shared_value_optimizer = optim.Adam(self.shared_value_net.parameters(), lr=lr_critic)
            
            for agent_id, agent in self.agents.items(): 
                self.agents[agent_id].value_net = self.shared_value_net
                self.agents[agent_id].value_optimizer = self.shared_value_optimizer 

            self.value_nets = {agent_id: self.shared_value_net for agent_id in range(num_agents)}
            self.value_optimizers = {agent_id: self.shared_value_optimizer for agent_id in range(num_agents)}

        else:
            self.value_nets = {agent_id: agent.value_net for agent_id, agent in self.agents.items()}
            self.value_optimizers = {agent_id: agent.value_optimizer for agent_id, agent in self.agents.items()}
        
        x_dim = obs_dim + (num_agents-1) * hidden_dim * recurrent_N 
        
        # [KEY CHANGE] ValueBridgeNetwork for predicting action values instead of rewards
        self.outcome_bridge_net = OutcomeBridgeNetwork( 
            action_dim=action_dim,
            num_agents=num_agents,
            w_dim=obs_dim * (num_agents-1),
            outcome_dim=num_agents,
            hidden_dim=hidden_dim,
            action_embedding_dim=8,
            x_dim=x_dim
        )
        
        self.action_bridge_net = ActionBridgeNetwork(
            z_dim=obs_dim * num_agents,
            action_dim=action_dim,
            num_agents=num_agents,
            hidden_dim=hidden_dim,
            action_embedding_dim=8
        )

        self.feature_projector = nn.Sequential(
            nn.Linear(num_agents*obs_dim + action_dim + x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        for parmam in self.feature_projector.parameters():
            parmam.requires_grad = False

        self.critic_kernel_w = RBFKernel(gamma=1.0).to(self.device)
        self.critic_kernel_z = RBFKernel(gamma=1.0).to(self.device)
        self.kernel_z_block = RBFKernel(gamma=1.0).to(self.device)
        self.kernel_a_block = RBFKernel(gamma=1.0).to(self.device)
        
        self.kernel_xobs_block = RBFKernel(gamma=1.0).to(self.device)
        self.kernel_xhid_block = RBFKernel(gamma=1.0).to(self.device)
        self.kernel_xact_block = RBFKernel(gamma=1.0).to(self.device)
        
        self.stabilizer_lambda = 1.0
        self.tikhonov_gamma = 1e-4

        self.outcome_bridge_optimizer = optim.Adam(self.outcome_bridge_net.parameters(), lr=lr_bridge)
        self.action_bridge_optimizer = optim.Adam(self.action_bridge_net.parameters(), lr=lr_critic)
        
        if self.train_naivepredictor:
            self.naive_predictor = NaivePredictor(
                action_dim=action_dim,
                num_agents=num_agents,
                w_dim=self.w_dim,
                x_dim=x_dim,
                hidden_dim=hidden_dim
            )
            self.naive_optimizer = optim.Adam(self.naive_predictor.parameters(), lr=lr_bridge)
        
        self.last_dones = {i: False for i in range(num_agents)}
        
        self._move_networks_to_device()
    
    def _move_networks_to_device(self):
        for agent in self.agents.values():
            agent.policy_net.to(self.device)
            if not self.use_value_parameter_sharing and not self.use_centralized_critic:
                agent.value_net.to(self.device)
        if self.use_centralized_critic:
            self.value_net.to(self.device)
        elif self.use_value_parameter_sharing:
            self.shared_value_net.to(self.device)
        self.outcome_bridge_net.to(self.device)
        self.action_bridge_net.to(self.device)
        self.critic_kernel_z.to(self.device)
        self.critic_kernel_w.to(self.device)
        self.feature_projector.to(self.device)
        if getattr(self, "train_naivepredictor", False):
            self.naive_predictor.to(self.device)
        self.kernel_z_block.to(self.device)
        self.kernel_a_block.to(self.device)
        self.kernel_xobs_block.to(self.device)
        self.kernel_xhid_block.to(self.device)
        self.kernel_xact_block.to(self.device)
    
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
    
    def update_outcome_bridge_networks(self, batch_data: List[Dict]) -> Dict[str, float]:
        """
        [KEY METHOD] Train ValueBridgeNetwork to predict Monte Carlo returns.

        Key points:
        - Y = Monte Carlo returns (pure discounted sum of rewards: r_t + gamma*r_{t+1} + ...)
        - NO bootstrap from value function estimates
        - This learns the TRUE expected return given state and action
        """
        if self.get_current_causal_lambda() == 0.0:
            return {'outcome_bridge_loss': 0.0, 'action_bridge_loss': 0.0}
        
        batch_size = len(batch_data)
        num_agents = self.num_agents
        device = self.device

        # === 1. Data Preparation (Vectorized) ===
        joint_o_prev = torch.stack([self._concatenate_observations(t['bold_o_t_minus_1']) for t in batch_data]).to(device) # Default Z
        joint_r_prev = torch.tensor([[t['bold_r_t_minus_1'][i] for i in range(num_agents)] for t in batch_data], dtype=torch.float32, device=device) # [SWITCH] prev rewards
        # joint_a_prev = torch.tensor([[t['bold_a_t_minus_1'][i] for i in range(num_agents)] for t in batch_data], dtype=torch.long, device=device)
        obs_batch = torch.stack([torch.stack(list(t['bold_o_t'].values())) for t in batch_data]).to(device)
        action_batch = torch.tensor([[t['bold_a_t'][i] for i in range(num_agents)] for t in batch_data], dtype=torch.long, device=device)
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
        obs_others_super = expanded_obs[mask].view(batch_size * num_agents, -1)

        A_onehot = F.one_hot(action_batch, self.action_dim).float()
        expanded_A = A_onehot.unsqueeze(1).expand(-1, num_agents, -1, -1)
        mask_A = ~torch.eye(num_agents, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.action_dim)
        A_others_super = expanded_A[mask_A].view(batch_size * num_agents, -1)

        W_super = obs_others_super # Default W
        # W_super = torch.cat([obs_others_super, A_others_super], dim=1) # W = complements obs and action
        
        # [KEY CHANGE] Extract returns instead of rewards for value targets
        return_batch = torch.tensor([[t['returns'][i] for i in range(num_agents)] for t in batch_data], dtype=torch.float32, device=device)
        Y_super = return_batch.repeat_interleave(num_agents, dim=0)  # (batch_size * num_agents, num_agents)

        # Constructing X_super = {o^i_t, h^{-i}_{t-1}}
        # 1. Gather all hidden states: (B, N, RecN*H)
        h_list = []
        for i in range(num_agents):
            if i in hidden_states_batch:
                h_list.append(hidden_states_batch[i].view(batch_size, -1))
            else:
                h_list.append(torch.zeros(batch_size, self.recurrent_N * self.hidden_dim, device=device))
        H_all = torch.stack(h_list, dim=1) # (B, N, RecN*H)

        # 2. Concat obs and hidden states
        ########################################################################################
        ## Option 1. x = {o^i_t, h^{-i}_{t-1}}
        expanded_H = H_all.unsqueeze(1).expand(-1, num_agents, -1, -1) # (B, N, N, RecN*H)
        mask_H = ~torch.eye(num_agents, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.recurrent_N * self.hidden_dim)
        H_others_super = expanded_H[mask_H].view(batch_size * num_agents, -1) # (B*N, (N-1)*RecN*H)
        obs_super = obs_batch.view(-1, self.obs_dim)
        X_super = torch.cat([obs_super, H_others_super], dim=1)

        ## Option 2. x = {o^i_t, \mathbf{h}_{t-1}}
        # H_all_expanded = H_all.unsqueeze(1).expand(-1, num_agents, -1, -1)  # (B, N, N, RecN*H)
        # H_all_super = H_all_expanded.reshape(batch_size * num_agents, -1)   # (B*N, N*RecN*H)
        # obs_super = obs_batch.view(-1, self.obs_dim)
        # X_super = torch.cat([obs_super, H_all_super], dim=1)

        ## Option 3. x = {o^i_t, \mathbf{h}_{t-1}, a^{-i}_{t}}
        # H_all_expanded = H_all.unsqueeze(1).expand(-1, num_agents, -1, -1)  # (B, N, N, RecN*H)
        # H_all_super = H_all_expanded.reshape(batch_size * num_agents, -1)   # (B*N, N*RecN*H)
        # obs_super = obs_batch.view(-1, self.obs_dim)
        # X_super = torch.cat([obs_super, H_all_super, A_others_super], dim=1)
        ########################################################################################
        
        Z_super = joint_o_prev.repeat_interleave(num_agents, dim=0) # Default Z
        # Z_super = joint_r_prev.repeat_interleave(num_agents, dim=0) # [SWITCH] prev rewards
        # Z_super = torch.cat([joint_o_prev, joint_r_prev], dim=1).repeat_interleave(num_agents, dim=0)
        # Z_super = torch.cat([joint_o_prev, joint_a_prev_onehot], dim=1).repeat_interleave(num_agents, dim=0)
        
        
        with torch.no_grad():
            log_probs_list = []
            for i in range(num_agents):
                obs_i = obs_batch[:, i, :]
                act_i = action_batch[:, i]
                
                if i in hidden_states_batch:                    
                    p_state = hidden_states_batch[i]
                    masks_i = torch.ones_like(dones_batch[:, i]).unsqueeze(1)
                    lp, _ = self.agents[i].policy_net.evaluate_actions(obs_i, p_state, act_i, masks_i)
                else:
                    bs = obs_i.size(0)
                    dummy_state = torch.zeros(bs, self.recurrent_N, self.hidden_dim, device=device)
                    dummy_mask = torch.ones(bs, 1, device=device)
                    lp, _ = self.agents[i].policy_net.evaluate_actions(obs_i, dummy_state, act_i, dummy_mask)

                log_probs_list.append(lp)
            
            policy_log_probs = torch.stack(log_probs_list, dim=1).view(-1)
            policy_probs_super = torch.exp(policy_log_probs) 
            
        obs_super = obs_batch.view(-1, self.obs_dim)

        total_samples = A_super.size(0) 
        indices = torch.arange(total_samples, device=device)

        avg_outcome_bridge_loss = 0.0
        avg_action_loss = 0.0
        avg_naive_loss = 0.0

        sum_off_mean = 0.0
        sum_off_std = 0.0
        num_updates = 0
        chunk_size = self.causal_mini_batch_size 

        for i in range(0, total_samples, chunk_size):
            idx = indices[i:i + chunk_size]
            current_batch_size = idx.size(0)
            
            a_mb = A_super[idx]; w_mb = W_super[idx]; y_mb = Y_super[idx]; z_mb = Z_super[idx]; x_mb = X_super[idx]; o_mb = obs_super[idx]
            pi_mb = policy_probs_super[idx]
            batch_agent_ids = idx % num_agents

            eps = 1e-6
            
            joint_obs_dim = self.obs_dim * self.num_agents
            if z_mb.size(1) == joint_obs_dim + self.num_agents:
                z_obs_norm = (z_mb[:, :joint_obs_dim] - z_mb[:, :joint_obs_dim].mean(0, keepdim=True)) \
                            / z_mb[:, :joint_obs_dim].std(0, keepdim=True, unbiased=False).clamp_min(eps)
                z_rew_norm = (z_mb[:, joint_obs_dim:] - z_mb[:, joint_obs_dim:].mean(0, keepdim=True)) \
                            / z_mb[:, joint_obs_dim:].std(0, keepdim=True, unbiased=False).clamp_min(eps)
                z_mb_norm = torch.cat([z_obs_norm, z_rew_norm], dim=1)
            else: 
                z_mean = z_mb.mean(dim=0, keepdim=True)
                z_std = z_mb.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
                z_mb_norm = (z_mb - z_mean) / z_std

            x_obs_mb_raw = x_mb[:, :self.obs_dim]
            x_hid_mb_raw = x_mb[:, self.obs_dim:self.obs_dim + self.recurrent_N * self.hidden_dim]

            x_obs_mean = x_obs_mb_raw.mean(dim=0, keepdim=True)
            x_obs_std = x_obs_mb_raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
            x_obs_mb = (x_obs_mb_raw - x_obs_mean) / x_obs_std

            x_hid_mean = x_hid_mb_raw.mean(dim=0, keepdim=True)
            x_hid_std = x_hid_mb_raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
            x_hid_mb = (x_hid_mb_raw - x_hid_mean) / x_hid_std

            x_mb_norm = torch.cat([x_obs_mb, x_hid_mb], dim=1)

            action_onehot_mb = F.one_hot(a_mb.long(), self.action_dim).float()
            
            with torch.no_grad():
                for feat, kernel in [
                    (z_mb_norm,        self.kernel_z_block),
                    (action_onehot_mb, self.kernel_a_block),
                    (x_obs_mb,         self.kernel_xobs_block),
                    (x_hid_mb,         self.kernel_xhid_block),
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
                off_mean = K_z[off_mask].mean().item()
                off_std = K_z[off_mask].std(unbiased=False).item()
            sum_off_mean += off_mean
            sum_off_std += off_std

            reg_K_z = self.tikhonov_gamma * torch.eye(current_batch_size, device=device) + self.stabilizer_lambda * K_z
            try:
                torch.linalg.solve(reg_K_z, y_mb)
            except RuntimeError:
                reg_K_z = reg_K_z + 1e-3 * torch.eye(current_batch_size, device=device)
            
            # [KEY CHANGE] Train outcome_bridge_net to predict action values (returns)
            epoch_outcome_bridge_loss = 0.0
            for _ in range(self.bridge_epoch):
                q_hat_mb = self.outcome_bridge_net(a_mb, w_mb, x=x_mb)
                residuals_mb = y_mb - q_hat_mb
                
                solved_residuals = torch.linalg.solve(reg_K_z, residuals_mb)
                per_sample_loss_h = (residuals_mb * (K_z @ solved_residuals)).sum(dim=1)
                
                loss_h = per_sample_loss_h.mean()
                loss_q = torch.tensor(0.0, device=device)

                total_loss = loss_h + loss_q
                
                self.outcome_bridge_optimizer.zero_grad()
                self.action_bridge_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.outcome_bridge_net.parameters(), self.max_grad_norm)
                self.outcome_bridge_optimizer.step()
                
                epoch_outcome_bridge_loss += loss_h.item()

            avg_outcome_bridge_loss += epoch_outcome_bridge_loss / self.bridge_epoch
            avg_action_loss += loss_q.item()
            
            if getattr(self, "train_naivepredictor", False):
                naive_output = self.naive_predictor(a_mb, w_mb, x_mb)
                naive_loss = F.mse_loss(naive_output, y_mb)
                self.naive_optimizer.zero_grad()
                naive_loss.backward()
                self.naive_optimizer.step()
                avg_naive_loss += naive_loss.item()
                
            num_updates += 1

        return_dict = {
            'outcome_bridge_loss': avg_outcome_bridge_loss / num_updates,
            'action_bridge_loss': avg_action_loss / num_updates,
            'Kz_offdiag_mean': sum_off_mean / num_updates,
            'Kz_offdiag_std': sum_off_std / num_updates,
        }
        
        if getattr(self, "train_naivepredictor", False):
            return_dict['naive_loss'] = avg_naive_loss / num_updates

        return return_dict

    def _compute_kw2_sum_minibatch(self, w_mb, kernel_feat_w, o_mb, batch_agent_ids, M):
        batch_size = w_mb.size(0)
        
        w_expanded = w_mb.repeat_interleave(M, dim=0)
        
        sampled_action_onehots = torch.zeros(
            batch_size * M, 
            self.action_dim, 
            device=self.device
        )
        
        with torch.no_grad():
            unique_agents = torch.unique(batch_agent_ids)
            for ag_id in unique_agents:
                mask = (batch_agent_ids == ag_id)
                if not mask.any(): continue
                
                obs_ag = o_mb[mask]
                curr_subset_size = obs_ag.size(0)
                
                obs_exp = obs_ag.repeat_interleave(M, dim=0)
                
                dummy_state = torch.zeros(curr_subset_size * M, self.recurrent_N, self.hidden_dim, device=self.device)
                dummy_mask = torch.ones(curr_subset_size * M, 1, device=self.device)
                
                logits, _ = self.agents[ag_id.item()].policy_net(obs_exp, dummy_state, dummy_mask)
                
                dist = torch.distributions.Categorical(logits=logits)
                actions_sampled = dist.sample()
                onehots = F.one_hot(actions_sampled, self.action_dim).float()
                
                mask_expanded = mask.unsqueeze(1).expand(-1, M).reshape(-1)
                sampled_action_onehots[mask_expanded] = onehots

        features_w_sampled = torch.cat([w_expanded, sampled_action_onehots], dim=1)
        
        K_w2_raw = self.critic_kernel_w.forward_asymmetric(kernel_feat_w, features_w_sampled)
        
        K_w2_mean = K_w2_raw.view(batch_size, batch_size, M).mean(dim=2)
        K_w2_sum = K_w2_mean.sum(dim=1).unsqueeze(1)
        
        return K_w2_sum

    def _concatenate_observations(self, observations: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Helper function to concatenate observations for all agents."""
        sorted_obs = [observations[i] for i in range(self.num_agents) if i in observations]
        if not sorted_obs:
            device = next(self.agents[0].policy_net.parameters()).device
            return torch.zeros(self.num_agents * self.obs_dim, device=device)
        return torch.cat(sorted_obs, dim=-1)
    
    def reset_agents(self):
        """Reset agent hidden states."""
        for agent in self.agents.values():
            agent.reset_rnn_state(device=self.device)
        if self.use_centralized_critic:
            self.rnn_state_value = torch.zeros(self.recurrent_N, self.hidden_dim).to(self.device)
        self.last_dones = {i: False for i in range(self.num_agents)}
    
    def reset_agents_batch(self, n_envs: int):
        """Reset agent hidden states for batch mode (parallel environments)."""
        for agent in self.agents.values():
            agent.reset_rnn_states_batch(n_envs, device=self.device)
        if self.use_centralized_critic:
            self.rnn_states_value_batch = torch.zeros(
                n_envs, self.recurrent_N, self.hidden_dim, device=self.device
            )
        self.last_dones_batch = {i: [False] * n_envs for i in range(self.num_agents)}
   
    def get_actions(self, observations: Dict[int, torch.Tensor], available_actions: Optional[Dict[int, List[int]]] = None) -> Dict[int, Tuple[int, float]]:
        actions = {}
        self.temp_hidden_states = {}
        self.temp_masks = {}
        self.temp_log_probs = {}
        
        if self.use_centralized_critic:
            mask0 = 1.0 if not self.last_dones.get(0, False) else 0.0
            mask_tensor0 = torch.tensor([mask0], dtype=torch.float32, device=self.device)
            global_obs = self._concatenate_observations(observations)
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
                _, v_state = self.agents[agent_id].get_value(
                    obs, 
                    mask_tensor, 
                    prev_action_onehot=agent_prev_action_onehot
                )
            
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
        """Get actions for a batch of observations (parallel environments)."""
        n_envs = len(observations_batch)
        actions_batch = [{} for _ in range(n_envs)]
        
        self.temp_hidden_states_batch = [{} for _ in range(n_envs)]
        self.temp_masks_batch = [{} for _ in range(n_envs)]
        
        if self.use_centralized_critic:
            global_obs_list = []
            for i in range(n_envs):
                global_obs_list.append(self._concatenate_observations(observations_batch[i]))
            global_obs_batch_tensor = torch.stack(global_obs_list).to(self.device)
            
            global_prev_actions_list = []
            for i in range(self.num_agents):
                global_prev_actions_list.append(self.agents[i].prev_actions_batch)
            global_prev_action_onehot_batch = torch.cat(global_prev_actions_list, dim=-1)
            
            central_masks = torch.tensor([
                [1.0 if not self.last_dones_batch[0][env_idx] else 0.0]
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
                observations_batch[env_idx][agent_id]
                for env_idx in range(n_envs)
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
                obs_stacked,
                masks_stacked,
                avail_actions_stacked
            )
            
            if not self.use_centralized_critic:
                _, v_states = self.agents[agent_id].get_value_batch(
                    obs_stacked,
                    masks_stacked,
                    prev_actions_onehot=agent_prev_actions_onehot
                )
            
            for env_idx in range(n_envs):
                actions_batch[env_idx][agent_id] = (
                    actions[env_idx].item(),
                    log_probs[env_idx].item()
                )
                if self.use_centralized_critic:
                    v_state_for_env = curr_states_value[env_idx]
                else:
                    v_state_for_env = v_states[env_idx].clone()
                self.temp_hidden_states_batch[env_idx][agent_id] = (
                    p_states[env_idx].clone(),
                    v_state_for_env
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
        available_actions: Optional[Dict[int, List[int]]] = None,
        env_id: int = 0
    ):
        self.last_dones = dones.copy()
        
        kwargs = {
            'prev_observations': prev_observations,
            'prev_rewards': prev_rewards,
            'prev_actions': prev_actions,
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_observations,
            'dones': dones,
            'masks': self.temp_masks,
            'hidden_states': self.temp_hidden_states, 
            'available_actions': available_actions,
            'log_probs': self.temp_log_probs,
            'info': info
        }
        
        if self.use_episode_buffer:
             kwargs['env_id'] = env_id
             
        self.buffer_manager.store_transition(**kwargs)
    
    def store_transition_from_env(
        self,
        env_idx: int,
        prev_observations: Dict[int, torch.Tensor],
        prev_rewards: Dict[int, float],
        prev_actions: Dict[int, int],
        observations: Dict[int, torch.Tensor],
        actions: Dict[int, Tuple[int, float]],
        rewards: Dict[int, float],
        next_observations: Dict[int, torch.Tensor],
        dones: Dict[int, bool],
        info: Dict = None,
        available_actions: Optional[Dict[int, List[int]]] = None
    ):
        """Store transition from a specific parallel environment."""
        actions_only = {agent_id: action for agent_id, (action, _) in actions.items()}
        log_probs_dict = {agent_id: lp for agent_id, (_, lp) in actions.items()}
        
        hidden_states = self.temp_hidden_states_batch[env_idx]
        masks = self.temp_masks_batch[env_idx]
        
        if info is None:
            info = {}
        info['env_id'] = env_idx
        
        kwargs = {
            'prev_observations': prev_observations,
            'prev_rewards': prev_rewards,
            'prev_actions': prev_actions,
            'observations': observations,
            'actions': actions_only,
            'rewards': rewards,
            'next_observations': next_observations,
            'dones': dones,
            'masks': masks,
            'hidden_states': hidden_states,
            'available_actions': available_actions,
            'log_probs': log_probs_dict,
            'info': info
        }
        
        if self.use_episode_buffer:
             kwargs['env_id'] = env_idx
             
        self.buffer_manager.store_transition(**kwargs)
        
        if self.use_episode_buffer:
            if any(dones.values()):
                self.buffer_manager.finish_episode(env_id=env_idx)
        
        for agent_id in range(self.num_agents):
            self.last_dones_batch[agent_id][env_idx] = dones[agent_id]
            
            if dones[agent_id]:
                self.agents[agent_id].rnn_states_policy_batch[env_idx] = torch.zeros(
                    self.recurrent_N, self.hidden_dim, device=self.device
                )
                self.agents[agent_id].prev_actions_batch[env_idx] = torch.zeros(
                    self.action_dim, device=self.device
                )
                if not self.use_centralized_critic:
                    self.agents[agent_id].rnn_states_value_batch[env_idx] = torch.zeros(
                        self.recurrent_N, self.hidden_dim, device=self.device
                    )
        
        if self.use_centralized_critic and any(dones.values()):
            self.rnn_states_value_batch[env_idx] = torch.zeros(
                self.recurrent_N, self.hidden_dim, device=self.device
            )

    def finish_episode(self, env_id: int = 0):
        """Finish current episode if using EpisodeBuffer."""
        if self.use_episode_buffer:
            self.buffer_manager.finish_episode(env_id=env_id)
    
    def compute_monte_carlo_returns(
        self,
        rewards: List[float],
        dones: List[bool]
    ) -> List[float]:
        """
        Compute Monte Carlo returns (pure discounted sum of rewards).

        This computes the TRUE value target without any bootstrap:
        G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

        Args:
            rewards: List of rewards for the trajectory
            dones: List of done flags

        Returns:
            List of Monte Carlo returns for each timestep
        """
        returns = []
        G = 0.0

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
            else:
                next_non_terminal = 1.0 - dones[step]

            # Pure Monte Carlo: G_t = r_t + gamma * G_{t+1}
            G = rewards[step] + self.gamma * G * next_non_terminal
            returns.insert(0, G)

        return returns

    def compute_monte_carlo_returns_vectorized(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Monte Carlo returns (vectorized for all agents).

        Args:
            rewards: (T, N) tensor of rewards
            dones: (T, N) tensor of done flags

        Returns:
            (T, N) tensor of Monte Carlo returns
        """
        T, N = rewards.shape
        returns = torch.zeros_like(rewards)
        G = torch.zeros(N, device=rewards.device)

        for step in reversed(range(T)):
            next_non_terminal = 1.0 - dones[step]
            G = rewards[step] + self.gamma * G * next_non_terminal
            returns[step] = G

        return returns

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float,
        dones: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """Helper function to compute Generalized Advantage Estimation."""
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

    def _compute_counterfactual_values_vectorized(
        self,
        transition: Dict
    ) -> torch.Tensor:
        """
        Compute counterfactual value baselines V_cf^i(s_t) for ALL agents (vectorized).

        Returns:
            v_cf: Counterfactual value baselines for all agents (num_agents,)
        """
        with torch.no_grad():
            device = self.device
            num_agents = self.num_agents
            action_dim = self.action_dim

            # 1. Gather all observations
            obs_all = torch.stack([
                transition['bold_o_t'][i].to(device) if isinstance(transition['bold_o_t'][i], torch.Tensor)
                else torch.tensor(transition['bold_o_t'][i], dtype=torch.float32, device=device)
                for i in range(num_agents)
            ])  # (N, obs_dim)

            # 2. Construct w for all agents: (N, (N-1)*obs_dim)
            w_all = torch.stack([
                torch.cat([obs_all[j] for j in range(num_agents) if j != i])
                for i in range(num_agents)
            ])  # (N, (N-1)*obs_dim)

            # 3. Construct x for all agents: (N, obs_dim + (N-1)*recurrent_N*hidden_dim)
            x_all = torch.stack([
                torch.cat([
                    obs_all[i]] +
                    [
                        transition['hidden_states'][j][0].flatten().to(device)
                        if 'hidden_states' in transition and j in transition['hidden_states']
                        else torch.zeros(self.recurrent_N * self.hidden_dim, device=device)
                        for j in range(num_agents) if j != i
                    ]
                )
                for i in range(num_agents)
            ])

            # 4. Get policy probabilities for all agents: (N, action_dim)
            probs_all = []
            for i in range(num_agents):
                obs_i = obs_all[i].unsqueeze(0)

                if 'hidden_states' in transition and i in transition['hidden_states']:
                    rnn_state = transition['hidden_states'][i][0]
                    if not isinstance(rnn_state, torch.Tensor):
                        rnn_state = torch.tensor(rnn_state, dtype=torch.float32, device=device)
                    else:
                        rnn_state = rnn_state.to(device)
                    rnn_state = rnn_state.unsqueeze(0)
                else:
                    rnn_state = torch.zeros(1, self.recurrent_N, self.hidden_dim, device=device)

                mask = torch.ones(1, 1, device=device)

                prev_action = torch.zeros(1, action_dim, device=device)
                if 'bold_a_t_minus_1' in transition and i in transition['bold_a_t_minus_1']:
                    prev_a = transition['bold_a_t_minus_1'][i]
                    prev_action = F.one_hot(torch.tensor(prev_a, device=device).long(), action_dim).float().unsqueeze(0)

                avail_actions = None
                if 'available_actions' in transition and transition['available_actions'] is not None and i in transition['available_actions']:
                    a = transition['available_actions'][i]
                    avail_actions = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)

                logits, _ = self.agents[i].policy_net(obs_i, rnn_state, mask, available_actions=avail_actions, prev_actions=prev_action)
                probs_all.append(F.softmax(logits, dim=-1).squeeze(0))

            probs_all = torch.stack(probs_all, dim=0)  # (N, action_dim)

            # 5. Counterfactual value estimate for each agent
            action_range = torch.arange(action_dim, device=device)

            v_cf_all = []
            for i in range(num_agents):
                # Evaluate all possible actions for agent i
                w_i = w_all[i].unsqueeze(0).repeat(action_dim, 1)
                x_i = x_all[i].unsqueeze(0).repeat(action_dim, 1)
                q_all = self.outcome_bridge_net(action_range, w_i, x_i)  # (action_dim, num_agents)
                q_sum_over_agents = q_all.sum(dim=1)  # (action_dim,)

                # Counterfactual policy for actions (exclude factual action)
                factual_action = transition['bold_a_t'][i]
                mask = torch.ones(action_dim, device=device)
                mask[factual_action] = 0.0

                probs_cf = probs_all[i] * mask

                if 'available_actions' in transition and transition['available_actions'] is not None and i in transition['available_actions']:
                    avail = transition['available_actions'][i]
                    avail_tensor = avail if isinstance(avail, torch.Tensor) else torch.tensor(avail, dtype=torch.float32, device=device)
                    probs_cf = probs_cf * avail_tensor

                probs_cf = probs_cf / (probs_cf.sum() + 1e-10)
                v_cf_all.append((probs_cf * q_sum_over_agents).sum())

            return torch.stack(v_cf_all)  # (N,)

    def compute_counterfactual_gae_vectorized(
        self,
        trajectory: List[Dict],
        next_bootstrap: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Counterfactual-Aware GAE for ALL agents (vectorized).

        Args:
            trajectory: List of transitions
            next_bootstrap: Bootstrap values V(s_{T+1}) for all agents (num_agents,)

        Returns:
            advantages: Counterfactual advantages for all agents (T, num_agents)
            returns: Target returns (not used, just for compatibility) (T, num_agents)
        """
        T = len(trajectory)
        num_agents = self.num_agents
        device = self.device

        if next_bootstrap is None:
            next_bootstrap = torch.zeros(num_agents, device=device)

        # Initialize GAE for all agents
        gae = torch.zeros(num_agents, device=device)
        advantages_all = []

        for t in reversed(range(T)):
            # 1. Get done flags
            dones = torch.tensor([trajectory[t]['dones'][i] for i in range(num_agents)], dtype=torch.float32, device=device)
            next_non_terminal = 1.0 - dones

            # 2. Get factual values at t for all agents
            v_factual_t = torch.tensor([trajectory[t]['values'][i] for i in range(num_agents)], device=device)

            # 3. Compute counterfactual baselines at t for all agents (vectorized!)
            v_cf_t = self._compute_counterfactual_values_vectorized(trajectory[t])

            # 4. Mix factual and counterfactual baselines with causal_lambda
            # shaped_value_t = V_factual - λ*V_cf
            current_lambda = self.get_current_causal_lambda()
            v_mixed_t = v_factual_t - current_lambda * v_cf_t

            # 5. Compute shaped_value_{t+1} (also mix factual and counterfactual)
            if t == T - 1:
                # Last timestep: use bootstrap value (assumed to be shaped already or zero)
                v_mixed_next = next_bootstrap
            else:
                # Get factual values at t+1
                v_factual_next = torch.tensor([trajectory[t+1]['values'][i] for i in range(num_agents)], device=device)

                # Compute counterfactual baselines at t+1
                v_cf_next = self._compute_counterfactual_values_vectorized(trajectory[t+1])

                # shaped_value_{t+1} = V_factual_{t+1} - λ*V_cf_{t+1}
                v_mixed_next = v_factual_next - current_lambda * v_cf_next

            # 6. Get GLOBAL rewards for all agents (sum of all agents' rewards)
            global_reward_t = sum([trajectory[t]['bold_r_t'][i] for i in range(num_agents)])
            rewards_t = torch.full((num_agents,), global_reward_t, device=device)  # Same global reward for all agents

            # 7. Shaped TD error for all agents
            # delta = r + γ * shaped_value_{t+1} - shaped_value_t
            delta_mixed = rewards_t + self.gamma * v_mixed_next * next_non_terminal - v_mixed_t

            # 8. GAE accumulation for all agents
            gae = delta_mixed + self.gamma * self.gae_lambda * next_non_terminal * gae

            advantages_all.insert(0, gae.clone())

        advantages_all = torch.stack(advantages_all)  # (T, num_agents)

        # Returns are not needed since we use Monte Carlo returns
        returns_dummy = torch.zeros_like(advantages_all)

        return advantages_all, returns_dummy

    def _classify_delta_scenario(self, transition: Dict, agent_id: int) -> str:
        """Helper function to classify delta scenario."""
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

    def _chunk_sequence(self, tensor: torch.Tensor, chunk_len: int) -> torch.Tensor:
        """Reshape (T, ...) tensor into (T//chunk_len, chunk_len, ...)."""
        T = tensor.size(0)
        num_chunks = T // chunk_len
        if num_chunks == 0:
            return tensor.unsqueeze(1)
        
        tensor = tensor[:num_chunks * chunk_len]
        shape = (num_chunks, chunk_len) + tensor.shape[1:]
        tensor = tensor.view(shape)
        
        permute_dims = (1, 0) + tuple(range(2, tensor.dim()))
        return tensor.permute(permute_dims).contiguous()

    def update_agents(self, batch_data: List[Dict]) -> Dict[str, float]:
        """Standard PPO agent update (same as r_pci)."""
        agent_losses = defaultdict(float)

        full_obs_list = [torch.stack(list(t['bold_o_t'].values())) for t in batch_data]
        full_obs_batch = torch.stack(full_obs_list).to(self.device, non_blocking=True)
        T, N, D = full_obs_batch.shape
        
        actions_batch = torch.tensor([[t['bold_a_t'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.long, device=self.device)
        
        available_actions_batch = {}
        if 'available_actions' in batch_data[0] and batch_data[0]['available_actions'] is not None:
            for agent_id in range(self.num_agents):
                if agent_id in batch_data[0]['available_actions']:
                     avail_list = [t['available_actions'][agent_id] for t in batch_data]
                     available_actions_batch[agent_id] = torch.tensor(avail_list, dtype=torch.float32, device=self.device)

        if 'hidden_states' in batch_data[0] and batch_data[0]['hidden_states'] is not None:
             init_hidden_states = batch_data[0]['hidden_states']
        else:
             init_hidden_states = {i: (torch.zeros(self.recurrent_N, self.hidden_dim), torch.zeros(self.recurrent_N, self.hidden_dim)) for i in range(self.num_agents)}
        
        masks_batch = {}
        for agent_id in range(self.num_agents):
            masks_batch[agent_id] = torch.tensor([t['masks'][agent_id] for t in batch_data], device=self.device).unsqueeze(1)
        
        agent_returns = {}
        agent_value_returns = {}
        agent_advantages = {}

        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue

            returns_list = [t['returns'][agent_id] for t in batch_data]  # Individual rewards for outcome bridge
            value_returns_list = [t['value_returns'][agent_id] for t in batch_data]  # Global rewards for value network
            advantages_list = [t['advantages'][agent_id] for t in batch_data]

            agent_returns[agent_id] = torch.tensor(returns_list, dtype=torch.float32, device=self.device)
            agent_value_returns[agent_id] = torch.tensor(value_returns_list, dtype=torch.float32, device=self.device)
            agent_advantages[agent_id] = torch.tensor(advantages_list, dtype=torch.float32, device=self.device)

        for i in range(self.num_agents):
            for t, transition in enumerate(batch_data):
                if 'delta' in transition and i in transition['delta']:
                    if self.env.__class__.__name__ == 'StagHuntEnv':
                        scenario = self._classify_delta_scenario(transition, i)
                        agent_losses[f'delta_agent_{i}_{scenario}_sum'] += transition['delta'][i]
                        agent_losses[f'delta_agent_{i}_{scenario}_count'] += 1
                    else:
                        others_rewards_sum = sum([transition['bold_r_t'][j] for j in range(self.num_agents) if j != i])
                        if others_rewards_sum == 0:
                            agent_losses[f'delta_agent{i}_no_reward_sum'] += transition['delta'][i]
                            agent_losses[f'delta_agent{i}_no_reward_count'] += 1
                        else:
                            agent_losses[f'delta_agent{i}_positive_reward_sum'] += transition['delta'][i]
                            agent_losses[f'delta_agent{i}_positive_reward_count'] += 1

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

            p_states_all = torch.stack([t['hidden_states'][agent_id][0] for t in batch_data]).to(self.device)
            v_states_all = torch.stack([t['hidden_states'][agent_id][1] for t in batch_data]).to(self.device)
            
            avail_actions_all = None
            if agent_id in available_actions_batch:
                avail_actions_all = available_actions_batch[agent_id]

            T_total = agent_obs.size(0)
            chunk_len = self.rnn_chunk_length
            num_chunks = T_total // chunk_len
            
            if num_chunks > 0:
                rnn_obs_chunked = self._chunk_sequence(agent_obs, chunk_len)
                my_actions_chunked = self._chunk_sequence(my_actions, chunk_len) 
                rnn_masks_main = masks_batch[agent_id]
                rnn_masks_chunked = self._chunk_sequence(rnn_masks_main, chunk_len) 
                
                prev_actions_raw = torch.tensor(
                    [t['bold_a_t_minus_1'][agent_id] for t in batch_data[:num_chunks * chunk_len]], 
                    dtype=torch.long, device=self.device
                )
                prev_actions_onehot_seq = F.one_hot(prev_actions_raw, self.action_dim).float()
                prev_actions_chunked = self._chunk_sequence(prev_actions_onehot_seq, chunk_len)
                
                if 'active_mask' in batch_data[0]:
                    active_masks = torch.tensor([t['active_mask'] for t in batch_data], device=self.device).unsqueeze(1)
                    active_masks_chunked = self._chunk_sequence(active_masks, chunk_len)
                else:
                    active_masks = torch.ones(T_total, 1, device=self.device)
                    active_masks_chunked = self._chunk_sequence(active_masks, chunk_len)

                avail_actions_chunked = None
                if avail_actions_all is not None:
                    avail_actions_chunked = self._chunk_sequence(avail_actions_all, chunk_len)
                
                chunk_indices = torch.arange(0, num_chunks * chunk_len, chunk_len)
                p_h0_chunked = p_states_all[chunk_indices] 
                v_h0_chunked = v_states_all[chunk_indices]
                
                adv_chunked = self._chunk_sequence(adv, chunk_len)
                ret_chunked = self._chunk_sequence(agent_value_returns[agent_id], chunk_len)  # Use global rewards
                
                p_h0_batch = p_h0_chunked
                v_h0_batch = v_h0_chunked
                rnn_obs_batch = rnn_obs_chunked
                rnn_masks_batch = rnn_masks_chunked
                my_actions_batch = my_actions_chunked.reshape(-1)
                adv_batch = adv_chunked.reshape(-1)
                ret_batch = ret_chunked.reshape(-1)
                active_masks_batch = active_masks_chunked.reshape(-1)
                prev_actions_batch = prev_actions_chunked
                prev_actions_batch_merged = prev_actions_chunked.reshape(-1, self.action_dim)
                
                avail_actions_batch_merged = None
                if avail_actions_chunked is not None:
                     avail_actions_batch_merged = avail_actions_chunked.reshape(-1, avail_actions_chunked.size(-1))

            else:
                rnn_obs_batch = agent_obs.unsqueeze(1)
                rnn_masks_batch = masks_batch[agent_id]
                p_h0_batch, _ = init_hidden_states[agent_id]
                p_h0_batch = p_h0_batch.to(self.device).unsqueeze(0)
                _, v_h0_batch = init_hidden_states[agent_id]
                v_h0_batch = v_h0_batch.to(self.device).unsqueeze(0)
                my_actions_batch = my_actions
                adv_batch = adv
                ret_batch = agent_value_returns[agent_id]  # Use global rewards
                active_masks_batch = active_masks.squeeze(1) if 'active_mask' in batch_data[0] else torch.ones(T_total, device=self.device)
                avail_actions_batch_merged = avail_actions_all
                prev_actions_raw = torch.tensor(
                    [t['bold_a_t_minus_1'][agent_id] for t in batch_data], 
                    dtype=torch.long, device=self.device
                )
                prev_actions_onehot_seq = F.one_hot(prev_actions_raw, self.action_dim).float()
                prev_actions_batch = prev_actions_onehot_seq.unsqueeze(1)
                prev_actions_batch_merged = prev_actions_onehot_seq
            
            old_log_probs_raw = torch.tensor([t['log_probs'][agent_id] for t in batch_data], dtype=torch.float32, device=self.device)
            
            if num_chunks > 0:
                old_log_probs = self._chunk_sequence(old_log_probs_raw, chunk_len).reshape(-1)
            else:
                old_log_probs = old_log_probs_raw
                
            for epoch in range(self.ppo_epochs):
                new_log_probs, entropy = self.agents[agent_id].policy_net.evaluate_actions(
                    rnn_obs_batch, p_h0_batch, my_actions_batch, rnn_masks_batch, 
                    available_actions=avail_actions_batch_merged, prev_actions=prev_actions_batch
                )
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
                
                policy_loss = -torch.min(surr1, surr2)
                policy_loss = (policy_loss * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)

                entropy_loss = -entropy
                entropy_loss = (entropy_loss * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-8)
                
                loss = policy_loss + self.entropy_coef * entropy_loss
                
                self.agents[agent_id].optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[agent_id].policy_net.parameters(), self.max_grad_norm)
                self.agents[agent_id].optimizer.step()
                
                agent_losses[f'agent_{agent_id}_policy_loss'] += policy_loss.item()
                agent_losses[f'agent_{agent_id}_entropy_loss'] += entropy_loss.item()
                
                if not self.use_centralized_critic:
                    new_vals, _ = self.agents[agent_id].value_net(
                        rnn_obs_batch, v_h0_batch, rnn_masks_batch, 
                        prev_actions=prev_actions_batch
                    )
                    
                    val_error = (new_vals.squeeze(1) - ret_batch) ** 2
                    val_loss = (val_error * active_masks_batch).sum()
                    
                    if not self.use_value_parameter_sharing:
                        mean_val_loss = val_loss / (active_masks_batch.sum() + 1e-8)
                        self.agents[agent_id].value_optimizer.zero_grad()
                        (self.value_coef * mean_val_loss).backward()
                        torch.nn.utils.clip_grad_norm_(self.agents[agent_id].value_net.parameters(), self.max_grad_norm)
                        self.agents[agent_id].value_optimizer.step()
                        agent_losses[f'agent_{agent_id}_value_loss'] += mean_val_loss.item()
                    else:
                        if not hasattr(self, '_temp_epoch_value_loss'):
                            self._temp_epoch_value_loss = 0.0
                            self._temp_epoch_mask_sum = 0.0
                        self._temp_epoch_value_loss += val_loss
                        self._temp_epoch_mask_sum += active_masks_batch.sum()
            
            if not self.use_centralized_critic and self.use_value_parameter_sharing:
                if hasattr(self, '_temp_epoch_mask_sum') and self._temp_epoch_mask_sum > 0:
                    mean_val_loss = self._temp_epoch_value_loss / self._temp_epoch_mask_sum
                    self.shared_value_optimizer.zero_grad()
                    (self.value_coef * mean_val_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.shared_value_net.parameters(), self.max_grad_norm)
                    self.shared_value_optimizer.step()
                    
                    agent_losses['value_loss'] += mean_val_loss.item()
                    
                    del self._temp_epoch_value_loss
                    del self._temp_epoch_mask_sum

        if self.use_centralized_critic:
            global_obs_seq = torch.stack([self._concatenate_observations(t['bold_o_t']) for t in batch_data]).to(self.device)
            global_masks_seq = torch.tensor([t['masks'][0] for t in batch_data], device=self.device).unsqueeze(1)
            
            v_states_all_central = torch.stack([t['hidden_states'][0][1] for t in batch_data]).to(self.device)
            
            # Use value_returns (global rewards) for centralized critic
            all_returns_list = [[t['value_returns'][i] for i in range(self.num_agents)] for t in batch_data]
            all_returns = torch.tensor(all_returns_list, dtype=torch.float32, device=self.device)

            T_total = global_obs_seq.size(0)
            chunk_len = self.rnn_chunk_length
            num_chunks = T_total // chunk_len
            
            if num_chunks > 0:
                global_obs_batch = self._chunk_sequence(global_obs_seq, chunk_len)
                global_masks_batch = self._chunk_sequence(global_masks_seq, chunk_len)
                
                if 'active_mask' in batch_data[0]:
                    active_masks = torch.tensor([t['active_mask'] for t in batch_data], device=self.device).unsqueeze(1)
                    active_masks_chunked = self._chunk_sequence(active_masks, chunk_len)
                    active_masks_batch = active_masks_chunked.reshape(-1)
                else:
                    active_masks_batch = torch.ones(num_chunks * chunk_len, device=self.device)
                
                chunk_indices = torch.arange(0, num_chunks * chunk_len, chunk_len)
                v_h0_batch = v_states_all_central[chunk_indices]
                global_ret_batch = self._chunk_sequence(all_returns, chunk_len).reshape(-1, self.num_agents)
            else:
                global_obs_batch = global_obs_seq.unsqueeze(1)
                global_masks_batch = global_masks_seq
                active_masks_batch = torch.ones(T_total, device=self.device)
                v_h0_batch = v_states_all_central[0].unsqueeze(0)
                global_ret_batch = all_returns
            
            for _ in range(self.ppo_epochs):
                agent_prev_actions_list = []
                for agent_id in range(self.num_agents):
                    agent_prev_actions = torch.tensor([t['bold_a_t_minus_1'][agent_id] for t in batch_data], dtype=torch.long, device=self.device)
                    agent_prev_actions_onehot = F.one_hot(agent_prev_actions, self.action_dim).float()
                    agent_prev_actions_list.append(agent_prev_actions_onehot)
                
                joint_prev_actions_batch_raw = torch.cat(agent_prev_actions_list, dim=-1)
                
                if num_chunks > 0:
                    global_prev_actions_batch = self._chunk_sequence(joint_prev_actions_batch_raw, chunk_len)
                else:
                    global_prev_actions_batch = joint_prev_actions_batch_raw.unsqueeze(1)

                new_vals, _ = self.value_net(global_obs_batch, v_h0_batch, global_masks_batch, prev_actions=global_prev_actions_batch)
                val_error = (new_vals.squeeze(1) - global_ret_batch) ** 2
                val_loss = (val_error * active_masks_batch.unsqueeze(1)).sum() / (active_masks_batch.sum() * self.num_agents + 1e-8)
                
                self.value_optimizer.zero_grad()
                (self.value_coef * val_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                agent_losses['value_loss'] += val_loss.item()

        for k in list(agent_losses.keys()):
            if 'delta' not in k:
                if k == 'value_loss':
                    if self.use_centralized_critic or self.use_value_parameter_sharing:
                        agent_losses['value_loss'] /= self.ppo_epochs
                    else:
                        total_value_loss = sum([agent_losses.get(f'agent_{i}_value_loss', 0.0) for i in range(self.num_agents)])
                        agent_losses['value_loss'] = total_value_loss / (self.num_agents * self.ppo_epochs)
                else:
                    agent_losses[k] /= self.ppo_epochs
            elif 'sum' in k:
                if 'delta' in k:
                    continue

                count_key = k.replace('sum', 'count')
                if agent_losses[count_key] > 0:
                    agent_losses[k.replace('sum', 'mean')] = agent_losses[k] / agent_losses[count_key]
                del agent_losses[k]
        
        return dict(agent_losses)

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

        for traj in trajectories:
            if len(traj) == 0:
                continue
            
            if self.use_centralized_critic:
                T = len(traj)
                global_obs_list = [self._concatenate_observations(t['bold_o_t']) for t in traj]
                global_obs_seq = torch.stack(global_obs_list).to(self.device, non_blocking=True)
                global_masks_seq = torch.tensor([t['masks'][0] for t in traj], device=self.device).unsqueeze(1)

                v_h0 = traj[0]['hidden_states'][0][1].to(self.device, non_blocking=True).unsqueeze(0)

                with torch.no_grad():
                    last_next_obs = self._concatenate_observations(traj[-1]['next_observations']).to(self.device, non_blocking=True)
                    last_mask = 1.0 - (1.0 if any(traj[-1]['dones'].values()) else 0.0)
                    
                    extended_obs = torch.cat([global_obs_seq.unsqueeze(1),
                                             last_next_obs.unsqueeze(0).unsqueeze(1)], dim=0)
                    extended_masks = torch.cat([global_masks_seq,
                                               torch.tensor([[last_mask]], device=self.device)], dim=0)
                    
                    agent_prev_actions_list = []
                    for agent_id in range(self.num_agents):
                        agent_prev_actions = torch.tensor([t['bold_a_t_minus_1'][agent_id] for t in traj], dtype=torch.long, device=self.device)
                        agent_prev_actions_onehot = F.one_hot(agent_prev_actions, self.action_dim).float()
                        last_prev_action_onehot = agent_prev_actions_onehot[-1].unsqueeze(0)
                        extended_agent_prev_actions = torch.cat([agent_prev_actions_onehot, last_prev_action_onehot], dim=0)
                        agent_prev_actions_list.append(extended_agent_prev_actions)
                    
                    joint_prev_actions = torch.cat(agent_prev_actions_list, dim=-1).unsqueeze(1)
                    
                    all_values, _ = self.value_net(extended_obs, v_h0, extended_masks, prev_actions=joint_prev_actions)
                    values = all_values[:-1].view(-1, self.num_agents)
                    next_val = all_values[-1].view(self.num_agents) * last_mask
                
                # Store values for all agents
                for agent_id in range(self.num_agents):
                    if agent_id not in traj[0]['bold_o_t']:
                        continue
                    for t_idx, t in enumerate(traj):
                        if 'values' not in t:
                            t['values'] = {}
                        t['values'][agent_id] = values[t_idx, agent_id].item()

                # Compute counterfactual advantages (vectorized for all agents)
                next_bootstrap_tensor = next_val if isinstance(next_val, torch.Tensor) else torch.tensor([next_val] * self.num_agents, device=self.device)
                advantages_all, _ = self.compute_counterfactual_gae_vectorized(
                    traj,
                    next_bootstrap=next_bootstrap_tensor
                )  # (T, num_agents)

                # Prepare all rewards and dones as tensors (T, N)
                T = len(traj)
                all_rewards = torch.tensor(
                    [[t['bold_r_t'][i] for i in range(self.num_agents)] for t in traj],
                    dtype=torch.float32, device=self.device
                )  # (T, N)
                all_dones = torch.tensor(
                    [[t['dones'][i] for i in range(self.num_agents)] for t in traj],
                    dtype=torch.float32, device=self.device
                )  # (T, N)

                # Compute global rewards (sum over agents) for value network
                global_rewards = all_rewards.sum(dim=1)  # (T,)
                global_dones = all_dones.max(dim=1)[0]  # (T,) - any agent done

                # Compute Monte Carlo returns vectorized
                # For individual agents (outcome bridge)
                individual_mc_returns = self.compute_monte_carlo_returns_vectorized(all_rewards, all_dones)  # (T, N)

                # For global rewards (value network)
                global_mc_returns = self.compute_monte_carlo_returns_vectorized(
                    global_rewards.unsqueeze(1), global_dones.unsqueeze(1)
                ).squeeze(1).tolist()

                # Store all computed values in trajectory
                for t_idx, t in enumerate(traj):
                    if 'advantages' not in t:
                        t['advantages'] = {}
                        t['returns'] = {}
                        t['value_returns'] = {}
                    for agent_id in range(self.num_agents):
                        if agent_id not in traj[0]['bold_o_t']:
                            continue
                        t['advantages'][agent_id] = advantages_all[t_idx, agent_id].item()
                        t['returns'][agent_id] = individual_mc_returns[t_idx, agent_id].item()
                        t['value_returns'][agent_id] = global_mc_returns[t_idx]
            else:
                # For decentralized critic, we need to compute values for each agent separately
                next_vals = []
                for agent_id in range(self.num_agents):
                    if agent_id not in traj[0]['bold_o_t']:
                        next_vals.append(0.0)
                        continue

                    agent_obs_list = [t['bold_o_t'][agent_id] for t in traj]
                    agent_obs_seq = torch.stack(agent_obs_list).to(self.device, non_blocking=True)
                    agent_masks_seq = torch.tensor([t['masks'][agent_id] for t in traj], device=self.device).unsqueeze(1)

                    _, v_h0 = traj[0]['hidden_states'][agent_id]
                    v_h0 = v_h0.to(self.device, non_blocking=True).unsqueeze(0)

                    with torch.no_grad():
                        last_next_obs = traj[-1]['next_observations'][agent_id].to(self.device, non_blocking=True)
                        last_mask = 1.0 - traj[-1]['dones'][agent_id]

                        extended_obs = torch.cat([agent_obs_seq.unsqueeze(1),
                                                last_next_obs.unsqueeze(0).unsqueeze(1)], dim=0)
                        extended_masks = torch.cat([agent_masks_seq,
                                                    torch.tensor([[last_mask]], dtype=torch.float32, device=self.device)], dim=0)

                        agent_prev_actions = torch.tensor([t['bold_a_t_minus_1'][agent_id] for t in traj], dtype=torch.long, device=self.device)
                        agent_prev_actions_onehot = F.one_hot(agent_prev_actions, self.action_dim).float()
                        last_prev_action_onehot = agent_prev_actions_onehot[-1].unsqueeze(0)
                        extended_agent_prev_actions = torch.cat([agent_prev_actions_onehot, last_prev_action_onehot], dim=0).unsqueeze(1)

                        all_values, _ = self.agents[agent_id].value_net(extended_obs, v_h0, extended_masks, prev_actions=extended_agent_prev_actions)
                        values = all_values[:-1].view(-1)
                        next_val = all_values[-1].item() * last_mask

                        # Store values for counterfactual GAE
                        for t_idx, t in enumerate(traj):
                            if 'values' not in t:
                                t['values'] = {}
                            t['values'][agent_id] = values[t_idx].item()

                        next_vals.append(next_val)

                # Compute counterfactual advantages (vectorized for all agents)
                next_bootstrap_tensor = torch.tensor(next_vals, device=self.device)
                advantages_all, _ = self.compute_counterfactual_gae_vectorized(
                    traj,
                    next_bootstrap=next_bootstrap_tensor
                )  # (T, num_agents)

                # Prepare all rewards and dones as tensors (T, N)
                T = len(traj)
                all_rewards = torch.tensor(
                    [[t['bold_r_t'][i] for i in range(self.num_agents)] for t in traj],
                    dtype=torch.float32, device=self.device
                )  # (T, N)
                all_dones = torch.tensor(
                    [[t['dones'][i] for i in range(self.num_agents)] for t in traj],
                    dtype=torch.float32, device=self.device
                )  # (T, N)

                # Compute global rewards (sum over agents) for value network
                global_rewards = all_rewards.sum(dim=1)  # (T,)
                global_dones = all_dones.max(dim=1)[0]  # (T,) - any agent done

                # Compute Monte Carlo returns vectorized
                # For individual agents (outcome bridge)
                individual_mc_returns = self.compute_monte_carlo_returns_vectorized(all_rewards, all_dones)  # (T, N)

                # For global rewards (value network)
                global_mc_returns = self.compute_monte_carlo_returns_vectorized(
                    global_rewards.unsqueeze(1), global_dones.unsqueeze(1)
                ).squeeze(1).tolist()

                # Store all computed values in trajectory
                for t_idx, t in enumerate(traj):
                    if 'advantages' not in t:
                        t['advantages'] = {}
                        t['returns'] = {}
                        t['value_returns'] = {}
                    for agent_id in range(self.num_agents):
                        if agent_id not in traj[0]['bold_o_t']:
                            continue
                        t['advantages'][agent_id] = advantages_all[t_idx, agent_id].item()
                        t['returns'][agent_id] = individual_mc_returns[t_idx, agent_id].item()
                        t['value_returns'][agent_id] = global_mc_returns[t_idx]
        
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
                end = start + chunk_len
                chunk = traj[start:end]
                all_chunks.append(chunk)
            
            if remainder > 0:
                last_chunk = traj[num_full_chunks * chunk_len:]
                
                padding_len = chunk_len - remainder
                last_valid_trans = last_chunk[-1]
                
                padding_chunk = []
                for _ in range(padding_len):
                    pad_t = last_valid_trans.copy() 
                    pad_t['active_mask'] = 0.0
                    padding_chunk.append(pad_t)
                
                full_last_chunk = last_chunk + padding_chunk
                all_chunks.append(full_last_chunk)

        n_chunks_per_batch = max(1, len(all_chunks) // self.num_batches)
        chunk_batches = [all_chunks[i:i + n_chunks_per_batch] 
                        for i in range(0, len(all_chunks), n_chunks_per_batch)]
        
        batches = []
        for b in chunk_batches:
            flat_batch = []
            for chunk in b:
                flat_batch.extend(chunk)
            if flat_batch:
                batches.append(flat_batch)
            
        if self.use_episode_buffer:
            self.buffer_manager.clear_completed_episodes()
        else:
            self.buffer_manager.clear_buffer()
            
        total_losses = defaultdict(float)
        
        for batch_data in batches:
            if not batch_data: continue
            
            if self.get_current_causal_lambda() != 0.0:
                outcome_bridge_losses = self.update_outcome_bridge_networks(batch_data)
            else:
                outcome_bridge_losses = {'outcome_bridge_loss': 0.0, 'action_bridge_loss': 0.0}
            
            agent_and_value_losses = self.update_agents(batch_data)

            for loss_name, loss_value in outcome_bridge_losses.items():
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
                continue
            else:
                 if num > 0:
                    final_losses[k] = v / num
                 else:
                    final_losses[k] = v
        
        return final_losses

    def _debug_causal_effects(
        self,
        q_hat_all: torch.Tensor,
        actions_flat: torch.Tensor,
        q_hat_factual: torch.Tensor,
        baseline: torch.Tensor,
        raw_effects: torch.Tensor,
        action_dim: int,
        alive_mask: torch.Tensor = None
    ):
        """Debug print for causal effect analysis."""
        import numpy as np
        np.set_printoptions(precision=4, suppress=True, linewidth=200)

        total = actions_flat.size(0)
        if alive_mask is None:
            alive_mask = torch.ones(total, dtype=torch.bool, device=actions_flat.device)
        n_alive = alive_mask.sum().item()
        n_dead = total - n_alive

        print(f"\n{'='*80}")
        print(f"[Causal Debug] samples={total}, alive={n_alive}, dead={n_dead}")
        print(f"[Causal Debug] Value Bridge Predictions (Q-hat) per action (alive only)")
        print(f"  q_hat_all shape: {q_hat_all.shape}  (batch*agents, action_dim, num_agents)")
        if n_alive > 0:
            q_hat_alive = q_hat_all[alive_mask]
            q_hat_mean = q_hat_alive.detach().mean(dim=0).cpu().numpy()
        else:
            q_hat_mean = q_hat_all.detach().mean(dim=0).cpu().numpy()
        action_labels = {0: 'noop', 1: 'stop', 2: 'north', 3: 'south', 4: 'east', 5: 'west'}
        for a_idx in range(min(action_dim, q_hat_mean.shape[0])):
            label = action_labels.get(a_idx, f'attack_{a_idx-6}') if a_idx < 6 else f'attack_{a_idx-6}'
            print(f"  action {a_idx:2d} ({label:>8s}): q_hat mean = {q_hat_mean[a_idx]}")
        
        print(f"\n[Causal Debug] Baseline & Raw Effects by action group (alive only)")
        alive_actions = actions_flat[alive_mask]
        alive_q_factual = q_hat_factual[alive_mask]
        alive_baseline = baseline[alive_mask]
        alive_raw_effects = raw_effects[alive_mask]

        move_mask = alive_actions < 6
        attack_mask = alive_actions >= 6
        if move_mask.any():
            print(f"  Movement (0~5): count={move_mask.sum().item()}")
            print(f"    q_factual mean: {alive_q_factual[move_mask].mean(dim=0).detach().cpu().numpy()}")
            print(f"    baseline   mean: {alive_baseline[move_mask].mean(dim=0).detach().cpu().numpy()}")
            print(f"    raw_effect mean: {alive_raw_effects[move_mask].mean(dim=0).detach().cpu().numpy()}")
        if attack_mask.any():
            print(f"  Attack   (6~ ): count={attack_mask.sum().item()}")
            print(f"    q_factual mean: {alive_q_factual[attack_mask].mean(dim=0).detach().cpu().numpy()}")
            print(f"    baseline   mean: {alive_baseline[attack_mask].mean(dim=0).detach().cpu().numpy()}")
            print(f"    raw_effect mean: {alive_raw_effects[attack_mask].mean(dim=0).detach().cpu().numpy()}")
        print(f"{'='*80}")

    def clear_buffer(self):
        self.buffer_manager.clear_buffer()
    
    def get_buffer_info(self) -> Dict[str, int]:
        return self.buffer_manager.get_buffer_info()
