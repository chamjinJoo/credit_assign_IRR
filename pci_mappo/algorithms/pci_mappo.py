import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

from ..networks import PolicyNetwork, BridgeNetwork, ActionBridgeNetwork, ValueNetwork
from ..utils.buffer import ExperienceBuffer


class RBFKernel(nn.Module):
    """
    An improved RBF Kernel implementation.
    - `forward` is simplified for efficiency and consistency.
    - Includes a helper method to set `gamma` dynamically based on data.
    """
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        # gamma can be initialized but it's best to set it from data.
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma)), requires_grad=False)

    def set_gamma(self, x: torch.Tensor):
        """
        Sets the gamma parameter based on the median heuristic for the input data.
        This should be called periodically (e.g., once per update) with a batch of features.
        
        Args:
            x: A batch of feature vectors, shape [batch_size, feature_dim].
        """
        with torch.no_grad():
            # We need at least 2 points to compute pairwise distances.
            if x.size(0) < 2:
                return

            # pdist computes pairwise distances and returns a condensed 1D vector.
            dists_sq = torch.pdist(x, p=2).pow(2)
            
            # The median provides a robust estimate of the typical distance scale.
            median_dist_sq = dists_sq.median()
            
            # The heuristic is gamma = 1 / (2 * median_dist^2)
            gamma = 1.0 / (2 * median_dist_sq + 1e-8) # Add epsilon for numerical stability
            
            # Update the non-trainable parameter.
            self.log_gamma.data.fill_(torch.log(gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the symmetric Gram matrix K(x, x).
        This is a special case of the asymmetric forward pass.
        """
        # Reusing the asymmetric method is cleaner and often more efficient.
        return self.forward_asymmetric(x, x)

    def forward_asymmetric(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Computes the asymmetric kernel matrix K(x1, x2) = exp(-gamma * ||x1 - x2||^2).
        This implementation is efficient and numerically stable.
        
        Args:
            x1: Tensor of shape [batch_size_1, feature_dim]
            x2: Tensor of shape [batch_size_2, feature_dim]
            
        Returns:
            Kernel matrix of shape [batch_size_1, batch_size_2]
        """
        gamma = torch.exp(self.log_gamma)
        # Compute squared Euclidean distances using the broadcast trick
        x1_norm_sq = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm_sq = x2.pow(2).sum(dim=-1, keepdim=True)
        # dists = ||x1||^2 - 2*<x1, x2> + ||x2||^2
        dists_sq = x1_norm_sq - 2 * torch.matmul(x1, x2.T) + x2_norm_sq.T
        # Clamp for numerical stability before applying exp
        dists_sq = torch.clamp(dists_sq, min=0)
        return torch.exp(-gamma * dists_sq)

class LinearKernel(nn.Module):
    """
    Linear Kernel for RKHS. Models linear relationships.
    K(x, y) = x^T * y
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the symmetric Gram matrix K(x, x) = x * x^T.
        """
        return torch.matmul(x, x.T)

    def forward_asymmetric(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Computes the asymmetric kernel matrix K(x1, x2) = x1 * x2^T.
        
        Args:
            x1: Tensor of shape [batch_size_1, feature_dim]
            x2: Tensor of shape [batch_size_2, feature_dim]
            
        Returns:
            Kernel matrix of shape [batch_size_1, batch_size_2]
        """
        return torch.matmul(x1, x2.T)

class PCIMAPPOAgent:
    """
    Single agent for PCI-MAPPO algorithm.
    Each agent has its own policy network.
    """
    
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Policy network
        self.policy_net = PolicyNetwork(obs_dim, action_dim, hidden_dim)
        
        # Optimizer for the policy network
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
    def get_action(self, obs: torch.Tensor) -> Tuple[int, float]:
        """
        Get action from policy.
        
        Args:
            obs: Observation tensor
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
        """
        with torch.no_grad():
            action, log_prob = self.policy_net.get_action_and_log_prob(obs.unsqueeze(0))
        
        return action.item(), log_prob.item()


class PCIMAPPOAlgorithm:
    """
    PCI-MAPPO Algorithm implementation following the pseudocode.
    z_t = \mathbf{o}_{t-1}, w_t = \mathbf{o}_t
    """
    
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        w_dim: int, 
        hidden_dim: int = 64,
        lr: float = 3e-4,
        lr_bridge: float = 3e-4,
        lr_critic: float = 3e-4,
        causal_lambda: float = 0.1,
        causal_lambda_schedule: bool = False,
        causal_lambda_start: float = 0.0,
        causal_lambda_final: float = 0.2,
        causal_lambda_warmup_steps: int = 10000,
        causal_lambda_startingpoint: int = 0,  # Starting iteration for causal lambda scheduling
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        num_batches: int = 4,  # Number of batches K for training
        # Epsilon greedy parameters for causal data collection
        epsilon_start: float = 0.6,
        epsilon_end: float = 0.0,
        epsilon_decay_steps: int = 1000,
        # Buffer size limits for FIFO queue behavior
        max_buffer_size: int = 10000,  # Maximum size for main buffer
        max_causal_buffer_size: int = 2000,  # Maximum size for causal buffer
        device: str = "cpu",  # Device for computation
        # RKHS Kernel parameters for analytical loss
        rkhs_gamma: float = 1.0,  # Gamma parameter for RBF kernel
        # Environment required_agents for scenario classification
        env_required_agents: Optional[Dict[int, List[int]]] = None,
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.w_dim = w_dim
        
        # Store environment's required_agents for accurate scenario classification
        self.env_required_agents = env_required_agents
        
        # Device setup
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        # Causal lambda scheduling parameters
        self.causal_lambda = causal_lambda
        self.causal_lambda_schedule = causal_lambda_schedule
        self.causal_lambda_start = causal_lambda_start
        self.causal_lambda_final = causal_lambda_final
        self.causal_lambda_warmup_steps = causal_lambda_warmup_steps
        self.causal_lambda_startingpoint = causal_lambda_startingpoint
        self.training_steps = 0  # Track training steps for scheduling
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.num_batches = num_batches
        
        # Epsilon greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.current_step = 0
        
        # Create experience buffer with FIFO queue behavior
        self.buffer_manager = ExperienceBuffer(
            max_buffer_size=max_buffer_size,
            max_causal_buffer_size=max_causal_buffer_size
        )
        
        # Create agents
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = PCIMAPPOAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                lr=lr
            )
        
        # Value netowrk option 1: no parameter sharing
        # self.value_nets = {
        #     i: ValueNetwork(
        #         obs_dim=obs_dim,
        #         hidden_dim=hidden_dim
        #     ) for i in range(num_agents)
        # }
        # # Optimizer per value network
        # self.value_optimizers = {
        #     i: optim.Adam(self.value_nets[i].parameters(), lr=lr) for i in range(num_agents)
        # }

        # Value network option 2: parameter sharing with agent ID embedding PS+id
        self.value_net = ValueNetwork(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
        )
        
        # Optimizer for the shared value network
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)
        
        # Causal inference networks (shared across agents)
        # h_φ(a, w) - Outcome Bridge network
        self.outcome_bridge_net = BridgeNetwork( 
            action_dim=action_dim,
            num_agents=num_agents,
            w_dim=obs_dim * num_agents,
            outcome_dim=num_agents,  # Predict rewards for all agents
            hidden_dim=hidden_dim,
            action_embedding_dim=8  # Action embedding dimension
        )
        
        # q_θ(z, a) - Action Bridge network
        self.action_bridge_net = ActionBridgeNetwork(
            z_dim=obs_dim * num_agents,  # Use concatenated observations as context
            action_dim=action_dim,
            num_agents=num_agents,
            hidden_dim=hidden_dim,
            action_embedding_dim=8  # Action embedding dimension
        )
        
        # RKHS Kernel
        self.critic_kernel_z = RBFKernel(gamma=1.0).to(self.device)
        self.critic_kernel_w = RBFKernel(gamma=1.0).to(self.device)

        # # Linear Kernel
        # self.critic_kernel_z = LinearKernel().to(self.device)
        # self.critic_kernel_w = LinearKernel().to(self.device)
        
        # Stabilizer
        self.stabilizer_lambda = 1.0  # Minimax 게임의 안정화 항 강도
        self.tikhonov_gamma = 1e-4    # 역행렬 계산 시 수치적 안정성을 위한 정규화 항

        # Action embedding for kernel input
        self.action_embedding_for_kernel = nn.Embedding(action_dim, 8)
        
        # Optimizers for causal networks
        self.outcome_bridge_optimizer = optim.Adam(self.outcome_bridge_net.parameters(), lr=lr_bridge)
        self.action_bridge_optimizer = optim.Adam(self.action_bridge_net.parameters(), lr=lr_critic)
        self.kernel_action_embed_optimizer = optim.Adam(self.action_embedding_for_kernel.parameters(), lr=lr_critic)
        
        # (Removed) single centralized value optimizer in favor of per-agent optimizers
        
        # Move all networks to device
        self._move_networks_to_device()
    
    def _move_networks_to_device(self):
        """Move all networks to the specified device."""
        # Move agent networks
        for agent in self.agents.values():
            agent.policy_net.to(self.device)
        
        # Move value network (shared across agents)
        self.value_net.to(self.device)
        self.outcome_bridge_net.to(self.device)
        self.action_bridge_net.to(self.device)
        self.critic_kernel_z.to(self.device)
        self.critic_kernel_w.to(self.device)
        self.action_embedding_for_kernel.to(self.device)
    
    def get_current_causal_lambda(self) -> float:
        """
        Get current causal lambda value based on scheduling.
        If scheduling is enabled, lambda increases from causal_lambda_start to causal_lambda_final over warmup_steps.
        Scheduling starts from causal_lambda_startingpoint iteration.
        """
        if not self.causal_lambda_schedule:
            return self.causal_lambda
        
        # Check if we should start scheduling yet
        if self.training_steps < self.causal_lambda_startingpoint:
            return self.causal_lambda_start
        
        # Calculate effective steps for scheduling (subtract starting point)
        effective_steps = self.training_steps - self.causal_lambda_startingpoint
        
        # Linear warmup from causal_lambda_start to causal_lambda_final
        if effective_steps >= self.causal_lambda_warmup_steps:
            return self.causal_lambda_final
        else:
            # Linear interpolation: causal_lambda_start -> causal_lambda_final over warmup_steps
            progress = effective_steps / self.causal_lambda_warmup_steps
            return self.causal_lambda_start + progress * (self.causal_lambda_final - self.causal_lambda_start)
    
    def update_causal_networks(self, batch_data: List[Dict], mc_samples_k_w2: int = 16) -> Dict[str, float]:
        """
        - The Outcome Bridge loss (for h_φ) correctly extends Eq. (24) to the multi-agent case.
        - The Action Bridge loss (for q_θ) is corrected to include a Monte Carlo approximation
        of the linear term from Eq. (25), making it more theoretically sound.
        
        Args:
            batch_data: A list of transition dictionaries from the mini-batch.
            mc_samples_k_w2 (int): The number of Monte Carlo samples to use for approximating the K_w2 matrix.

        Returns:
            A dictionary containing loss values for monitoring.
        """
        # Skip causal network updates if causal lambda is 0
        if self.get_current_causal_lambda() == 0.0:
            return {
                'outcome_bridge_loss': 0.0,
                'action_bridge_loss': 0.0,
            }
        
        batch_size = len(batch_data)
        # === Vectorized Data Preparation ===
        # Swap: o_{t-1} becomes z_t, o_t becomes w_t
        z_t_batch = torch.stack([self._concatenate_observations(t['bold_o_t_minus_1']) for t in batch_data])
        w_t_batch = torch.stack([self._concatenate_observations(t['bold_o_t']) for t in batch_data])
        a_t_batch = torch.tensor([[t['bold_a_t'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.long, device=self.device)
        r_t_batch = torch.tensor([[t['bold_r_t'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.float32, device=self.device)

        # === Part 1: Update Outcome Bridge Network h_φ ===
        r_hat_outcome = self.outcome_bridge_net(a_t_batch, w_t_batch)
        residuals = r_t_batch - r_hat_outcome
    
        action_embeds = self.action_embedding_for_kernel(a_t_batch).view(batch_size, -1)
        kernel_input_features_z = torch.cat([z_t_batch, action_embeds], dim=1)
        # self.critic_kernel_z.set_gamma(kernel_input_features_z) # for RBF kernel
        K_z = self.critic_kernel_z(kernel_input_features_z) # Shape: [batch_size, batch_size]

        # 1-1. Loss without Stabilizer
        # quadratic_forms = torch.matmul(residuals.T, torch.matmul(K_z, residuals))
        # outcome_bridge_loss = torch.trace(quadratic_forms) / (self.num_agents * batch_size)

        # 1-2. Loss with Stabilizer
        # 이전 버전 (bottleneck of training time)
        # ------------------------------------------------------------
        # reg_K_z_inv = torch.linalg.inv(self.tikhonov_gamma * torch.eye(batch_size, device=K_z.device) + self.stabilizer_lambda * K_z) # (γI + λK_z)^-1
        # loss_matrix_h = residuals.T @ K_z @ reg_K_z_inv @ residuals
        # outcome_bridge_loss = torch.trace(loss_matrix_h) / batch_size
        # ------------------------------------------------------------
        # 역행렬 대신 Cholesky 분해를 사용한 선형 시스템 풀이 (더 빠르고 수치적으로 안정적)
        reg_K_z = self.tikhonov_gamma * torch.eye(batch_size, device=K_z.device) + self.stabilizer_lambda * K_z
        # residuals.T @ K_z @ inv(reg_K_z) @ residuals 계산
        # = (residuals.T @ K_z) @ solve(reg_K_z, residuals)
        K_z_residuals = residuals.T @ K_z  # Shape: [num_agents, batch_size]
        solved_residuals = torch.linalg.solve(reg_K_z, residuals)  # Shape: [batch_size, num_agents]
        loss_matrix_h = K_z_residuals @ solved_residuals  # Shape: [num_agents, num_agents]
        outcome_bridge_loss = torch.trace(loss_matrix_h) / batch_size

        # === Part 2: Update Action Bridge Network q_θ ===
        # 1. Compute joint policy probabilities π(a_t|o_t)
        with torch.no_grad():
            log_probs_list = [self.agents[i].policy_net.evaluate_actions(
                torch.stack([t['bold_o_t'][i] for t in batch_data]), a_t_batch[:, i]
            )[0] for i in range(self.num_agents)]
            joint_probs = torch.exp(torch.stack(log_probs_list, dim=1).sum(dim=1))

        # 2. Compute the weighted IPW term φ_q
        action_bridge_output = self.action_bridge_net(z_t_batch, a_t_batch) # Shape: [batch_size, num_agents]
        phi_q = joint_probs * action_bridge_output.squeeze() # Shape: [batch_size]
        # 3. Calculate Gram matrix K_w1 (Quadratic term)
        kernel_input_features_w = torch.cat([w_t_batch, action_embeds], dim=1)
        # self.critic_kernel_w.set_gamma(kernel_input_features_w) # for RBF kernel
        K_w1 = self.critic_kernel_w(kernel_input_features_w)

        # 4. Vectorized Approximation of Gram matrix K_w2
        with torch.no_grad():
            # a. Expand batch for parallel sampling
            # Shape: [batch_size * M, obs_dim] for each agent
            expanded_obs_per_agent = {
                i: torch.stack([t['bold_o_t'][i] for t in batch_data]).repeat_interleave(mc_samples_k_w2, dim=0)
                for i in range(self.num_agents)
            }

            # b. Sample M actions for each state in the batch in parallel
            # Shape: [batch_size * M, num_agents]
            sampled_actions_list = []
            for i in range(self.num_agents):
                logits = self.agents[i].policy_net(expanded_obs_per_agent[i])
                dist = torch.distributions.Categorical(logits=logits)
                sampled_actions_list.append(dist.sample())
            
            sampled_actions_tensor = torch.stack(sampled_actions_list, dim=1)

            # c. Create feature vectors for all sampled actions
            w_t_expanded = w_t_batch.repeat_interleave(mc_samples_k_w2, dim=0) # Shape: [batch_size * M, w_dim]
            sampled_action_embeds = self.action_embedding_for_kernel(sampled_actions_tensor).view(batch_size * mc_samples_k_w2, -1)
            features_w_sampled = torch.cat([w_t_expanded, sampled_action_embeds], dim=1) # Shape: [batch_size * M, feature_dim]
            
            # d. Compute all kernel values in one go
            # kernel_input_features_w: [batch_size, feature_dim]
            # features_w_sampled: [batch_size * M, feature_dim]
            # K_w2_raw: [batch_size, batch_size * M]
            # self.critic_kernel_w.set_gamma(features_w_sampled) # for RBF kernel
            K_w2_raw = self.critic_kernel_w.forward_asymmetric(kernel_input_features_w, features_w_sampled)
            
            # e. Reshape and average to get the final K_w2
            # Reshape to -> [batch_size, batch_size, M]
            K_w2_reshaped = K_w2_raw.view(batch_size, batch_size, mc_samples_k_w2)
            K_w2 = K_w2_reshaped.mean(dim=2) # Average over the M samples -> [batch_size, batch_size]

        # 5. Compute the full Action Bridge loss (as before)
        phi_q_col = phi_q.unsqueeze(1)
        # 5-1. Without Stabilizer
        # quadratic_term = torch.matmul(phi_q_col.T, torch.matmul(K_w1, phi_q_col)).squeeze()
        # linear_term = torch.matmul(phi_q, K_w2.sum(dim=1))
        # action_bridge_loss = (quadratic_term - 2.0 * linear_term) / batch_size

        # 5-2. With Stabilizer
        # 이전 버전 (bottleneck of training time)
        # ------------------------------------------------------------
        # reg_K_w1_inv = torch.linalg.inv(self.tikhonov_gamma * torch.eye(batch_size, device=K_w1.device) + self.stabilizer_lambda * K_w1)
        # quadratic_term = phi_q_col.T @ K_w1 @ reg_K_w1_inv @ phi_q_col
        # linear_term = phi_q_col.T @ reg_K_w1_inv @ K_w2.sum(dim=1).unsqueeze(1)
        # action_bridge_loss = (quadratic_term - 2.0 * linear_term).squeeze() / batch_size
        # ------------------------------------------------------------
        # 역행렬 대신 선형 시스템 풀이 사용
        reg_K_w1 = self.tikhonov_gamma * torch.eye(batch_size, device=K_w1.device) + self.stabilizer_lambda * K_w1
        # quadratic_term = phi_q_col.T @ K_w1 @ inv(reg_K_w1) @ phi_q_col
        # = (phi_q_col.T @ K_w1) @ solve(reg_K_w1, phi_q_col)
        K_w1_phi_q = phi_q_col.T @ K_w1  # Shape: [1, batch_size]
        solved_phi_q = torch.linalg.solve(reg_K_w1, phi_q_col)  # Shape: [batch_size, 1]
        quadratic_term = K_w1_phi_q @ solved_phi_q  # Shape: [1, 1]

        # linear_term = phi_q_col.T @ inv(reg_K_w1) @ K_w2.sum(dim=1).unsqueeze(1)
        # = phi_q_col.T @ solve(reg_K_w1, K_w2.sum(dim=1).unsqueeze(1))
        K_w2_sum = K_w2.sum(dim=1).unsqueeze(1)  # Shape: [batch_size, 1]
        solved_K_w2_sum = torch.linalg.solve(reg_K_w1, K_w2_sum)  # Shape: [batch_size, 1]
        linear_term = phi_q_col.T @ solved_K_w2_sum  # Shape: [1, 1]

        action_bridge_loss = (quadratic_term - 2.0 * linear_term).squeeze() / batch_size 

        # --- Combine Losses and Backward Pass ---
        # We now have two distinct loss terms for the shared action embedding.
        total_loss = outcome_bridge_loss + action_bridge_loss

        self.outcome_bridge_optimizer.zero_grad()
        self.action_bridge_optimizer.zero_grad()
        self.kernel_action_embed_optimizer.zero_grad()

        total_loss.backward()

        self.outcome_bridge_optimizer.step()
        self.action_bridge_optimizer.step()
        self.kernel_action_embed_optimizer.step()
        
        return {
            'outcome_bridge_loss': outcome_bridge_loss.item(),
            'action_bridge_loss': action_bridge_loss.item(),
        }
    
    def _concatenate_observations(self, observations: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Concatenate all agents' observations into a single tensor.
        
        Args:
            observations: Dict mapping agent_id to observation tensor
            
        Returns:
            Concatenated observations [num_agents * obs_dim]
        """
        # Sort by agent_id to ensure consistent ordering
        sorted_obs = [observations[i] for i in range(self.num_agents) if i in observations]
        
        if not sorted_obs:
            # If no observations, return zero tensor
            device = next(self.agents[0].policy_net.parameters()).device
            return torch.zeros(self.num_agents * self.obs_dim, device=device)
        
        # Concatenate all observations
        return torch.cat(sorted_obs, dim=-1)
    
    def get_current_epsilon(self) -> float:
        """
        Get current epsilon value for epsilon-greedy exploration.
        Epsilon decays linearly from epsilon_start to epsilon_end over epsilon_decay_steps.
        """
        progress = min(self.current_step / self.epsilon_decay_steps, 1.0)
        current_epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
        return current_epsilon
    
    def get_epsilon_greedy_action(self, agent_id: int, obs: torch.Tensor) -> Tuple[int, float]:
        """
        Get action using epsilon-greedy strategy for causal data collection.
        
        Args:
            agent_id: Agent ID
            obs: Observation tensor
        
        Returns:
            action: Selected action (random with probability epsilon, policy action otherwise)
            log_prob: Log probability of the action (from policy network)
        """
        epsilon = self.get_current_epsilon()
        
        # Get policy action first
        policy_action, policy_log_prob = self.agents[agent_id].get_action(obs)
        
        # Epsilon greedy decision
        if np.random.random() < epsilon:
            # Random action
            random_action = np.random.randint(0, self.action_dim)
            # Keep the log prob from policy for consistency in learning
            return random_action, policy_log_prob
        else:
            # Policy action
            return policy_action, policy_log_prob
        
    def reset_agents(self):
        """Reset agents for new episode (no RNN state to reset)."""
        pass
    
    def get_actions(self, observations: Dict[int, torch.Tensor]) -> Dict[int, Tuple[int, float]]:
        """
        Get actions from all agents.
        
        Args:
            observations: Dict mapping agent_id to observation tensor
            
        Returns:
            Dict mapping agent_id to (action, log_prob)
        """
        actions = {}
        
        # Get actions from all agents
        for agent_id, obs in observations.items():
            action, log_prob = self.agents[agent_id].get_action(obs)
            actions[agent_id] = (action, log_prob)
        
        return actions
    
    def get_epsilon_greedy_actions(self, observations: Dict[int, torch.Tensor]) -> Dict[int, Tuple[int, float]]:
        """Get actions from all agents using epsilon-greedy strategy for causal data collection."""
        actions = {}
        
        # Get actions from all agents using epsilon-greedy strategy
        for agent_id, obs in observations.items():
            action, log_prob = self.get_epsilon_greedy_action(agent_id, obs)
            actions[agent_id] = (action, log_prob)
        
        return actions
    
    def store_transition(
        self,
        prev_observations: Dict[int, torch.Tensor],  # o_{t-1}
        observations: Dict[int, torch.Tensor],       # o_t
        actions: Dict[int, int],                     # a_t
        rewards: Dict[int, float],                   # r_t
        next_observations: Dict[int, torch.Tensor],  # o_{t+1}
        dones: Dict[int, bool],
        info: Dict = None
    ):
        """Store transition in buffer with proper causal structure."""
        # Store transition using buffer manager with (o_{t-1}, o_t, a_t, r_t, o_{t+1}) structure
        self.buffer_manager.store_transition(
            prev_observations=prev_observations,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            hidden_states={i: None for i in range(self.num_agents)},
            info=info  # No hidden states needed
        )
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float,
        dones: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation.
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
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
        Delta statistics를 분류하기 위해 현재 transition의 상황을 분석합니다.
        
        required_agents 정보를 활용하여 정확한 협력 관계를 판단:
        - 'stag_reward': 현재 agent가 required_agents에 포함된 stag 사냥에 참여해서 
                        다른 agent가 stag reward를 받음
        - 'forage_reward': 다른 agent가 hare를 사냥함 (협력 없음)
        - 'no_reward': 아무도 reward를 받지 않음
        
        Args:
            transition: 현재 transition 데이터
            agent_id: 분석 대상 agent ID
            
        Returns:
            scenario: 'stag_reward', 'forage_reward', 또는 'no_reward'
        """
        info = transition.get('info', {})
        stag_hunts = info.get('stag_hunts', {})  # {stag_id: [participating_agent_ids]}
        
        # 다른 agent들의 보상 확인
        other_agents_stag_reward = False
        other_agents_forage_reward = False
        
        for other_agent_id in range(self.num_agents):
            if other_agent_id != agent_id:
                if transition['bold_r_t'][other_agent_id] == 5.0:
                    other_agents_stag_reward = True
                elif transition['bold_r_t'][other_agent_id] == 3.0:
                    other_agents_forage_reward = True
        
        # required_agents 정보를 활용한 협력 관계 판단
        if other_agents_stag_reward and self.env_required_agents is not None:
            # Stag 사냥이 성공한 경우, 현재 agent가 해당 stag의 required_agents에 포함되는지 확인
            for stag_id, participants in stag_hunts.items():
                # 이 stag 사냥에 필요한 agents
                required = self.env_required_agents.get(stag_id, [])
                
                # 현재 agent가 required_agents에 포함되어 있는지 확인
                if agent_id in required:
                    # 실제로 참여했는지도 확인
                    if agent_id in participants:
                        # 현재 agent가 협력에 참여해서 다른 agent가 stag reward를 받음
                        return 'stag_reward'
        
        if other_agents_forage_reward:
            # 다른 agent가 hare를 사냥함 (협력 없음)
            return 'forage_reward'
        
        # 아무도 reward를 받지 않음
        return 'no_reward'
    
    def causal_intrinsic_reward(
        self,
        agent_id: int,
        w_t: torch.Tensor,      # Observed proxy variable w_t = o_t
        z_t: torch.Tensor,      # Context variable z_t = o_{t-1}
        o_t: torch.Tensor,      # Observation for agent i
        current_actions: Dict[int, int],  # Current joint action a_t
        current_rewards: Dict[int, float],  # Current joint reward r_t
    ) -> Tuple[float, np.ndarray]:
        """
        Computes a causal reward shaping term Δ^i_t using a policy-weighted difference method.

        This method weights both the factual and counterfactual reward predictions by their
        respective policy probabilities before calculating the difference.

        Args:
            agent_id: ID of the agent i for whom the shaping term is calculated.
            w_t: Proxy w_t, typically the concatenated current observation o_t.
            z_t: Proxy z_t, typically the concatenated previous observation o_(t-1).
            o_t: The individual observation for agent i.
            current_actions: The joint action a_t actually taken at the timestep.
            current_rewards: The joint reward r_t actually received.

        Returns:
            Delta_i_t: The final scalar causal reward shaping term for agent i.
            r_hat_all_numpy: A numpy array of all counterfactual reward predictions for monitoring.
        """
        # Return 0 if causal lambda is 0 (no causal inference)
        if self.get_current_causal_lambda() == 0.0:
            return 0.0, np.zeros((self.action_dim, self.num_agents))
        
        with torch.no_grad():
            # === 1. Compute All Counterfactual Outcomes (including the factual one) ===
            # DR Correction Term (calculated once based on the factual action)
            current_actions_tensor = torch.tensor([current_actions[j] for j in range(self.num_agents)],
                                                dtype=torch.long, device=self.device)
            r_pred_t = self.outcome_bridge_net(current_actions_tensor.unsqueeze(0), w_t.unsqueeze(0))
            current_rewards_tensor = torch.tensor([current_rewards[j] for j in range(self.num_agents)],
                                                dtype=torch.float32, device=self.device)
            error_t = current_rewards_tensor - r_pred_t.squeeze()
            weight_t = self.action_bridge_net(z_t.unsqueeze(0), current_actions_tensor.unsqueeze(0))
            
            DR_correction_t = weight_t * error_t # DR_correction_t: [1, num_agents]
            
            # Create a matrix of all possible counterfactual joint actions for agent i
            action_matrix = current_actions_tensor.unsqueeze(0).repeat(self.action_dim, 1)
            action_matrix[:, agent_id] = torch.arange(self.action_dim, device=self.device)
            w_t_expanded = w_t.unsqueeze(0).expand(self.action_dim, -1)
            # r_hat_all contains predicted reward vectors for each of agent i's possible actions.
            reg_estimates = self.outcome_bridge_net(action_matrix, w_t_expanded) # Shape: [action_dim, num_agents]
            
            r_hat_all = reg_estimates + DR_correction_t  # Shape: [action_dim, num_agents]

            # === 2. Weight Predictions by Policy Probabilities ===
            # Get agent i's current policy probabilities for each action
            action_logits = self.agents[agent_id].policy_net(o_t.unsqueeze(0))
            action_probs = torch.softmax(action_logits, dim=-1).squeeze()  # Shape: [action_dim]

            # Identify the action agent i actually took
            factual_action_idx = current_actions[agent_id]

            # a) Get the weighted reward prediction for the action that was actually taken
            prob_factual = action_probs[factual_action_idx]
            r_hat_factual = r_hat_all[factual_action_idx] # Shape: [num_agents]
            r_hat_factual_weighted = prob_factual * r_hat_all[factual_action_idx] # Shape: [num_agents]
            
            # b) Get the weighted average reward prediction for all other possible actions
            # Create a mask to select all actions EXCEPT the factual one
            mask = torch.ones(self.action_dim, dtype=torch.bool, device=self.device)
            mask[factual_action_idx] = False
            
            r_hat_counterfactuals = r_hat_all[mask] # Shape: [action_dim - 1, num_agents]
            
            # === 3. Compute the Difference-Based Causal Effect ===
            # Baseline using policy probabilities (commented out - not used)
            # probs_counterfactuals = action_probs[mask]  # Shape: [action_dim - 1]
            # sum_probs_counterfactuals = probs_counterfactuals.sum()
            # if sum_probs_counterfactuals > 1e-8:
            #     # weighted_sum = (확률 * 보상).sum()
            #     weighted_sum_counterfactuals = (probs_counterfactuals.unsqueeze(1) * r_hat_counterfactuals).sum(dim=0)
            #     # baseline = weighted_sum / 확률의 합
            #     baseline = weighted_sum_counterfactuals / sum_probs_counterfactuals
            # else:
            #     # 비교할 반사실적 행동이 없는 경우 (행동이 1개뿐인 경우 등)
            #     baseline = torch.zeros_like(r_hat_factual)
            
            # Baseline using simple average (same as causal_intrinsic_reward_batch)
            counterfactual_count = mask.sum().clamp(min=1).to(r_hat_counterfactuals.dtype)
            baseline = r_hat_counterfactuals.sum(dim=0) / (counterfactual_count + 1e-8)  # Shape: [num_agents]
                
            # === 3. 최종 인과 효과 벡터 계산 === #
            causal_effect_vector = r_hat_factual - baseline # Shape: [num_agents]
            # causal_effect_vector = current_rewards - baseline # Shape: [num_agents]
            # === 4. Calculate Final Shaping Term Δ^i_t ===
            # Average the causal effects over all *other* agents.
            other_agents = [j for j in range(self.num_agents) if j != agent_id]
            if other_agents:
                Delta_i_t = causal_effect_vector[other_agents].max().item()
                #Delta_i_t = causal_effect_vector[other_agents].mean().item()
            else:
                Delta_i_t = 0.0
            
            # if agent_id == 0 and current_rewards[2] == 5.0:
            #     Delta_i_t = 5.0
            # elif agent_id == 1 and current_rewards[0] == 5.0:
            #     Delta_i_t = 5.0
            # elif agent_id == 2 and current_rewards[1] == 5.0:
            #     Delta_i_t = 5.0
            # else:
            #     Delta_i_t = 0.0

        # Check bridge networks
        # if current_rewards[0] != 0 or current_rewards[1] != 0 or current_rewards[2] != 0:
        #     print(f"agent_id: {agent_id}")
        #     print(f"current_rewards: {current_rewards}")
        #     print(f"DR_corrections (q): {DR_correction_t}", weight_t.squeeze(), error_t.squeeze())
        #     print(f"reg_estimates (h): {reg_estimates}")
        #     print(f"r_hat_all: {r_hat_all}")
        #     print(f"Delta_i_t: {Delta_i_t}")
        
        return Delta_i_t, r_hat_all.cpu().numpy()
    
    def causal_intrinsic_reward_batch(
        self,
        agent_id: int,
        w_t_batch: torch.Tensor,      # [batch_size, w_dim]
        z_t_batch: torch.Tensor,      # [batch_size, z_dim]
        o_t_batch: torch.Tensor,      # [batch_size, obs_dim]
        actions_batch: torch.Tensor,  # [batch_size, num_agents]
        rewards_batch: torch.Tensor,  # [batch_size, num_agents]
    ) -> torch.Tensor:
        """
        Vectorized batch version of causal_intrinsic_reward.
        
        Args:
            agent_id: ID of the agent i for whom the shaping term is calculated.
            w_t_batch: Batch of proxy w_t, [batch_size, w_dim]
            z_t_batch: Batch of proxy z_t, [batch_size, z_dim]
            o_t_batch: Batch of individual observations for agent i, [batch_size, obs_dim]
            actions_batch: Batch of joint actions, [batch_size, num_agents]
            rewards_batch: Batch of joint rewards, [batch_size, num_agents]
        
        Returns:
            Delta_batch: Causal reward shaping terms, [batch_size]
        """
        # Return zeros if causal lambda is 0
        if self.get_current_causal_lambda() == 0.0:
            return torch.zeros(w_t_batch.size(0), device=self.device)
        
        batch_size = w_t_batch.size(0)
        
        with torch.no_grad():
            # === 1. Compute DR Correction Term for all samples ===
            r_pred_batch = self.outcome_bridge_net(actions_batch, w_t_batch)  # [batch_size, num_agents]
            error_batch = rewards_batch - r_pred_batch  # [batch_size, num_agents]
            weight_batch = self.action_bridge_net(z_t_batch, actions_batch)  # [batch_size, num_agents]
            DR_correction_batch = weight_batch * error_batch  # [batch_size, num_agents]
            
            # === 2. Create action matrices for all counterfactual actions ===
            # For each sample, create action_dim versions with different actions for agent_id
            # Shape: [batch_size, action_dim, num_agents]
            action_matrix = actions_batch.unsqueeze(1).repeat(1, self.action_dim, 1)
            action_range = torch.arange(self.action_dim, device=self.device).unsqueeze(0).repeat(batch_size, 1)
            action_matrix[:, :, agent_id] = action_range
            
            # Reshape for batch processing: [batch_size * action_dim, num_agents]
            action_matrix_flat = action_matrix.view(batch_size * self.action_dim, self.num_agents)
            
            # Expand w_t for all counterfactual actions: [batch_size * action_dim, w_dim]
            w_t_expanded = w_t_batch.unsqueeze(1).repeat(1, self.action_dim, 1).view(batch_size * self.action_dim, -1)
            
            # Get all counterfactual reward predictions: [batch_size * action_dim, num_agents]
            reg_estimates_flat = self.outcome_bridge_net(action_matrix_flat, w_t_expanded)
            
            # Add DR correction: [batch_size, action_dim, num_agents]
            reg_estimates = reg_estimates_flat.view(batch_size, self.action_dim, self.num_agents)
            r_hat_all = reg_estimates + DR_correction_batch.unsqueeze(1)
            
            # === 3. Get policy probabilities for agent i ===
            action_logits = self.agents[agent_id].policy_net(o_t_batch)  # [batch_size, action_dim]
            action_probs = torch.softmax(action_logits, dim=-1)  # [batch_size, action_dim]
            
            # === 4. Extract factual and counterfactual predictions ===
            # Factual action indices: [batch_size]
            factual_actions = actions_batch[:, agent_id]
            
            # Gather factual predictions: [batch_size, num_agents]
            # predicted reward vector for factual action
            r_hat_factual = r_hat_all[torch.arange(batch_size, device=self.device), factual_actions]
            # Get factual probabilities: [batch_size]
            # prob_factual = action_probs[torch.arange(batch_size, device=self.device), factual_actions]
            
            # === 5. Compute counterfactual baseline ===
            # Create mask for counterfactual actions: [batch_size, action_dim]
            mask = torch.ones(batch_size, self.action_dim, dtype=torch.bool, device=self.device)
            mask[torch.arange(batch_size, device=self.device), factual_actions] = False
            
            # Zero out the rewards of the factual actions as well to ensure they don't contribute
            r_hat_all_masked = r_hat_all.clone()
            '''
            # Baseline using policy probabilities
            probs_counterfactuals = action_probs.clone()
            probs_counterfactuals[~mask] = 0.0
            sum_probs_counterfactuals = probs_counterfactuals.sum(dim=1, keepdim=True)
            weighted_sum_rewards = (probs_counterfactuals.unsqueeze(2) * r_hat_all_masked).sum(dim=1)
            baseline = weighted_sum_rewards / (sum_probs_counterfactuals + 1e-8)
            '''
            # mask is [B, A], needs to be expanded for [B, A, N]
            r_hat_all_masked[~mask.unsqueeze(-1).expand_as(r_hat_all)] = 0.0
            # Compute the number of counterfactual actions for each batch element
            counterfactual_count = mask.sum(dim=1, keepdim=True).clamp(min=1).to(r_hat_all_masked.dtype)
            # Compute the final baseline as the simple average of counterfactual rewards
            baseline = r_hat_all_masked.sum(dim=1) / (counterfactual_count + 1e-8) # Shape: [B, N]
            
            # === 6. Compute causal effect vector ===
            causal_effect_vector = r_hat_factual - baseline  # [B, N]
            # causal_effect_vector = rewards_batch - baseline  # [B, N]
            
            # # TEST: consider my reward
            # Delta_batch = causal_effect_vector.mean(dim=1)
            
            # === 7. Extract effects for other agents ===
            # Create mask for other agents: [num_agents]
            other_agents_mask = torch.ones(self.num_agents, dtype=torch.bool, device=self.device)
            other_agents_mask[agent_id] = False
            
            # Get effects for other agents: [batch_size, num_agents-1]
            causal_effects_others = causal_effect_vector[:, other_agents_mask]
            if causal_effects_others.size(1) > 0:
                Delta_batch = causal_effects_others.max(dim=1)[0] # why [0]? because max returns a tuple (value, index)
                # Delta_batch = causal_effects_others.mean(dim=1) 
            else:
                Delta_batch = torch.zeros(batch_size, device=self.device)

        # TEST: Optimal Delta_batch to test how policy is updated with assumption that bridge networks are optimal.
        # -------------------------------------------------------------
        # Delta_batch = torch.zeros(batch_size, device=self.device)
        # find where the rewards are 5.0 and set corresponding Delta_batch element to 5.0
        # rule: if agent_id is zero, check last agent's reward and else check (agent_id-1) agent reward 
        # if agent_id == 0:
        #     Delta_batch[rewards_batch[:, (self.num_agents-1)] == 5.0] = 5.0
        # else:
        #     Delta_batch[rewards_batch[:, (agent_id-1)] == 5.0] = 5.0    
        # -------------------------------------------------------------
       
        # Check performance of bridge networks
        # if agent_id == 0:
        #     print("r_pred_batch", r_pred_batch[rewards_batch[:, (self.num_agents-1)] == 5.0])
        #     print("error_batch", error_batch[rewards_batch[:, (self.num_agents-1)] == 5.0])
        #     print("r_hat_all", r_hat_all[rewards_batch[:, (self.num_agents-1)] == 5.0])
        #     print("r_hat_factual", r_hat_factual[rewards_batch[:, (self.num_agents-1)] == 5.0])
        #     print("baseline", baseline[rewards_batch[:, (self.num_agents-1)] == 5.0])
        #     print("Delta_batch", Delta_batch[rewards_batch[:, (self.num_agents-1)] == 5.0])
        # else:
        #     print("r_pred_batch", r_pred_batch[rewards_batch[:, (agent_id-1)] == 5.0])    
        #     print("error_batch", error_batch[rewards_batch[:, (agent_id-1)] == 5.0])
        #     print("r_hat_all", r_hat_all[rewards_batch[:, (agent_id-1)] == 5.0])
        #     print("factual_actions", factual_actions[rewards_batch[:, (agent_id-1)] == 5.0])
        #     print("baseline", baseline[rewards_batch[:, (agent_id-1)] == 5.0])
        #     print("Delta_batch", Delta_batch[rewards_batch[:, (agent_id-1)] == 5.0])

        return Delta_batch
    '''
    def hamilton_divide_agent_i_action(
        self,
        agent_id: int,
        w_t_batch: torch.Tensor,      # [batch_size, w_dim]
        z_t_batch: torch.Tensor,      # [batch_size, z_dim]
        o_t_batch: torch.Tensor,      # [batch_size, obs_dim]
        actions_batch: torch.Tensor,  # [batch_size, num_agents]
        rewards_batch: torch.Tensor,  # [batch_size, num_agents]
    ) -> torch.Tensor:
        """
        all operation should be vectorized like causal_intrinsic_reward_batch
        """
        # Compute expected reward by counter factual actions using bridge networks
        # E_{(a^i_t)'~\pi^i}[1^T r(o_t, {(a^i_t)', a^{-i}_t})]
        
        # Compute term (A): actual reward 1^T r - E_{(a^i_t)'~\pi^i}[1^T r(o_t, {(a^i_t)', a^{-i}_t})]

        # Compute counterfatual value using current value function
        # E_{(a^i_t)'~\pi^i, o^i_{t+1}~T(o^i_t, a^i_t)}[V^i(o^i_{t+1})]

        # Compute term (B): 
        # V^i(o^i_{t+1}) - E_{(a^i_t)'~\pi^i, o^i_{t+1}~T(o^i_t, a^i_t)}[V^i(o^i_{t+1})]

        # Compute output:
        # delta^i_[t_1:t_T] = delta_batch = term (A) - term (B) (length of batch is T)
        
        return delta_batch
    '''
    def update_agents(self, batch_data: List[Dict]) -> Dict[str, float]:
        """
        Updates agent policies and the centralized value function.
        MODIFIED: The value function is now trained using returns from SHAPED rewards.
        The entropy loss bug is also fixed.
        """
        agent_losses = defaultdict(float)
        batch_size = len(batch_data)

        # === Batch-level Data Preparation ===
        # Prepare individual agent observations for value network
        agent_obs_tensors = {}
        for agent_id in range(self.num_agents):
            if agent_id in batch_data[0]['bold_o_t']:
                agent_obs_tensors[agent_id] = torch.stack([t['bold_o_t'][agent_id] for t in batch_data]).to(self.device)

        # === 1. Collect Agent-specific Shaped Rewards and Delta Statistics ===
        all_agent_shaped_rewards = {}
        all_agent_rewards = {}  # Original rewards for value network
        all_agent_dones = {}
        
        # Delta statistics for wandb logging
        delta_stats = {
            'no_reward': [],
            'forage_reward': [],
            'stag_reward': []
        }
        
        # Prepare batch data for vectorized computation
        w_t_batch = torch.stack([self._concatenate_observations(t['bold_o_t']) for t in batch_data]).to(self.device)
        z_t_batch = torch.stack([self._concatenate_observations(t['bold_o_t_minus_1']) for t in batch_data]).to(self.device)
        actions_batch = torch.tensor([[t['bold_a_t'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.long, device=self.device)
        rewards_batch = torch.tensor([[t['bold_r_t'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.float32, device=self.device)
        
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue
                
            agent_obs = torch.stack([t['bold_o_t'][agent_id] for t in batch_data]).to(self.device)
            agent_rewards = torch.tensor([t['bold_r_t'][agent_id] for t in batch_data], dtype=torch.float32, device=self.device)
            agent_dones = torch.tensor([t['dones'][agent_id] for t in batch_data], dtype=torch.float32, device=self.device)
            
            shaped_rewards = []
            # Use vectorized batch computation
            with torch.no_grad():
                Delta_batch = self.causal_intrinsic_reward_batch(
                    agent_id, w_t_batch, z_t_batch, agent_obs, actions_batch, rewards_batch
                )
                
                # Compute shaped rewards
                shaped_rewards = agent_rewards + self.get_current_causal_lambda() * Delta_batch
                
                # Collect delta statistics for wandb logging
                for t, transition in enumerate(batch_data):
                    scenario = self._classify_delta_scenario(transition, agent_id)
                    delta_stats[scenario].append(Delta_batch[t].item())
            
            all_agent_shaped_rewards[agent_id] = shaped_rewards
            all_agent_rewards[agent_id] = agent_rewards
            all_agent_dones[agent_id] = agent_dones

        # === 2. Compute Individual Agent Returns/Advantages using SHAPED rewards ===
        agent_returns = {}
        agent_advantages = {}
        agent_values = {}
        next_agent_values = {}
        
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']:
                continue
                
            # Get current values for this agent
            with torch.no_grad():
                agent_values[agent_id] = self.value_net(agent_obs_tensors[agent_id]).squeeze()
            
            # Estimate next state value for bootstrapping the SHAPED-return calculation
            with torch.no_grad():
                last = batch_data[-1]
                is_terminal = all_agent_dones[agent_id][-1].item()
                if not is_terminal:
                    next_obs_agent = last['next_observations'][agent_id]
                    next_agent_values[agent_id] = self.value_net(next_obs_agent.unsqueeze(0)).item()
                else:
                    next_agent_values[agent_id] = 0.0

            advantages_shaped, returns_agent_shaped = self.compute_gae(
                all_agent_shaped_rewards[agent_id].tolist(), 
                agent_values[agent_id].tolist(), 
                next_agent_values[agent_id], 
                all_agent_dones[agent_id].tolist()
            )
            agent_returns[agent_id] = torch.tensor(returns_agent_shaped, dtype=torch.float32, device=self.device)
            agent_advantages[agent_id] = torch.tensor(advantages_shaped, dtype=torch.float32, device=self.device)

        # === Agent-Specific Policy Updates ===
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']:
                continue
            
            agent_obs = torch.stack([t['bold_o_t'][agent_id] for t in batch_data]).to(self.device)
            agent_actions = torch.tensor([t['bold_a_t'][agent_id] for t in batch_data], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.agents[agent_id].policy_net(agent_obs)
                old_log_probs_tensor = F.log_softmax(logits, dim=-1).gather(1, agent_actions.unsqueeze(1)).squeeze(1)

            # 3. Advantage for the POLICY update (using agent-specific SHAPED rewards)
            advantages_tensor = agent_advantages[agent_id]
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

            # === PPO Policy Update Loop (for this agent) ===
            for _ in range(self.ppo_epochs):
                new_log_probs, entropy = self.agents[agent_id].policy_net.evaluate_actions(agent_obs, agent_actions)
                ratio = torch.exp(new_log_probs - old_log_probs_tensor.detach())

                # Policy loss uses the agent-specific advantage from SHAPED rewards
                surr1 = ratio * advantages_tensor
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
                policy_loss = -torch.min(surr1, surr2).mean()
                
                entropy_loss = -entropy.mean()
                
                # Use '+' to maximize entropy.
                agent_policy_loss = policy_loss + self.entropy_coef * entropy_loss
                
                self.agents[agent_id].optimizer.zero_grad()
                agent_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[agent_id].policy_net.parameters(), self.max_grad_norm)
                self.agents[agent_id].optimizer.step()

                agent_losses[f'agent_{agent_id}_policy_loss'] += policy_loss.item()
                agent_losses[f'agent_{agent_id}_entropy_loss'] += entropy_loss.item()
        
        # === Centralized Value Function Update (using individual agent data) ===
        # The same value network is trained on data from all agents, ensuring parameter sharing
        for _ in range(self.ppo_epochs):
            total_value_loss = 0.0
            
            # Train value network on individual agent returns
            # All agents use the same network parameters (centralized training)
            for agent_id in range(self.num_agents):
                if agent_id not in batch_data[0]['bold_o_t']:
                    continue
                    
                # Get new values for this agent using the shared value network
                new_agent_values = self.value_net(agent_obs_tensors[agent_id]).squeeze()
                agent_value_loss = F.mse_loss(new_agent_values, agent_returns[agent_id])
                total_value_loss += agent_value_loss
            
            # Average the value loss across all agents
            if len(agent_returns) > 0:
                avg_value_loss = total_value_loss / len(agent_returns)
                
                self.value_optimizer.zero_grad()
                (self.value_coef * avg_value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

                agent_losses['value_loss'] += avg_value_loss.item()

        # Average losses for logging
        for key in list(agent_losses.keys()):
            agent_losses[key] /= self.ppo_epochs
        
        # Add delta statistics to losses for wandb logging
        for scenario, deltas in delta_stats.items():
            if deltas:  # Only add if we have data
                agent_losses[f'delta_{scenario}_mean'] = np.mean(deltas)
                agent_losses[f'delta_{scenario}_std'] = np.std(deltas)
                agent_losses[f'delta_{scenario}_count'] = len(deltas)
        return dict(agent_losses)

    def update(self, **kwargs):
        """
        Main update function that orchestrates the training process for one iteration.
        """
        if self.buffer_manager.is_main_buffer_empty():
            return {}

        transitions = self.buffer_manager.get_main_buffer_transitions()
        
        # Partition into K mini-batches
        batch_size = max(1, len(transitions) // self.num_batches)
        batches = [transitions[i:i + batch_size] for i in range(0, len(transitions), batch_size)]
        
        total_losses = defaultdict(float)
        
        # Loop through each mini-batch to perform updates
        for batch_data in batches:
            if not batch_data: continue
            
            # Phase 1: Update Causal Networks (only if causal lambda > 0)
            if self.get_current_causal_lambda() > 0.0:
                causal_losses = self.update_causal_networks(batch_data, mc_samples_k_w2=16)
            else:
                causal_losses = {
                    'outcome_bridge_loss': 0.0,
                    'action_bridge_loss': 0.0,
                }
            
            # Phase 2: Update Agent Policies and Value Function
            agent_and_value_losses = self.update_agents(batch_data)

            # Accumulate losses from both phases for logging
            for loss_name, loss_value in causal_losses.items():
                total_losses[loss_name] += loss_value
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

    # Buffer management
    def clear_buffer(self):
        """Clear main experience buffer."""
        self.buffer_manager.clear_buffer()
    
    def clear_causal_buffer(self):
        """Clear causal experience buffer."""
        self.buffer_manager.clear_causal_buffer()
    
    def get_buffer_info(self) -> Dict[str, int]:
        """Get current buffer sizes and limits for monitoring."""
        return self.buffer_manager.get_buffer_info()
