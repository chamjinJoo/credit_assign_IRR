import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

from ..networks import PolicyNetwork, BridgeNetwork, ActionBridgeNetwork, CentralizedValueNetwork
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
    z_t = \mathbf{o}_t, w_t = \mathbf{o}_{t-1}
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
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.w_dim = w_dim
        
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
        
        # Centralized value network (shared parameters across all agents)
        # All agents use the same value network instance, ensuring parameter sharing
        self.value_net = CentralizedValueNetwork(
            obs_dim=obs_dim,
            num_agents=num_agents,
            hidden_dim=hidden_dim
        )
        
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
        
        # RKHS Kernel for analytical loss
        self.critic_kernel_z = RBFKernel(gamma=1.0).to(self.device)
        self.critic_kernel_w = RBFKernel(gamma=1.0).to(self.device)

        # # Linear Kernel for analytical loss
        # self.critic_kernel_z = LinearKernel().to(self.device)
        # self.critic_kernel_w = LinearKernel().to(self.device)
        
        # Action embedding for kernel input
        self.action_embedding_for_kernel = nn.Embedding(action_dim, 8)
        
        # Optimizers for causal networks
        self.outcome_bridge_optimizer = optim.Adam(self.outcome_bridge_net.parameters(), lr=lr_bridge)
        self.action_bridge_optimizer = optim.Adam(self.action_bridge_net.parameters(), lr=lr_critic)
        self.kernel_action_embed_optimizer = optim.Adam(self.action_embedding_for_kernel.parameters(), lr=lr_critic)
        
        # Optimizer for centralized value network
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Move all networks to device
        self._move_networks_to_device()
    
    def _move_networks_to_device(self):
        """Move all networks to the specified device."""
        # Move agent networks
        for agent in self.agents.values():
            agent.policy_net.to(self.device)
        
        # Set device for all networks
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
    
    def update_causal_networks(self, batch_data: List[Dict], mc_samples_k_w2: int = 10) -> Dict[str, float]:
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
        batch_size = len(batch_data)
        # === Vectorized Data Preparation ===
        z_t_batch = torch.stack([self._concatenate_observations(t['bold_o_t']) for t in batch_data])
        w_t_batch = torch.stack([self._concatenate_observations(t['bold_o_t_minus_1']) for t in batch_data])
        a_t_batch = torch.tensor([[t['bold_a_t'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.long, device=self.device)
        r_t_batch = torch.tensor([[t['bold_r_t'][i] for i in range(self.num_agents)] for t in batch_data], dtype=torch.float32, device=self.device)

        # === Part 1: Update Outcome Bridge Network h_φ ===
        r_hat_outcome = self.outcome_bridge_net(a_t_batch, w_t_batch)
        residuals = r_hat_outcome - r_t_batch
        
        action_embeds = self.action_embedding_for_kernel(a_t_batch).view(batch_size, -1)
        kernel_input_features_z = torch.cat([z_t_batch, action_embeds], dim=1)
        # self.critic_kernel_z.set_gamma(kernel_input_features_z) # for RBF kernel
        K_z = self.critic_kernel_z(kernel_input_features_z) # Shape: [batch_size, batch_size]
        quadratic_forms = torch.matmul(residuals.T, torch.matmul(K_z, residuals))
        outcome_bridge_loss = torch.trace(quadratic_forms) / (self.num_agents * batch_size)

        # === Part 2: Update Action Bridge Network q_θ ===
        # 1. Compute joint policy probabilities π(a_t|o_t)
        with torch.no_grad():
            log_probs_list = [self.agents[i].policy_net.evaluate_actions(
                torch.stack([t['bold_o_t'][i] for t in batch_data]), a_t_batch[:, i]
            )[0] for i in range(self.num_agents)]
            joint_probs = torch.exp(torch.stack(log_probs_list, dim=1).sum(dim=1))

        # 2. Compute the weighted IPW term φ_q
        action_bridge_output = self.action_bridge_net(z_t_batch, a_t_batch)
        phi_q = joint_probs * action_bridge_output.squeeze()

        # 3. Calculate Gram matrix K_w1 (Quadratic term)
        kernel_input_features_w = torch.cat([w_t_batch, action_embeds], dim=1) # Detach to stop grads from h-update
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
        quadratic_term = torch.matmul(phi_q_col.T, torch.matmul(K_w1, phi_q_col)).squeeze()
        linear_term = torch.matmul(phi_q, K_w2.sum(dim=1))
        action_bridge_loss = quadratic_term - 2.0 * linear_term / batch_size

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
        dones: Dict[int, bool]
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
            hidden_states={i: None for i in range(self.num_agents)}  # No hidden states needed
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
    
    def causal_intrinsic_reward(
        self,
        agent_id: int,
        w_t: torch.Tensor,      # Observed proxy variable w_t = o_{t-1}
        z_t: torch.Tensor,      # Context variable z_t = o_t
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
            w_t: Proxy w_t, typically the concatenated previous observation o_{t-1}.
            z_t: Proxy z_t, typically the concatenated current observation o_t.
            o_t: The individual observation for agent i.
            current_actions: The joint action a_t actually taken at the timestep.
            current_rewards: The joint reward r_t actually received.

        Returns:
            Delta_i_t: The final scalar causal reward shaping term for agent i.
            r_hat_all_numpy: A numpy array of all counterfactual reward predictions for monitoring.
        """
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
            r_hat_factual_weighted = prob_factual * r_hat_all[factual_action_idx] # Shape: [num_agents]
  
            # b) Get the weighted average reward prediction for all other possible actions
            # Create a mask to select all actions EXCEPT the factual one
            mask = torch.ones(self.action_dim, dtype=torch.bool, device=self.device)
            mask[factual_action_idx] = False
            
            r_hat_counterfactuals = r_hat_all[mask] # Shape: [action_dim - 1, num_agents]
            probs_counterfactuals = action_probs[mask]  # Shape: [action_dim - 1]
            # === 3. Compute the Difference-Based Causal Effect ===
            if r_hat_counterfactuals.shape[0] == 0:
                # If there are no other actions, the causal effect is just the weighted factual outcome.
                # This is a design choice; another option is to return zero.
                causal_effect_vector = r_hat_factual_weighted
            else:
                # Weight the counterfactual rewards by their probabilities
                # Shape: [action_dim - 1, num_agents]
                weighted_r_hat_counterfactuals = probs_counterfactuals.unsqueeze(1) * r_hat_counterfactuals
                
                # Normalize to get the expected value over the counterfactual action distribution
                sum_probs_counterfactuals = probs_counterfactuals.sum()
                r_hat_counterfactuals_mean_weighted = weighted_r_hat_counterfactuals.sum(dim=0) / (sum_probs_counterfactuals + 1e-8)
                # The causal effect is the difference between the weighted factual outcome and the weighted counterfactual average
                causal_effect_vector = r_hat_factual_weighted - r_hat_counterfactuals_mean_weighted # Shape: [num_agents]
            
            # === 4. Calculate Final Shaping Term Δ^i_t ===
            # Average the causal effects over all *other* agents.
            other_agents = [j for j in range(self.num_agents) if j != agent_id]
            if other_agents:
                Delta_i_t = causal_effect_vector[other_agents].mean().item()
            else:
                Delta_i_t = 0.0
                
        return Delta_i_t, r_hat_all.cpu().numpy()

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
        all_agent_dones = {}
        
        # Delta statistics for wandb logging
        delta_stats = {
            'no_reward': [],
            'forage_reward': [],
            'stag_reward': []
        }
        
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue
                
            agent_obs = torch.stack([t['bold_o_t'][agent_id] for t in batch_data]).to(self.device)
            agent_rewards = torch.tensor([t['bold_r_t'][agent_id] for t in batch_data], dtype=torch.float32, device=self.device)
            agent_dones = torch.tensor([t['dones'][agent_id] for t in batch_data], dtype=torch.float32, device=self.device)
            
            shaped_rewards = []
            with torch.no_grad():
                for t, transition in enumerate(batch_data):
                    w_t = self._concatenate_observations(transition['bold_o_t_minus_1'])
                    z_t = self._concatenate_observations(transition['bold_o_t'])
                    Delta_t, _ = self.causal_intrinsic_reward(
                        agent_id, w_t, z_t, agent_obs[t],
                        transition['bold_a_t'], transition['bold_r_t']
                    )
                    shaped_reward = agent_rewards[t] + self.get_current_causal_lambda() * Delta_t
                    shaped_rewards.append(shaped_reward)
                    
                    # Collect delta statistics for wandb logging
                    other_agent_id = 1 - agent_id  # 2-agent environment
                    other_reward = transition['bold_r_t'][other_agent_id]
                    
                    # Classify reward scenario
                    if other_reward == 0.0:
                        delta_stats['no_reward'].append(Delta_t)
                    elif other_reward == 3.0:
                        delta_stats['forage_reward'].append(Delta_t)
                    elif other_reward == 5.0:
                        delta_stats['stag_reward'].append(Delta_t)
            
            all_agent_shaped_rewards[agent_id] = torch.tensor(shaped_rewards, dtype=torch.float32, device=self.device)
            all_agent_dones[agent_id] = agent_dones

        # === 2. Compute Individual Agent Returns for Value Network (from SHAPED rewards) ===
        agent_returns = {}
        agent_values = {}
        next_agent_values = {}
        
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']:
                continue
                
            # Get current values for this agent
            with torch.no_grad():
                agent_values[agent_id] = self.value_net(agent_obs_tensors[agent_id]).squeeze()
            
            # Compute returns for this agent using its shaped rewards
            with torch.no_grad():
                last = batch_data[-1]
                is_terminal = all_agent_dones[agent_id][-1].item()
                if not is_terminal:
                    next_obs_agent = last['next_observations'][agent_id]
                    next_agent_values[agent_id] = self.value_net(next_obs_agent.unsqueeze(0)).item()
                else:
                    next_agent_values[agent_id] = 0.0

            _, returns_agent_shaped = self.compute_gae(
                all_agent_shaped_rewards[agent_id].tolist(), 
                agent_values[agent_id].tolist(), 
                next_agent_values[agent_id], 
                all_agent_dones[agent_id].tolist()
            )
            agent_returns[agent_id] = torch.tensor(returns_agent_shaped, dtype=torch.float32, device=self.device)

        # === Agent-Specific Policy Updates ===
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']:
                continue
            
            agent_obs = torch.stack([t['bold_o_t'][agent_id] for t in batch_data]).to(self.device)
            agent_actions = torch.tensor([t['bold_a_t'][agent_id] for t in batch_data], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.agents[agent_id].policy_net(agent_obs)
                old_log_probs_tensor = F.log_softmax(logits, dim=-1).gather(1, agent_actions.unsqueeze(1)).squeeze(1)

            shaped_rewards_tensor = all_agent_shaped_rewards[agent_id]
            agent_dones = all_agent_dones[agent_id]
            
            # 3. GAE for the POLICY update (using agent-specific SHAPED rewards)
            # The baseline for this GAE is the individual agent value function estimate.
            advantages_shaped, _ = self.compute_gae(
                shaped_rewards_tensor.tolist(), 
                agent_values[agent_id].tolist(), # Use individual agent value as baseline
                next_agent_values[agent_id], 
                agent_dones.tolist()
            )
            advantages_tensor = torch.tensor(advantages_shaped, dtype=torch.float32, device=self.device)
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
            
            # Phase 1: Update Causal Networks
            causal_losses = self.update_causal_networks(batch_data)
            
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