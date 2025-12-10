import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

from algorithms.networks import PolicyNetwork, OutcomeBridgeNetwork, ActionBridgeNetwork, ValueNetwork
from utils.buffer import ExperienceBuffer

class IPPOAgent:
    """
    Single agent for IPPO algorithm.
    Each agent has its own policy network and value network.
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


class IPPOAlgorithm:
    """
    IPPO Algorithm implementation
    """
    
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        num_batches: int = 4,  # Number of batches K for training
        epsilon_start: float = 0.6,
        epsilon_end: float = 0.0,
        epsilon_decay_steps: int = 1000,
        device: str = "cpu",  # Device for computation
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Device setup
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
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
        self.training_steps = 0
        
        # Create experience buffer
        self.buffer_manager = ExperienceBuffer(
            max_buffer_size=10000,
            max_causal_buffer_size=2000
        )
        # Create agents
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = IPPOAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                lr=lr
            )

        # Value network: parameter sharing and local obs
        self.value_net = ValueNetwork(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
        )
        
        # Optimizer for the shared value network
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)
        
        # Move all networks to device
        self._move_networks_to_device()
    
    def _move_networks_to_device(self):
        for agent in self.agents.values():
            agent.policy_net.to(self.device)
        self.value_net.to(self.device)
    
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

        all_agent_rewards = {}  # Original rewards for value network
        all_agent_dones = {}
        
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']: continue

            agent_obs = torch.stack([t['bold_o_t'][agent_id] for t in batch_data]).to(self.device)
            agent_rewards = torch.tensor([t['bold_r_t'][agent_id] for t in batch_data], dtype=torch.float32, device=self.device)
            agent_dones = torch.tensor([t['dones'][agent_id] for t in batch_data], dtype=torch.float32, device=self.device)
            
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
                all_agent_rewards[agent_id].tolist(), 
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
            
            agent_and_value_losses = self.update_agents(batch_data)

            for loss_name, loss_value in agent_and_value_losses.items():
                total_losses[loss_name] += loss_value

        self.buffer_manager.clear_buffer()
        self.training_steps += 1
        
        # Average losses over the number of mini-batches for accurate logging
        num_valid_batches = len([b for b in batches if b])
        if num_valid_batches == 0: return {}

        final_losses = {name: val / num_valid_batches for name, val in total_losses.items()}

        return final_losses

    # Buffer management
    def clear_buffer(self):
        """Clear main experience buffer."""
        self.buffer_manager.clear_buffer()
    
    def get_buffer_info(self) -> Dict[str, int]:
        """Get current buffer sizes and limits for monitoring."""
        return self.buffer_manager.get_buffer_info()
