import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

from algorithms.networks import PolicyNetwork, OutcomeBridgeNetwork, ActionBridgeNetwork, ValueNetwork
from utils.buffer import TransitionBuffer


class MAPPOAgent:
    """
    Single agent for MAPPO algorithm.
    Each agent has its own policy network and central value network is shared,
    which takes concatenated observations of all agents as input.
    """
    
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr_actor: float = 3e-4
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Policy network
        self.policy_net = PolicyNetwork(obs_dim, action_dim, hidden_dim)
        
        # Optimizer for the policy network
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_actor)
        
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


class MAPPOAlgorithm:
    """
    MAPPO Algorithm implementation
    """
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr_actor: float = 3e-4,
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
        device: str = "cpu",  # Device for computation
        global_reward: bool = True,
    ):
        self.global_reward = global_reward
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

        # Create experience buffer
        self.buffer_manager = TransitionBuffer(
            max_buffer_size=10000
        )

        # Create agents
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = MAPPOAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                lr_actor=lr_actor
            )
        # Central Value network
        self.value_net = ValueNetwork(
            obs_dim=obs_dim * num_agents,
            hidden_dim=hidden_dim,
            out_dim=1 if global_reward else num_agents
        )
        # Optimizer for the shared value network
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_critic)
        # Move all networks to device
        self._move_networks_to_device()

        self.training_steps = 0
    
    def _move_networks_to_device(self):
        """Move all networks to the specified device."""
        # Move agent networks
        for agent in self.agents.values():
            agent.policy_net.to(self.device)
        
        # Move value network (shared across agents)
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
    
    def store_transition(
        self,
        prev_observations: Dict[int, torch.Tensor],  # o_{t-1}
        prev_rewards: Dict[int, float],               # r_{t-1}
        prev_actions: Dict[int, int],                 # a_{t-1}
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
            prev_rewards=prev_rewards,
            prev_actions=prev_actions,
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
        """
        agent_losses = defaultdict(float)
        batch_size = len(batch_data)

        # === Batch-level Data Preparation ===
        # Prepare individual agent observations for value network
        global_obs_tensor = torch.stack([self._concatenate_observations(t['bold_o_t']) for t in batch_data]).to(self.device)
        agent_obs_tensors = {}
        for agent_id in range(self.num_agents):
            if agent_id in batch_data[0]['bold_o_t']:
                agent_obs_tensors[agent_id] = torch.stack([t['bold_o_t'][agent_id] for t in batch_data]).to(self.device)

        # === 1. Compute Rewards & Returns ===
        if self.global_reward:
            # === Global Rewards (sum of all agents) ===
            global_rewards = []
            global_dones = []
            for t in batch_data:
                global_reward = sum(t['bold_r_t'].values()) / (self.num_agents)
                global_rewards.append(global_reward)
                global_done = any(t['dones'].values())
                global_dones.append(global_done)
            
            all_rewards = torch.tensor(global_rewards, dtype=torch.float32, device=self.device)
            all_dones_tensor = torch.tensor(global_dones, dtype=torch.float32, device=self.device)

            # Centralized Critic for Global Value
            with torch.no_grad():
                # [Batch, 1] -> [Batch]
                global_values = self.value_net(global_obs_tensor).squeeze()
            
            # Next Value
            with torch.no_grad():
                last = batch_data[-1]
                is_terminal = all_dones_tensor[-1].item()
                if not is_terminal:
                    next_global_obs = self._concatenate_observations(last.get('next_observations', last['bold_o_t']))
                    next_val_item = self.value_net(next_global_obs.unsqueeze(0)).item()
                else:
                    next_val_item = 0.0

            advantages_global, returns_global = self.compute_gae(
                all_rewards.tolist(), 
                global_values.tolist(), 
                next_val_item, 
                all_dones_tensor.tolist()
            )
            all_returns = torch.tensor(returns_global, dtype=torch.float32, device=self.device)
            all_advantages = torch.tensor(advantages_global, dtype=torch.float32, device=self.device)

        else:
            # === Individual Rewards ===
            rewards_list = []
            dones_list = []
            
            # Reconstruct batch-wise individual rewards
            for t in batch_data:
                step_rewards = [t['bold_r_t'].get(i, 0.0) for i in range(self.num_agents)]
                rewards_list.append(step_rewards)
                
                # For individual value estimation, what is 'done'? 
                # Usually episode done is shared, but if agents die individually... 
                # Let's assume shared episode done for now for simplicity or use individual 'dones' if available.
                step_dones = [t['dones'].get(i, False) for i in range(self.num_agents)]
                dones_list.append(step_dones)

            all_rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device) # [B, N]
            all_dones_tensor = torch.tensor(dones_list, dtype=torch.float32, device=self.device) # [B, N]

            # Centralized Critic for Individual Values
            # output: [Batch, N]
            with torch.no_grad():
                all_values = self.value_net(global_obs_tensor) 

            # Next Values
            with torch.no_grad():
                last = batch_data[-1]
                # Check terminal per agent?
                # all_values is [B, N], next_val should be [1, N] or list
                next_vals_list = []
                next_global_obs = self._concatenate_observations(last.get('next_observations', last['bold_o_t']))
                next_pred = self.value_net(next_global_obs.unsqueeze(0)).squeeze(0) # [N]
                
                for i in range(self.num_agents):
                    if all_dones_tensor[-1, i].item():
                        next_vals_list.append(0.0)
                    else:
                        next_vals_list.append(next_pred[i].item())
            
            # Compute GAE per agent
            returns_list = []
            advantages_list = []
            
            # all_rewards: [B, N], all_values: [B, N]
            for i in range(self.num_agents):
                adv, ret = self.compute_gae(
                    all_rewards[:, i].tolist(),
                    all_values[:, i].tolist(),
                    next_vals_list[i],
                    all_dones_tensor[:, i].tolist()
                )
                returns_list.append(ret)
                advantages_list.append(adv)
            
            # Stack -> [N, B] -> Transpose -> [B, N]
            all_returns = torch.tensor(returns_list, dtype=torch.float32, device=self.device).transpose(0, 1)
            all_advantages = torch.tensor(advantages_list, dtype=torch.float32, device=self.device).transpose(0, 1)

        # Normalize Advantages
        if self.global_reward:
             advantages_tensor = all_advantages
             if advantages_tensor.numel() > 1:
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        else:
             # Normalize per batch or per agent?
             advantages_tensor = all_advantages # [B, N]
             if advantages_tensor.numel() > 1:
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # === Agent-Specific Policy Updates ===
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']:
                continue
            
            agent_obs = torch.stack([t['bold_o_t'][agent_id] for t in batch_data]).to(self.device)
            agent_actions = torch.tensor([t['bold_a_t'][agent_id] for t in batch_data], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.agents[agent_id].policy_net(agent_obs)
                old_log_probs_tensor = F.log_softmax(logits, dim=-1).gather(1, agent_actions.unsqueeze(1)).squeeze(1)

            # 3. Advantage for the POLICY update
            if self.global_reward:
                target_advantage = advantages_tensor
            else:
                target_advantage = advantages_tensor[:, agent_id]

            # === PPO Policy Update Loop (for this agent) ===
            for _ in range(self.ppo_epochs):
                new_log_probs, entropy = self.agents[agent_id].policy_net.evaluate_actions(agent_obs, agent_actions)
                ratio = torch.exp(new_log_probs - old_log_probs_tensor.detach())

                # Policy loss
                surr1 = ratio * target_advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * target_advantage
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
        
        # === Centralized Value Function Update ===
        for _ in range(self.ppo_epochs):
            # output: [Batch, 1] (global) or [Batch, N] (individual)
            new_values = self.value_net(global_obs_tensor)
            
            if self.global_reward:
                 new_values = new_values.squeeze() # [Batch]
                 # all_returns is [Batch]
            
            # MSE Loss
            value_loss = F.mse_loss(new_values, all_returns)
            
            self.value_optimizer.zero_grad()
            (self.value_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.value_optimizer.step()

            agent_losses['value_loss'] += value_loss.item()

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
