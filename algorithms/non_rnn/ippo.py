import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

from algorithms.networks import PolicyNetwork, OutcomeBridgeNetwork, ActionBridgeNetwork, ValueNetwork
from utils.buffer import TransitionBuffer, EpisodeBuffer

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

    def get_action_batch(self, obs_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get actions for a batch of environments.
        """
        with torch.no_grad():
            action, log_prob = self.policy_net.get_action_and_log_prob(obs_batch)
        return action, log_prob


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
        global_reward: bool = False,
        use_episode_buffer: bool = False,
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
    
        self.training_steps = 0
        self.use_episode_buffer = use_episode_buffer
        
        # Create experience buffer
        if use_episode_buffer:
            self.buffer_manager = EpisodeBuffer(max_episodes=100)
        else:
            self.buffer_manager = TransitionBuffer(max_buffer_size=10000)
        # Create agents
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = IPPOAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                lr_actor=lr_actor
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
    
    def reset_agents(self):
        """Reset agents for new episode (no RNN state to reset)."""
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
        Get actions for a batch of environments.
        """
        n_envs = len(observations_batch)
        actions_batch = [{} for _ in range(n_envs)]
        
        for agent_id in range(self.num_agents):
            agent_obs_list = []
            for i in range(n_envs):
                obs = observations_batch[i][agent_id]
                agent_obs_list.append(obs)
            
            agent_obs_tensor = torch.stack(agent_obs_list).to(self.device)
            
            actions, log_probs = self.agents[agent_id].get_action_batch(agent_obs_tensor)
            
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
            info=info
        )
        
        if self.use_episode_buffer:
            if any(dones.values()):
                self.buffer_manager.finish_episode(env_id=env_idx)

    def finish_episode(self, env_id: int = 0):
        """Finish current episode if using EpisodeBuffer."""
        if self.use_episode_buffer:
            self.buffer_manager.finish_episode(env_id=env_id)
    
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
            
        # Calculate global rewards if enabled
        if self.global_reward:
             global_rewards = []
             for t in batch_data:
                 global_r = sum(t['bold_r_t'].values()) / self.num_agents
                 global_rewards.append(global_r)
             
             global_rewards_tensor = torch.tensor(global_rewards, dtype=torch.float32, device=self.device)
             
             # Override individual rewards with global reward
             for agent_id in all_agent_rewards:
                 all_agent_rewards[agent_id] = global_rewards_tensor

        # === 2. Extract Pre-computed Advantages and Returns ===
        agent_returns = {}
        agent_advantages = {}
        
        for agent_id in range(self.num_agents):
            if agent_id not in batch_data[0]['bold_o_t']:
                continue
            
            returns_list = [t['returns'][agent_id] for t in batch_data]
            advantages_list = [t['advantages'][agent_id] for t in batch_data]
            
            agent_returns[agent_id] = torch.tensor(returns_list, dtype=torch.float32, device=self.device)
            agent_advantages[agent_id] = torch.tensor(advantages_list, dtype=torch.float32, device=self.device)

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
            
            # 2. Compute GAE for each trajectory
            for env_id, traj in env_trajectories.items():
                if len(traj) == 0:
                    continue
                
                # For each agent, compute GAE
                for agent_id in range(self.num_agents):
                    if agent_id not in traj[0]['bold_o_t']:
                        continue
                    
                    # Extract agent data
                    agent_obs_seq = torch.stack([t['bold_o_t'][agent_id] for t in traj]).to(self.device)
                    
                    if self.global_reward:
                        agent_rewards_seq = torch.tensor([sum(t['bold_r_t'].values()) / self.num_agents for t in traj],
                                                        dtype=torch.float32, device=self.device)
                    else:
                        agent_rewards_seq = torch.tensor([t['bold_r_t'][agent_id] for t in traj],
                                                        dtype=torch.float32, device=self.device)
                    
                    agent_dones_seq = torch.tensor([t['dones'][agent_id] for t in traj], dtype=torch.float32, device=self.device)
                    
                    # Compute values
                    with torch.no_grad():
                        agent_values = self.value_net(agent_obs_seq).squeeze()
                        last_next_obs = traj[-1]['next_observations'][agent_id]
                        is_terminal = agent_dones_seq[-1].item()
                        next_val = 0.0 if is_terminal else self.value_net(last_next_obs.unsqueeze(0)).item()
                    
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
            print(f"[ERROR IPPO] Failed to restructure rollout data: {e}")
            import traceback
            traceback.print_exc()
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
