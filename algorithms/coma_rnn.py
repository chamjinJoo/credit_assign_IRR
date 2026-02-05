import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import copy

from algorithms.networks import RNNPolicyNetwork
from algorithms.networks.coma_rnn_critic import COMARNNCritic
from utils.buffer import EpisodeBuffer


class COMARNNAgent:
    """
    Single agent for COMA-RNN algorithm.
    Uses RNN-based policy network for partial observability.
    """
    
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        recurrent_N: int = 1,
        lr_actor: float = 3e-4,
        device: str = "cpu"
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.recurrent_N = recurrent_N
        self.device = device
        
        # RNN Policy Network
        self.policy_net = RNNPolicyNetwork(obs_dim, action_dim, hidden_dim, recurrent_N)
        self.policy_net.to(device)
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr_actor, alpha=0.99, eps=1e-5)
        
        # RNN hidden state
        self.rnn_state_policy = None
        self.reset_rnn_state()
        
    def reset_rnn_state(self):
        """Reset RNN hidden state (single environment)."""
        self.rnn_state_policy = torch.zeros(self.recurrent_N, self.hidden_dim).to(self.device)
        
    def reset_rnn_states_batch(self, n_envs: int):
        """Reset RNN states for multiple environments (batch mode)."""
        self.n_envs = n_envs
        self.rnn_states_policy_batch = torch.zeros(n_envs, self.recurrent_N, self.hidden_dim).to(self.device)

    def get_action(
        self, 
        obs: torch.Tensor, 
        masks: torch.Tensor = None, 
        available_actions: torch.Tensor = None,
        return_rnn_state: bool = True
    ) -> Tuple[int, float, Optional[torch.Tensor]]:
        """Get action from RNN policy (single environment)."""
        with torch.no_grad():
            obs_in = obs.unsqueeze(0) if obs.dim() == 1 else obs
            
            if masks is None:
                batch_size = obs_in.size(0)
                masks_in = torch.ones(batch_size, 1, device=self.device)
                return_rnn_state = False
            else:
                masks_in = masks.unsqueeze(0) if masks.dim() == 1 else masks
            
            curr_rnn_state = self.rnn_state_policy.unsqueeze(0)
            
            if available_actions is not None:
                available_actions = available_actions.unsqueeze(0) if available_actions.dim() == 1 else available_actions

            action, log_prob, next_rnn_state = self.policy_net.get_action_and_log_prob(
                obs_in, 
                curr_rnn_state, 
                masks_in,
                available_actions
            )
            
            self.rnn_state_policy = next_rnn_state.squeeze(0)
        
        if return_rnn_state:
            return action.item(), log_prob.item(), curr_rnn_state.squeeze(0).clone()
        else:
            return action.item(), log_prob.item()

    def get_action_batch(
        self, 
        obs_batch: torch.Tensor, 
        masks_batch: torch.Tensor, 
        available_actions_batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get actions for a batch of environments."""
        with torch.no_grad():
            prev_states = self.rnn_states_policy_batch.clone()
            
            action, log_prob, next_states = self.policy_net.get_action_and_log_prob(
                obs_batch, 
                self.rnn_states_policy_batch, 
                masks_batch, 
                available_actions_batch
            )
            
            self.rnn_states_policy_batch = next_states
            
        return action, log_prob, prev_states


class COMARNNAlgorithm:
    """
    COMA (Counterfactual Multi-Agent Policy Gradients) Algorithm with RNN support.
    
    Based on pymarl implementation:
    - RNN-based policy for partial observability
    - Centralized critic with counterfactual baseline
    - TD-lambda target for value estimation
    - Episode-based training
    """
    
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        recurrent_N: int = 1,
        lr_actor: float = 5e-4,
        lr_critic: float = 5e-4,
        gamma: float = 0.99,
        td_lambda: float = 0.8,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 10.0,
        target_update_interval: int = 200,
        critic_train_steps: int = 1,
        device: str = "cpu",
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.recurrent_N = recurrent_N
        
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_update_interval = target_update_interval
        self.critic_train_steps = critic_train_steps
        
        # Episode buffer for COMA (episode-based training)
        self.buffer_manager = EpisodeBuffer(max_episodes=32)
        
        # Create agents (with shared policy parameters like pymarl MAC)
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = COMARNNAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                recurrent_N=recurrent_N,
                lr_actor=lr_actor,
                device=self.device
            )
        
        # Centralized Critic (pymarl style)
        self.critic = COMARNNCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            hidden_dim=128,  # pymarl uses 128
            device=self.device
        ).to(self.device)
        
        # Target Critic
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.critic_optimizer = optim.RMSprop(
            self.critic.parameters(), 
            lr=lr_critic, 
            alpha=0.99, 
            eps=1e-5
        )
        
        self.training_steps = 0
        self.critic_training_steps = 0
        self.last_target_update_step = 0
        
        # Temp storage for hidden states and masks
        self.temp_hidden_states = {}
        self.temp_masks = {}
        self.last_dones = {i: False for i in range(self.num_agents)}
        
        # For parallel environments
        self.temp_hidden_states_batch = []
        self.temp_masks_batch = []
        self.last_dones_batch = []

    def reset_agents(self):
        """Reset all agent RNN states (single environment)."""
        for agent in self.agents.values():
            agent.reset_rnn_state()
        self.last_dones = {i: False for i in range(self.num_agents)}

    def reset_agents_batch(self, n_envs: int):
        """Reset agents for multiple environments (batch mode)."""
        for agent in self.agents.values():
            agent.reset_rnn_states_batch(n_envs)
        self.last_dones_batch = [{i: False for i in range(self.num_agents)} for _ in range(n_envs)]

    def _concatenate_observations(self, observations: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Concatenate all agent observations to form joint observation."""
        sorted_obs = [observations[i] for i in range(self.num_agents) if i in observations]
        if not sorted_obs:
            device = next(self.agents[0].policy_net.parameters()).device
            return torch.zeros(self.num_agents * self.obs_dim, device=device)
        return torch.cat(sorted_obs, dim=-1)

    def get_actions(
        self, 
        observations: Dict[int, torch.Tensor], 
        available_actions: Optional[Dict[int, List[int]]] = None
    ) -> Dict[int, Tuple[int, float]]:
        """Get actions from all agents (single environment)."""
        actions = {}
        self.temp_hidden_states = {}
        self.temp_masks = {}
        
        for agent_id, obs in observations.items():
            mask = 1.0 if not self.last_dones.get(agent_id, False) else 0.0
            mask_tensor = torch.tensor([mask], dtype=torch.float32, device=self.device)
            
            avail_actions_tensor = None
            if available_actions is not None and agent_id in available_actions:
                avail_actions_tensor = torch.tensor(
                    available_actions[agent_id], 
                    dtype=torch.float32, 
                    device=self.device
                )

            action, log_prob, p_state = self.agents[agent_id].get_action(
                obs, mask_tensor, available_actions=avail_actions_tensor
            )
            
            actions[agent_id] = (action, log_prob)
            self.temp_hidden_states[agent_id] = p_state
            self.temp_masks[agent_id] = mask
            
        return actions

    def get_actions_batch(
        self,
        observations_batch: List[Dict[int, torch.Tensor]],
        available_actions_batch: Optional[List[Dict[int, List[int]]]] = None
    ) -> List[Dict[int, Tuple[int, float]]]:
        """Get actions for a batch of environments (ParallelEnv execution)."""
        n_envs = len(observations_batch)
        actions_batch = [{} for _ in range(n_envs)]
        
        self.temp_hidden_states_batch = [{} for _ in range(n_envs)]
        self.temp_masks_batch = [{} for _ in range(n_envs)]
        
        for agent_id in range(self.num_agents):
            agent_obs_list = []
            agent_masks_list = []
            agent_avail_list = []
            has_avail = (available_actions_batch is not None)
            
            for i in range(n_envs):
                obs = observations_batch[i][agent_id]
                agent_obs_list.append(obs)
                
                mask = 1.0 if not self.last_dones_batch[i].get(agent_id, False) else 0.0
                agent_masks_list.append(mask)
                self.temp_masks_batch[i][agent_id] = mask
                
                if has_avail and available_actions_batch[i] is not None and agent_id in available_actions_batch[i]:
                    agent_avail_list.append(available_actions_batch[i][agent_id])
                else:
                    agent_avail_list.append(None)
                    
            agent_obs_batch = torch.stack(agent_obs_list).to(self.device)
            agent_masks_batch = torch.tensor(agent_masks_list, dtype=torch.float32, device=self.device).unsqueeze(1)
            
            agent_avail_tensor = None
            if any(x is not None for x in agent_avail_list):
                agent_avail_tensor = torch.tensor(agent_avail_list, dtype=torch.float32, device=self.device)
            
            actions, log_probs, prev_p_states = self.agents[agent_id].get_action_batch(
                agent_obs_batch,
                agent_masks_batch,
                available_actions_batch=agent_avail_tensor
            )
            
            for i in range(n_envs):
                if isinstance(actions[i], torch.Tensor):
                    act = actions[i].item()
                    lp = log_probs[i].item()
                else:
                    act = actions[i]
                    lp = log_probs[i]
                    
                actions_batch[i][agent_id] = (act, lp)
                self.temp_hidden_states_batch[i][agent_id] = prev_p_states[i]
                
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
        available_actions: Optional[Dict[int, List[int]]] = None
    ):
        """Store transition in episode buffer."""
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
            hidden_states=self.temp_hidden_states.copy(),
            info=info,
            env_id=0  # Default to 0 for single env
        )

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
        hidden_states = self.temp_hidden_states_batch[env_idx]
        
        if info is None:
            info = {}
        info['env_id'] = env_idx
        
        self.buffer_manager.store_transition(
            prev_observations=prev_observations,
            prev_rewards=prev_rewards,
            prev_actions=prev_actions,
            observations=observations,
            actions=actions_only,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            hidden_states=hidden_states,
            info=info,
            env_id=env_idx  # Pass env_id
        )
        
        # Update done status and reset RNN if done
        env_done = False
        for agent_id in range(self.num_agents):
            self.last_dones_batch[env_idx][agent_id] = dones[agent_id]
            if dones[agent_id]:
                self.agents[agent_id].rnn_states_policy_batch[env_idx] = torch.zeros(
                   self.recurrent_N, self.hidden_dim, device=self.device
                )
        
        if any(dones.values()):
             env_done = True
        
        if env_done:
            self.buffer_manager.finish_episode(env_id=env_idx)

    def finish_episode(self, env_id: int = 0) -> Optional[List[Dict]]:
        """Finish current episode and move it to completed episodes."""
        return self.buffer_manager.finish_episode(env_id=env_id)

    def _to_one_hot(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert actions to one-hot encoding."""
        return F.one_hot(actions, num_classes=self.action_dim).float()

    def _build_td_lambda_targets(
        self,
        rewards: torch.Tensor,       # [Batch, T]
        terminated: torch.Tensor,    # [Batch, T]
        target_qs: torch.Tensor,     # [Batch, T, num_agents]
        mask: torch.Tensor,          # [Batch, T] (padding mask)
        gamma: float,
        td_lambda: float
    ) -> torch.Tensor:
        """
        Build TD-lambda targets with batch support.
        
        Args:
            rewards: Team rewards [Batch, T]
            terminated: Terminated flags [Batch, T]
            target_qs: Target Q-values for taken actions [Batch, T, num_agents]
            mask: Padding mask (1.0 for valid, 0.0 for padding) [Batch, T]
            gamma: Discount factor
            td_lambda: TD-lambda coefficient
            
        Returns:
            TD-lambda targets [Batch, T, num_agents]
        """
        B, T = rewards.shape
        targets = torch.zeros(B, T, self.num_agents, device=self.device)
        
        # Last step target
        running_target = target_qs[:, -1, :] if T > 0 else torch.zeros(B, self.num_agents, device=self.device)
        
        for t in reversed(range(T)):
            # Valid mask for this step (not terminated AND not padded)
            # If terminated, next value should be 0 (handled by mask logic in formula usually, but here explicit)
            # Target calculation uses NEXT step's value.
            
            mask_t = mask[:, t].unsqueeze(1) # [B, 1]
            term_t = terminated[:, t].unsqueeze(1) # [B, 1]
            reward_t = rewards[:, t].unsqueeze(1) # [B, 1]
            
            # If terminated at t, target is just reward (bootstrapping handled by 0 mask)
            # Actually, standard formula: r + gamma * (1-term) * ( ... )
             
            if t == T - 1:
                # Last step in sequence (might be terminal or truncated)
                target_val = reward_t + gamma * (1 - term_t) * running_target
            else:
                # TD-lambda: G_t^λ = r_t + γ * (1-d_t) * ((1-λ) * Q_{t+1} + λ * G_{t+1}^λ)
                next_q = target_qs[:, t + 1, :]
                target_val = reward_t + gamma * (1 - term_t) * (
                    (1 - td_lambda) * next_q + td_lambda * running_target
                )
            
            targets[:, t, :] = target_val
            running_target = target_val
            
            # Apply padding mask (zero out targets for padded steps to avoid garbage)
            # Though loss masking handles this, it's cleaner to zero it out.
            # targets[:, t, :] *= mask_t # Optional
        
        return targets

    def _pad_and_stack(self, episodes: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        """
        Convert list of episodes to batch of tensors with padding.
        
        Args:
            episodes: List of episodes (each is list of transitions)
            
        Returns:
            Batch dictionary with tensors [Batch, T_max, ...]
        """
        batch_size = len(episodes)
        max_t = max(len(ep) for ep in episodes)
        
        # Pre-allocate tensors
        joint_obs = torch.zeros(batch_size, max_t, self.obs_dim * self.num_agents, device=self.device)
        agent_obs = torch.zeros(batch_size, max_t, self.num_agents, self.obs_dim, device=self.device)
        actions = torch.zeros(batch_size, max_t, self.num_agents, dtype=torch.long, device=self.device)
        rewards = torch.zeros(batch_size, max_t, device=self.device)
        terminated = torch.zeros(batch_size, max_t, device=self.device)
        masks = torch.zeros(batch_size, max_t, device=self.device)
        
        # Pre-allocate for hidden states: [Batch, Num_Agents, Recurrent_N, Hidden_Dim]
        # We only need initial hidden states for the batch forward pass
        # But we might want to store all if needed, but for now just h0 is enough for re-rollout if we do full seq.
        # Actually, for training, we re-run the RNN on the full sequence starting from 0 (or stored h0).
        # Assuming episodes start from beginning, h0 is 0. 
        # If we use chunking, we need stored h0. Here we assume full episodes.
        
        for b, episode in enumerate(episodes):
            t_len = len(episode)
            
            # Collect episode data
            # Note: This loop might be slow in python for very long episodes/many episodes.
            # Optimized by gathering lists first then converting to tensor is usually faster.
            e_joint_obs = []
            e_agent_obs = []
            e_actions = []
            e_rewards = []
            e_terminated = []
            
            for t, trans in enumerate(episode):
                e_joint_obs.append(self._concatenate_observations(trans['bold_o_t']))
                e_agent_obs.append(torch.stack([trans['bold_o_t'][i] for i in range(self.num_agents)]))
                e_actions.append(torch.tensor([trans['bold_a_t'][i] for i in range(self.num_agents)], dtype=torch.long))
                e_rewards.append(sum(trans['bold_r_t'].values()) / self.num_agents)
                e_terminated.append(1.0 if any(trans['dones'].values()) else 0.0)
            
            # Stack and assign
            if t_len > 0:
                joint_obs[b, :t_len] = torch.stack(e_joint_obs).to(self.device)
                agent_obs[b, :t_len] = torch.stack(e_agent_obs).to(self.device)
                actions[b, :t_len] = torch.stack(e_actions).to(self.device)
                rewards[b, :t_len] = torch.tensor(e_rewards, device=self.device)
                terminated[b, :t_len] = torch.tensor(e_terminated, device=self.device)
                masks[b, :t_len] = 1.0
        
        return {
            'joint_obs': joint_obs,      # [B, T, Joint_Obs]
            'agent_obs': agent_obs,      # [B, T, N, Obs]
            'actions': actions,          # [B, T, N]
            'rewards': rewards,          # [B, T]
            'terminated': terminated,    # [B, T]
            'masks': masks               # [B, T]
        }

    def _train_critic_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Train critic using batch data.
        
        Args:
            batch: Batch dictionary from _pad_and_stack
            
        Returns:
            Critic training statistics
        """
        joint_obs = batch['joint_obs']      # [B, T, Joint_Diff]
        agent_obs = batch['agent_obs']      # [B, T, N, Obs]
        actions = batch['actions']          # [B, T, N]
        rewards = batch['rewards']          # [B, T]
        terminated = batch['terminated']    # [B, T]
        masks = batch['masks']              # [B, T]
        
        B, T, N = actions.shape
        
        # Prepare actions onehot
        actions_onehot = self._to_one_hot(actions) # [B, T, N, A]
        
        # Prepare last actions (shifted right, pad with 0 at start)
        zero_action = torch.zeros(B, 1, N, self.action_dim, device=self.device)
        last_actions_onehot = torch.cat([zero_action, actions_onehot[:, :-1]], dim=1) # [B, T, N, A]
        
        # Calculate Target Q-values (Batch forward)
        with torch.no_grad():
            target_qs = torch.zeros(B, T, N, device=self.device)
            # Optimized: Compute all agents in parallel if possible, or loop agents
            # COMACritic takes agent_id. We loop over agents.
            for i in range(N):
                # Target Critic Forward: [B, T, A]
                target_q_vals = self.target_critic.forward_batch(
                    joint_obs, 
                    agent_obs[:, :, i], 
                    actions_onehot, 
                    last_actions_onehot, 
                    i
                )
                # Gather taken actions: [B, T, 1]
                taken_target_q = target_q_vals.gather(2, actions[:, :, i:i+1])
                target_qs[:, :, i] = taken_target_q.squeeze(2)
        
        # Build TD-lambda targets
        targets = self._build_td_lambda_targets(
            rewards, terminated, target_qs, masks, self.gamma, self.td_lambda
        ) # [B, T, N]
        
        # Calculate Current Q-values and Loss
        critic_loss = 0
        
        for i in range(N):
            # Critic Forward: [B, T, A]
            q_vals = self.critic.forward_batch(
                joint_obs,
                agent_obs[:, :, i],
                actions_onehot,
                last_actions_onehot,
                i
            )
            # Gather taken actions: [B, T]
            q_taken = q_vals.gather(2, actions[:, :, i:i+1]).squeeze(2)
            
            # Masked Bellman Error
            td_error = (q_taken - targets[:, :, i].detach()) * masks
            
            # Mean over valid steps (sum / sum(mask))
            loss = (td_error ** 2).sum() / masks.sum().clamp(min=1.0)
            critic_loss += loss
        
        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        self.critic_training_steps += 1
        return {'critic_loss': critic_loss.item()}

    def _train_actors_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Train actors using batch data and counterfactual baseline.
        """
        joint_obs = batch['joint_obs']      # [B, T, Joint]
        agent_obs = batch['agent_obs']      # [B, T, N, Obs]
        actions = batch['actions']          # [B, T, N]
        masks = batch['masks']              # [B, T]
        
        B, T, N = actions.shape
        
        # Actions onehot
        actions_onehot = self._to_one_hot(actions) # [B, T, N, A]
        
        # Last actions
        zero_action = torch.zeros(B, 1, N, self.action_dim, device=self.device)
        last_actions_onehot = torch.cat([zero_action, actions_onehot[:, :-1]], dim=1) # [B, T, N, A]
        
        actor_losses = {}
        
        # Loop over agents
        for i in range(N):
            # 1. Get Policy Outputs (Batch RNN forward)
            # Reshape inputs for RNN: [B, T, Obs]
            agent_obs_i = agent_obs[:, :, i] 
            
            # Initial hidden state: Zeros [B, Recurrent_N, Hidden]
            h0 = torch.zeros(B, self.recurrent_N, self.hidden_dim, device=self.device)
            
            # Fast way: Pass [T, B, Obs] through GRU directly.
            # Accessing internal GRU: self.agents[i].policy_net.rnn.rnn
            
            logits_list = []
            h_t = h0 # [B, recurrent_N, H]
            
            for t_step in range(T):
                obs_t = agent_obs[:, t_step, i] # [B, Obs]
                mask_t = masks[:, t_step].unsqueeze(1) # [B, 1]
                
                # Check RNNPolicyNetwork forward: usually returns logits or action/logprob.
                # We need logits/probs for analytical gradient.
                # RNNPolicyNetwork call: x, h = net(x, h, mask) -> x is logits
                
                step_logits, h_t = self.agents[i].policy_net(obs_t, h_t, mask_t)
                logits_list.append(step_logits)
            
            logits = torch.stack(logits_list, dim=1) # [B, T, A]
            
            # 2. Compute Probabilities
            probs = F.softmax(logits, dim=-1) # [B, T, A]
            log_probs = F.log_softmax(logits, dim=-1) # [B, T, A]
            
            # 3. Get Q-values (Critic)
            q_vals = self.critic.forward_batch(
                joint_obs, agent_obs[:, :, i], actions_onehot, last_actions_onehot, i
            ) # [B, T, A]
            
            # 4. Counterfactual Baseline
            # ∑_a π(a) * Q(s, u^{-i}, a)
            baseline = (probs * q_vals).sum(dim=2, keepdim=True) # [B, T, 1]
            
            # 5. Advantage
            q_taken = q_vals.gather(2, actions[:, :, i:i+1]) # [B, T, 1]
            advantages = (q_taken - baseline).detach() # [B, T, 1]
            
            # 6. Loss
            # Log prob of taken action
            log_prob_taken = log_probs.gather(2, actions[:, :, i:i+1]) # [B, T, 1]
            
            # Masked Mean PG Loss
            pg_loss = -(log_prob_taken * advantages * masks.unsqueeze(2)).sum() / masks.sum().clamp(min=1.0)
            
            # Entropy
            entropy = -(probs * log_probs).sum(dim=2) # [B, T]
            entropy_loss = -(entropy * masks).sum() / masks.sum().clamp(min=1.0)
            
            total_loss = pg_loss + self.entropy_coef * entropy_loss
            
            # Update
            self.agents[i].optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[i].policy_net.parameters(), self.max_grad_norm)
            self.agents[i].optimizer.step()
            
            actor_losses[f'agent_{i}_pg_loss'] = pg_loss.item()
            actor_losses[f'agent_{i}_entropy'] = -entropy_loss.item()
            
        return actor_losses

    def update(self, **kwargs) -> Dict[str, float]:
        """
        Update agents using completed episodes with batch processing.
        """
        if not self.buffer_manager.has_completed_episodes():
            return {}

        completed_episodes = self.buffer_manager.get_completed_episodes()
        
        if not completed_episodes:
            return {}
        
        # Prepare batch
        batch = self._pad_and_stack(completed_episodes)
        
        losses = {}
        
        # Train critic (Batch)
        critic_stats = self._train_critic_batch(batch)
        losses.update(critic_stats)
        
        # Train actors (Batch)
        actor_stats = self._train_actors_batch(batch)
        losses.update(actor_stats)
        
        # Update target network
        if (self.critic_training_steps - self.last_target_update_step) >= self.target_update_interval:
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.last_target_update_step = self.critic_training_steps
        
        self.training_steps += 1
        
        # Clear completed episodes after processing
        self.buffer_manager.clear_completed_episodes()
                
        return losses

    def clear_buffer(self):
        """Clear all buffers."""
        self.buffer_manager.clear_all()
    
    def get_buffer_info(self) -> Dict[str, int]:
        """Get buffer information for monitoring."""
        return self.buffer_manager.get_buffer_info()
