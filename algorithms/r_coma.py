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


class rCOMAAgent:
    """
    Single agent for COMA-RNN algorithm.
    Uses shared RNN-based policy network (pymarl-style parameter sharing).
    Agent ID is concatenated as one-hot to the observation input.
    """
    
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        recurrent_N: int = 1,
        num_agents: int = 1,
        policy_net: RNNPolicyNetwork = None,
        device: str = "cpu"
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.recurrent_N = recurrent_N
        self.num_agents = num_agents
        self.device = device
        
        # Shared RNN Policy Network (pymarl-style parameter sharing)
        self.policy_net = policy_net
        
        # RNN hidden state
        self.rnn_state_policy = None
        self.reset_rnn_state()
    
    def _add_agent_id(self, obs: torch.Tensor) -> torch.Tensor:
        """Concatenate agent_id one-hot encoding to observation (pymarl-style)."""
        leading_shape = obs.shape[:-1]
        agent_id_onehot = torch.zeros(*leading_shape, self.num_agents, device=obs.device)
        agent_id_onehot[..., self.agent_id] = 1.0
        return torch.cat([obs, agent_id_onehot], dim=-1)
        
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
        """Get action from shared RNN policy (single environment)."""
        with torch.no_grad():
            obs_with_id = self._add_agent_id(obs)
            obs_in = obs_with_id.unsqueeze(0) if obs_with_id.dim() == 1 else obs_with_id
            
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
            obs_with_id = self._add_agent_id(obs_batch)
            prev_states = self.rnn_states_policy_batch.clone()
            
            action, log_prob, next_states = self.policy_net.get_action_and_log_prob(
                obs_with_id, 
                self.rnn_states_policy_batch, 
                masks_batch, 
                available_actions_batch
            )
            
            self.rnn_states_policy_batch = next_states
            
        return action, log_prob, prev_states


class rCOMAAlgorithm:
    """
    COMA (Counterfactual Multi-Agent Policy Gradients) Algorithm with RNN support.
    
    Based on pymarl implementation:
    - RNN-based policy with parameter sharing (agent_id one-hot concatenated)
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
        target_update_interval: int = 10,
        critic_train_steps: int = 1,
        device: str = "cpu",
        use_episode_buffer: bool = True,
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
        
        self.use_episode_buffer = use_episode_buffer
        
        # Episode buffer for COMA (episode-based training)
        self.buffer_manager = EpisodeBuffer(max_episodes=100)
        
        # Shared Policy Network (pymarl-style parameter sharing)
        # Input: obs + agent_id one-hot
        self.shared_policy = RNNPolicyNetwork(
            obs_dim + num_agents, action_dim, hidden_dim, recurrent_N
        )
        self.shared_policy.to(self.device)
        
        self.policy_optimizer = optim.RMSprop(
            self.shared_policy.parameters(), lr=lr_actor, alpha=0.99, eps=1e-5
        )
        
        # Create agents with shared policy
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = rCOMAAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                recurrent_N=recurrent_N,
                num_agents=num_agents,
                policy_net=self.shared_policy,
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
            device = next(self.shared_policy.parameters()).device
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
            available_actions=available_actions,
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
            available_actions=available_actions,
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
        mask: torch.Tensor,          # [Batch, T]
        target_qs: torch.Tensor,     # [Batch, T+1, num_agents]
        gamma: float,
        td_lambda: float
    ) -> torch.Tensor:
        """
        Build TD-lambda targets     
        Args:
            rewards: [B, T]
            terminated: [B, T]
            mask: [B, T] (1=valid, 0=padded/post-terminal)
            target_qs: [B, T+1, N] — Q-values including next state
            gamma, td_lambda: scalars
        Returns:
            [B, T, N]
        """
        B, Tp1, N = target_qs.shape
        T = Tp1 - 1
        
        ret = target_qs.new_zeros(B, Tp1, N)
        # Initialize last lambda-return for not-terminated episodes
        ret[:, -1] = target_qs[:, -1] * (1 - terminated.sum(dim=1, keepdim=True))
        
        # Backwards recursive update
        for t in range(T-1, -1, -1):
            ret[:, t] = td_lambda * gamma * ret[:, t+1] + mask[:, t:t+1] * (
                rewards[:, t:t+1] + (1 - td_lambda) * gamma * target_qs[:, t+1] * (1 - terminated[:, t:t+1])
            )
        
        # Return from t=0 to t=T-1
        return ret[:, :-1]

    def _pad_and_stack(self, episodes: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        """
        Convert list of episodes to batch of tensors with padding.
        """
        batch_size = len(episodes)
        max_t = max(len(ep) for ep in episodes)  # T transitions
        Tp1 = max_t + 1  # T+1 for obs/actions
        
        # T+1 length tensors (obs, actions, avail_actions)
        joint_obs = torch.zeros(batch_size, Tp1, self.obs_dim * self.num_agents, device=self.device)
        agent_obs = torch.zeros(batch_size, Tp1, self.num_agents, self.obs_dim, device=self.device)
        actions = torch.zeros(batch_size, Tp1, self.num_agents, dtype=torch.long, device=self.device)
        available_actions = torch.ones(batch_size, Tp1, self.num_agents, self.action_dim, device=self.device)
        has_available_actions = False
        
        # T length tensors (rewards, terminated, masks)
        rewards = torch.zeros(batch_size, max_t, device=self.device)
        terminated = torch.zeros(batch_size, max_t, device=self.device)
        masks = torch.zeros(batch_size, max_t, device=self.device)
        
        for b, episode in enumerate(episodes):
            t_len = len(episode)
            
            for t, trans in enumerate(episode):
                # Obs and actions at time t
                joint_obs[b, t] = self._concatenate_observations(trans['bold_o_t']).to(self.device)
                agent_obs[b, t] = torch.stack([trans['bold_o_t'][i] for i in range(self.num_agents)]).to(self.device)
                actions[b, t] = torch.tensor([trans['bold_a_t'][i] for i in range(self.num_agents)], dtype=torch.long, device=self.device)
                rewards[b, t] = sum(trans['bold_r_t'].values()) / self.num_agents
                terminated[b, t] = 1.0 if any(trans['dones'].values()) else 0.0
                
                # Available actions
                avail = trans.get('available_actions', None)
                if avail is not None:
                    has_available_actions = True
                    for i in range(self.num_agents):
                        if i in avail:
                            available_actions[b, t, i] = torch.tensor(avail[i], dtype=torch.float32, device=self.device)
                
                # Fill T+1 slot with next_observations from last transition
                if t == t_len - 1 and 'next_observations' in trans:
                    next_obs = trans['next_observations']
                    joint_obs[b, t+1] = self._concatenate_observations(next_obs).to(self.device)
                    agent_obs[b, t+1] = torch.stack([next_obs[i] for i in range(self.num_agents)]).to(self.device)
            
            # Masks: valid timesteps, zeroed after termination (pymarl style)
            masks[b, :t_len] = 1.0
        
        # pymarl: mask[:, 1:] *= (1 - terminated[:, :-1])
        if max_t > 1:
            masks[:, 1:] = masks[:, 1:] * (1 - terminated[:, :-1])
        
        result = {
            'joint_obs': joint_obs,              # [B, T+1, Joint_Obs]
            'agent_obs': agent_obs,              # [B, T+1, N, Obs]
            'actions': actions,                  # [B, T+1, N]
            'rewards': rewards,                  # [B, T]
            'terminated': terminated,            # [B, T]
            'masks': masks,                      # [B, T]
        }
        
        if has_available_actions:
            result['available_actions'] = available_actions  # [B, T+1, N, A]
        
        return result

    def _train_critic_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Train critic using batch data (pymarl style: per-timestep reverse update).
        """
        joint_obs = batch['joint_obs']        # [B, T+1, Joint]
        agent_obs = batch['agent_obs']        # [B, T+1, N, Obs]
        actions = batch['actions']            # [B, T+1, N]
        rewards = batch['rewards']            # [B, T] (T = T+1 - 1)
        terminated = batch['terminated']      # [B, T]
        masks = batch['masks']                # [B, T]
        
        B, Tp1, N = actions.shape
        T = Tp1 - 1
        
        # Prepare actions onehot [B, T+1, N, A]
        actions_onehot = self._to_one_hot(actions)
        
        # Last actions (shifted right, pad with 0)
        zero_action = torch.zeros(B, 1, N, self.action_dim, device=self.device)
        last_actions_onehot = torch.cat([zero_action, actions_onehot[:, :-1]], dim=1)
        
        # Target Q-values [B, T+1, N] for TD-lambda
        with torch.no_grad():
            target_qs = torch.zeros(B, Tp1, N, device=self.device)
            for i in range(N):
                tq = self.target_critic.forward_batch(
                    joint_obs, agent_obs[:, :, i], actions_onehot, last_actions_onehot, i
                )  # [B, T+1, A]
                target_qs[:, :, i] = tq.gather(2, actions[:, :, i:i+1]).squeeze(2)
        
        # Build TD-lambda targets [B, T, N]
        targets = self._build_td_lambda_targets(
            rewards, terminated, masks, target_qs, self.gamma, self.td_lambda
        )
        
        # Per-timestep reverse critic update (pymarl style)
        # Compute Q-values fresh at each timestep to avoid retain_graph issues
        total_critic_loss = 0.0
        for t in reversed(range(T)):
            if masks[:, t].sum() == 0:
                continue
            
            # Fresh forward pass for this timestep only
            critic_loss = 0
            for i in range(N):
                q_t = self.critic(
                    joint_obs[:, t], agent_obs[:, t, i],
                    actions_onehot[:, t], last_actions_onehot[:, t], i
                )  # [B, A]
                q_taken_t = q_t.gather(1, actions[:, t, i:i+1]).squeeze(1)  # [B]
                target_t = targets[:, t, i]  # [B]
                td_error = (q_taken_t - target_t.detach()) * masks[:, t]
                loss = (td_error ** 2).sum() / masks[:, t].sum().clamp(min=1.0)
                critic_loss = critic_loss + loss
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            self.critic_training_steps += 1
            total_critic_loss += critic_loss.item()
        
        return {'critic_loss': total_critic_loss / max(T, 1)}

    def _train_actors_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Train actors using batch data and counterfactual baseline (pymarl style).
        Uses shared policy network with batch RNN forward.
        """
        joint_obs = batch['joint_obs'][:, :-1]      # [B, T, Joint] (exclude last)
        agent_obs = batch['agent_obs'][:, :-1]      # [B, T, N, Obs]
        actions = batch['actions'][:, :-1]           # [B, T, N]
        masks = batch['masks']                       # [B, T]
        avail_actions = batch.get('available_actions', None)  # [B, T+1, N, A] or None
        
        B, T, N = actions.shape
        
        # Actions onehot [B, T, N, A]
        actions_onehot = self._to_one_hot(actions)
        
        # Last actions
        zero_action = torch.zeros(B, 1, N, self.action_dim, device=self.device)
        last_actions_onehot = torch.cat([zero_action, actions_onehot[:, :-1]], dim=1)
        
        # Flatten mask for pymarl style: [B*T*N]
        mask_flat = masks.unsqueeze(2).expand(-1, -1, N).reshape(-1)
        
        actor_losses = {}
        total_pg_loss = 0
        total_entropy_loss = 0
        
        for i in range(N):
            # 1. Prepare observation with agent_id one-hot
            agent_obs_i = agent_obs[:, :, i]  # [B, T, Obs]
            agent_id_onehot = torch.zeros(B, T, N, device=self.device)
            agent_id_onehot[:, :, i] = 1.0
            obs_with_id = torch.cat([agent_obs_i, agent_id_onehot], dim=-1)  # [B, T, Obs+N]

            # 2. Batch RNN forward
            obs_flat = obs_with_id.transpose(0, 1).contiguous().reshape(T * B, -1)
            masks_rnn = masks.transpose(0, 1).contiguous().reshape(T * B, 1)
            h0 = torch.zeros(B, self.recurrent_N, self.hidden_dim, device=self.device)
            
            x = self.shared_policy.feature_extractor(obs_flat)
            x, _ = self.shared_policy.rnn(x, h0, masks_rnn)
            logits = self.shared_policy.output_layer(x)
            logits = logits.view(T, B, -1).transpose(0, 1).contiguous()  # [B, T, A]
            
            # 3. pymarl style: softmax -> mask -> renormalize (no in-place ops)
            probs = F.softmax(logits, dim=-1)  # [B, T, A]
            if avail_actions is not None:
                agent_avail = avail_actions[:, :-1, i]  # [B, T, A] (exclude last)
                probs = probs * agent_avail
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                probs = probs * agent_avail
            
            # 4. Get Q-values with no_grad
            with torch.no_grad():
                q_vals = self.critic.forward_batch(
                    joint_obs, agent_obs[:, :, i], actions_onehot, last_actions_onehot, i
                )  # [B, T, A]
            
            # 5. Counterfactual Baseline
            baseline = (probs * q_vals).sum(dim=-1).detach()  # [B, T]
            q_taken = q_vals.gather(2, actions[:, :, i:i+1]).squeeze(2)  # [B, T]
            advantages = (q_taken - baseline).detach()  # [B, T]
            
            # 6. Policy gradient loss (pymarl style: log(pi_taken))
            pi_taken = probs.gather(2, actions[:, :, i:i+1]).squeeze(2)  # [B, T]
            pi_taken = torch.where(masks > 0, pi_taken, torch.ones_like(pi_taken))  # log(1)=0 for masked
            log_pi_taken = torch.log(pi_taken + 1e-8)
            
            pg_loss = -((advantages * log_pi_taken) * masks).sum() / masks.sum().clamp(min=1.0)
            
            # Entropy
            log_probs = torch.log(probs + 1e-8)
            entropy = -(probs * log_probs).sum(dim=-1)  # [B, T]
            entropy_loss = -(entropy * masks).sum() / masks.sum().clamp(min=1.0)
            
            total_pg_loss += pg_loss
            total_entropy_loss += entropy_loss
            
            actor_losses[f'agent_{i}_pg_loss'] = pg_loss.item()
            actor_losses[f'agent_{i}_entropy'] = -entropy_loss.item()
        
        # Single update for shared policy
        total_loss = total_pg_loss + self.entropy_coef * total_entropy_loss
        
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.shared_policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
            
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
