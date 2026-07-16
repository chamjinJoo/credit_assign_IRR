import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import copy

from .networks.qmix_mixer import QMixer
from .qmix_preprocessing import (
    qmix_pack_episode,
    qmix_pad_and_stack_legacy,
    qmix_stack_packed_episodes,
)
from utils.buffer import EpisodeBuffer


class QMIXRNNAgent(nn.Module):
    """
    RNN-based Q-network agent for QMIX.
    
    Architecture follows PyMARL's RNNAgent:
        obs (with agent_id one-hot) → fc1 → ReLU → GRUCell → fc2 → Q-values
    
    Uses parameter sharing across agents (agent_id is concatenated as one-hot).
    """
    
    def __init__(
        self,
        input_shape: int,
        action_dim: int,
        hidden_dim: int = 64,
    ):
        super(QMIXRNNAgent, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def init_hidden(self) -> torch.Tensor:
        """Initialize hidden state. Returns shape [1, hidden_dim]."""
        return self.fc1.weight.new(1, self.hidden_dim).zero_()
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            inputs: [B, input_shape]
            hidden_state: [B, hidden_dim]
            
        Returns:
            q_values: [B, action_dim]
            hidden_state: [B, hidden_dim]
        """
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class rQMIXAgent:
    """
    Single agent wrapper for QMIX.
    Manages RNN hidden state and epsilon-greedy action selection.
    Uses shared Q-network (parameter sharing via agent_id one-hot).
    """
    
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_agents: int = 1,
        q_network: QMIXRNNAgent = None,
        device: str = "cpu"
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.device = device
        
        # Shared Q-network (parameter sharing)
        self.q_network = q_network
        # Alias for Trainer compatibility (Trainer accesses agent.policy_net)
        self.policy_net = q_network
        
        # RNN hidden state (single env)
        self.rnn_state = None
        # Batch RNN states (parallel envs)
        self.rnn_states_batch = None
        # Previous action tracking
        self.prev_action = None
        self.prev_actions_batch = None
        
        self.reset_rnn_state()
    
    def _build_input(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Build input: [obs (potentially with agent_id from trainer), prev_action_onehot].

        Note: agent_id can be added by trainer if config.algorithm.add_agent_id=True.
        This method only adds previous action one-hot encoding.
        """
        # Append previous action one-hot
        if self.prev_action is not None:
            prev_act = self.prev_action
            if prev_act.dim() < obs.dim():
                prev_act = prev_act.unsqueeze(0)
            return torch.cat([obs, prev_act], dim=-1)
        else:
            zeros = torch.zeros(*obs.shape[:-1], self.action_dim, device=obs.device)
            return torch.cat([obs, zeros], dim=-1)

    def _build_input_batch(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """
        Build input for batch: [obs (potentially with agent_id from trainer), prev_action_onehot].

        Note: agent_id can be added by trainer if config.algorithm.add_agent_id=True.
        This method only adds previous action one-hot encoding.
        """
        if self.prev_actions_batch is not None:
            return torch.cat([obs_batch, self.prev_actions_batch], dim=-1)
        else:
            zeros = torch.zeros(obs_batch.size(0), self.action_dim, device=obs_batch.device)
            return torch.cat([obs_batch, zeros], dim=-1)
    
    def reset_rnn_state(self):
        """Reset RNN hidden state (single environment)."""
        self.rnn_state = torch.zeros(1, self.hidden_dim, device=self.device)
        self.prev_action = torch.zeros(self.action_dim, device=self.device)
    
    def reset_rnn_states_batch(self, n_envs: int):
        """Reset RNN states for multiple environments (batch mode)."""
        self.n_envs = n_envs
        self.rnn_states_batch = torch.zeros(n_envs, self.hidden_dim, device=self.device)
        self.prev_actions_batch = torch.zeros(n_envs, self.action_dim, device=self.device)
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        epsilon: float = 0.0,
        available_actions: torch.Tensor = None,
    ) -> Tuple[int, float]:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            obs: observation tensor
            epsilon: exploration rate
            available_actions: binary mask of available actions
            
        Returns:
            (action, log_prob) — log_prob is always 0.0 for compatibility
        """
        with torch.no_grad():
            inputs = self._build_input(obs)
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            
            q_values, self.rnn_state = self.q_network(inputs, self.rnn_state)
            q_values = q_values.squeeze(0)  # [action_dim]
            
            # Mask unavailable actions
            if available_actions is not None:
                q_values[available_actions == 0] = -1e10
            
            # Epsilon-greedy
            if np.random.random() < epsilon:
                if available_actions is not None:
                    avail_indices = torch.nonzero(available_actions, as_tuple=False).squeeze(-1)
                    action = avail_indices[torch.randint(len(avail_indices), (1,))].item()
                else:
                    action = np.random.randint(0, self.action_dim)
            else:
                action = q_values.argmax().item()
            
            # Update prev_action
            self.prev_action = F.one_hot(
                torch.tensor(action, device=self.device), self.action_dim
            ).float()
        
        return action, 0.0
    
    def get_action_batch(
        self, 
        obs_batch: torch.Tensor,
        epsilon: float = 0.0,
        available_actions_batch: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get actions for a batch of environments."""
        with torch.no_grad():
            inputs = self._build_input_batch(obs_batch)
            q_values, self.rnn_states_batch = self.q_network(inputs, self.rnn_states_batch)
            
            # Mask unavailable actions
            if available_actions_batch is not None:
                q_values[available_actions_batch == 0] = -1e10
            
            # Epsilon-greedy: random mask
            n_envs = obs_batch.size(0)
            random_mask = (torch.rand(n_envs, device=self.device) < epsilon)
            
            # Greedy actions
            greedy_actions = q_values.argmax(dim=-1)
            
            # Random actions
            if available_actions_batch is None or bool((available_actions_batch > 0).all().item()):
                random_actions = torch.randint(0, self.action_dim, (n_envs,), device=self.device)
            else:
                avail_probs = available_actions_batch.float().clamp_min(0.0)
                no_available = avail_probs.sum(dim=-1) <= 0
                if no_available.any():
                    avail_probs = avail_probs.clone()
                    avail_probs[no_available] = 1.0
                random_actions = torch.multinomial(avail_probs, num_samples=1).squeeze(-1)
            
            # Combine
            actions = torch.where(random_mask, random_actions, greedy_actions)
            
            # Update prev_actions
            self.prev_actions_batch = F.one_hot(actions, self.action_dim).float()
            
            # Dummy log_probs for compatibility
            log_probs = torch.zeros(n_envs, device=self.device)
        
        return actions, log_probs


class rQMIXAlgorithm:
    """
    QMIX Algorithm with RNN support.
    
    Based on PyMARL implementation (https://github.com/oxwhirl/pymarl):
    - RNN-based Q-network with parameter sharing (agent_id one-hot concatenated)
    - QMixer hypernetwork for value decomposition (monotonic mixing)
    - Double Q-learning for stable training
    - Epsilon-greedy exploration with linear annealing
    - Episode-based training with EpisodeBuffer
    
    Follows the same interface pattern as rCOMAAlgorithm for Trainer compatibility.
    """
    
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.99,
        max_grad_norm: float = 10.0,
        target_update_interval: int = 200,
        epsilon_start: float = 1.0,
        epsilon_finish: float = 0.05,
        epsilon_anneal_time: int = 50000,
        double_q: bool = True,
        mixing_embed_dim: int = 32,
        hypernet_layers: int = 2,
        hypernet_embed: int = 64,
        mixer_init_scale: float = 1.0e-3,
        reward_scale: float = 1.0,
        replay_buffer_size: int = 5000,   # max episodes to keep in replay buffer
        train_batch_size: int = 32,       # number of episodes per training batch
        qmix_update_epochs: Optional[int] = None,
        learning_starts: int = 0,
        device: str = "cpu",
        use_episode_buffer: bool = True,
        # Additional params for compatibility with the config system
        gae_lambda: float = 0.95,        # unused, for config compat
        clip_epsilon: float = 0.2,       # unused, for config compat
        entropy_coef: float = 0.01,      # unused, for config compat
        value_coef: float = 0.5,         # unused, for config compat
        ppo_epochs: int = 4,             # unused, for config compat
        batch_size: int = 32,            # unused, for config compat
        num_batches: int = 1,            # unused, for config compat
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.target_update_interval = target_update_interval
        self.double_q = double_q
        
        # Epsilon-greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_finish = epsilon_finish
        self.epsilon_anneal_time = epsilon_anneal_time
        self.epsilon = epsilon_start
        
        self.use_episode_buffer = use_episode_buffer
        self.train_batch_size = train_batch_size
        self.qmix_update_epochs = qmix_update_epochs
        self.learning_starts = max(0, int(learning_starts))
        self.reward_scale = float(reward_scale)
        self.mixer_init_scale = float(mixer_init_scale)
        
        # Episode replay buffer (off-policy: episodes are kept across iterations)
        self.buffer_manager = EpisodeBuffer(max_episodes=replay_buffer_size)

        # Input shape: obs (potentially with agent_id from trainer) + prev_action_onehot
        # Note: trainer adds agent_id if config.algorithm.add_agent_id=True,
        # so obs_dim already includes agent_id dimension in that case
        input_shape = obs_dim + action_dim

        # Shared Q-network (PyMARL-style RNNAgent)
        self.shared_q_network = QMIXRNNAgent(
            input_shape=input_shape,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        
        # Target Q-network
        self.target_q_network = copy.deepcopy(self.shared_q_network)
        
        # State dimension for mixer (concatenation of all agent observations)
        state_dim = obs_dim * num_agents
        
        # Mixing networks
        self.mixer = QMixer(
            n_agents=num_agents,
            state_dim=state_dim,
            mixing_embed_dim=mixing_embed_dim,
            hypernet_layers=hypernet_layers,
            hypernet_embed=hypernet_embed,
            init_scale=self.mixer_init_scale,
        ).to(self.device)
        
        self.target_mixer = copy.deepcopy(self.mixer)
        
        # Single optimizer for Q-network + mixer.
        self.params = list(self.shared_q_network.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(self.params, lr=lr)
        
        # Create agent wrappers
        self.agents = {}
        for i in range(num_agents):
            self.agents[i] = rQMIXAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_agents=num_agents,
                q_network=self.shared_q_network,
                device=self.device,
            )
        
        # Training step counter
        self.training_steps = 0
        self.total_env_steps = 0
        self.last_target_update_step = 0
        
        # Tracking for Trainer compatibility
        self.last_dones = {i: False for i in range(self.num_agents)}
        self.temp_hidden_states = {}
        self.temp_masks = {}
        
        # For parallel environments
        self.last_dones_batch = []
        self.temp_hidden_states_batch = []
        self.temp_masks_batch = []
    
    def _get_epsilon(self) -> float:
        """Get current epsilon with linear annealing."""
        delta = (self.epsilon_start - self.epsilon_finish) / self.epsilon_anneal_time
        self.epsilon = max(
            self.epsilon_finish,
            self.epsilon_start - delta * self.total_env_steps
        )
        return self.epsilon
    
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
        """Concatenate all agent observations to form global state."""
        sorted_obs = [observations[i] for i in range(self.num_agents) if i in observations]
        if not sorted_obs:
            return torch.zeros(self.num_agents * self.obs_dim, device=self.device)
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
        
        epsilon = self._get_epsilon()
        
        for agent_id, obs in observations.items():
            avail_actions_tensor = None
            if available_actions is not None and agent_id in available_actions:
                avail_actions_tensor = torch.tensor(
                    available_actions[agent_id],
                    dtype=torch.float32,
                    device=self.device
                )
            
            action, log_prob = self.agents[agent_id].get_action(
                obs, epsilon=epsilon, available_actions=avail_actions_tensor
            )
            
            actions[agent_id] = (action, log_prob)
            self.temp_hidden_states[agent_id] = self.agents[agent_id].rnn_state.clone()
            mask = 1.0 if not self.last_dones.get(agent_id, False) else 0.0
            self.temp_masks[agent_id] = mask
        
        self.total_env_steps += 1
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
        
        epsilon = self._get_epsilon()
        
        for agent_id in range(self.num_agents):
            # Gather obs for this agent across all envs
            agent_obs_list = []
            agent_avail_list = []
            has_avail = (available_actions_batch is not None)
            
            for i in range(n_envs):
                agent_obs_list.append(observations_batch[i][agent_id])
                self.temp_masks_batch[i][agent_id] = (
                    1.0 if not self.last_dones_batch[i].get(agent_id, False) else 0.0
                )
                if has_avail and available_actions_batch[i] is not None and agent_id in available_actions_batch[i]:
                    agent_avail_list.append(available_actions_batch[i][agent_id])
                else:
                    agent_avail_list.append(None)
            
            agent_obs_batch = torch.stack(agent_obs_list).to(self.device)
            
            agent_avail_tensor = None
            if any(x is not None for x in agent_avail_list):
                agent_avail_tensor = torch.tensor(
                    agent_avail_list, dtype=torch.float32, device=self.device
                )
            
            actions, log_probs = self.agents[agent_id].get_action_batch(
                agent_obs_batch, epsilon=epsilon,
                available_actions_batch=agent_avail_tensor
            )
            
            for i in range(n_envs):
                act = actions[i].item() if isinstance(actions[i], torch.Tensor) else actions[i]
                lp = log_probs[i].item() if isinstance(log_probs[i], torch.Tensor) else log_probs[i]
                actions_batch[i][agent_id] = (act, lp)
                self.temp_hidden_states_batch[i][agent_id] = (
                    self.agents[agent_id].rnn_states_batch[i].clone()
                )

        # NOTE: Count actual environment steps, not parallelism
        # Each parallel env contributes 1 step to total_env_steps
        self.total_env_steps += n_envs
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
            env_id=0
        )

        # Reset RNN states when episode ends (single environment)
        for agent_id in range(self.num_agents):
            if dones[agent_id]:
                self.agents[agent_id].rnn_state = torch.zeros(
                    1, self.hidden_dim, device=self.device
                )
                self.agents[agent_id].prev_action = torch.zeros(
                    self.action_dim, device=self.device
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
            env_id=env_idx
        )
        
        # Update done status and reset RNN if done
        env_done = False
        for agent_id in range(self.num_agents):
            self.last_dones_batch[env_idx][agent_id] = dones[agent_id]
            if dones[agent_id]:
                self.agents[agent_id].rnn_states_batch[env_idx] = torch.zeros(
                    self.hidden_dim, device=self.device
                )
                self.agents[agent_id].prev_actions_batch[env_idx] = torch.zeros(
                    self.action_dim, device=self.device
                )
        
        if any(dones.values()):
            env_done = True
        
        if env_done:
            self.finish_episode(env_id=env_idx)
    
    def finish_episode(self, env_id: int = 0) -> Optional[List[Dict]]:
        """Finish current episode and move it to completed episodes."""
        completed = self.buffer_manager.finish_episode(env_id=env_id)
        if completed is None:
            return None
        packed = self._pack_episode(completed)
        if self.buffer_manager.completed_episodes:
            self.buffer_manager.completed_episodes[-1] = packed
        return packed

    def _pack_episode(self, episode: List[Dict]) -> Dict[str, torch.Tensor]:
        return qmix_pack_episode(
            episode,
            obs_dim=self.obs_dim,
            num_agents=self.num_agents,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            reward_scale=self.reward_scale,
            env_required_agents=getattr(getattr(self, "env", None), "required_agents", None),
        )

    def _stack_packed_episodes(self, episodes: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return qmix_stack_packed_episodes(
            episodes,
            obs_dim=self.obs_dim,
            num_agents=self.num_agents,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            device=self.device,
        )
    
    def _pad_and_stack(self, episodes: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        """
        Convert list of episodes to batch of padded tensors.
        
        Returns dict with:
            joint_obs: [B, T+1, obs_dim * n_agents]   (global state for mixer)
            agent_obs: [B, T+1, n_agents, obs_dim]     (per-agent obs)
            actions:   [B, T+1, n_agents]               (long)
            rewards:   [B, T]                            (team reward)
            terminated: [B, T]
            masks:     [B, T]                            (1=valid, 0=padded)
            available_actions: [B, T+1, n_agents, action_dim] (optional)
        """
        if episodes and isinstance(episodes[0], dict) and episodes[0].get("packed", False):
            return self._stack_packed_episodes(episodes)
        return qmix_pad_and_stack_legacy(
            episodes,
            obs_dim=self.obs_dim,
            num_agents=self.num_agents,
            action_dim=self.action_dim,
            reward_scale=self.reward_scale,
            device=self.device,
        )
    
    def _forward_agent_q_values(
        self,
        batch: Dict[str, torch.Tensor],
        q_network: QMIXRNNAgent,
    ) -> torch.Tensor:
        """
        Compute Q-values for all agents across all timesteps using sequential RNN forward.
        
        Args:
            batch: padded batch dict
            q_network: Q-network to use (online or target)
            
        Returns:
            mac_out: [B, T+1, n_agents, action_dim]
        """
        agent_obs = batch['agent_obs']          # [B, T+1, N, obs_dim]
        actions = batch['actions']              # [B, T+1, N]
        B, Tp1, N, _ = agent_obs.shape
        
        mac_out = []
        
        # Initialize hidden states for all agents: [B*N, hidden_dim]
        hidden = q_network.init_hidden().expand(B * N, -1).contiguous()
        
        for t in range(Tp1):
            # Build inputs for all agents at timestep t
            obs_t = agent_obs[:, t]  # [B, N, obs_dim]
            # Note: obs_t may already include agent_id if config.algorithm.add_agent_id=True

            # Previous actions one-hot
            if t == 0:
                prev_act = torch.zeros(B, N, self.action_dim, device=self.device)
            else:
                prev_act = F.one_hot(actions[:, t - 1], self.action_dim).float()  # [B, N, A]

            # Concatenate: [obs (potentially with agent_id), prev_action]
            inputs = torch.cat([obs_t, prev_act], dim=-1)  # [B, N, input_shape]
            inputs = inputs.reshape(B * N, -1)  # [B*N, input_shape]
            
            q_vals, hidden = q_network(inputs, hidden)  # [B*N, A], [B*N, H]
            q_vals = q_vals.view(B, N, -1)  # [B, N, A]
            
            mac_out.append(q_vals)
        
        # Stack: [B, T+1, N, A]
        mac_out = torch.stack(mac_out, dim=1)
        return mac_out
    
    def _sample_batch_from_replay(self, completed_episodes: List[Dict]) -> Dict[str, torch.Tensor]:
        n_available = len(completed_episodes)
        sample_size = min(self.train_batch_size, n_available)
        indices = np.random.choice(n_available, sample_size, replace=False)
        sampled_episodes = [completed_episodes[i] for i in indices]
        return self._pad_and_stack(sampled_episodes)

    def _train_step_from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single gradient update step (PyMARL's QLearner.train equivalent).
        """
        rewards = batch['rewards']          # [B, T]
        actions = batch['actions']          # [B, T+1, N]
        terminated = batch['terminated']    # [B, T]
        masks = batch['masks']              # [B, T]
        joint_obs = batch['joint_obs']      # [B, T+1, state_dim]
        avail_actions = batch.get('available_actions', None)  # [B, T+1, N, A] or None
        
        B, Tp1, N = actions.shape
        T = Tp1 - 1
        
        # ---- Compute Q-values from online network ----
        mac_out = self._forward_agent_q_values(batch, self.shared_q_network)  # [B, T+1, N, A]
        
        # Pick Q-values for chosen actions: [B, T, N]
        chosen_action_qvals = torch.gather(
            mac_out[:, :-1], dim=3, index=actions[:, :-1].unsqueeze(-1)
        ).squeeze(3)
        
        # ---- Compute target Q-values ----
        with torch.no_grad():
            # Compute target Q-values using target network
            target_mac_out = self._forward_agent_q_values(batch, self.target_q_network)
            # t = 1 ~ T+1 for targets
            target_mac_out_future = target_mac_out[:, 1:]  # [B, T, N, A]
            
            # Mask unavailable actions
            if avail_actions is not None:
                target_mac_out_future[avail_actions[:, 1:] == 0] = -9999999
            
            if self.double_q:
                # Double Q-learning: use online network to select actions
                mac_out_detach = mac_out.clone().detach()
                if avail_actions is not None:
                    mac_out_detach[avail_actions == 0] = -9999999
                # choose actions (index/action-id of q-value) from the online q-netowork output
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_max_qvals = torch.gather(
                    target_mac_out_future, 3, cur_max_actions
                ).squeeze(3)
            else:
                target_max_qvals = target_mac_out_future.max(dim=3)[0]
        
        # ---- Mix ----
        # chosen_action_qvals: [B, T, N] → Q_tot
        chosen_action_q_tot = self.mixer(
            chosen_action_qvals, joint_obs[:, :-1]
        )  # [B, T, 1]
        
        with torch.no_grad():
            target_max_q_tot = self.target_mixer(
                target_max_qvals, joint_obs[:, 1:]
            )  # [B, T, 1]
        
        # ---- 1-step TD target ----
        rewards_expanded = rewards.unsqueeze(-1)        # [B, T, 1]
        terminated_expanded = terminated.unsqueeze(-1)   # [B, T, 1]
        masks_expanded = masks.unsqueeze(-1)             # [B, T, 1]
        
        targets = rewards_expanded + self.gamma * (1 - terminated_expanded) * target_max_q_tot
        
        # TD error
        td_error = (chosen_action_q_tot - targets.detach())
        masked_td_error = td_error * masks_expanded
        
        # L2 loss
        loss = (masked_td_error ** 2).sum() / masks_expanded.sum().clamp(min=1.0)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
        
        self.training_steps += 1
        
        # Update target network
        if (self.training_steps - self.last_target_update_step) >= self.target_update_interval:
            self._update_targets()
            self.last_target_update_step = self.training_steps
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'td_error_abs': masked_td_error.abs().sum().item() / masks_expanded.sum().clamp(min=1.0).item(),
            'q_taken_mean': (chosen_action_q_tot * masks_expanded).sum().item() / (masks_expanded.sum().item() * self.num_agents),
            'target_mean': (targets * masks_expanded).sum().item() / (masks_expanded.sum().item() * self.num_agents),
        }

    def _train_step(self, completed_episodes: List[Dict]) -> Dict[str, float]:
        batch = self._sample_batch_from_replay(completed_episodes)
        return self._train_step_from_batch(batch)
    
    def update(self, **kwargs) -> Dict[str, float]:
        """
        Update Q-network and mixer using completed episodes from replay buffer.
        """
        if not self.buffer_manager.has_completed_episodes():
            return {}
        
        completed_episodes = self.buffer_manager.get_completed_episodes()
        if not completed_episodes:
            return {}
        
        n_available = len(completed_episodes)
        if n_available < self.train_batch_size:
            # Not enough episodes to form a full batch yet — skip training
            # (matches PyMARL's buffer.can_sample(args.batch_size) check)
            return {}
        
        metadata = {
            'epsilon': self.epsilon,
            'total_env_steps': self.total_env_steps,
            'replay_buffer_episodes': n_available,
            'qmix_update_epochs': float(self.qmix_update_epochs or 0),
            'learning_starts': float(self.learning_starts),
            'learning_started': float(self.total_env_steps >= self.learning_starts),
            'reward_scale': float(self.reward_scale),
            'mixer_init_scale': float(self.mixer_init_scale),
        }

        if self.total_env_steps < self.learning_starts:
            metadata['num_updates'] = 0
            return metadata

        num_updates = (
            self.qmix_update_epochs
            if self.qmix_update_epochs is not None
            else max(1, kwargs.get('num_episodes_collected', 1))
        )
        
        # Accumulate metrics across all gradient steps
        accumulated = defaultdict(float)
        
        for _ in range(num_updates):
            batch = self._sample_batch_from_replay(completed_episodes)
            step_metrics = self._train_step_from_batch(batch)
            for k, v in step_metrics.items():
                accumulated[k] += v
        
        # Average metrics over all gradient steps
        losses = {k: v / num_updates for k, v in accumulated.items()}
        
        losses.update(metadata)
        losses['num_updates'] = num_updates
        
        # NOTE: Do NOT clear completed_episodes here — QMIX is off-policy.
        # Episodes remain in the replay buffer until it reaches max capacity (FIFO).
        
        return losses
    
    def _update_targets(self):
        """Hard update target networks."""
        self.target_q_network.load_state_dict(self.shared_q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
    
    def clear_buffer(self):
        """Clear all buffers."""
        self.buffer_manager.clear_all()
    
    def get_buffer_info(self) -> Dict[str, int]:
        """Get buffer information for monitoring."""
        return self.buffer_manager.get_buffer_info()
