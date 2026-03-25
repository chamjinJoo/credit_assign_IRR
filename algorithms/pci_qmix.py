import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import copy

from algorithms.networks.rnn import RNNLayer
from algorithms.networks.qmix_mixer import QMixer
from algorithms.networks import OutcomeBridgeNetwork, ActionBridgeNetwork, NaivePredictor
from utils.buffer import EpisodeBuffer


class RBFKernel(nn.Module):
    """RBF Kernel for PCI bridge function learning."""
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
        gamma = torch.exp(self.log_gamma)
        dists_sq = torch.sum((x1 - x2)**2, dim=-1)
        return torch.exp(-gamma * dists_sq)

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
            if available_actions_batch is not None:
                # Sample random available actions
                random_actions = torch.zeros(n_envs, dtype=torch.long, device=self.device)
                for i in range(n_envs):
                    avail = torch.nonzero(available_actions_batch[i], as_tuple=False).squeeze(-1)
                    random_actions[i] = avail[torch.randint(len(avail), (1,))]
            else:
                random_actions = torch.randint(0, self.action_dim, (n_envs,), device=self.device)

            # Combine
            actions = torch.where(random_mask, random_actions, greedy_actions)

            # Update prev_actions
            self.prev_actions_batch = F.one_hot(actions, self.action_dim).float()

            # Dummy log_probs for compatibility
            log_probs = torch.zeros(n_envs, device=self.device)

        return actions, log_probs


class PCIQMIXAlgorithm:
    """
    QMIX Algorithm with PCI-based Reward Shaping.

    Combines QMIX's value decomposition with PCI theory's outcome bridge network
    for learning causal influence between agents and providing intrinsic rewards.

    Key differences from vanilla QMIX:
    - Adds OutcomeBridgeNetwork for learning agent causal effects
    - Computes causal intrinsic rewards (delta) during training
    - Shapes rewards using learned causal influence before Q-learning updates
    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 5e-4,
        lr_bridge: float = 3e-4,
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
        replay_buffer_size: int = 5000,
        train_batch_size: int = 64,
        device: str = "cpu",
        use_episode_buffer: bool = True,
        # PCI-specific parameters
        w_dim: int = None,  # Will be set to obs_dim * (num_agents - 1)
        causal_lambda: float = 0.1,
        causal_lambda_schedule: bool = False,
        causal_lambda_start: float = 0.0,
        causal_lambda_final: float = 0.1,
        causal_lambda_warmup_steps: int = 10000,
        causal_lambda_startingpoint: int = 0,
        bridge_epoch: int = 4,
        clip_delta: float = 0.2,
        causal_mini_batch_size: int = 10000,
        train_naivepredictor: bool = False,
        bridge_update_interval: int = 10,  # Update bridge network every N QMIX updates
        # Additional params for compatibility with the config system
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 32,
        num_batches: int = 1,
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

        # PCI parameters
        self.causal_lambda = causal_lambda
        self.causal_lambda_schedule = causal_lambda_schedule
        self.causal_lambda_start = causal_lambda_start
        self.causal_lambda_final = causal_lambda_final
        self.causal_lambda_warmup_steps = causal_lambda_warmup_steps
        self.causal_lambda_startingpoint = causal_lambda_startingpoint
        self.bridge_epoch = bridge_epoch
        self.clip_delta = clip_delta
        self.causal_mini_batch_size = causal_mini_batch_size
        self.train_naivepredictor = train_naivepredictor
        self.bridge_update_interval = bridge_update_interval

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
        ).to(self.device)

        self.target_mixer = copy.deepcopy(self.mixer)

        # Single optimizer for Q-network + mixer (PyMARL style)
        self.params = list(self.shared_q_network.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.RMSprop(
            self.params, lr=lr, alpha=0.99, eps=1e-5
        )

        # ====== PCI Bridge Networks ======
        # x_dim: for QMIX we use (o^i_t, h^{-i}_{t-1}) where h is GRU hidden state
        x_dim = obs_dim + (num_agents - 1) * hidden_dim

        self.outcome_bridge_net = OutcomeBridgeNetwork(
            action_dim=action_dim,
            num_agents=num_agents,
            w_dim=obs_dim * (num_agents - 1),  # Other agents' observations
            outcome_dim=num_agents,
            hidden_dim=hidden_dim,
            action_embedding_dim=8,
            x_dim=x_dim
        ).to(self.device)

        self.action_bridge_net = ActionBridgeNetwork(
            z_dim=obs_dim * num_agents,  # Previous joint observation
            action_dim=action_dim,
            num_agents=num_agents,
            hidden_dim=hidden_dim,
            action_embedding_dim=8
        ).to(self.device)

        # RBF Kernels for bridge function learning
        self.critic_kernel_w = RBFKernel(gamma=1.0).to(self.device)
        self.critic_kernel_z = RBFKernel(gamma=1.0).to(self.device) # kept for checkpoint compatibility

        self.kernel_z_block = RBFKernel(gamma=1.0).to(self.device)
        self.kernel_a_block = RBFKernel(gamma=1.0).to(self.device)
        self.kernel_xobs_block = RBFKernel(gamma=1.0).to(self.device)
        self.kernel_xhid_block = RBFKernel(gamma=1.0).to(self.device)

        self.stabilizer_lambda = 1.0
        self.tikhonov_gamma = 1e-4

        self.outcome_bridge_optimizer = optim.Adam(self.outcome_bridge_net.parameters(), lr=lr_bridge)
        self.action_bridge_optimizer = optim.Adam(self.action_bridge_net.parameters(), lr=lr_bridge)

        if self.train_naivepredictor:
            self.naive_predictor = NaivePredictor(
                action_dim=action_dim,
                num_agents=num_agents,
                x_dim=x_dim,
                hidden_dim=hidden_dim
            ).to(self.device)
            self.naive_optimizer = optim.Adam(self.naive_predictor.parameters(), lr=lr_bridge)

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
        self.bridge_training_steps = 0

        # Tracking for Trainer compatibility
        self.last_dones = {i: False for i in range(self.num_agents)}
        self.temp_hidden_states = {}
        self.temp_masks = {}

        # For parallel environments
        self.last_dones_batch = []
        self.temp_hidden_states_batch = []
        self.temp_masks_batch = []

    def get_current_causal_lambda(self) -> float:
        """Get current causal lambda value with optional scheduling."""
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
            # Store hidden state (note: for QMIX this is GRU hidden, shape [1, hidden_dim])
            self.temp_hidden_states[agent_id] = self.agents[agent_id].rnn_state.clone().squeeze(0)
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
                # Store hidden state for bridge network (shape: [hidden_dim])
                self.temp_hidden_states_batch[i][agent_id] = (
                    self.agents[agent_id].rnn_states_batch[i].clone()
                )

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
            self.buffer_manager.finish_episode(env_id=env_idx)

    def finish_episode(self, env_id: int = 0) -> Optional[List[Dict]]:
        """Finish current episode and move it to completed episodes."""
        return self.buffer_manager.finish_episode(env_id=env_id)

    def compute_causal_rewards(
        self,
        w_t_batch: torch.Tensor,
        z_t_batch: torch.Tensor,
        full_obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        rewards_batch: torch.Tensor,
        hidden_states_batch: Optional[Dict[int, torch.Tensor]] = None,
        available_actions_batch: Optional[Dict[int, torch.Tensor]] = None,
        debug: bool = False
    ) -> torch.Tensor:
        """
        Computes the pairwise Causal Effect Matrix C using outcome bridge network.
        C[b, i, j] = Effect of Agent i on Agent j in batch b.

        Policy probability는 QMIX off-policy 특성에 맞춰 Q-value softmax로 근사.
        Baseline은 r_pci.py와 동일하게 factual action을 제외한 counterfactual
        policy-weighted expectation으로 계산.

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
                r_b = rewards_batch[i:end]
                obs_b = full_obs_batch[i:end]
                curr_bs = act_b.size(0)

                # Vectorized logic
                actions_flat = act_b.view(-1)  # (curr_bs * N,)
                w_t_flat = w_b.view(-1, w_b.size(-1))
                # rewards_flat: (curr_bs * N, N)
                rewards_flat = r_b.unsqueeze(1).expand(-1, num_agents, -1).reshape(-1, num_agents)

                actions_range = torch.arange(action_dim, device=self.device).repeat(curr_bs * num_agents)
                w_t_expanded = w_t_flat.repeat_interleave(action_dim, dim=0)

                # ── X 구성: (o^i_t, h^{-i}_{t-1}) ──────────────────────────────────
                h_list_inf = []
                for k in range(num_agents):
                    if hidden_states_batch is not None and k in hidden_states_batch:
                        # hidden_states_batch[k]: (batch_size, hidden_dim) for QMIX
                        h_list_inf.append(hidden_states_batch[k][i:end])
                    else:
                        h_list_inf.append(torch.zeros(curr_bs, self.hidden_dim, device=self.device))
                H_all_inf = torch.stack(h_list_inf, dim=1)  # (curr_bs, N, hidden_dim)

                expanded_H_inf = H_all_inf.unsqueeze(1).expand(-1, num_agents, -1, -1)
                mask_H_inf = ~torch.eye(num_agents, dtype=torch.bool, device=self.device) \
                    .unsqueeze(0).unsqueeze(-1).expand(curr_bs, -1, -1, self.hidden_dim)
                H_others_inf = expanded_H_inf[mask_H_inf].view(curr_bs * num_agents, -1)
                obs_flat = obs_b.view(-1, self.obs_dim)
                x_flat = torch.cat([obs_flat, H_others_inf], dim=1)

                x_flat_expanded = x_flat.repeat_interleave(action_dim, dim=0)

                # ── Bridge network: 모든 counterfactual action에 대한 보상 예측 ──────
                r_pred_expanded = self.outcome_bridge_net(actions_range, w_t_expanded, x=x_flat_expanded)
                # r_hat_all: (curr_bs, N, action_dim, N)
                #   [batch, acting_agent, cf_action, outcome_agent]
                r_hat_all = r_pred_expanded.view(curr_bs, num_agents, action_dim, num_agents)

                # ── Policy probability: Q-value softmax (QMIX off-policy 근사) ──────
                # Q-network input shape: [obs (potentially with agent_id), prev_action_onehot]
                # hidden_states_batch[k]는 해당 시점 직전의 GRU hidden state.
                # Vectorized Q-network forward for all agents
                obs_flat = obs_b.view(-1, self.obs_dim)  # (curr_bs * N, obs_dim)
                prev_act_onehot = torch.zeros(curr_bs * num_agents, action_dim, device=self.device)
                inputs_all = torch.cat([obs_flat, prev_act_onehot], dim=-1)
                hidden_all = H_all_inf.view(-1, self.hidden_dim)  # (curr_bs * N, hidden_dim)
                q_values_all, _ = self.shared_q_network(inputs_all, hidden_all)
                q_values_all = q_values_all.view(curr_bs, num_agents, -1)  # (curr_bs, N, action_dim)

                probs_list = []
                for k in range(num_agents):
                    q_values_k = q_values_all[:, k, :]
                    # available_actions 마스킹: 불가능한 action의 Q-value를 -inf로
                    if available_actions_batch is not None and k in available_actions_batch:
                        avail_k = available_actions_batch[k][i:end]  # (curr_bs, action_dim)
                        q_values_k = q_values_k.masked_fill(avail_k == 0, -1e10)
                    probs_list.append(q_values_k)  # logits 저장

                # action_probs_flat: (curr_bs * N, action_dim)
                q_logits_flat = torch.stack(probs_list, dim=1).view(-1, action_dim)
                action_probs_flat = torch.softmax(q_logits_flat, dim=-1)

                # ── Counterfactual baseline (r_pci.py 방식) ─────────────────────────
                # factual action을 제외한 나머지에 대해 renormalize 후 policy-weighted sum
                factual_mask = torch.zeros_like(action_probs_flat).scatter_(
                    1, actions_flat.unsqueeze(1), 1.0
                )  # (curr_bs*N, action_dim)
                non_factual_mask = 1.0 - factual_mask

                probs_counterfactual = action_probs_flat * non_factual_mask
                renormalized_probs = probs_counterfactual / (
                    probs_counterfactual.sum(dim=1, keepdim=True) + 1e-10
                )  # (curr_bs*N, action_dim)

                # r_hat_all_flat: (curr_bs*N, action_dim, N)
                r_hat_all_flat = r_hat_all.view(curr_bs * num_agents, action_dim, num_agents)

                # baseline: (curr_bs*N, N) — counterfactual policy-weighted expectation
                baseline = (renormalized_probs.unsqueeze(2) * r_hat_all_flat).sum(dim=1)

                # ── Causal effect = factual reward - counterfactual baseline ─────────
                raw_effects = rewards_flat - baseline  # (curr_bs*N, N)
                C_chunk = raw_effects.view(curr_bs, num_agents, num_agents)

                # ── Dead agent 마스킹 ────────────────────────────────────────────────
                # no-op(action 0)만 가능한 에이전트 = dead → causal effect를 0으로
                if available_actions_batch is not None:
                    alive_list = []
                    for kk in range(num_agents):
                        if kk in available_actions_batch:
                            alive_list.append(
                                (available_actions_batch[kk][i:end][:, 1:].sum(dim=-1) > 0)
                            )
                        else:
                            alive_list.append(torch.ones(curr_bs, dtype=torch.bool, device=self.device))
                    alive = torch.stack(alive_list, dim=1).float()  # (curr_bs, N)
                    C_chunk = C_chunk * alive.unsqueeze(2) * alive.unsqueeze(1)

                C_matrix[i:end] = C_chunk

        return C_matrix

    def update_causal_networks(self, batch_data: List[Dict]) -> Dict[str, float]:
        """Update outcome bridge network using batch data from replay buffer."""
        if self.get_current_causal_lambda() == 0.0:
            return {'outcome_bridge_loss': 0.0, 'action_bridge_loss': 0.0}

        batch_size = len(batch_data)
        num_agents = self.num_agents
        device = self.device

        # === 1. Data Preparation ===
        joint_o_prev = torch.stack([self._concatenate_observations(t['bold_o_t_minus_1']) for t in batch_data]).to(device)
        obs_batch = torch.stack([torch.stack(list(t['bold_o_t'].values())) for t in batch_data]).to(device)
        action_batch = torch.tensor([[t['bold_a_t'][i] for i in range(num_agents)] for t in batch_data], dtype=torch.long, device=device)
        reward_batch = torch.tensor([[t['bold_r_t'][i] for i in range(num_agents)] for t in batch_data], dtype=torch.float32, device=device)

        # Extract hidden states (GRU hidden states from QMIX agents)
        hidden_states_batch = {}
        if 'hidden_states' in batch_data[0] and batch_data[0]['hidden_states'] is not None:
            for agent_id in range(self.num_agents):
                if agent_id in batch_data[0]['hidden_states']:
                    # For QMIX: hidden state is (hidden_dim,) tensor per timestep
                    h_states = torch.stack([t['hidden_states'][agent_id] for t in batch_data]).to(self.device)
                    hidden_states_batch[agent_id] = h_states

        # Prepare flattened data for bridge network training
        A_super = action_batch.view(-1)
        expanded_obs = obs_batch.unsqueeze(1).expand(-1, num_agents, -1, -1)
        mask = ~torch.eye(num_agents, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.obs_dim)
        obs_others_super = expanded_obs[mask].view(batch_size * num_agents, -1)

        W_super = obs_others_super  # W = other agents' observations
        Y_super = reward_batch.repeat_interleave(num_agents, dim=0)  # (batch_size * num_agents, num_agents)

        # Construct X_super = {o^i_t, h^{-i}_{t-1}}
        h_list = []
        for i in range(num_agents):
            if i in hidden_states_batch:
                h_list.append(hidden_states_batch[i])
            else:
                h_list.append(torch.zeros(batch_size, self.hidden_dim, device=device))
        H_all = torch.stack(h_list, dim=1)  # (B, N, hidden_dim)

        # Get others' hidden states
        expanded_H = H_all.unsqueeze(1).expand(-1, num_agents, -1, -1)
        mask_H = ~torch.eye(num_agents, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.hidden_dim)
        H_others_super = expanded_H[mask_H].view(batch_size * num_agents, -1)
        obs_super = obs_batch.view(-1, self.obs_dim)
        X_super = torch.cat([obs_super, H_others_super], dim=1)

        Z_super = joint_o_prev.repeat_interleave(num_agents, dim=0)  # Z = previous joint observation

        # === 2. Mini-batch Training Loop ===
        total_samples = A_super.size(0)
        indices = torch.arange(total_samples, device=device)

        avg_outcome_loss = 0.0
        avg_action_loss = 0.0
        avg_naive_loss = 0.0
        sum_off_mean = 0.0
        sum_off_std = 0.0
        num_updates = 0
        chunk_size = self.causal_mini_batch_size

        for i in range(0, total_samples, chunk_size):
            idx = indices[i:i + chunk_size]
            current_batch_size = idx.size(0)

            a_mb = A_super[idx]
            w_mb = W_super[idx]
            y_mb = Y_super[idx]
            z_mb = Z_super[idx]
            x_mb = X_super[idx]
            o_mb = obs_super[idx]

            # Standardize features for kernel
            eps = 1e-6
            z_mean = z_mb.mean(dim=0, keepdim=True)
            z_std = z_mb.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
            z_mb_norm = (z_mb - z_mean) / z_std

            # Split x into obs and hid components
            x_obs_mb_raw = x_mb[:, :self.obs_dim]
            x_hid_mb_raw = x_mb[:, self.obs_dim:]

            x_obs_mean = x_obs_mb_raw.mean(dim=0, keepdim=True)
            x_obs_std = x_obs_mb_raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
            x_obs_mb = (x_obs_mb_raw - x_obs_mean) / x_obs_std

            x_hid_mean = x_hid_mb_raw.mean(dim=0, keepdim=True)
            x_hid_std = x_hid_mb_raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
            x_hid_mb = (x_hid_mb_raw - x_hid_mean) / x_hid_std

            # Action one-hot
            action_onehot_mb = F.one_hot(a_mb.long(), self.action_dim).float()

            # Update kernel gammas
            with torch.no_grad():
                for feat, kernel in [
                    (z_mb_norm, self.kernel_z_block),
                    (action_onehot_mb, self.kernel_a_block),
                    (x_obs_mb, self.kernel_xobs_block),
                    (x_hid_mb, self.kernel_xhid_block),
                ]:
                    n = feat.shape[1]
                    kernel.log_gamma.data.fill_(torch.log(torch.tensor(1.0 / n)))

            # Product kernel
            K_z = (self.kernel_z_block(z_mb_norm)
                   * self.kernel_a_block(action_onehot_mb)
                   * self.kernel_xobs_block(x_obs_mb)
                   * self.kernel_xhid_block(x_hid_mb))

            # Check kernel collapse
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

            # Update Outcome Bridge
            epoch_outcome_loss = 0.0
            for _ in range(self.bridge_epoch):
                r_hat_mb = self.outcome_bridge_net(a_mb, w_mb, x=x_mb)
                residuals_mb = y_mb - r_hat_mb

                solved_residuals = torch.linalg.solve(reg_K_z, residuals_mb)
                per_sample_loss_h = (residuals_mb * (K_z @ solved_residuals)).sum(dim=1)

                loss_h = per_sample_loss_h.mean()
                loss_q = torch.tensor(0.0, device=device)
                total_loss = loss_h + loss_q

                self.outcome_bridge_optimizer.zero_grad()
                self.action_bridge_optimizer.zero_grad()
                total_loss.backward()
                self.outcome_bridge_optimizer.step()

                epoch_outcome_loss += loss_h.item()

            avg_outcome_loss += epoch_outcome_loss / self.bridge_epoch
            avg_action_loss += loss_q.item()

            # Update Naive Predictor
            if getattr(self, "train_naivepredictor", False):
                naive_output = self.naive_predictor(a_mb, x_mb)
                naive_loss = F.mse_loss(naive_output, y_mb)
                self.naive_optimizer.zero_grad()
                naive_loss.backward()
                self.naive_optimizer.step()
                avg_naive_loss += naive_loss.item()

            num_updates += 1

        return_dict = {
            'outcome_bridge_loss': avg_outcome_loss / num_updates,
            'action_bridge_loss': avg_action_loss / num_updates,
            'Kz_offdiag_mean': sum_off_mean / num_updates,
            'Kz_offdiag_std': sum_off_std / num_updates,
        }

        if getattr(self, "train_naivepredictor", False):
            return_dict['naive_loss'] = avg_naive_loss / num_updates

        return return_dict

    def _apply_causal_reward_shaping(self, episodes: List[List[Dict]]) -> List[List[Dict]]:
        """
        Apply PCI-based reward shaping to episodes before Q-learning.
        Computes causal intrinsic rewards (delta) and adds them to environment rewards.

        Args:
            episodes: List of episodes, each episode is a list of transitions

        Returns:
            Episodes with shaped rewards
        """
        current_lambda = self.get_current_causal_lambda()
        if current_lambda == 0.0:
            return episodes

        shaped_episodes = []

        for episode in episodes:
            if len(episode) == 0:
                shaped_episodes.append(episode)
                continue

            B = len(episode)
            N = self.num_agents

            # Prepare batch data
            full_obs_batch = torch.stack([torch.stack(list(t['bold_o_t'].values())) for t in episode]).to(self.device)
            actions_batch = torch.tensor([[t['bold_a_t'][i] for i in range(N)] for t in episode], dtype=torch.long, device=self.device)
            rewards_batch = torch.tensor([[t['bold_r_t'][i] for i in range(N)] for t in episode], dtype=torch.float32, device=self.device)

            # Z: previous joint observations
            z_t_batch = torch.stack([self._concatenate_observations(t['bold_o_t_minus_1']) for t in episode]).to(self.device)

            # W: other agents' observations
            expanded_obs = full_obs_batch.unsqueeze(1).expand(-1, N, -1, -1)
            mask = ~torch.eye(N, dtype=torch.bool, device=self.device).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, self.obs_dim)
            obs_others_all = expanded_obs[mask].view(B, N, (N-1)*self.obs_dim)

            # Hidden states
            hidden_states_batch = {}
            if 'hidden_states' in episode[0] and episode[0]['hidden_states'] is not None:
                for agent_id in range(self.num_agents):
                    if agent_id in episode[0]['hidden_states']:
                        h_states = torch.stack([t['hidden_states'][agent_id] for t in episode]).to(self.device)
                        hidden_states_batch[agent_id] = h_states

            # Compute causal rewards
            # Build available_actions_batch: Dict[agent_id, Tensor(B, action_dim)]
            avail_act_batch = None
            if 'available_actions' in episode[0] and episode[0]['available_actions'] is not None:
                avail_act_batch = {}
                for agent_id in range(N):
                    avail_list = []
                    for t in episode:
                        avail = t.get('available_actions', None)
                        if avail is not None and agent_id in avail:
                            avail_list.append(torch.tensor(avail[agent_id], dtype=torch.float32))
                        else:
                            avail_list.append(torch.ones(self.action_dim))
                    avail_act_batch[agent_id] = torch.stack(avail_list).to(self.device)  # (B, action_dim)

            C_matrix = self.compute_causal_rewards(
                obs_others_all, z_t_batch, full_obs_batch,
                actions_batch, rewards_batch,
                hidden_states_batch=hidden_states_batch,
                available_actions_batch=avail_act_batch,
            )

            # Shape rewards: convex combination of team reward and causal delta (r_pci.py 방식)
            shaped_episode = []
            for t, transition in enumerate(episode):
                shaped_transition = transition.copy()
                shaped_rewards = {}

                # Team reward: sum of all individual env rewards at this timestep
                team_reward = sum(transition['bold_r_t'].values())

                for agent_id in range(N):
                    # Delta: sum of causal effects this agent has on others
                    delta = C_matrix[t, agent_id].sum().item()
                    # Clip delta
                    delta = np.clip(delta, -self.clip_delta, self.clip_delta)

                    # Shaped reward = (1-λ) * team_reward/N + λ * delta
                    # λ=0: 팀 보상 공유 (env 신호 보존)
                    # λ=1: causal delta만 남음
                    shaped_reward = (1 - current_lambda) * (team_reward / N) + current_lambda * delta
                    shaped_rewards[agent_id] = shaped_reward

                    # Store delta for logging
                    if 'delta' not in shaped_transition:
                        shaped_transition['delta'] = {}
                    shaped_transition['delta'][agent_id] = delta

                shaped_transition['bold_r_t'] = shaped_rewards
                shaped_episode.append(shaped_transition)

            shaped_episodes.append(shaped_episode)

        return shaped_episodes

    def _pad_and_stack(self, episodes: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        """
        Convert list of episodes to batch of padded tensors.

        Returns dict with:
            joint_obs: [B, T+1, obs_dim * n_agents]
            agent_obs: [B, T+1, n_agents, obs_dim]
            actions:   [B, T+1, n_agents]
            rewards:   [B, T]
            terminated: [B, T]
            masks:     [B, T]
            available_actions: [B, T+1, n_agents, action_dim] (optional)
        """
        batch_size = len(episodes)
        max_t = max(len(ep) for ep in episodes)
        Tp1 = max_t + 1

        # T+1 length tensors
        joint_obs = torch.zeros(batch_size, Tp1, self.obs_dim * self.num_agents, device=self.device)
        agent_obs = torch.zeros(batch_size, Tp1, self.num_agents, self.obs_dim, device=self.device)
        actions = torch.zeros(batch_size, Tp1, self.num_agents, dtype=torch.long, device=self.device)
        available_actions = torch.ones(batch_size, Tp1, self.num_agents, self.action_dim, device=self.device)
        has_available_actions = False

        # T length tensors
        rewards = torch.zeros(batch_size, max_t, device=self.device)
        terminated = torch.zeros(batch_size, max_t, device=self.device)
        masks = torch.zeros(batch_size, max_t, device=self.device)

        for b, episode in enumerate(episodes):
            t_len = len(episode)

            for t, trans in enumerate(episode):
                # Observations and actions at time t
                joint_obs[b, t] = self._concatenate_observations(trans['bold_o_t']).to(self.device)
                agent_obs[b, t] = torch.stack(
                    [trans['bold_o_t'][i] for i in range(self.num_agents)]
                ).to(self.device)
                actions[b, t] = torch.tensor(
                    [trans['bold_a_t'][i] for i in range(self.num_agents)],
                    dtype=torch.long, device=self.device
                )

                rewards[b, t] = sum(trans['bold_r_t'].values())
                terminated[b, t] = 1.0 if any(trans['dones'].values()) else 0.0

                # Available actions
                avail = trans.get('available_actions', None)
                if avail is not None:
                    has_available_actions = True
                    for i in range(self.num_agents):
                        if i in avail:
                            available_actions[b, t, i] = torch.tensor(
                                avail[i], dtype=torch.float32, device=self.device
                            )

                # Fill T+1 slot with next_observations from last transition
                if t == t_len - 1 and 'next_observations' in trans:
                    next_obs = trans['next_observations']
                    joint_obs[b, t + 1] = self._concatenate_observations(next_obs).to(self.device)
                    agent_obs[b, t + 1] = torch.stack(
                        [next_obs[i] for i in range(self.num_agents)]
                    ).to(self.device)

            # Masks: valid timesteps
            masks[b, :t_len] = 1.0

        # pymarl style: mask[:, 1:] *= (1 - terminated[:, :-1])
        if max_t > 1:
            masks[:, 1:] = masks[:, 1:] * (1 - terminated[:, :-1])

        result = {
            'joint_obs': joint_obs,
            'agent_obs': agent_obs,
            'actions': actions,
            'rewards': rewards,
            'terminated': terminated,
            'masks': masks,
        }

        if has_available_actions:
            result['available_actions'] = available_actions

        return result

    def _forward_agent_q_values(
        self,
        batch: Dict[str, torch.Tensor],
        q_network: QMIXRNNAgent,
    ) -> torch.Tensor:
        """
        Compute Q-values for all agents across all timesteps using sequential RNN forward.

        Returns:
            mac_out: [B, T+1, n_agents, action_dim]
        """
        agent_obs = batch['agent_obs']
        actions = batch['actions']
        B, Tp1, N, _ = agent_obs.shape

        mac_out = []

        # Initialize hidden states
        hidden = q_network.init_hidden().expand(B * N, -1).contiguous()

        for t in range(Tp1):
            obs_t = agent_obs[:, t]
            # Note: obs_t may already include agent_id if config.algorithm.add_agent_id=True

            # Previous actions one-hot
            if t == 0:
                prev_act = torch.zeros(B, N, self.action_dim, device=self.device)
            else:
                prev_act = F.one_hot(actions[:, t - 1], self.action_dim).float()

            # Concatenate inputs: [obs (potentially with agent_id), prev_action]
            inputs = torch.cat([obs_t, prev_act], dim=-1)
            inputs = inputs.reshape(B * N, -1)

            q_vals, hidden = q_network(inputs, hidden)
            q_vals = q_vals.view(B, N, -1)

            mac_out.append(q_vals)

        mac_out = torch.stack(mac_out, dim=1)
        return mac_out

    def _train_step(self, sampled_episodes: List[List[Dict]]) -> Dict[str, float]:
        """
        Perform a single gradient update step.

        1. Apply PCI reward shaping
        2. Compute Q-learning loss
        3. Update Q-network and mixer
        """
        # Apply causal reward shaping
        shaped_episodes = self._apply_causal_reward_shaping(sampled_episodes)

        batch = self._pad_and_stack(shaped_episodes)

        rewards = batch['rewards']
        actions = batch['actions']
        terminated = batch['terminated']
        masks = batch['masks']
        joint_obs = batch['joint_obs']
        avail_actions = batch.get('available_actions', None)

        B, Tp1, N = actions.shape
        T = Tp1 - 1

        # Compute Q-values
        mac_out = self._forward_agent_q_values(batch, self.shared_q_network)

        # Pick Q-values for chosen actions
        chosen_action_qvals = torch.gather(
            mac_out[:, :-1], dim=3, index=actions[:, :-1].unsqueeze(-1)
        ).squeeze(3)

        # Compute target Q-values
        with torch.no_grad():
            target_mac_out = self._forward_agent_q_values(batch, self.target_q_network)
            target_mac_out_future = target_mac_out[:, 1:]

            if avail_actions is not None:
                target_mac_out_future[avail_actions[:, 1:] == 0] = -9999999

            if self.double_q:
                mac_out_detach = mac_out.clone().detach()
                if avail_actions is not None:
                    mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_max_qvals = torch.gather(
                    target_mac_out_future, 3, cur_max_actions
                ).squeeze(3)
            else:
                target_max_qvals = target_mac_out_future.max(dim=3)[0]

        # Mix
        chosen_action_q_tot = self.mixer(
            chosen_action_qvals, joint_obs[:, :-1]
        )

        with torch.no_grad():
            target_max_q_tot = self.target_mixer(
                target_max_qvals, joint_obs[:, 1:]
            )

        # TD target
        rewards_expanded = rewards.unsqueeze(-1)
        terminated_expanded = terminated.unsqueeze(-1)
        masks_expanded = masks.unsqueeze(-1)

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

        # Collect delta statistics
        delta_stats = defaultdict(float)
        delta_count = 0
        for episode in shaped_episodes:
            for trans in episode:
                if 'delta' in trans:
                    for agent_id in range(self.num_agents):
                        if agent_id in trans['delta']:
                            delta_stats[f'delta_agent_{agent_id}'] += trans['delta'][agent_id]
                            delta_count += 1

        metrics = {
            'loss': loss.item(),
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'td_error_abs': masked_td_error.abs().sum().item() / masks_expanded.sum().clamp(min=1.0).item(),
            'q_taken_mean': (chosen_action_q_tot * masks_expanded).sum().item() / (masks_expanded.sum().item() * self.num_agents),
            'target_mean': (targets * masks_expanded).sum().item() / (masks_expanded.sum().item() * self.num_agents),
        }

        # Add delta stats
        if delta_count > 0:
            for k, v in delta_stats.items():
                metrics[k] = v / delta_count

        return metrics

    def update(self, **kwargs) -> Dict[str, float]:
        """
        Update Q-network, mixer, and bridge networks using replay buffer.

        QMIX updates:
        - Performs gradient steps on Q-network using replay buffer
        - Does NOT clear buffer (off-policy)

        PCI updates:
        - Updates outcome bridge network periodically
        - Reward shaping is applied during Q-learning
        """
        if not self.buffer_manager.has_completed_episodes():
            return {}

        completed_episodes = self.buffer_manager.get_completed_episodes()
        if not completed_episodes:
            return {}

        n_available = len(completed_episodes)
        if n_available < self.train_batch_size:
            return {}

        # Sample episodes once for both bridge and Q-learning
        sample_size = min(self.train_batch_size, n_available)
        indices = np.random.choice(n_available, sample_size, replace=False)
        sampled_episodes = [completed_episodes[i] for i in indices]

        # Update bridge network (less frequently than Q-network)
        bridge_metrics = {}
        if self.training_steps % self.bridge_update_interval == 0:
            # Flatten sampled episodes to transitions for bridge network training
            all_transitions = []
            for ep in sampled_episodes:
                all_transitions.extend(ep)

            if len(all_transitions) > 0:
                # Sample for bridge training
                sample_size_trans = min(len(all_transitions), 10000)
                sampled_transitions = all_transitions if len(all_transitions) <= sample_size_trans else \
                    [all_transitions[i] for i in np.random.choice(len(all_transitions), sample_size_trans, replace=False)]

                bridge_metrics = self.update_causal_networks(sampled_transitions)
                self.bridge_training_steps += 1

        # Perform Q-learning updates using the same sampled episodes
        num_updates = max(1, kwargs.get('num_episodes_collected', 1))
        accumulated = defaultdict(float)

        for _ in range(num_updates):
            step_metrics = self._train_step(sampled_episodes)
            for k, v in step_metrics.items():
                accumulated[k] += v

        # Average metrics
        losses = {k: v / num_updates for k, v in accumulated.items()}

        # Add bridge metrics
        losses.update(bridge_metrics)

        # Add metadata
        losses['epsilon'] = self.epsilon
        losses['causal_lambda'] = self.get_current_causal_lambda()
        losses['total_env_steps'] = self.total_env_steps
        losses['replay_buffer_episodes'] = n_available
        losses['num_updates'] = num_updates
        losses['bridge_training_steps'] = self.bridge_training_steps

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
