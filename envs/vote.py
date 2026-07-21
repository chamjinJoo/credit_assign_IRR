from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import MultiAgentEnv


class VoteEnv(MultiAgentEnv):
    """Cooperative voting Dec-POMDP with one-hot clues and shared rewards."""

    def __init__(self, num_agents: int = 4, episode_limit: int = 3,
                 observation_correct_prob: float = 0.7,
                 seed: Optional[int] = 0, **kwargs: Any):
        if num_agents < 2:
            raise ValueError("num_agents must be at least 2")
        if episode_limit < 1:
            raise ValueError("episode_limit must be at least 1")
        if not 0.0 <= observation_correct_prob <= 1.0:
            raise ValueError("observation_correct_prob must be between 0 and 1")
        super().__init__(num_agents=num_agents, seed=seed)
        if self._np_random is None:
            self._np_random = np.random.RandomState()
        self.n_agents = num_agents
        self.n_actions = num_agents
        self.action_dim = num_agents
        self.obs_dim = num_agents
        self.state_dim = num_agents + 1
        self.episode_limit = episode_limit
        self.observation_correct_prob = float(observation_correct_prob)
        self.timestep = 0
        self.hidden_leader: Optional[int] = None
        self.current_clues: Optional[np.ndarray] = None
        self.last_actions: Optional[np.ndarray] = None
        self.last_reward: Optional[float] = None
        self._terminated = True

    def reset(self, hidden_leader: Optional[int] = None) -> Dict[int, np.ndarray]:
        """Start an episode; ``hidden_leader`` is a deterministic test control."""
        if hidden_leader is None:
            hidden_leader = int(self._np_random.randint(self.num_agents))
        elif not isinstance(hidden_leader, (int, np.integer)):
            raise TypeError("hidden_leader must be an integer")
        elif not 0 <= int(hidden_leader) < self.num_agents:
            raise ValueError(f"hidden_leader must be in [0, {self.num_agents - 1}]")
        self.hidden_leader = int(hidden_leader)
        self.timestep = 0
        self.last_actions = None
        self.last_reward = None
        self._terminated = False
        self._sample_clues()
        return self.get_obs()

    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[str, Any]
    ]:
        """Apply one candidate choice per agent and return the shared reward."""
        if self._terminated:
            raise RuntimeError("step() called after episode termination; call reset()")
        action_array = self._validate_actions(actions)
        reward = float(np.mean(action_array == self.hidden_leader))
        self.last_actions = action_array.copy()
        self.last_reward = reward
        self.timestep += 1
        self._terminated = self.timestep >= self.episode_limit
        if not self._terminated:
            self._sample_clues()
        observations = self.get_obs()
        rewards = {agent_id: reward for agent_id in self.agents}
        dones = {agent_id: self._terminated for agent_id in self.agents}
        info = {
            "timestep": self.timestep,
            "hidden_leader": self.hidden_leader,
            "actions": action_array.copy(),
            "correct_votes": (action_array == self.hidden_leader).copy(),
            "episode_limit": self._terminated,
        }
        return observations, rewards, dones, info

    def _validate_actions(self, actions: Dict[int, int]) -> np.ndarray:
        if not isinstance(actions, dict):
            raise TypeError("actions must be a dictionary keyed by agent ID")
        if set(actions) != set(self.agents):
            raise ValueError(f"actions must contain exactly agent IDs {self.agents}")
        validated = np.empty(self.num_agents, dtype=np.int64)
        for agent_id in self.agents:
            action = actions[agent_id]
            if hasattr(action, "detach"):
                action = action.detach().cpu().item()
            elif isinstance(action, np.ndarray):
                if action.size != 1:
                    raise TypeError("each action must be a scalar integer")
                action = action.item()
            if not isinstance(action, (int, np.integer)):
                raise TypeError("each action must be an integer-compatible scalar")
            if not 0 <= int(action) < self.n_actions:
                raise ValueError(f"actions must be in [0, {self.n_actions - 1}]")
            validated[agent_id] = int(action)
        return validated

    def _sample_clues(self) -> None:
        clues = self._np_random.randint(self.num_agents - 1, size=self.num_agents)
        clues += clues >= self.hidden_leader
        correct = self._np_random.random_sample(self.num_agents) < self.observation_correct_prob
        clues[correct] = self.hidden_leader
        self.current_clues = clues.astype(np.int64, copy=False)

    def get_obs(self) -> Dict[int, np.ndarray]:
        """Return one ``float32`` one-hot clue per agent (dimension ``N``)."""
        if self.current_clues is None:
            raise RuntimeError("reset() must be called before retrieving observations")
        one_hot = np.eye(self.num_agents, dtype=np.float32)[self.current_clues]
        return {agent_id: one_hot[agent_id].copy() for agent_id in self.agents}

    def get_state(self) -> np.ndarray:
        """Return leader one-hot and normalized timestep (dimension ``N + 1``)."""
        if self.hidden_leader is None:
            raise RuntimeError("reset() must be called before retrieving state")
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[self.hidden_leader] = 1.0
        state[-1] = self.timestep / self.episode_limit
        return state

    def get_avail_actions(self) -> np.ndarray:
        return np.ones((self.num_agents, self.n_actions), dtype=np.int32)

    def get_env_info(self) -> Dict[str, int]:
        return {"n_agents": self.num_agents, "n_actions": self.n_actions,
                "obs_shape": self.obs_dim, "state_shape": self.state_dim,
                "episode_limit": self.episode_limit}

    def get_obs_dim(self) -> int:
        return self.obs_dim

    def get_action_dim(self) -> int:
        return self.action_dim

