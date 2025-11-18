import numpy as np
import torch
import random
import math
from typing import Dict, Tuple, List, Any, Optional
from abc import ABC, abstractmethod

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Visualization features will not work.")

# Optional imports with fallbacks
# try:
#     import jax
#     import jax.random as jr
#     JAX_AVAILABLE = True
# except ImportError:
#     JAX_AVAILABLE = False
#     print("Warning: JAX not available. MPE environments will not work.")

# try:
#     from jaxmarl import make as jaxmarl_make
#     JAXMARL_AVAILABLE = True
# except ImportError:
#     JAXMARL_AVAILABLE = False
#     print("Warning: JaxMARL not available. MPE environments will not work.")


class MultiAgentEnv(ABC):
    """
    Abstract base class for multi-agent environments.
    """
    
    def __init__(self, num_agents: int, seed: int = None):
        self.num_agents = num_agents
        self.agents = list(range(num_agents))
        self._seed = seed
        self._np_random = None
        
        # Initialize random number generator if seed is provided
        if seed is not None:
            self.seed(seed)
        
    def seed(self, seed: int = None):
        """Set seed for environment random number generator."""
        if seed is not None:
            self._seed = seed
            self._np_random = np.random.RandomState(seed)
        return [self._seed]
    
    @abstractmethod
    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment and return initial observations."""
        pass
    
    @abstractmethod  
    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],  # observations
        Dict[int, float],       # rewards
        Dict[int, bool],        # dones
        Dict[str, Any]          # info
    ]:
        """Take actions and return next observations, rewards, dones, info."""
        pass
    
    @abstractmethod
    def get_obs_dim(self) -> int:
        """Get observation dimension."""
        pass
    
    @abstractmethod
    def get_action_dim(self) -> int:
        """Get action dimension."""
        pass


class SimpleCooperativeEnv(MultiAgentEnv):
    """
    Simple cooperative environment for testing PCI-MAPPO.
    Agents need to coordinate to maximize collective reward.
    """
    
    def __init__(
        self, 
        num_agents: int = 3,
        grid_size: int = 5,
        max_steps: int = 50,
        cooperation_bonus: float = 2.0,
        seed: int = None
    ):
        super().__init__(num_agents, seed)
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.cooperation_bonus = cooperation_bonus
        
        # State
        self.agent_positions = {}
        self.target_positions = {}
        self.step_count = 0
        
        # Action space: 0=stay, 1=up, 2=down, 3=left, 4=right
        self.action_dim = 5
        
        # Observation space: [agent_x, agent_y, target_x, target_y, other_agent_positions...]
        self.obs_dim = 2 + 2 + 2 * (num_agents - 1)
        
    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment."""
        self.step_count = 0
        
        # Random initial positions
        for agent_id in self.agents:
            if self._np_random is not None:
                self.agent_positions[agent_id] = [
                    self._np_random.randint(0, self.grid_size),
                    self._np_random.randint(0, self.grid_size)
                ]
                
                self.target_positions[agent_id] = [
                    self._np_random.randint(0, self.grid_size),
                    self._np_random.randint(0, self.grid_size)
                ]
            else:
                self.agent_positions[agent_id] = [
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                ]
                
                self.target_positions[agent_id] = [
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                ]
        
        return self._get_observations()
    
    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float], 
        Dict[int, bool],
        Dict[str, Any]
    ]:
        """Take step in environment."""
        self.step_count += 1
        
        # Update positions based on actions
        for agent_id, action in actions.items():
            pos = self.agent_positions[agent_id]
            
            if action == 1:  # up
                pos[1] = min(pos[1] + 1, self.grid_size - 1)
            elif action == 2:  # down
                pos[1] = max(pos[1] - 1, 0)
            elif action == 3:  # left
                pos[0] = max(pos[0] - 1, 0)
            elif action == 4:  # right
                pos[0] = min(pos[0] + 1, self.grid_size - 1)
            # action == 0 is stay
        
        # Compute rewards
        rewards = self._compute_rewards()
        
        # Check if done
        done = self.step_count >= self.max_steps
        dones = {agent_id: done for agent_id in self.agents}
        
        # Get observations
        observations = self._get_observations()
        
        # Info
        info = {
            'step_count': self.step_count,
            'cooperation_events': self._count_cooperation_events()
        }
        
        return observations, rewards, dones, info
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all agents."""
        observations = {}
        
        for agent_id in self.agents:
            obs = []
            
            # Agent position
            obs.extend(self.agent_positions[agent_id])
            
            # Target position
            obs.extend(self.target_positions[agent_id])
            
            # Other agents' positions
            for other_id in self.agents:
                if other_id != agent_id:
                    obs.extend(self.agent_positions[other_id])
            
            observations[agent_id] = np.array(obs, dtype=np.float32)
        
        return observations
    
    def _compute_rewards(self) -> Dict[int, float]:
        """Compute rewards for all agents."""
        rewards = {}
        
        for agent_id in self.agents:
            # Distance to target
            agent_pos = np.array(self.agent_positions[agent_id])
            target_pos = np.array(self.target_positions[agent_id])
            distance = np.linalg.norm(agent_pos - target_pos)
            
            # Base reward (negative distance)
            reward = -distance / self.grid_size
            
            # Cooperation bonus
            cooperation_reward = self._get_cooperation_reward(agent_id)
            
            rewards[agent_id] = reward + cooperation_reward
        
        return rewards
    
    def _get_cooperation_reward(self, agent_id: int) -> float:
        """Compute cooperation reward for an agent."""
        agent_pos = np.array(self.agent_positions[agent_id])
        
        cooperation_reward = 0.0
        for other_id in self.agents:
            if other_id != agent_id:
                other_pos = np.array(self.agent_positions[other_id])
                distance = np.linalg.norm(agent_pos - other_pos)
                
                # Reward for being close to other agents
                if distance <= 1.5:  # Adjacent or same cell
                    cooperation_reward += self.cooperation_bonus
        
        return cooperation_reward
    
    def _count_cooperation_events(self) -> int:
        """Count cooperation events (agents in close proximity)."""
        count = 0
        for i, agent_id in enumerate(self.agents):
            for other_id in self.agents[i+1:]:
                agent_pos = np.array(self.agent_positions[agent_id])
                other_pos = np.array(self.agent_positions[other_id])
                distance = np.linalg.norm(agent_pos - other_pos)
                
                if distance <= 1.5:
                    count += 1
        
        return count
    
    def get_obs_dim(self) -> int:
        return self.obs_dim
    
    def get_action_dim(self) -> int:
        return self.action_dim
    
    def render(self, mode='human'):
        """Render the environment (optional)."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        
        # Place agents
        for agent_id in self.agents:
            pos = self.agent_positions[agent_id]
            if grid[pos[1], pos[0]] == '.':
                grid[pos[1], pos[0]] = str(agent_id)
            else:
                grid[pos[1], pos[0]] += str(agent_id)
        
        # Place targets
        for agent_id in self.agents:
            pos = self.target_positions[agent_id]
            if grid[pos[1], pos[0]] == '.':
                grid[pos[1], pos[0]] = f'T{agent_id}'
            else:
                grid[pos[1], pos[0]] += f'T{agent_id}'
        
        print(f"Step: {self.step_count}")
        for row in reversed(grid):
            print(' '.join(f'{cell:>4}' for cell in row))
        print()


class PettingZooWrapper(MultiAgentEnv):
    """
    Wrapper for PettingZoo environments.
    """
    
    def __init__(self, env_name: str, **kwargs):
        try:
            from pettingzoo import AECEnv
            from pettingzoo.utils import parallel_to_aec
            import importlib
            
            # Import the environment
            module_path, env_func = env_name.rsplit('.', 1)
            module = importlib.import_module(module_path)
            env_class = getattr(module, env_func)
            
            # Create environment
            self.env = env_class(**kwargs)
            
            # Convert to parallel if needed
            if hasattr(self.env, 'parallel_env'):
                self.env = self.env.parallel_env()
            
            # Get number of agents
            self.env.reset()
            num_agents = len(self.env.possible_agents)
            super().__init__(num_agents)
            
            self.agent_mapping = {i: agent for i, agent in enumerate(self.env.possible_agents)}
            self.reverse_mapping = {agent: i for i, agent in self.agent_mapping.items()}
            
        except ImportError:
            raise ImportError("PettingZoo not installed. Install with: pip install pettingzoo")
    
    def reset(self) -> Dict[int, np.ndarray]:
        obs_dict = self.env.reset()
        
        # Convert to our format
        observations = {}
        for agent_name, obs in obs_dict.items():
            if agent_name in self.reverse_mapping:
                agent_id = self.reverse_mapping[agent_name]
                observations[agent_id] = obs.astype(np.float32)
        
        return observations
    
    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool], 
        Dict[str, Any]
    ]:
        # Convert actions to PettingZoo format
        pz_actions = {}
        for agent_id, action in actions.items():
            agent_name = self.agent_mapping[agent_id]
            pz_actions[agent_name] = action
        
        # Step environment
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(pz_actions)
        
        # Convert back to our format
        observations = {}
        rewards = {}
        dones = {}
        
        for agent_name in self.env.possible_agents:
            if agent_name in self.reverse_mapping:
                agent_id = self.reverse_mapping[agent_name]
                
                if agent_name in obs_dict:
                    observations[agent_id] = obs_dict[agent_name].astype(np.float32)
                if agent_name in reward_dict:
                    rewards[agent_id] = float(reward_dict[agent_name])
                if agent_name in done_dict:
                    dones[agent_id] = bool(done_dict[agent_name])
        
        return observations, rewards, dones, info_dict
    
    def get_obs_dim(self) -> int:
        return self.env.observation_space(self.env.possible_agents[0]).shape[0]
    
    def get_action_dim(self) -> int:
        return self.env.action_space(self.env.possible_agents[0]).n


class ToyCausalEnv(MultiAgentEnv):
    """
    Toy environment for testing PCI-MAPPO with causal inference.
    
    Features:
    - 2 agents, 5-step episodes
    - Left/Right actions with 0.9 success probability
    - Causal events: 20% chance of high reward (+3) with observable precursors
    - Cross-agent causal signals: one agent's reward precursor visible to other agent
    """
    
    def __init__(self, num_agents: int = 2, episode_length: int = 5, seed: int = None):
        super().__init__(num_agents, seed)
        
        if num_agents != 2:
            raise ValueError("ToyCausalEnv only supports 2 agents")
        
        self.episode_length = episode_length
        self.step_count = 0
        
        # Action space: 0=left, 1=right
        self.action_dim = 2
        
        # Observation space: [high_reward_signal, reward_location]
        # high_reward_signal: 0 (normal) or 1 (high reward precursor)
        # reward_location: -1 (left) or 1 (right)
        self.obs_dim = 2       
        # Probabilities
        self.action_success_prob = 0.9
        self.reward_locations = []
        self.reward_events = np.zeros(self.episode_length)

    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment for new episode."""
        self.step_count = 0
        
        # 리스트 초기화 (누적 방지)
        self.reward_locations = []
        self.reward_events = np.zeros(self.episode_length)
        
        for t in range(self.episode_length):
            if self._np_random is not None:
                self.reward_locations.append([self._np_random.choice([-1, 1]), self._np_random.choice([-1, 1])])
                event_prob = self._np_random.random()
            else:
                self.reward_locations.append([np.random.choice([-1, 1]), np.random.choice([-1, 1])])
                event_prob = np.random.random()
            
            if event_prob < 0.8:  # 80% normal events
                pass
            elif event_prob < 0.9:  # 10% agent 1's high reward signal to agent 2
                self.reward_events[t] = 1
            else:   # 10% agent 2's high reward signal to agent 1
                self.reward_events[t] = 2

        return self._get_observations()
    
    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[str, Any]
    ]:
        """Take step in environment."""

        rewards = {}
        
        # according to event type, reward is determined
        # for normal event, reward is +1 if agent chooses correct action
        # for high reward event, reward is +3 if other agent chooses correct action, the other agent does not get reward.
        
        def _action_matches_location(action, location):
            """Convert between action space (0=left, 1=right) and location space (-1=left, 1=right)"""
            return (action == 0 and location == -1) or (action == 1 and location == 1)
        
        if self.reward_events[self.step_count] == 0: # normal
            for agent_id in self.agents:
                if _action_matches_location(actions[agent_id], self.reward_locations[self.step_count][agent_id]):
                    rewards[agent_id] = 1.0
                else:
                    rewards[agent_id] = 0.0

        elif self.reward_events[self.step_count] == 1: # agent 1's high reward signal to agent 2
            if (_action_matches_location(actions[1], self.reward_locations[self.step_count][1]) and 
                _action_matches_location(actions[0], self.reward_locations[self.step_count][0])):
                rewards[0] = 3.0
            else:
                rewards[0] = 0.0
            rewards[1] = 0.0

        elif self.reward_events[self.step_count] == 2: # agent 2's high reward signal to agent 1
            if (_action_matches_location(actions[0], self.reward_locations[self.step_count][0]) and 
                _action_matches_location(actions[1], self.reward_locations[self.step_count][1])):
                rewards[1] = 3.0
            else:
                rewards[1] = 0.0
            rewards[0] = 0.0

        self.step_count += 1
        
        # Check if episode is done
        done = self.step_count >= self.episode_length
        dones = {agent_id: done for agent_id in self.agents}
        
        # Get observations (only if episode is not done to avoid index error)
        if done:
            # Return dummy observations when episode is done
            observations = {agent_id: np.zeros(self.obs_dim, dtype=np.float32) for agent_id in self.agents}
        else:
            observations = self._get_observations()
        # Info
        info = {
            'step_count': self.step_count,
            'reward_events': self.reward_events.copy(),
            'reward_locations': self.reward_locations.copy()
        }
        
        return observations, rewards, dones, info
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all agents."""
        observations = {}
        if self.reward_events[self.step_count] == 0: # normal
            observations[0] = np.array([0, self.reward_locations[self.step_count][0]], dtype=np.float32)
            observations[1] = np.array([0, self.reward_locations[self.step_count][1]], dtype=np.float32)
        elif self.reward_events[self.step_count] == 1: # agent 1's high reward signal to agent 2
            observations[0] = np.array([0, self.reward_locations[self.step_count][0]], dtype=np.float32)
            observations[1] = np.array([1, self.reward_locations[self.step_count][1]], dtype=np.float32)
        else: # agent 2's high reward signal to agent 1
            observations[0] = np.array([1, self.reward_locations[self.step_count][0]], dtype=np.float32)
            observations[1] = np.array([0, self.reward_locations[self.step_count][1]], dtype=np.float32)
        
        return observations
    
    def get_obs_dim(self) -> int:
        return self.obs_dim
    
    def get_action_dim(self) -> int:
        return self.action_dim
    
    def render(self, mode='human'):
        """Render the environment state."""
        print(f"Step: {self.step_count}")
        print(f"Reward events: {self.reward_events}")
        print(f"Reward locations: {self.reward_locations}")


class DecTigerEnv(MultiAgentEnv):
    """
    Dec-Tiger 환경을 PCI-MAPPO의 MultiAgentEnv 인터페이스에 맞게 구현합니다.
    두 에이전트가 협력하여 호랑이가 있는 문을 찾아 열어야 하는 협력 게임입니다.
    """
    
    # Action constants
    OPEN_LEFT = 0
    OPEN_RIGHT = 1
    LISTEN = 2
    
    def __init__(self, 
                 seed: int = 0,
                 max_steps: int = 4,
                 failure_obs_prob: float = 0.15,
                 fully_observable: bool = False,
                 **kwargs):
        """
        Dec-Tiger 환경 초기화
        
        Args:
            seed: 랜덤 시드
            max_steps: 최대 스텝 수
            failure_obs_prob: 관찰 실패 확률
            fully_observable: 완전 관찰 가능 여부
        """
        # 에이전트 수는 항상 2
        super().__init__(num_agents=2)
        
        # Actions: 0:open-left, 1:open-right, 2:listen
        self._actions = ["open-left", "open-right", "listen"]
        self.maxtimestep = max_steps
        
        # Observation space: each agent has 2-dimensional observation (hear-left, hear-right)
        self.obs_dim = 2
        
        # Action space: each agent has 3 actions
        self.action_dim = 3
        
        # Environment parameters
        self.failure_obs_prob = failure_obs_prob
        self.fully_observable = fully_observable
        
        # State: tiger position (0: left, 1: right)
        self.state = None
        self.steps = 0
        
        # RNG setup
        self.py_random = random.Random(seed)
        np.random.seed(seed)
        
        self.reset()
    
    def seed(self, seed: int):
        """Seed both Python random and NumPy RNG."""
        self.py_random.seed(seed)
        np.random.seed(seed)
        return [seed]
    
    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment and return initial observations."""
        self.steps = 0
        
        # Randomize tiger position (0: left, 1: right)
        self.state = np.zeros(2, dtype=np.float32)
        tiger_position = np.random.randint(0, 2)
        self.state[tiger_position] = 1
        
        # Generate completely random observations for each agent
        obs = np.zeros((self.num_agents, 2), dtype=np.float32)
        tiger_observations = np.rint(np.random.rand(self.num_agents))
        
        for agent in range(self.num_agents):
            if self.fully_observable:
                # Fully observable: agents know the true tiger position
                obs[agent][tiger_position] = 1
            else:
                # Completely random observations (independent of true state)
                obs_index = int(tiger_observations[agent])
                obs[agent][obs_index] = 1
        
        # Convert to PCI-MAPPO format
        observations = {}
        for agent_id in range(self.num_agents):
            observations[agent_id] = obs[agent]
        
        return observations
    
    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],  # observations
        Dict[int, float],       # rewards
        Dict[int, bool],        # dones
        Dict[str, Any]          # info
    ]:
        """
        Execute actions in the environment.
        actions: Dict[int, int] - {agent_id: action}
        returns: (observations, rewards, dones, info)
        """
        # Convert PCI-MAPPO format to Dec-Tiger format
        a0, a1 = int(actions[0]), int(actions[1])
        ja = (self._actions[a0], self._actions[a1])
        
        # Get tiger position
        tiger_position = np.argmax(self.state)
        
        # Initialize reward and observations
        reward = 0.0
        obs = np.zeros((self.num_agents, 2), dtype=np.float32)
        done = False  # Initialize done variable
        # Check if both agents choose the same action
        if a0 == a1:
            if a0 == self.LISTEN:
                # Both agents listen
                reward = -0.2
                
                # Generate observations for both agents
                for agent in range(self.num_agents):
                    rand = np.random.rand()
                    if rand <= self.failure_obs_prob:
                        # Failed observation: wrong position
                        other_position = (tiger_position + 1) % 2
                        obs[agent][other_position] = 1
                    else:
                        # Successful observation: correct position
                        obs[agent][tiger_position] = 1
            else:
                # Both agents open the same door
                done = True
                if tiger_position == a0:
                    # Opened door with tiger: penalty
                    reward = -5.0
                else:
                    # Opened door without tiger: reward
                    reward = 2.0
                
                # Generate random observations after opening
                for agent in range(self.num_agents):
                    random_index = np.random.randint(0, 2)
                    obs[agent][random_index] = 1
        else:
            # Agents choose different actions
            rewards = []
            
            for agent, action_agent in enumerate([a0, a1]):
                if action_agent == self.LISTEN:
                    # Agent listens: small penalty
                    reward -= 0.1
                else:
                    # Agent opens a door
                    done = True
                    if tiger_position == action_agent:
                        # Opened door with tiger: penalty
                        rewards.append(-10.0)
                    else:
                        # Opened door without tiger: reward
                        rewards.append(1.0)
                
                # Generate random observation for this agent
                random_index = np.random.randint(0, 2)
                obs[agent][random_index] = 1
            
            # If any agent opened a door, add the minimum reward
            if rewards:
                reward += min(rewards)
        
        # Update step counter
        self.steps += 1
        
        # Check episode limit
        if self.steps >= self.maxtimestep:
            done = True
        
        info = {
            'state': self.state,
            'action': ja,
            'tiger_position': tiger_position,
            'steps': self.steps,
            'episode_limit': self.steps >= self.maxtimestep
        }
        
        # Convert to PCI-MAPPO format
        observations = {}
        rewards = {}
        dones = {}
        
        for agent_id in range(self.num_agents):
            observations[agent_id] = obs[agent_id]
            rewards[agent_id] = float(reward)  # Same reward for all agents
            dones[agent_id] = done
        
        return observations, rewards, dones, info
    
    def get_obs_dim(self) -> int:
        """Get observation dimension."""
        return self.obs_dim
    
    def get_action_dim(self) -> int:
        """Get action dimension."""
        return self.action_dim
    
    def render(self, mode='human'):
        """Render the environment state."""
        if mode == 'human':
            tiger_pos = "left" if np.argmax(self.state) == 0 else "right"
            print(f"Tiger position: {tiger_pos}")
            print(f"Step: {self.steps}/{self.maxtimestep}")
        return None


class StagHuntEnv(MultiAgentEnv):
    """
    Reference: https://github.com/NullDefault/Gym-Stag-Hunt/tree/master
    Stag-Hunt 환경을 PCI-MAPPO의 MultiAgentEnv 인터페이스에 맞게 구현합니다.
    두 에이전트가 협력하여 Stag를 사냥하거나 개별적으로 Hare를 사냥할 수 있는 게임입니다.
    ╔════════════════════════════╗
    ║ ·2   ·   H·    ·    · 1  · ║
    ║ ·    ·    ·    ·    ·    · ║
    ║ ·    ·    ·  S ·    ·    · ║
    ║ ·    ·    ·    ·   H·    · ║
    ║ ·    ·    ·    ·    ·    · ║
    ╚════════════════════════════╝
    """
    # Action constants
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    
    def __init__(
        self, 
        num_agents: int = 2,
        max_steps: int = 50,
        stag_follows: bool = False,
        run_away_after_maul: bool = False,
        forage_quantity: int = None,
        forage_reward: float = 3.0,
        stag_reward: float = 5.0,
        mauling_punishment: float = -1.0,
        seed: int = None,
        **kwargs):
        """
        Stag-Hunt 환경 초기화 (원본 게임 로직 기반)
        
        Args:
            num_agents: 에이전트 수 (기본 2, n명으로 설정 가능)
            max_steps: 최대 스텝 수 (기본 50)
            stag_follows: Stag가 가장 가까운 에이전트를 추적할지 여부 (True) 또는 랜덤 이동 (False)
            run_away_after_maul: Stag가 에이전트를 공격한 후 도망갈지 여부
            forage_quantity: 식물(개별 사냥 대상)의 개수 (None이면 num_agents와 동일하게 설정)
            forage_reward: 식물 수확 시 보상
            stag_reward: Stag 협력 사냥 시 보상
            mauling_punishment: 혼자 Stag를 사냥하려 할 때 페널티
        """
        # 에이전트 수를 유연하게 설정
        super().__init__(num_agents=num_agents, seed=seed)
        
        self.max_steps = max_steps
        self.stag_follows = stag_follows
        self.run_away_after_maul = run_away_after_maul
        # forage_quantity가 None이면 num_agents와 동일하게 설정
        self.forage_quantity = forage_quantity if forage_quantity is not None else num_agents
        self.step_count = 0
        
        # Grid size (7x7)
        self.grid_size_x = 7
        self.grid_size_y = 7
        
        # Action space: 0=stay, 1=up, 2=down, 3=left, 4=right
        self.action_dim = 5
        
        # Observation space: coordinate(2) + 3x3 grid(9*(num_agents+2)) + relative_positions(2*(num_agents-1))
        # 동적으로 계산: 2 + 9*(num_agents+2) + 2*(num_agents-1)
        self.obs_dim = 2 + 9 * (2*num_agents + 1) + 2 * (num_agents - 1)
        
        # Environment state
        self.agent_positions = {}  # {agent_id: (x, y)}
        self.stag_positions = {}  # {stag_id: (x, y)} - stag 1과 stag 2
        self.stag_alive = {}  # {stag_id: bool} - stag 생존 상태 추적
        self.hare_positions = {}  # {hare_id: (x, y)} - hares for individual hunting
        self.hare_alive = {}  # {hare_id: bool} - hare 생존 상태 추적
        
        # spwan positions should be symmetrically distributed around the center
        self.agent_spawn_positions = self._get_symmetric_spawn_positions(num_agents)

        # self.agent_spawn_positions = [
        #     (0, self.grid_size_y-1), 
        #     (self.grid_size_x-1, self.grid_size_y-1), 
        #     (self.grid_size_x-1, 0),
        #     (0, 0),
        #     (self.grid_size_x//2, self.grid_size_y//2)
        # ]  
        
        # Rewards
        self.forage_reward = forage_reward
        self.stag_reward = stag_reward
        self.mauling_punishment = mauling_punishment

        # Allocate agents to each stag: stag_i -> (agent i, agent i+1), 마지막 stag는 (agent N, agent 1)
        self.required_agents = {}
        for stag_id in range(num_agents):
            if stag_id == num_agents - 1:  # 마지막 stag
                self.required_agents[stag_id] = [stag_id, 0]
            else:
                self.required_agents[stag_id] = [stag_id, stag_id + 1]

        print(f"mauling_punishment: {self.mauling_punishment}")
        print(f"stag allocated agents: {self.required_agents}")
        
    def _get_empty_positions(self):
        """비어있는 위치들을 반환합니다."""
        occupied_positions = set()
        
        for agent_id in self.agents:
            if agent_id in self.agent_positions:
                occupied_positions.add(self.agent_positions[agent_id])
        for stag_pos in self.stag_positions.values():
            occupied_positions.add(stag_pos)
        for hare_pos in self.hare_positions.values():
            occupied_positions.add(hare_pos)
        
        # 모든 가능한 위치에서 점유된 위치 제외
        empty_positions = []
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                if (x, y) not in occupied_positions:
                    empty_positions.append((x, y))
        return empty_positions

    def _get_symmetric_spawn_positions(self, num_agents):
        """대칭적으로 분포된 스폰 위치를 반환합니다."""
        spawn_positions = []

        center_x = (self.grid_size_x - 1) / 2
        center_y = (self.grid_size_y - 1) / 2
        max_radius = (min(self.grid_size_x, self.grid_size_y) - 1) / 2
        radius = max(1.0, max_radius)

        for i in range(num_agents):
            angle = 2 * math.pi * i / num_agents
            attempt_radius = radius

            while attempt_radius >= 0:
                x = int(round(center_x + attempt_radius * math.cos(angle)))
                y = int(round(center_y + attempt_radius * math.sin(angle)))

                x = min(max(0, x), self.grid_size_x - 1)
                y = min(max(0, y), self.grid_size_y - 1)

                if (x, y) not in spawn_positions:
                    spawn_positions.append((x, y))
                    break

                attempt_radius -= 1
            if len(spawn_positions) <= i:
                # 모든 반지름에서도 중복이 발생하면 아직 사용되지 않은 격자 좌표를 순차적으로 선택
                for x in range(self.grid_size_x):
                    for y in range(self.grid_size_y):
                        if (x, y) not in spawn_positions:
                            spawn_positions.append((x, y))
                            break
                    if len(spawn_positions) > i:
                        break

        # 여전히 부족하다면 남은 격자를 순서대로 채웁니다.
        if len(spawn_positions) < num_agents:
            for x in range(self.grid_size_x):
                for y in range(self.grid_size_y):
                    if (x, y) not in spawn_positions:
                        spawn_positions.append((x, y))
                    if len(spawn_positions) == num_agents:
                        break
                if len(spawn_positions) == num_agents:
                    break

        return spawn_positions

    def _spawn_object_randomly(self):
        """비어있는 위치에 객체를 랜덤하게 스폰합니다."""
        empty_positions = self._get_empty_positions()
        if empty_positions:
            if self._np_random is not None:
                return empty_positions[self._np_random.randint(len(empty_positions))]
            else:
                return empty_positions[np.random.randint(len(empty_positions))]
        else:
            # 비어있는 위치가 없으면 에러 발생 (충돌 방지)
            raise ValueError("No empty positions available for spawning object")
    
    def _spawn_stags_with_separation(self, num_stags):
        """stag들을 서로 상하좌우로 붙어있지 않게 스폰합니다."""
        stag_positions = []
        empty_positions = self._get_empty_positions()
        
        if len(empty_positions) < num_stags:
            raise ValueError(f"Not enough empty positions for {num_stags} stags")
        
        # 첫 번째 stag는 랜덤하게 선택
        if self._np_random is not None:
            first_stag_pos = empty_positions[self._np_random.randint(len(empty_positions))]
        else:
            first_stag_pos = empty_positions[np.random.randint(len(empty_positions))]
        
        stag_positions.append(first_stag_pos)
        available_positions = [pos for pos in empty_positions if pos != first_stag_pos]
        
        # 나머지 stag들을 상하좌우로 붙어있지 않게 배치
        for _ in range(num_stags - 1):
            valid_positions = []
            
            for pos in available_positions:
                # 기존 stag들과 상하좌우로 붙어있지 않은지 확인
                is_valid = True
                for existing_pos in stag_positions:
                    x_diff = abs(pos[0] - existing_pos[0])
                    y_diff = abs(pos[1] - existing_pos[1])
                    # 상하좌우로 붙어있으면 (맨하탄 거리가 1이면) 제외
                    if x_diff + y_diff == 1:
                        is_valid = False
                        break
                
                if is_valid:
                    valid_positions.append(pos)
            
            if not valid_positions:
                # 유효한 위치가 없으면 에러 발생
                raise ValueError("No valid positions available for stag separation")
            
            # 유효한 위치 중에서 랜덤 선택
            if self._np_random is not None:
                selected_pos = valid_positions[self._np_random.randint(len(valid_positions))]
            else:
                selected_pos = valid_positions[np.random.randint(len(valid_positions))]
            stag_positions.append(selected_pos)
            available_positions.remove(selected_pos)
        
        return stag_positions

    def reset(self) -> Dict[int, np.ndarray]:
        """환경을 리셋하고 초기 관찰을 반환합니다."""
        self.step_count = 0
       
        for i, agent_id in enumerate(self.agents):
            self.agent_positions[agent_id] = self._spawn_object_randomly() # random spawn
            # self.agent_positions[agent_id] = self.agent_spawn_positions[i] # fixed spawn
        # n개의 Stag를 서로 상하좌우로 붙어있지 않게 배치 (이동하지 않음)
        self.stag_positions = {}
        self.stag_alive = {}
        stag_positions = self._spawn_stags_with_separation(self.num_agents)
        for stag_id in range(self.num_agents):  # num_agents만큼의 stag 생성
            self.stag_positions[stag_id] = stag_positions[stag_id]
            self.stag_alive[stag_id] = True  # 모든 stag를 살아있는 상태로 초기화
        
        # 식물들을 빈 위치에 랜덤 스폰
        try:
            self.hare_positions = {}
            self.hare_alive = {}
            for hare_id in range(self.forage_quantity):
                hare_pos = self._spawn_object_randomly()
                self.hare_positions[hare_id] = hare_pos
                self.hare_alive[hare_id] = True  # 모든 hare를 살아있는 상태로 초기화
        except ValueError as e:
            # 매우 드문 경우지만, 빈 위치가 없으면 에러 처리
            print(f"Warning during reset: {e}")
            # 기본 위치로 설정
            self.hare_positions = {0: (1, 1), 1: (3, 3)}  # 대각선 위치
            self.hare_alive = {0: True, 1: True}
        
        return self._get_observations()
    
    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[str, Any]
    ]:
        """환경에서 한 스텝을 진행합니다."""
        self.step_count += 1
        
        # Stag는 더 이상 이동하지 않음 (고정 위치)
        
        # 에이전트 이동 처리
        self._process_movement_actions(actions)
        
        # 사냥 처리 직전의 사냥 가능 객체 상태를 캡처 (의사결정 기준 시점)
        huntable_objects_pre = self._get_huntable_objects()
        
        # 사냥 처리 및 보상 계산
        rewards = {agent_id: 0.0 for agent_id in self.agents}
        hunting_info = self._process_hunting(rewards)
        
        # 에피소드 종료 조건 확인
        alive_stags = sum(1 for alive in self.stag_alive.values() if alive)
        alive_hares = sum(1 for alive in self.hare_alive.values() if alive)
        done = (self.step_count >= self.max_steps or 
                (alive_stags == 0 and alive_hares == 0))
        
        # 관찰 획득
        observations = self._get_observations()
        
        # 사냥 가능한 objects 정보 (사냥 처리 이전 상태를 사용해 일관된 로깅 보장)
        huntable_objects = huntable_objects_pre
        
        # Info - 협력 사냥 정보 포함
        info = {
            'step_count': self.step_count,
            'agent_positions': self.agent_positions.copy(),
            'stag_positions': self.stag_positions.copy(),
            'stag_alive': self.stag_alive.copy(),
            'hare_positions': self.hare_positions.copy(),
            'hare_alive': self.hare_alive.copy(),
            'stag_hunts': hunting_info['stag_hunts'],  # {stag_id: [participating_agent_ids]}
            'stag_attempts': hunting_info['stag_attempts'],  # {stag_id: [agents who attempted to hunt]}
            'hare_hunts': hunting_info['hare_hunts'],  # [agent_ids who hunted hares]
            'hare_attempts': hunting_info['hare_attempts'],  # [agent_ids who attempted to hunt hares]
            'huntable_objects': huntable_objects  # 사냥 가능한 objects 정보
        }
        dones = {agent_id: done for agent_id in self.agents}
            
        return observations, rewards, dones, info
    
    def _get_huntable_objects(self) -> Dict[str, Any]:
        """
        현재 시점에서 사냥 가능한 objects를 판단합니다.
        
        Returns:
            Dict containing huntable stags and hares with their requirements
            - huntable_stags: required_agents에 포함된 stag들 (my_stag, coop_stag)
            - other_stags_by_agent: 각 agent별로 자신이 required_agents에 포함되지 않은 stag들 (other_stag)
            - huntable_hares: 사냥 가능한 hare들
        """
        huntable_stags = {}
        huntable_hares = []
        # agent별 other_stags 정보 추가
        other_stags_by_agent = {agent_id: {} for agent_id in self.agents}
        
        # Stag 사냥 가능성 판단
        for stag_id, (stag_x, stag_y) in self.stag_positions.items():
            if not self.stag_alive[stag_id]:
                continue
                
            # Stag i를 사냥하려면 required_agents에서 지정된 agents가 필요
            required_agents = self.required_agents[stag_id]
            
            # 사냥 가능한 위치들 (stag의 상하좌우 + stag 자리)
            huntable_positions = [
                (stag_x, stag_y),      # stag 자리
                (stag_x, stag_y + 1),  # 상
                (stag_x, stag_y - 1),  # 하
                (stag_x + 1, stag_y),  # 우
                (stag_x - 1, stag_y)   # 좌
            ]
            
            # 필요한 에이전트들이 모두 존재하는지 확인
            if all(agent_id in self.agent_positions for agent_id in required_agents):
                agent_positions = {agent_id: self.agent_positions[agent_id] for agent_id in required_agents}
                
                # 모든 필요한 에이전트가 사냥 가능한 위치에 있는지 확인
                all_agents_can_hunt = all(
                    agent_pos in huntable_positions 
                    for agent_pos in agent_positions.values()
                )
                
                if all_agents_can_hunt:
                    huntable_stags[stag_id] = {
                        'position': (stag_x, stag_y),
                        'required_agents': required_agents,
                        'agent_positions': agent_positions,
                        'huntable_positions': huntable_positions
                    }
            
            # 각 agent별로 other_stag 정보 수집 (required_agents에 포함되지 않은 agent들)
            # 모든 agent가 이 stag를 볼 수 있는지 확인 (사냥 가능한 위치에 있는지)
            for agent_id in self.agents:
                if agent_id not in required_agents:  # 자신이 required_agents에 포함되지 않은 stag
                    agent_pos = self.agent_positions.get(agent_id)
                    if agent_pos and agent_pos in huntable_positions:
                        # agent가 이 stag의 사냥 가능한 위치에 있으면 other_stag 기회로 간주
                        other_stags_by_agent[agent_id][stag_id] = {
                            'position': (stag_x, stag_y),
                            'required_agents': required_agents,
                            'agent_position': agent_pos,
                            'huntable_positions': huntable_positions
                        }
        
        # Hare 사냥 가능성 판단 (단독 사냥)
        for hare_id, (hare_x, hare_y) in self.hare_positions.items():
            if not self.hare_alive[hare_id]:
                continue
                
            # Hare는 단독으로 사냥 가능 (상하좌우 + hare 자리)
            huntable_positions = [
                (hare_x, hare_y + 1),  # 상
                (hare_x, hare_y - 1),  # 하
                (hare_x + 1, hare_y),  # 우
                (hare_x - 1, hare_y)   # 좌
            ]
            
            # 사냥 가능한 에이전트들 찾기
            hunting_agents = []
            for agent_id, agent_pos in self.agent_positions.items():
                if agent_pos in huntable_positions:
                    hunting_agents.append(agent_id)
            
            if hunting_agents:
                huntable_hares.append({
                    'hare_id': hare_id,
                    'position': (hare_x, hare_y),
                    'hunting_agents': hunting_agents,
                    'huntable_positions': huntable_positions
                })
        
        return {
            'huntable_stags': huntable_stags,  # 기존 로직 유지 (my_stag, coop_stag)
            'other_stags_by_agent': other_stags_by_agent,  # 새로운 필드 (other_stag)
            'huntable_hares': huntable_hares,
            'total_huntable_stags': len(huntable_stags),
            'total_huntable_hares': len(huntable_hares)
        }

    def _process_movement_actions(self, actions: Dict[int, int]):
        """이동 액션을 처리합니다."""
        new_positions = {}
        
        for agent_id, action in actions.items():
            current_x, current_y = self.agent_positions[agent_id]
            
            # 이동 방향 계산
            if action == self.STAY:
                new_x, new_y = current_x, current_y
            elif action == self.UP:
                new_x, new_y = current_x, current_y + 1
            elif action == self.DOWN:
                new_x, new_y = current_x, current_y - 1
            elif action == self.LEFT:
                new_x, new_y = current_x - 1, current_y
            elif action == self.RIGHT:
                new_x, new_y = current_x + 1, current_y
            else:
                # 잘못된 액션인 경우 stay
                new_x, new_y = current_x, current_y
            
            # 이동 가능성 확인 
            # 경계 체크
            if (0 <= new_x < self.grid_size_x and 0 <= new_y < self.grid_size_y):
                new_positions[agent_id] = (new_x, new_y)
                # target_pos = (new_x, new_y)
                # Stag 위치로의 이동 제한: stag 사냥 조건을 만족할 때만 허용
                # if target_pos in self.stag_positions.values():
                #     # 목표 위치가 stag 위치인지 확인
                #     stag_at_target = None
                #     for stag_id, stag_pos in self.stag_positions.items():
                #         if stag_pos == target_pos:
                #             stag_at_target = stag_id
                #             break
                    
                #     if stag_at_target is not None:
                #         # 해당 stag 위치로 이동하는 에이전트들 확인
                #         agents_moving_to_stag = []
                #         for other_agent_id, other_action in actions.items():
                #             other_current_x, other_current_y = self.agent_positions[other_agent_id]
                #             other_new_x, other_new_y = other_current_x, other_current_y
                            
                #             # 다른 에이전트의 이동 계산
                #             if other_action == self.UP:
                #                 other_new_x, other_new_y = other_current_x, other_current_y + 1
                #             elif other_action == self.DOWN:
                #                 other_new_x, other_new_y = other_current_x, other_current_y - 1
                #             elif other_action == self.LEFT:
                #                 other_new_x, other_new_y = other_current_x - 1, other_current_y
                #             elif other_action == self.RIGHT:
                #                 other_new_x, other_new_y = other_current_x + 1, other_current_y
                            
                #             # 같은 stag 위치로 이동하는지 확인
                #             if (other_new_x, other_new_y) == target_pos:
                #                 agents_moving_to_stag.append(other_agent_id)
                        
                #         # stag 사냥 조건 확인: stag i는 agent i와 agent i+1이 필요 (마지막 stag n은 agent n과 agent 1)
                #         required_agents = self.required_agents[stag_at_target]
                        
                #         # 필수 에이전트들이 모두 stag 위치로 이동하는지 확인
                #         if not all(agent_id in agents_moving_to_stag for agent_id in required_agents):
                #             # 조건을 만족하지 않으면 이동 차단
                #             new_x, new_y = current_x, current_y
                
                # new_positions[agent_id] = (new_x, new_y)
            else:
                # 경계 밖으로 이동하려 하면 현재 위치 유지
                new_positions[agent_id] = (current_x, current_y)
        
        # 에이전트들이 모든 위치를 공유할 수 있도록 허용
        final_positions = {}
        
        # 모든 에이전트가 원하는 위치로 이동할 수 있음 (위치 공유 허용)
        for agent_id, new_pos in new_positions.items():
            final_positions[agent_id] = new_pos
        
        # 위치 업데이트
        self.agent_positions.update(final_positions)

    def _process_hunting(self, rewards: Dict[int, float]) -> Dict:
        """사냥 처리: stag i는 agent i와 agent i+1이 동시에 사냥해야 성공
        stag i가 agent i에게만 reward 제공
        
        Returns:
            hunting_info: 사냥 정보 딕셔너리
                - 'stag_hunts': {stag_id: [participating_agent_ids]}
                - 'hare_hunts': [agent_ids who hunted hares]
        """
        # 사냥 정보 추적
        hunting_info = {
            'stag_hunts': {},  # {stag_id: [agents who participated successfully]}
            'stag_attempts': {},  # {stag_id: [agents who attempted to hunt]}
            'hare_hunts': [],   # [agents who hunted hares successfully]
            'hare_attempts': []  # [agents who attempted to hunt hares]
        }
        
        # 각 에이전트가 식물을 수확했는지 추적
        agent_foraged = {agent_id: False for agent_id in self.agents}
        
        # 식물 수확 체크 - 각 hare별로 해당 위치에 있는 에이전트들 확인
        for hare_id, hare_pos in self.hare_positions.items():
            if not self.hare_alive.get(hare_id, False):
                continue  # 이미 수확된 hare는 건너뛰기
                
            agents_at_hare = []
            for agent_id in self.agents:
                if self.agent_positions[agent_id] == hare_pos:
                    agents_at_hare.append(agent_id)
            
            if agents_at_hare:
                # 모든 시도한 에이전트들을 기록
                hunting_info['hare_attempts'].extend(agents_at_hare)
                
                # 여러 에이전트가 같은 hare를 사냥할 때 무작위로 하나 선택
                if self._np_random is not None:
                    selected_agent = agents_at_hare[self._np_random.randint(len(agents_at_hare))]
                else:
                    selected_agent = agents_at_hare[np.random.randint(len(agents_at_hare))]
                
                rewards[selected_agent] += self.forage_reward
                self.hare_alive[hare_id] = False  # hare를 수확된 상태로 표시
                agent_foraged[selected_agent] = True
                hunting_info['hare_hunts'].append(selected_agent)
        
        # Stag 사냥 체크 (새로운 로직: stag i는 agent i와 agent i+1이 동시에 사냥해야 성공)
        # 각 stag별로 체크 (살아있는 stag만)
        for stag_id, stag_pos in self.stag_positions.items():
            if not self.stag_alive.get(stag_id, False):
                continue
            # 해당 stag 위치에 있는 에이전트들 확인
            agents_at_stag = []
            for agent_id in self.agents:
                if self.agent_positions[agent_id] == stag_pos:
                    agents_at_stag.append(agent_id)
            
            # stag i를 사냥하기 위한 필수 에이전트들
            required_agents = self.required_agents[stag_id]

            # 모든 stag 위치에 있는 에이전트들의 시도를 기록
            if agents_at_stag:
                hunting_info['stag_attempts'][stag_id] = agents_at_stag.copy()
            
            # required_agents에 속하지 않은 agent가 stag에 접근할 경우 페널티
            for agent_id in agents_at_stag:
                if agent_id not in required_agents:
                    rewards[agent_id] += self.mauling_punishment
            # 필수 에이전트들이 모두 stag 위치에 있는지 확인
            if all(agent_id in agents_at_stag for agent_id in required_agents):
                # 협력 사냥 성공 - stag i는 agent i에게만 reward 제공
                rewards[stag_id] += self.stag_reward
                # 협력 사냥 정보 기록
                hunting_info['stag_hunts'][stag_id] = agents_at_stag.copy()
                # 사냥된 stag를 죽은 상태로 표시 
                self.stag_alive[stag_id] = False
            
            # Stag가 에이전트를 공격한 후 도망가는 경우
            if self.run_away_after_maul and len(agents_at_stag) > 0:
                self.stag_alive[stag_id] = False
        
        return hunting_info
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """각 에이전트의 관찰을 반환합니다 (좌표 + 3x3 그리드)."""
        observations = {}
        
        for agent_id in self.agents:
            agent_x, agent_y = self.agent_positions[agent_id]
            
            coord_obs = [agent_x/(self.grid_size_x-1), agent_y/(self.grid_size_y-1)] # 좌표 정보 (2차원)
            grid_obs = [] # 3x3 그리드 정보 
            
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    check_x, check_y = agent_x + dx, agent_y + dy
                    check_pos = (check_x, check_y)
                    
                    # n+num_agents+1차원 튜플: (agent0, agent1, ..., agent_n-1, stag0, stag1, ..., stag_n-1, hare)
                    cell_info = [0.0] * (self.num_agents + self.num_agents + 1)
                    
                    # 경계 밖인지 확인
                    if not (0 <= check_x < self.grid_size_x and 0 <= check_y < self.grid_size_y):
                        # 경계 밖은 -1로 표현 (모든 요소를 -1로)
                        cell_info = [-1.0] * (self.num_agents + self.num_agents + 1)
                    else:
                        # 에이전트 확인 (여러 agent가 같은 위치에 있을 수 있음)
                        for other_agent_id in self.agents:
                            if self.agent_positions[other_agent_id] == check_pos:
                                cell_info[other_agent_id] = 1.0
                        
                        # Stag 확인 (각 stag별로 개별 인덱스 할당) - 살아있는 stag만 확인
                        for stag_id, stag_pos in self.stag_positions.items():
                            if stag_pos == check_pos and self.stag_alive.get(stag_id, False):
                                cell_info[self.num_agents + stag_id] = 1.0  # stag_id에 해당하는 인덱스를 1로 설정 
                        
                        # hare 확인 (식물들 체크) - 살아있는 hare만
                        for hare_id, hare_pos in self.hare_positions.items():
                            if hare_pos == check_pos and self.hare_alive.get(hare_id, False):
                                cell_info[self.num_agents + self.num_agents] = 1.0
                                break
                    
                    grid_obs.extend(cell_info)
            
            # 다른 에이전트의 상대적 위치 정보 (2차원)
            relative_positions = []
            for other_agent_id in self.agents:
                if other_agent_id != agent_id:
                    other_x, other_y = self.agent_positions[other_agent_id]
                    # 상대적 위치 계산 (x, y 차이)
                    rel_x = (other_x - agent_x)/(self.grid_size_x-1)
                    rel_y = (other_y - agent_y)/(self.grid_size_y-1)
                    relative_positions.extend([rel_x, rel_y])
            # 전체 관찰 결합 (2 + 9*(num_agents+2) + 2*(num_agents-1) 차원)
            full_obs = coord_obs + grid_obs + relative_positions
            observations[agent_id] = np.array(full_obs, dtype=np.float32)
        
        return observations
    
    def get_obs_dim(self) -> int:
        """관찰 차원을 반환합니다."""
        return self.obs_dim
    
    def get_action_dim(self) -> int:
        """액션 차원을 반환합니다."""
        return self.action_dim
    
    def render(self, mode='human'):
        """환경 상태를 렌더링합니다."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib이 설치되지 않아 렌더링을 할 수 없습니다.")
            return None
            
        if mode == 'human':
            self._render_human()
        elif mode == 'rgb_array':
            return self._render_rgb_array()
        else:
            raise ValueError(f"지원하지 않는 렌더링 모드: {mode}")
    
    def _render_human(self):
        """인간이 볼 수 있는 형태로 렌더링합니다."""
        if not hasattr(self, '_fig') or self._fig is None:
            # 초기 설정
            self._fig, self._ax = plt.subplots(figsize=(8, 8))
            self._ax.set_xlim(-0.5, self.grid_size_x - 0.5)
            self._ax.set_ylim(-0.5, self.grid_size_y - 0.5)
            self._ax.set_aspect('equal')
            self._ax.grid(True, alpha=0.3)
            self._ax.set_title('Stag Hunt Environment', fontsize=14, fontweight='bold')
            
            # 축 레이블 설정
            self._ax.set_xlabel('X coordinate')
            self._ax.set_ylabel('Y coordinate')
            
            # 격자 표시
            self._ax.set_xticks(range(self.grid_size_x))
            self._ax.set_yticks(range(self.grid_size_y))
            
            # 범례를 위한 빈 플롯 요소들
            self._legend_elements = []
            
        # 기존 플롯 요소들 제거
        for element in self._ax.collections + self._ax.patches + self._ax.texts:
            if hasattr(element, 'remove') and element is not None:
                try:
                    element.remove()
                except:
                    pass  # 제거 실패 시 무시
        
        # 축 범위 재설정
        self._ax.set_xlim(-0.5, self.grid_size_x - 0.5)
        self._ax.set_ylim(-0.5, self.grid_size_y - 0.5)
        
        # 색상 팔레트 정의 (agent-stag pair별)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Agent 렌더링 (원형, 번호 표시, 3x3 시야)
        for agent_id in self.agents:
            if agent_id in self.agent_positions:
                x, y = self.agent_positions[agent_id]
                color = colors[agent_id % len(colors)]
                
                # Agent의 3x3 시야 영역 그리기 (연한 색상)
                vision_alpha = 0.2
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        vision_x, vision_y = x + dx, y + dy
                        # 그리드 범위 내에 있는지 확인
                        if (0 <= vision_x < self.grid_size_x and 
                            0 <= vision_y < self.grid_size_y):
                            # 3x3 시야 영역을 연한 색상으로 표시
                            vision_rect = plt.Rectangle((vision_x-0.5, vision_y-0.5), 1, 1, 
                                                     color=color, alpha=vision_alpha, 
                                                     ec='none', linewidth=0)
                            self._ax.add_patch(vision_rect)
                
                # Agent 원형 그리기
                circle = plt.Circle((x, y), 0.3, color=color, alpha=0.8, ec='black', linewidth=2)
                self._ax.add_patch(circle)
                
                # Agent 번호 표시
                self._ax.text(x, y, f'A{agent_id}', ha='center', va='center', 
                            fontsize=10, fontweight='bold', color='white')
        
        # Stag 렌더링 (삼각형, 번호 표시) - 살아있는 stag만
        for stag_id in self.stag_positions:
            if not self.stag_alive.get(stag_id, False):
                continue
            x, y = self.stag_positions[stag_id]
            color = colors[stag_id % len(colors)]  # 같은 색상 사용
            
            # Stag 삼각형 그리기
            triangle = plt.Polygon([(x, y-0.3), (x-0.3, y+0.3), (x+0.3, y+0.3)], 
                                 color=color, alpha=0.8, ec='black', linewidth=2)
            self._ax.add_patch(triangle)
            
            # Stag 번호 표시
            self._ax.text(x, y+0.1, f'S{stag_id}', ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white')
        
        # hare (Rabbit) 렌더링 (사각형) - 살아있는 hare만
        for hare_id, (x, y) in self.hare_positions.items():
            if not self.hare_alive.get(hare_id, False):
                continue  # 수확된 hare는 렌더링하지 않음
                
            # hare는 회색 사각형으로 표시
            rect = plt.Rectangle((x-0.25, y-0.25), 0.5, 0.5, 
                               color='gray', alpha=0.7, ec='black', linewidth=1)
            self._ax.add_patch(rect)
            
            # hare 번호 표시
            self._ax.text(x, y, f'H{hare_id}', ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='white')
        
        # 범례 생성
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Agent'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                      markersize=10, label='Stag'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                      markersize=10, label='Hare'),
            plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.2, label='Agent Vision (3x3)')
        ]
        
        # 기존 범례 제거
        if hasattr(self._ax, 'legend_') and self._ax.legend_ is not None:
            self._ax.legend_.remove()
        
        # 새 범례 추가
        self._ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        # 스텝 정보 표시
        info_text = f"Step: {self.step_count}/{self.max_steps}"
        self._ax.text(0.02, 0.98, info_text, transform=self._ax.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.draw()
        plt.pause(0.02)  # 부드러운 실시간 업데이트를 위한 지연
    
    def _render_rgb_array(self):
        """RGB 배열 형태로 렌더링합니다."""
        # Animation을 위한 별도 figure 생성 (화면에 표시하지 않음)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-0.5, self.grid_size_x - 0.5)
        ax.set_ylim(-0.5, self.grid_size_y - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Stag Hunt Environment', fontsize=14, fontweight='bold')
        
        # 축 레이블 설정
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        
        # 격자 표시
        ax.set_xticks(range(self.grid_size_x))
        ax.set_yticks(range(self.grid_size_y))
        
        # 색상 팔레트 정의 (agent-stag pair별)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Agent 렌더링 (원형, 번호 표시, 3x3 시야)
        for agent_id in self.agents:
            if agent_id in self.agent_positions:
                x, y = self.agent_positions[agent_id]
                color = colors[agent_id % len(colors)]
                
                # Agent의 3x3 시야 영역 그리기 (연한 색상)
                vision_alpha = 0.2
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        vision_x, vision_y = x + dx, y + dy
                        # 그리드 범위 내에 있는지 확인
                        if (0 <= vision_x < self.grid_size_x and 
                            0 <= vision_y < self.grid_size_y):
                            # 3x3 시야 영역을 연한 색상으로 표시
                            vision_rect = plt.Rectangle((vision_x-0.5, vision_y-0.5), 1, 1, 
                                                     color=color, alpha=vision_alpha, 
                                                     ec='none', linewidth=0)
                            ax.add_patch(vision_rect)
                
                # Agent 원형 그리기
                circle = plt.Circle((x, y), 0.3, color=color, alpha=0.8, ec='black', linewidth=2)
                ax.add_patch(circle)
                
                # Agent 번호 표시
                ax.text(x, y, f'A{agent_id}', ha='center', va='center', 
                            fontsize=10, fontweight='bold', color='white')
        
        # Stag 렌더링 (삼각형, 번호 표시) - 살아있는 stag만
        for stag_id in self.stag_positions:
            if not self.stag_alive.get(stag_id, False):
                continue
            x, y = self.stag_positions[stag_id]
            color = colors[stag_id % len(colors)]  # 같은 색상 사용
            
            # Stag 삼각형 그리기
            triangle = plt.Polygon([(x, y-0.3), (x-0.3, y+0.3), (x+0.3, y+0.3)], 
                                 color=color, alpha=0.8, ec='black', linewidth=2)
            ax.add_patch(triangle)
            
            # Stag 번호 표시
            ax.text(x, y+0.1, f'S{stag_id}', ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white')
        
        # hare (Rabbit) 렌더링 (사각형) - 살아있는 hare만
        for hare_id, (x, y) in self.hare_positions.items():
            if not self.hare_alive.get(hare_id, False):
                continue  # 수확된 hare는 렌더링하지 않음
                
            # hare는 회색 사각형으로 표시
            rect = plt.Rectangle((x-0.25, y-0.25), 0.5, 0.5, 
                               color='gray', alpha=0.7, ec='black', linewidth=1)
            ax.add_patch(rect)
            
            # hare 번호 표시
            ax.text(x, y, f'H{hare_id}', ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='white')
        
        # 스텝 정보 표시
        info_text = f"Step: {self.step_count}/{self.max_steps}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # RGB 배열로 변환
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        arr = np.asarray(buf)
        
        # Figure 정리
        plt.close(fig)
        
        return arr[:, :, :3]  # Alpha 채널 제거
    
    def close_render(self):
        """렌더링 창을 닫습니다."""
        if hasattr(self, '_fig') and self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None


class MPESimpleSpreadEnv(MultiAgentEnv):
    """
    MPE Simple Spread 환경을 PCI-MAPPO의 MultiAgentEnv 인터페이스에 맞게 구현합니다.
    N개의 에이전트가 N개의 랜드마크에 분산되어 위치해야 하는 협력 게임입니다.
    """
    
    def __init__(
        self,
        n_agents: int = 2,
        local_ratio: float = 0.5,
        max_cycles: int = 25,
        continuous_actions: bool = False,
        seed: int = 0,
        **kwargs
    ):
        """
        MPE Simple Spread 환경 초기화
        
        Args:
            n_agents: 에이전트 수
            local_ratio: 지역 보상 비율 (0.0: 글로벌 보상만, 1.0: 지역 보상만)
            max_cycles: 최대 스텝 수
            continuous_actions: 연속 액션 사용 여부
            seed: 랜덤 시드
        """
        # if not JAXMARL_AVAILABLE:
        #     raise ImportError("JaxMARL is required for MPESimpleSpreadEnv. Install with: pip install git+https://github.com/instadeepai/jaxmarl.git")
        
        super().__init__(num_agents=n_agents)
        
        self.local_ratio = local_ratio
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.step_count = 0
        
        # RNG key
        self.key = jr.PRNGKey(seed)
        
        # JaxMARL MPE Simple Spread 환경 생성
        self.env = jaxmarl_make(
            "MPE_simple_spread_v3",
            N=n_agents,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        
        self.state = None
        self.agents_list = self.env.agents  # ["agent_0", "agent_1", ...]
        
        # 실제 생성된 에이전트 수로 업데이트
        actual_num_agents = len(self.agents_list)
        if actual_num_agents != n_agents:
            print(f"Warning: Requested {n_agents} agents, but MPE environment created {actual_num_agents} agents")
            # 부모 클래스의 num_agents와 agents 업데이트
            self.num_agents = actual_num_agents
            self.agents = list(range(actual_num_agents))
        
        # Observation & action dimensions
        sample_obs = self.env.observation_space(self.agents_list[0])
        self.obs_dim = sample_obs.shape[0]
        self.action_dim = self.env.action_space(self.agents_list[0]).n
    
    def reset(self) -> Dict[int, np.ndarray]:
        """환경을 리셋하고 초기 관찰을 반환합니다."""
        self.step_count = 0
        self.key, sub = jr.split(self.key)
        obs_dict, self.state = self.env.reset(sub)
        
        # JaxMARL 형식을 PCI-MAPPO 형식으로 변환
        observations = {}
        for i, agent_name in enumerate(self.agents_list):
            observations[i] = np.asarray(obs_dict[agent_name], dtype=np.float32)
        
        return observations
    
    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[str, Any]
    ]:
        """환경에서 한 스텝을 진행합니다."""
        self.step_count += 1
        
        # PCI-MAPPO 형식을 JaxMARL 형식으로 변환
        self.key, sub = jr.split(self.key)
        act_dict = {}
        for i, action in actions.items():
            if i < len(self.agents_list):
                agent_name = self.agents_list[i]
                act_dict[agent_name] = int(action)
            else:
                print(f"Warning: Skipping action for agent {i}, only {len(self.agents_list)} agents available")
        
        # 환경 스텝 실행
        obs_dict, self.state, rew_dict, done_dict, info = self.env.step(
            sub, self.state, act_dict
        )
        
        # JaxMARL 형식을 PCI-MAPPO 형식으로 변환
        observations = {}
        rewards = {}
        dones = {}
        
        for i, agent_name in enumerate(self.agents_list):
            observations[i] = np.asarray(obs_dict[agent_name], dtype=np.float32)
            rewards[i] = float(rew_dict[agent_name])
            dones[i] = bool(done_dict.get(agent_name, False))
        
        # 전체 에피소드 종료 확인
        episode_done = bool(done_dict.get("__all__", False)) or self.step_count >= self.max_cycles
        for agent_id in self.agents:
            dones[agent_id] = episode_done
        
        # Info 정보
        info_dict = {
            'step_count': self.step_count,
            'max_cycles': self.max_cycles,
            'episode_done': episode_done,
            'raw_info': info
        }
        
        return observations, rewards, dones, info_dict
    
    def get_obs_dim(self) -> int:
        """관찰 차원을 반환합니다."""
        return self.obs_dim
    
    def get_action_dim(self) -> int:
        """액션 차원을 반환합니다."""
        return self.action_dim
    
    def render(self, mode='human'):
        """환경 상태를 렌더링합니다."""
        if mode == 'human':
            print(f"MPE Simple Spread Environment:")
            print(f"Step: {self.step_count}/{self.max_cycles}")
            print(f"Agents: {self.num_agents}")
            print(f"Local ratio: {self.local_ratio}")
            print(f"Continuous actions: {self.continuous_actions}")
        return None

class SimpleStagHuntEnv(MultiAgentEnv):
    """
    Simple Stag Hunt 환경을 PCI-MAPPO의 MultiAgentEnv 인터페이스에 맞게 구현합니다.
    2x2 Stag Hunt 게임으로, 게임 이론에서 일반적으로 설명되는 형태입니다.
    
    게임 규칙:
    - 두 에이전트가 모두 협력(COOPERATE)하면 각각 cooperation_reward 획득
    - 한 에이전트가 협력하고 다른 에이전트가 배신(DEFECT)하면 협력한 에이전트는 failed_cooperation_punishment
    - 배신한 에이전트는 defect_alone_reward 획득
    - 두 에이전트가 모두 배신하면 각각 defect_together_reward 획득
    
    참고: https://github.com/NullDefault/Gym-Stag-Hunt/blob/master/gym_stag_hunt/envs/gym/simple.py
    """
    
    # Action constants
    COOPERATE = 0  # 협력 (Stag 사냥)
    DEFECT = 1     # 배신 (Hare 사냥)
    
    def __init__(
        self,
        cooperation_reward: int = 5,
        defect_alone_reward: int = 3,
        defect_together_reward: int = 1,
        failed_cooperation_punishment: int = 0,
        eps_per_game: int = 1,
        seed: int = None,
        **kwargs
    ):
        """
        Simple Stag Hunt 환경 초기화
        
        Args:
            cooperation_reward: 두 에이전트가 모두 협력할 때의 보상
            defect_alone_reward: 배신했을 때 다른 에이전트가 협력한 경우의 보상
            defect_together_reward: 두 에이전트가 모두 배신했을 때의 보상
            failed_cooperation_punishment: 협력했는데 다른 에이전트가 배신한 경우의 페널티
            eps_per_game: 게임당 에피소드 수
        """
        # 에이전트 수는 항상 2
        super().__init__(num_agents=2, seed=seed)
        
        # 보상 설정
        self.cooperation_reward = cooperation_reward
        self.defect_alone_reward = defect_alone_reward
        self.defect_together_reward = defect_together_reward
        self.failed_cooperation_punishment = failed_cooperation_punishment
        self.eps_per_game = eps_per_game
        
        # 에피소드 카운터
        self.episode_count = 0
        
        # Action space: 0=COOPERATE, 1=DEFECT
        self.action_dim = 2
        
        # Observation space: 마지막 액션들 (2차원: [agent0_last_action, agent1_last_action])
        self.obs_dim = 2
        
        # 마지막 액션 기록
        self.last_actions = {0: 0, 1: 0}  # 초기값은 모두 COOPERATE
        
        # 에피소드 종료 플래그
        self.done = False
        
        self.reset()
    
    def reset(self) -> Dict[int, np.ndarray]:
        """환경을 리셋하고 초기 관찰을 반환합니다."""
        self.episode_count += 1
        
        # 에피소드 종료 조건 확인
        if self.episode_count > self.eps_per_game:
            self.done = True
        else:
            self.done = False
        
        # 마지막 액션을 초기값으로 설정 (모두 COOPERATE)
        self.last_actions = {0: 0, 1: 0}
        
        # 초기 관찰 반환 (마지막 액션들)
        observations = {}
        for agent_id in self.agents:
            observations[agent_id] = np.array([
                self.last_actions[0],  # agent 0의 마지막 액션
                self.last_actions[1]   # agent 1의 마지막 액션
            ], dtype=np.float32)
        
        return observations
    
    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[str, Any]
    ]:
        """환경에서 한 스텝을 진행합니다."""
        # 액션 추출
        action_0 = actions[0]
        action_1 = actions[1]
        
        # 보상 계산
        rewards = self._compute_rewards(action_0, action_1)
        
        # 마지막 액션 업데이트
        self.last_actions[0] = action_0
        self.last_actions[1] = action_1
        
        # 관찰 생성 (업데이트된 마지막 액션들)
        observations = {}
        for agent_id in self.agents:
            observations[agent_id] = np.array([
                self.last_actions[0],  # agent 0의 마지막 액션
                self.last_actions[1]   # agent 1의 마지막 액션
            ], dtype=np.float32)
        
        # 에피소드 종료 조건
        episode_done = self.done or self.episode_count >= self.eps_per_game
        dones = {agent_id: episode_done for agent_id in self.agents}
        
        # Info
        info = {
            'episode_count': self.episode_count,
            'eps_per_game': self.eps_per_game,
            'actions': {0: action_0, 1: action_1},
            'last_actions': self.last_actions.copy(),
            'done': episode_done
        }
        
        return observations, rewards, dones, info
    
    def _compute_rewards(self, action_0: int, action_1: int) -> Dict[int, float]:
        """액션에 따른 보상을 계산합니다."""
        rewards = {}
        
        if action_0 == self.COOPERATE and action_1 == self.COOPERATE:
            # 둘 다 협력: cooperation_reward
            rewards[0] = float(self.cooperation_reward)
            rewards[1] = float(self.cooperation_reward)
        elif action_0 == self.COOPERATE and action_1 == self.DEFECT:
            # agent 0 협력, agent 1 배신
            rewards[0] = float(self.failed_cooperation_punishment)
            rewards[1] = float(self.defect_alone_reward)
        elif action_0 == self.DEFECT and action_1 == self.COOPERATE:
            # agent 0 배신, agent 1 협력
            rewards[0] = float(self.defect_alone_reward)
            rewards[1] = float(self.failed_cooperation_punishment)
        else:  # action_0 == DEFECT and action_1 == DEFECT
            # 둘 다 배신: defect_together_reward
            rewards[0] = float(self.defect_together_reward)
            rewards[1] = float(self.defect_together_reward)
        
        return rewards
    
    def get_obs_dim(self) -> int:
        """관찰 차원을 반환합니다."""
        return self.obs_dim
    
    def get_action_dim(self) -> int:
        """액션 차원을 반환합니다."""
        return self.action_dim
    
    def render(self, mode='human'):
        """환경 상태를 렌더링합니다."""
        if mode == 'human':
            action_names = {0: 'C', 1: 'D'}  # C=Cooperate, D=Defect
            
            print("Simple Stag Hunt Environment:")
            print(f"Episode: {self.episode_count}/{self.eps_per_game}")
            print(f"Last actions: Agent 0={action_names[self.last_actions[0]]}, Agent 1={action_names[self.last_actions[1]]}")
            print()
            
            # 게임 매트릭스 표시
            print("      B   ")
            print("    C   D ")
            print("   ╔══╦══╗")
            print(f" C ║AB║  ║")
            print("   ║  ║  ║")
            print("A  ╠══╬══╣")
            print("   ║  ║  ║")
            print(f" D ║  ║  ║")
            print("   ╚══╩══╝")
            print()
            
            # 현재 상태 표시
            current_action_0 = action_names[self.last_actions[0]]
            current_action_1 = action_names[self.last_actions[1]]
            print(f"Current state: A={current_action_0}, B={current_action_1}")
            
            # 보상 정보 표시
            if self.last_actions[0] == self.COOPERATE and self.last_actions[1] == self.COOPERATE:
                print(f"Both cooperating: Reward = {self.cooperation_reward} each")
            elif self.last_actions[0] == self.COOPERATE and self.last_actions[1] == self.DEFECT:
                print(f"A cooperates, B defects: A={self.failed_cooperation_punishment}, B={self.defect_alone_reward}")
            elif self.last_actions[0] == self.DEFECT and self.last_actions[1] == self.COOPERATE:
                print(f"A defects, B cooperates: A={self.defect_alone_reward}, B={self.failed_cooperation_punishment}")
            else:
                print(f"Both defecting: Reward = {self.defect_together_reward} each")
            print()
        
        return None
        
def dict_to_tensor(obs_dict: Dict[int, np.ndarray], device: str = "cpu") -> Dict[int, torch.Tensor]:
    """Convert observation dictionary to tensor dictionary."""
    return {agent_id: torch.tensor(obs, dtype=torch.float32, device=device) 
            for agent_id, obs in obs_dict.items()}