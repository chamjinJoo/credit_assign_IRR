"""
Backup of unused ENVs: MPESimpleSpread, SimpleStagHunt, SimpleCooperative, PettingZoo, DecTiger
"""

import numpy as np
import torch
import random
import math
from typing import Dict, Tuple, List, Any, Optional
from abc import ABC, abstractmethod
import lbforaging
import gymnasium as gym

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
MATPLOTLIB_AVAILABLE = True

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

def dict_to_tensor(obs_dict: Dict[int, np.ndarray], device: str = "cpu") -> Dict[int, torch.Tensor]:
    """Convert observation dictionary to tensor dictionary."""
    return {agent_id: torch.tensor(obs, dtype=torch.float32, device=device) 
            for agent_id, obs in obs_dict.items()}