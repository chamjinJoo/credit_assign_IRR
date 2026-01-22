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

class StagHuntEnv(MultiAgentEnv):
    """
    Reference: https://github.com/NullDefault/Gym-Stag-Hunt/tree/master
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
        coop_bonus: float = 0.0,
        required_agents: Dict[int, List[int]] = None,
        seed: int = None,
        grid_observation: bool = False,
        vision: int = 2,
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
            mauling_punishment: 혼자 Stag를 사냥하려 할 때 페널티
            grid_observation: True면 기존 grid observation 사용, False면 좌표 기반 observation 사용
            vision: 에이전트의 시야 범위 
        """
        # 에이전트 수를 유연하게 설정
        super().__init__(num_agents=num_agents, seed=seed)

        self.grid_observation = grid_observation
        self.vision = vision

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
        
        # Observation space: coordinate(2) + 5x5 grid
        if self.grid_observation:
            # self.obs_dim = 2 + (2 * self.vision + 1)**2 * 3 # bitmask obs
            self.obs_dim = 2 + (2 * self.vision + 1)**2 * (2*num_agents + 1) # original grid obs
        else:
            # num_agents + num_stags + num_hares. All have (x, y) coordinates
            self.obs_dim = (num_agents + num_agents + self.forage_quantity) * 2
        
        # Environment state
        self.agent_positions = {}  # {agent_id: (x, y)}
        self.stag_positions = {}  # {stag_id: (x, y)} - stag 1과 stag 2
        self.stag_alive = {}  # {stag_id: bool} - stag 생존 상태 추적
        self.hare_positions = {}  # {hare_id: (x, y)} - hares for individual hunting
        self.hare_alive = {}  # {hare_id: bool} - hare 생존 상태 추적
              
        # Rewards
        self.forage_reward = forage_reward
        self.stag_reward = stag_reward
        self.mauling_punishment = mauling_punishment
        self.coop_bonus = coop_bonus

        # Allocate agents to each stag: stag_i -> (agent i, agent i+1), 마지막 stag는 (agent N, agent 1)
        if required_agents is not None:
            self.required_agents = required_agents
        else:
            self.required_agents = {}
            for stag_id in range(num_agents):
                if stag_id == num_agents - 1:  # 마지막 stag
                    self.required_agents[stag_id] = [stag_id, 0]
                else:
                    self.required_agents[stag_id] = [stag_id, stag_id + 1]
    
    def print_env_setting(self):
        print(f"mauling_punishment: {self.mauling_punishment}, coop_bonus: {self.coop_bonus}")
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
        self.stag_positions = {}
        self.stag_alive = {}
   
        for i, agent_id in enumerate(self.agents):
            self.agent_positions[agent_id] = self._spawn_object_randomly() # random spawn
        
        stag_positions = self._spawn_stags_with_separation(self.num_agents)
        for i, stag_id in enumerate(self.agents):  # num_agents만큼의 stag 생성
            self.stag_positions[stag_id] = stag_positions[i]
            self.stag_alive[stag_id] = True  # 모든 stag를 살아있는 상태로 초기화
    
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
        stag i가 agent i에게 stag_reward 제공, 협력에 참여한 agent에게는 coop_bonus 제공
        
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
                
                # 1. 여러 에이전트가 같은 hare를 사냥할 때 무작위로 하나 선택
                if self._np_random is not None:
                    selected_agent = agents_at_hare[self._np_random.randint(len(agents_at_hare))]
                else:
                    selected_agent = agents_at_hare[np.random.randint(len(agents_at_hare))]
                rewards[selected_agent] += self.forage_reward
                agent_foraged[selected_agent] = True
                hunting_info['hare_hunts'].append(selected_agent)

                # # 2. 모든 agent에게 보상 분배
                # for agent_id in agents_at_hare:
                #     rewards[agent_id] += self.forage_reward
                #     agent_foraged[agent_id] = True
                #     hunting_info['hare_hunts'].append(agent_id)

                self.hare_alive[hare_id] = False  # hare를 수확된 상태로 표시
        
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
                # 협력 사냥 성공 - stag i는 agent i에게 stag_reward 제공, 협력에 참여한 agent에게는 coop_bonus 제공
                rewards[stag_id] += self.stag_reward
                for agent_id in agents_at_stag:
                    if agent_id != stag_id:
                        rewards[agent_id] += self.coop_bonus

                # 협력 사냥 정보 기록
                hunting_info['stag_hunts'][stag_id] = agents_at_stag.copy()
                # 사냥된 stag를 죽은 상태로 표시 
                self.stag_alive[stag_id] = False
            
            # Stag가 에이전트를 공격한 후 도망가는 경우
            if self.run_away_after_maul and len(agents_at_stag) > 0:
                self.stag_alive[stag_id] = False
        
        return hunting_info
    
        return observations

    '''
    def _get_observations_from_grid(self) -> Dict[int, np.ndarray]:
        """
        비트마스크 인코딩을 사용하여 차원을 압축한 그리드 기반 관찰값 생성.
        에이전트 수(N)에 관계없이 각 셀은 [Agent_Bitmask, Stag_ID, Hare_Presence] 3개 채널로 고정됨.
        """
        observations = {}
        # N에 따른 최대 비트마스크 값 (정규화용: 2^N)
        # 예: N=3이면 8, N=4이면 16
        max_bit_val = 2 ** self.num_agents
        
        for agent_id in self.agents:
            agent_x, agent_y = self.agent_positions[agent_id]
            
            # 1. 자신의 상대적 좌표 (0~1 정규화)
            coord_obs = [
                agent_x / (self.grid_size_x - 1), 
                agent_y / (self.grid_size_y - 1)
            ]
            
            grid_obs = []
            # 2. 시야 범위 내의 그리드 정보 수집
            for dy in range(-self.vision, self.vision + 1):
                for dx in range(-self.vision, self.vision + 1):
                    check_x, check_y = agent_x + dx, agent_y + dy
                    check_pos = (check_x, check_y)
                    
                    # 채널 구성: [Agent_Bitmask, Stag_ID, Hare_Presence] -> 3차원 고정
                    cell_info = [0.0, 0.0, 0.0]
                    
                    # 그리드 경계 체크
                    if not (0 <= check_x < self.grid_size_x and 0 <= check_y < self.grid_size_y):
                        # 시야 밖은 -1.0으로 마스킹
                        cell_info = [-1.0, -1.0, -1.0]
                    else:
                        # A. 에이전트 조합 비트마스크 인코딩
                        bitmask = 0
                        for other_id, pos in self.agent_positions.items():
                            if pos == check_pos:
                                # 각 에이전트 ID를 비트 자릿수로 변환 (2^0, 2^1, ...)
                                bitmask += (2 ** other_id)
                        cell_info[0] = bitmask / max_bit_val
                        
                        # B. Stag 존재 여부 및 ID (살아있는 Stag만)
                        for s_id, s_pos in self.stag_positions.items():
                            if s_pos == check_pos and self.stag_alive.get(s_id, False):
                                # ID 식별을 위해 (ID+1)/N으로 정규화
                                cell_info[1] = (s_id + 1) / self.num_agents
                                break
                                
                        # C. Hare 존재 여부 (살아있는 Hare만)
                        for h_id, h_pos in self.hare_positions.items():
                            if h_pos == check_pos and self.hare_alive.get(h_id, False):
                                cell_info[2] = 1.0
                                break
                                
                    grid_obs.extend(cell_info)
            # observation: self-coordinate + vision-range grid information
            full_obs = coord_obs + grid_obs 
            observations[agent_id] = np.array(full_obs, dtype=np.float32)
            
        return observations
    '''
    def _get_observations_from_grid(self) -> Dict[int, np.ndarray]:
        """기존 그리드 기반 관찰"""
        observations = {}
        
        for agent_id in self.agents:
            agent_x, agent_y = self.agent_positions[agent_id]
            
            coord_obs = [agent_x/(self.grid_size_x-1), agent_y/(self.grid_size_y-1)] # 좌표 정보 (2차원)
            grid_obs = [] 
            
            # collect grid information over the vision range around an agent
            for dy in range(-self.vision, self.vision + 1):
                for dx in range(-self.vision, self.vision + 1):
                    check_x, check_y = agent_x + dx, agent_y + dy
                    check_pos = (check_x, check_y)
                    
                    # n+num_agents+1차원 튜플: (agent0, agent1, ..., agent_n-1, stag0, stag1, ..., stag_n-1, hare)
                    cell_info = [0.0] * (self.num_agents + self.num_agents + 1)
                    
                    # 경계 밖은 -1로 masking
                    if not (0 <= check_x < self.grid_size_x and 0 <= check_y < self.grid_size_y):
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
            
            # 전체 관찰 결합 (2 + 25*(num_agents+2) 차원)
            full_obs = coord_obs + grid_obs
            observations[agent_id] = np.array(full_obs, dtype=np.float32)

            # add stochasticity to obs
            # self.obs_noise = 0.01
            # observations[agent_id] = observations[agent_id] + np.random.normal(0, self.obs_noise, size=observations[agent_id].shape)
            
        return observations
    
    def _get_observations_coordinate(self) -> Dict[int, np.ndarray]:
        """새로운 좌표 기반 관찰
        obs = [Agents, Stags, Hares] locations
        out of 5x5 vision masked as -1
        normalized with full grid size
        """
        observations = {}
        
        # Grid sizes for normalization
        norm_x = self.grid_size_x - 1
        norm_y = self.grid_size_y - 1
        
        # Vision range check (5x5 centered on agent => dx, dy in [-2, 2])
        vision_range = 2

        for agent_id in self.agents:
            agent_x, agent_y = self.agent_positions[agent_id]
            obs_list = []

            # 1. Agents - Self first, then others by index
            # Self
            if agent_id in self.agent_positions:
                ax, ay = self.agent_positions[agent_id]
                obs_list.extend([ax / norm_x, ay / norm_y]) # Always visible to self
            else:
                obs_list.extend([-1.0, -1.0])

            # Others
            for target_id in self.agents:
                if target_id == agent_id:
                    continue
                    
                target_pos = self.agent_positions.get(target_id)
                if target_pos:
                    tx, ty = target_pos
                    # Check vision
                    if abs(tx - agent_x) <= vision_range and abs(ty - agent_y) <= vision_range:
                        obs_list.extend([tx / norm_x, ty / norm_y])
                    else:
                        obs_list.extend([-1.0, -1.0])
                else:
                    obs_list.extend([-1.0, -1.0])

            # 2. Stags
            # self.stag_positions uses agent_id as key (0 to num_agents-1)
            # We should iterate in a fixed order. keys are 0..num_agents-1
            for stag_id in range(self.num_agents): # stag keys are same as agents
                if self.stag_alive.get(stag_id, False):
                    sx, sy = self.stag_positions[stag_id]
                    # Check vision
                    if abs(sx - agent_x) <= vision_range and abs(sy - agent_y) <= vision_range:
                        obs_list.extend([sx / norm_x, sy / norm_y])
                    else:
                        obs_list.extend([-1.0, -1.0])
                else:
                    # Dead stag
                    obs_list.extend([-1.0, -1.0])

            # 3. Hares
            for hare_id in range(self.forage_quantity):
                if self.hare_alive.get(hare_id, False):
                    hx, hy = self.hare_positions[hare_id]
                    # Check vision
                    if abs(hx - agent_x) <= vision_range and abs(hy - agent_y) <= vision_range:
                        obs_list.extend([hx / norm_x, hy / norm_y])
                    else:
                        obs_list.extend([-1.0, -1.0])
                else:
                    # Dead hare
                    obs_list.extend([-1.0, -1.0])
            
            observations[agent_id] = np.array(obs_list, dtype=np.float32)
            
        return observations

    def _get_observations(self) -> Dict[int, np.ndarray]:
        if self.grid_observation:
            return self._get_observations_from_grid()
        else:
            return self._get_observations_coordinate()
    
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
                
                # Agent의 5x5 시야 영역 그리기 (연한 색상)
                vision_alpha = 0.2
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        vision_x, vision_y = x + dx, y + dy
                        # 그리드 범위 내에 있는지 확인
                        if (0 <= vision_x < self.grid_size_x and 
                            0 <= vision_y < self.grid_size_y):
                            # 5x5 시야 영역을 연한 색상으로 표시
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
            plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.2, label='Agent Vision (5x5)')
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
                
                # Agent의 5x5 시야 영역 그리기 (연한 색상)
                vision_alpha = 0.2
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
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

class LBForagingWrapper(MultiAgentEnv):
    """
    LBForaging environment wrapper
    """
    def __init__(
        self,
        seed: int = None,
        field_size: int = 7,
        num_agents: int = 2,
        max_food: int = 2,
        sight: int = 2,
        max_episode_steps: int = 50,
        force_coop: bool = False,
        player_level: Tuple[int, int] = (1, 1),
        food_level: Tuple[int, int] = (1, 3),
    ):
        super().__init__(num_agents=num_agents, seed=seed)
        
        self.field_size = field_size
        self.max_food = max_food
        self.sight = sight
        self.player_level = player_level
        self.food_level = food_level
        self.max_episode_steps = max_episode_steps
        self.force_coop = force_coop
        
        # Gymnasium 환경 생성
        from gymnasium.envs.registration import register
        env_id = f"Foraging-{field_size}x{field_size}-{num_agents}p-{max_food}f"
        if force_coop:
            env_id += "-coop"
        env_id += "-v3"
        self.env_id = env_id
        try:
            env_registered = env_id in gym.envs.registry
        except:
            env_registered = False

        self.kwargs = {"players": num_agents,
                    "min_player_level": player_level[0],
                    "max_player_level": player_level[1],
                    "min_food_level": food_level[0],
                    "max_food_level": food_level[1],
                    "field_size": (field_size, field_size),
                    "max_num_food": max_food,
                    "sight": sight,
                    "max_episode_steps": max_episode_steps,
                    "force_coop": force_coop,
                    "grid_observation": True
        }
        # 환경이 이미 등록되어 있으면 unregister하고 다시 등록
        if env_registered:
            try:
                from gymnasium.envs.registration import registry
                del registry[env_id]
            except:
                pass
        
        # 항상 새로 등록하여 올바른 설정으로 환경 생성
        register(
            id=env_id,
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs=self.kwargs,
        )
        self.env = gym.make(env_id)
        self.env_id = env_id
        
        # print(f"LBF env: {self.env_id}")
        
        self.agent_names = [f"agent_{i}" for i in range(self.num_agents)]
        self.agent_mapping = {i: f"agent_{i}" for i in range(self.num_agents)}
        self.reverse_mapping = {f"agent_{i}": i for i in range(self.num_agents)}

        # Observation 및 action 차원 확인
        # Gymnasium의 reset 표준에 따라 seed를 전달
        obs_array, _ = self.env.reset(seed=self._seed) 
        # LBForaging의 관찰은 (N, obs_dim) 배열이므로, obs_dim은 두 번째 차원입니다.
        # grid_observation인 경우 shape의 전체 크기를 사용
        if self.kwargs.get("grid_observation", False):
            self.obs_dim = int(np.prod(obs_array[0].shape))
        else:
            self.obs_dim = len(obs_array[0]) 
        if hasattr(self.env.action_space, '__getitem__'):
            self.action_dim = self.env.action_space[0].n
        else:
            self.action_dim = self.env.action_space.n

    def reset(self) -> Dict[int, np.ndarray]:
        """환경을 리셋하고 초기 관찰을 반환합니다."""
        obs_array, _ = self.env.reset() # (observation_array, info_dict)
        # numpy 배열을 Dict[agent_id: np.ndarray] 형식으로 변환
        # grid_observation인 경우 3차원 텐서를 1차원 벡터로 flatten
        observations = {i: obs_array[i].astype(np.float32).flatten() for i in range(self.num_agents)}
        return observations

    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[str, Any]
    ]:
        """환경에서 한 스텝을 진행합니다."""
        # 1. PCI-MAPPO Dict[int: int] 액션을 LBForaging의 np.ndarray[int] 형식으로 변환
        # LBForaging은 (N,) 형태의 np.int64 배열을 기대합니다.
        gym_actions = np.array([actions[i] for i in range(self.num_agents)], dtype=np.int64)
        
        # 2. 환경 스텝 실행
        # Gymnasium의 step 표준: (obs_array, reward_array, terminated, truncated, info)
        step_result = self.env.step(gym_actions)
        obs_array, reward_array, terminated, truncated, info = step_result
        
        # 3. LBForaging의 배열 결과를 PCI-MAPPO의 Dict[int: value] 형식으로 변환
        observations = {}
        rewards = {}
        dones = {}
        
        for i in range(self.num_agents):
            # 관찰: (obs_dim,) 형태로 변환
            observations[i] = obs_array[i].astype(np.float32).flatten()
            # reward: each agent's maximum reward in one episode is 1.0
            rewards[i] = float(reward_array[i])
        # 4. Done 처리 (terminated와 truncated를 합쳐서 episode_done 계산)
        episode_done = bool(terminated) or bool(truncated)
        
        # 모든 에이전트에 동일하게 적용
        for agent_id in self.agents:
            dones[agent_id] = episode_done

        # Info 정보
        info_dict = {
            # 'step_count': self.env._elapsed_steps,
            'max_cycles': self.max_episode_steps,
            'episode_done': episode_done,
            'raw_info': info
        }
        # for debugging
        # self.env.render()
        return observations, rewards, dones, info_dict
    
    # get_obs_dim, get_action_dim, render, close는 기존 로직을 유지합니다.
    def get_obs_dim(self) -> int:
        """관찰 차원을 반환합니다."""
        return self.obs_dim
    
    def get_action_dim(self) -> int:
        """액션 차원을 반환합니다."""
        return self.action_dim
    
    def render(self, mode='human'):
        """환경 상태를 렌더링합니다."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode)
        return None
    
    def close(self):
        """환경을 닫습니다."""
        if hasattr(self.env, 'close'):
            self.env.close()

class SMACWrapper(MultiAgentEnv):
    """
    Wrapper for StarCraft Multi-Agent Challenge (SMAC) environment.
    """
    def __init__(self, map_name: str = "3m", seed: int = None, **kwargs):
        try:
            from smac.env import StarCraft2Env
        except ImportError:
            raise ImportError("SMAC is not installed. Please install it using `pip install git+https://github.com/oxwhirl/smac.git`")
            
        self._env = StarCraft2Env(map_name=map_name, seed=seed, **kwargs)
        self._env_info = self._env.get_env_info()
        # env_info: {'state_shape', 'obs_shape', 'n_actions', 'n_agents', 'episode_limit', 
        #               'agent_features': ['health', 'energy/cooldown', 'rel_x', 'rel_y'], 
        #               'enemy_features': ['health', 'rel_x', 'rel_y']}

        self.num_agents = self._env_info["n_agents"]
        self.agents = list(range(self.num_agents))
        self.obs_dim = self._env_info["obs_shape"]
        self.action_dim = self._env_info["n_actions"]

        super().__init__(num_agents=self.num_agents, seed=seed)
        
    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment and return initial observations."""
        self._env.reset()
        obs_list = self._env.get_obs()
        return {i: np.array(obs, dtype=np.float32) for i, obs in enumerate(obs_list)}
        
    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[str, Any]
    ]:
        """Take actions and return next observations, rewards, dones, info."""
        # Convert dict actions to list for SMAC
        action_list = [actions[i] for i in range(self.num_agents)]
        
        reward, terminated, info = self._env.step(action_list)
        obs_list = self._env.get_obs()
        observations = {i: np.array(obs, dtype=np.float32) for i, obs in enumerate(obs_list)}
        
        # SMAC returns a single scalar reward (shared reward)
        rewards = {i: float(reward) for i in self.agents}
        
        # SMAC returns a single terminated flag
        dones = {i: terminated for i in self.agents}
        
        # Add battle_won to info if available (SMAC specific)
        if "battle_won" in info:
            info["battle_won"] = info["battle_won"]
            
        return observations, rewards, dones, info
    
    def get_avail_actions(self) -> List[List[int]]:
        avail_actions = []
        for i in range(self.num_agents):
            avail_actions.append(self._env.get_avail_agent_actions(i))
        return avail_actions

    def get_obs_dim(self) -> int:
        return self.obs_dim
        
    def get_action_dim(self) -> int:
        return self.action_dim

    def close(self):
        self._env.close()

class SMACv2Wrapper(MultiAgentEnv):
    """
    Wrapper for StarCraft Multi-Agent Challenge (SMAC) environment.
    To address the lack of stochasticity in SMAC.
    Actually, some SMAC task can be completed using open-loop policy which is conditioned only on
    the timestep and agent id, without observations (from SMACv2 paper). 
        "One limitation of SMAC is the constant starting positions and types of units, 
        allowing methods to memorise action sequences for solving individual scenarios."
    """
    def __init__(self, map_name: str = "10gen_protoss", seed: int = None, **kwargs):
        try:
            from smacv2.env import StarCraftCapabilityEnvWrapper # for SMACv2
        except ImportError:
            raise ImportError("SMACv2 is not installed. Please install it using `pip install git+https://github.com/oxwhirl/smac.git`")
            
        self._env = StarCraftCapabilityEnvWrapper(map_name=map_name, seed=seed, **kwargs)
        self._env_info = self._env.get_env_info()
        # env_info: {'state_shape', 'obs_shape', 'n_actions', 'n_agents', 'episode_limit', 
        #               'agent_features': ['health', 'energy/cooldown', 'rel_x', 'rel_y'], 
        #               'enemy_features': ['health', 'rel_x', 'rel_y']}

        self.num_agents = self._env_info["n_agents"]
        self.agents = list(range(self.num_agents))
        self.obs_dim = self._env_info["obs_shape"]
        self.action_dim = self._env_info["n_actions"]

        super().__init__(num_agents=self.num_agents, seed=seed)
        
    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment and return initial observations."""
        self._env.reset()
        obs_list = self._env.get_obs()
        return {i: np.array(obs, dtype=np.float32) for i, obs in enumerate(obs_list)}
        
    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[str, Any]
    ]:
        """Take actions and return next observations, rewards, dones, info."""
        # Convert dict actions to list for SMAC
        action_list = [actions[i] for i in range(self.num_agents)]
        
        reward, terminated, info = self._env.step(action_list)
        
        obs_list = self._env.get_obs()
        observations = {i: np.array(obs, dtype=np.float32) for i, obs in enumerate(obs_list)}
        
        # SMAC returns a single scalar reward (shared reward)
        rewards = {i: float(reward) for i in self.agents}
        
        # SMAC returns a single terminated flag
        dones = {i: terminated for i in self.agents}
        
        # Add battle_won to info if available (SMAC specific)
        if "battle_won" in info:
            info["battle_won"] = info["battle_won"]
            
        return observations, rewards, dones, info
    
    def get_avail_actions(self) -> List[List[int]]:
        avail_actions = []
        for i in range(self.num_agents):
            avail_actions.append(self._env.get_avail_agent_actions(i))
        return avail_actions

    def get_obs_dim(self) -> int:
        return self.obs_dim
        
    def get_action_dim(self) -> int:
        return self.action_dim

    def close(self):
        self._env.close()

def dict_to_tensor(obs_dict: Dict[int, np.ndarray], device: str = "cpu") -> Dict[int, torch.Tensor]:
    """Convert observation dictionary to tensor dictionary."""
    return {agent_id: torch.tensor(obs, dtype=torch.float32, device=device) 
            for agent_id, obs in obs_dict.items()}