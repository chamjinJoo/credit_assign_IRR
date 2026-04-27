import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from abc import ABC, abstractmethod

# import lbforaging
import gymnasium as gym
import matplotlib.pyplot as plt
import copy
import torch
from envs.smax_target_env import swap_to_target_aware_inner_smax
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
        num_agents: int = 3,
        max_steps: int = 50,
        stag_follows: bool = False,
        run_away_after_maul: bool = False,
        forage_quantity: int = None,
        global_reward: bool = False,
        forage_reward: float = 3.0,
        stag_reward: float = 5.0,
        mauling_punishment: float = 0.0,
        coop_bonus: float = 0.0,
        required_agents: Dict[int, List[int]] = None,
        seed: int = None,
        grid_observation: bool = False,
        vision: int = 2,
        obs_noise_std: float = 0.0,
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
            # self.obs_dim = (2 * self.vision + 1)**2 * 3 # bitmask obs
            self.obs_dim = (2 * self.vision + 1)**2 * (2*num_agents + 1) 
        else:
            # num_agents + num_stags + num_hares. All have (x, y) coordinates
            self.obs_dim = (num_agents + num_agents + self.forage_quantity) * 3
            
        # add noise to observation
        self.obs_noise_std = obs_noise_std

        # Environment state
        self.agent_positions = {}  # {agent_id: (x, y)}
        self.stag_positions = {}  # {stag_id: (x, y)} - stag 1과 stag 2
        self.stag_alive = {}  # {stag_id: bool} - stag 생존 상태 추적
        self.hare_positions = {}  # {hare_id: (x, y)} - hares for individual hunting
        self.hare_alive = {}  # {hare_id: bool} - hare 생존 상태 추적
              
        # Rewards
        self.global_reward = global_reward
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
            
    def get_env_state(self):
        """현재 환경의 내부 상태(에이전트 위치, 사슴/토끼 위치, 스텝 수 등)를 깊은 복사하여 반환합니다."""
        # 만약 numpy array나 기본 자료형으로만 이루어져 있다면 __dict__ 복사가 가장 빠릅니다.
        return copy.deepcopy(self.__dict__)

    def set_env_state(self, state):
        """저장된 상태로 환경을 강제 복원(Reset 없이 되돌리기)합니다."""
        self.__dict__.update(copy.deepcopy(state))

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
        
        if self.global_reward: # all agents share the same reward
            g_reward = sum(rewards.values())
            for agent_id in self.agents:
                rewards[agent_id] = g_reward

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
    
    # bitmask observation
    def _get_bitmask_grid_observations(self) -> Dict[int, np.ndarray]:
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
       
            grid_obs = []
            for dy in range(-self.vision, self.vision + 1):
                for dx in range(-self.vision, self.vision + 1):
                    check_x, check_y = agent_x + dx, agent_y + dy
                    check_pos = (check_x, check_y)
            
                    cell_info = [0.0, 0.0, 0.0]
                    
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

            observations[agent_id] = np.array(grid_obs, dtype=np.float32)
            
        return observations

    def _get_grid_observations(self) -> Dict[int, np.ndarray]:
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
                    
                    # n+num_agents+1 dimensional tuple: (agent0, agent1, ..., agent_n-1, stag0, stag1, ..., stag_n-1, hare)
                    cell_info = [0.0] * (self.num_agents + self.num_agents + 1)
                    
                    # out of the grid is mask with -1
                    if not (0 <= check_x < self.grid_size_x and 0 <= check_y < self.grid_size_y):
                        cell_info = [-1.0] * (self.num_agents + self.num_agents + 1)
                    else:
                        # check agents (allow overlapping in a cell)
                        for other_agent_id in self.agents:
                            if self.agent_positions[other_agent_id] == check_pos:
                                cell_info[other_agent_id] = 1.0
                        
                        # check alive stags
                        for stag_id, stag_pos in self.stag_positions.items():
                            if stag_pos == check_pos and self.stag_alive.get(stag_id, False):
                                cell_info[self.num_agents + stag_id] = 1.0  # stag_id에 해당하는 인덱스를 1로 설정 
                        
                        # check alive hares
                        for hare_id, hare_pos in self.hare_positions.items():
                            if hare_pos == check_pos and self.hare_alive.get(hare_id, False):
                                cell_info[self.num_agents + self.num_agents] = 1.0
                                break
                    grid_obs.extend(cell_info)
     
            observations[agent_id] = np.array(grid_obs, dtype=np.float32)

        return observations
    
    def _get_observations_coordinate(self) -> Dict[int, np.ndarray]:
        observations = {}
        norm_x = self.grid_size_x - 1
        norm_y = self.grid_size_y - 1
        vision_range = 2

        for agent_id in self.agents:
            agent_x, agent_y = self.agent_positions[agent_id]
            obs_list = []

            # --- helper function: return [x, y, exist_flag] ---
            def get_entity_obs(pos, alive, is_agent=False):
                if pos and (alive or is_agent):
                    tx, ty = pos
                    # 시야 체크
                    if abs(tx - agent_x) <= vision_range and abs(ty - agent_y) <= vision_range:
                        return [tx / norm_x, ty / norm_y, 1.0]
                    else:
                        return [0.0, 0.0, -1.0]
                else:
                    return [0.0, 0.0, 0.0]

            # 1. Agents
            for target_id in self.agents:
                target_pos = self.agent_positions.get(target_id)
                # 자기 자신은 항상 보임
                if target_id == agent_id:
                    obs_list.extend([agent_x / norm_x, agent_y / norm_y, 1.0])
                else:
                    obs_list.extend(get_entity_obs(target_pos, True, is_agent=True))

            # 2. Stags
            for stag_id in range(self.num_agents):
                s_pos = self.stag_positions.get(stag_id)
                s_alive = self.stag_alive.get(stag_id, False)
                obs_list.extend(get_entity_obs(s_pos, s_alive))

            # 3. Hares
            for hare_id in range(self.forage_quantity):
                h_pos = self.hare_positions.get(hare_id)
                h_alive = self.hare_alive.get(hare_id, False)
                obs_list.extend(get_entity_obs(h_pos, h_alive))

            observations[agent_id] = np.array(obs_list, dtype=np.float32)
        return observations

    def _get_observations(self) -> Dict[int, np.ndarray]:
        if self.grid_observation:
            observations = self._get_grid_observations()
            # return self._get_bitmask_grid_observations()
        else:
            observations = self._get_observations_coordinate()
        
        # add noise to observations if std > 0
        if self.obs_noise_std > 0:
            for agent_id in observations:
                observations[agent_id] = observations[agent_id] + np.random.normal(0, self.obs_noise_std, size=observations[agent_id].shape)
                
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
        grid_observation: bool = True,
        obs_noise_std: float = 0.0
    ):
        super().__init__(num_agents=num_agents, seed=seed)
        
        self.field_size = field_size
        self.max_food = max_food
        self.sight = sight
        self.player_level = player_level
        self.food_level = food_level
        self.max_episode_steps = max_episode_steps
        self.force_coop = force_coop
        self.grid_observation = grid_observation
        self.obs_noise_std = obs_noise_std

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
                    "grid_observation": grid_observation
        }
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

        obs_array, _ = self.env.reset(seed=self._seed) 
        # LBForaging observation is array of shape (N, obs_dim)

        if self.kwargs.get("grid_observation", False):
            self.obs_dim = int(np.prod(obs_array[0].shape))
        else:
            self.obs_dim = len(obs_array[0]) 
        if hasattr(self.env.action_space, '__getitem__'):
            self.action_dim = self.env.action_space[0].n
        else:
            self.action_dim = self.env.action_space.n

    def reset(self) -> Dict[int, np.ndarray]:
        obs_array, _ = self.env.reset() # (observation_array, info_dict)
        observations = {i: obs_array[i].astype(np.float32).flatten() for i in range(self.num_agents)}
        
        # add noise to observations if std > 0
        if self.obs_noise_std > 0:
            for agent_id in observations:
                observations[agent_id] = observations[agent_id] + np.random.normal(0, self.obs_noise_std, size=observations[agent_id].shape)
        return observations

    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[str, Any]
    ]:
        gym_actions = np.array([actions[i] for i in range(self.num_agents)], dtype=np.int64)
        
        step_result = self.env.step(gym_actions)
        obs_array, reward_array, terminated, truncated, info = step_result
        
        observations = {}
        rewards = {}
        dones = {}
        
        for i in range(self.num_agents):
            observations[i] = obs_array[i].astype(np.float32).flatten()
            # add noise to observations if std > 0
            if self.obs_noise_std > 0:
                observations[i] = observations[i] + np.random.normal(0, self.obs_noise_std, size=observations[i].shape)
            # reward: each agent's maximum reward in one episode is 1.0
            rewards[i] = float(reward_array[i])
     
        episode_done = bool(terminated) or bool(truncated)
        
        for agent_id in self.agents:
            dones[agent_id] = episode_done

        # Info
        info_dict = {
            # 'step_count': self.env._elapsed_steps,
            'max_cycles': self.max_episode_steps,
            'episode_done': episode_done,
            'raw_info': info
        }
        return observations, rewards, dones, info_dict

    def get_obs_dim(self) -> int:
        return self.obs_dim
    
    def get_action_dim(self) -> int:
        return self.action_dim
    
    def render(self, mode='human'):
        if hasattr(self.env, 'render'):
            return self.env.render(mode)
        return None
    
    def close(self):
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

    def save_replay(self, dir=None):
        if dir is not None:
            # StarCraftCapabilityEnvWrapper 내의 StarCraft2Env 접근
            if hasattr(self._env, 'replay_dir'):
                self._env.replay_dir = dir
                
        if hasattr(self._env, 'save_replay'):
            self._env.save_replay()

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
    def __init__(self, map_name: str = "10gen_protoss", seed: int = None, use_individual_rewards: bool = True, **kwargs):
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
        self.num_enemies = self._infer_num_enemies()
        self.use_individual_rewards = use_individual_rewards

        # Track previous ally health for individual damage calculation
        # (prev enemy health is read directly from sc_env.previous_enemy_units)
        self.prev_enemy_health = None
        self.prev_ally_health = None

        super().__init__(num_agents=self.num_agents, seed=seed)

    def _infer_num_enemies(self) -> Optional[int]:
        """Infer static enemy count for the current SMACv2 scenario."""
        sc_env_inner = self._env.env if hasattr(self._env, 'env') else self._env
        candidates = [
            getattr(sc_env_inner, 'n_enemies', None),
            getattr(sc_env_inner, 'num_enemies', None),
            self._env_info.get("n_enemies"),
            self._env_info.get("num_enemies"),
        ]
        for cand in candidates:
            try:
                if cand is not None and int(cand) > 0:
                    return int(cand)
            except (TypeError, ValueError):
                continue
        return None

    def _get_health_info(self):
        """Get current enemy and ally health from the environment."""
        sc_env = self._env.env
        enemies = sc_env.enemies # dict
        agents = sc_env.agents   # dict

        enemy_health = np.array([e.health + e.shield for e in enemies.values()]) if enemies else np.zeros(0)
        ally_health = np.array([a.health + a.shield for a in agents.values()]) if agents else np.zeros(self.num_agents)
        return enemy_health, ally_health

    def _compute_individual_rewards(self, global_reward, terminated, info):
        """
        Compute individual rewards based on damage dealt by each agent.

        Reward structure:
        1. Damage-based reward: Each agent gets reward proportional to damage they dealt
        2. Win bonus: Shared equally among all agents when battle is won

        Args:
            global_reward: The global reward from SMAC
            terminated: Whether the episode terminated
            info: Info dict from SMAC (may contain battle_won)

        Returns:
            Dict mapping agent_id to individual reward
        """
        # Get current health
        curr_enemy_health, curr_ally_health = self._get_health_info()

        # Compute total damage dealt using SC2Env's own previous_enemy_units.
        # SC2Env saves a deepcopy of enemies at the START of update_units() each step,
        # so previous_enemy_units is guaranteed to reflect the pre-step state regardless
        # of whether enemies died (health set to 0 in-place) or survived.
        sc_env_inner = self._env.env if hasattr(self._env, 'env') else self._env
        if hasattr(sc_env_inner, 'previous_enemy_units') and sc_env_inner.previous_enemy_units:
            total_damage_dealt = sum(
                max(0.0, (prev.health + prev.shield) -
                         (sc_env_inner.enemies[eid].health + sc_env_inner.enemies[eid].shield))
                for eid, prev in sc_env_inner.previous_enemy_units.items()
                if eid in sc_env_inner.enemies
            )
        elif self.prev_enemy_health is not None and len(curr_enemy_health) == len(self.prev_enemy_health):
            total_damage_dealt = np.maximum(0, self.prev_enemy_health - curr_enemy_health).sum()
        else:
            total_damage_dealt = 0.0

        # Try to get per-agent damage statistics from SMAC
        individual_damages = self._get_agent_damage_stats()

        # Compute individual damage-based rewards
        rewards = {}
        if total_damage_dealt > 0 and individual_damages is not None:
            for i in range(self.num_agents):
                total_credited_damage = individual_damages.sum()
                if total_credited_damage > 0:
                    rewards[i] = (individual_damages[i] / total_credited_damage) * global_reward
                else:
                    rewards[i] = 0.0
        else:
            # Fallback: equal distribution if no damage info available
            for i in range(self.num_agents):
                rewards[i] = global_reward

        # Add win bonus (shared equally) when battle is won
        battle_won = info.get("battle_won", False)
        if battle_won and terminated:
            for i in range(self.num_agents):
                rewards[i] = global_reward

        # Update ally health tracking for next step
        self.prev_enemy_health = curr_enemy_health.copy()
        self.prev_ally_health = curr_ally_health.copy()

        return rewards, individual_damages

    def _was_in_attack_range(self, sc_env, agent_id, enemy_id):
        """
        Check if an agent was in attack range of an enemy.

        Args:
            sc_env: StarCraft2 environment
            agent_id: Agent ID
            enemy_id: Enemy ID

        Returns:
            True if agent could attack enemy, False otherwise
        """
        # Get agent and enemy units
        if not hasattr(sc_env, 'agents') or not hasattr(sc_env, 'enemies'):
            return True  # Fallback: assume in range if we can't check

        if agent_id not in sc_env.agents or enemy_id not in sc_env.enemies:
            # Enemy may have been killed this step and removed from the dict.
            # If the agent targeted it and it's now gone, assume the attack landed.
            return True

        agent_unit = sc_env.agents[agent_id]
        enemy_unit = sc_env.enemies[enemy_id]

        # Get agent position
        agent_x = agent_unit.pos.x
        agent_y = agent_unit.pos.y

        # Get enemy position
        enemy_x = enemy_unit.pos.x
        enemy_y = enemy_unit.pos.y

        # Calculate distance
        import math
        distance = math.hypot(enemy_x - agent_x, enemy_y - agent_y)

        # Get attack range
        attack_range = sc_env.unit_shoot_range(agent_id)
    
        # Add a small margin for inherent sampling frequency of SMACv2
        return distance <= (attack_range * 1)

    def _get_agent_damage_stats(self):
        """
        Extract per-agent damage statistics from SMAC environment.

        Uses sc_env.previous_enemy_units (maintained by StarCraft2Env.update_units)
        as the pre-step health source and sc_env.enemies as the post-step source.
        This avoids all timing issues: SC2Env deepcopies enemies at the very start of
        update_units(), so the diff is always exactly one step of damage regardless of
        whether enemies died (health set to 0 in-place) or survived.

        Returns:
            np.array of shape (num_agents,) with damage dealt by each agent,
            or np.ones if stats are unavailable.
        """
        sc_env = self._env.env if hasattr(self._env, 'env') else self._env

        damages = np.zeros(self.num_agents)

        # Require SC2Env's built-in previous-state snapshot
        if not hasattr(sc_env, 'previous_enemy_units') or sc_env.previous_enemy_units is None:
            return np.ones(self.num_agents)
        if not hasattr(sc_env, 'enemies'):
            return np.ones(self.num_agents)

        # Per-enemy damage using SC2Env's own pre/post step snapshots.
        # Dead enemies have health=0 in sc_env.enemies (set in-place by update_units).
        enemy_damage_taken = {}
        for eid, prev_unit in sc_env.previous_enemy_units.items():
            prev_hp = prev_unit.health + prev_unit.shield
            curr_unit = sc_env.enemies.get(eid)
            curr_hp = (curr_unit.health + curr_unit.shield) if curr_unit is not None else 0.0
            dmg = max(0.0, prev_hp - curr_hp)
            if dmg > 0:
                enemy_damage_taken[eid] = dmg

        # Get last actions and n_actions_no_attack from SMAC
        if not hasattr(sc_env, 'last_action') or not hasattr(sc_env, 'n_actions_no_attack'):
            return np.ones(self.num_agents)

        last_actions = sc_env.last_action  # (n_agents, n_actions) one-hot
        n_actions_no_attack = sc_env.n_actions_no_attack

        # Map agents to their attack targets (enemy_id = action_idx - n_actions_no_attack,
        # which matches sc_env.enemies keys since SMAC indexes enemies 0..n_enemies-1)
        attack_targets = {}
        for agent_id in range(self.num_agents):
            if agent_id < len(last_actions):
                action_idx = np.argmax(last_actions[agent_id])
                if action_idx >= n_actions_no_attack:
                    target_id = action_idx - n_actions_no_attack
                    attack_targets[agent_id] = target_id

        # Filter to feasible attacks: target took damage AND agent was in range
        feasible_attacks = {}
        for agent_id, target_id in attack_targets.items():
            if target_id in enemy_damage_taken:
                if self._was_in_attack_range(sc_env, agent_id, target_id):
                    feasible_attacks[agent_id] = target_id

        # Assign damage credit to feasible attackers
        for agent_id, target_id in feasible_attacks.items():
            damages[agent_id] = enemy_damage_taken[target_id]

        # No damage was registered for any agent.
        # Two causes, both meaning no agent actually dealt damage this step:
        #   - Path A: no one chose an attack action (step_mul carryover can produce
        #             global_reward > 0 from the previous step's lingering SC2 commands)
        #   - Path B: attack actions were chosen but get_agent_action() returned None
        #             because can_shoot=False or stochastic_attack failed, so SC2 never
        #             sent the command — last_action recorded the intent, not the execution
        # Either way: no actual damage → no individual reward.
        if damages.sum() == 0:
            return np.zeros(self.num_agents)

        return damages

    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment and return initial observations."""
        self._env.reset()
        obs_list = self._env.get_obs()

        # Initialize health tracking for individual rewards
        if self.use_individual_rewards:
            self.prev_enemy_health, self.prev_ally_health = self._get_health_info()

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

        # === Individual Reward Computation ===
        individual_damages = None
        if self.use_individual_rewards:
            rewards, individual_damages = self._compute_individual_rewards(reward, terminated, info)
        else:
            rewards = {i: float(reward) for i in self.agents}
            sc_env_inner = self._env.env if hasattr(self._env, 'env') else self._env
            has_damage_stats = (
                hasattr(sc_env_inner, 'previous_enemy_units')
                and sc_env_inner.previous_enemy_units is not None
                and hasattr(sc_env_inner, 'last_action')
                and hasattr(sc_env_inner, 'n_actions_no_attack')
            )
            if has_damage_stats:
                individual_damages = self._get_agent_damage_stats()

        # SMAC returns a single terminated flag
        dones = {i: terminated for i in self.agents}
        
        if individual_damages is not None:
            info["individual_damages"] = [float(x) for x in np.asarray(individual_damages).reshape(-1)]

        sc_env_inner = self._env.env if hasattr(self._env, 'env') else self._env
        n_actions_no_attack = getattr(sc_env_inner, 'n_actions_no_attack', None)
        if n_actions_no_attack is not None:
            info["n_actions_no_attack"] = int(n_actions_no_attack)

        n_enemies = self._infer_num_enemies()
        if n_enemies is not None:
            self.num_enemies = int(n_enemies)
            info["n_enemies"] = int(n_enemies)

        attack_threshold = None
        if n_enemies is not None:
            attack_threshold = self.action_dim - int(n_enemies)
        elif n_actions_no_attack is not None:
            attack_threshold = int(n_actions_no_attack)
        if attack_threshold is not None:
            attack_threshold = max(0, min(int(attack_threshold), self.action_dim - 1))
            info["attack_action_threshold"] = int(attack_threshold)
        info["team_reward"] = float(reward)

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

    def save_replay(self, dir=None):
        if dir is not None:
            # StarCraftCapabilityEnvWrapper 내의 StarCraft2Env 접근
            if hasattr(self._env, 'env') and hasattr(self._env.env, 'replay_dir'):
                self._env.env.replay_dir = dir
                
        if hasattr(self._env, 'save_replay'):
            self._env.save_replay()

    def close(self):
        self._env.close()

class MPESimpleSpreadEnv(MultiAgentEnv):
    """
    Wrapper for MPE Simple Spread environment using PettingZoo.
    """
    def __init__(self, n_agents: int = 3, local_ratio: float = 0.5, max_cycles: int = 25, continuous_actions: bool = False, seed: int = None, **kwargs):
        try:
            from pettingzoo.mpe import simple_spread_v3
        except ImportError:
            raise ImportError("PettingZoo MPE is not installed. Please install it using `pip install pettingzoo[mpe]`")
        
        self.num_agents = n_agents
        self.max_cycles = max_cycles
        self.local_ratio = local_ratio
        
        # Initialize the environment
        self._env = simple_spread_v3.parallel_env(
            N=n_agents, 
            local_ratio=local_ratio, 
            max_cycles=max_cycles, 
            continuous_actions=continuous_actions
        )
        self._env.reset(seed=seed)
        
        # agents are 'agent_0', 'agent_1', etc.
        self._agent_names = self._env.agents
        
        self.agents = list(range(self.num_agents))
        self.obs_dim = self._env.observation_space(self._agent_names[0]).shape[0]
        
        # For discrete actions, action_space is Discrete(5)
        self.action_dim = self._env.action_space(self._agent_names[0]).n
        
        super().__init__(num_agents=self.num_agents, seed=seed)
        self.step_count = 0

    def reset(self) -> Dict[int, np.ndarray]:
        obs_dict, info = self._env.reset()
        self.step_count = 0
        return {i: np.array(obs_dict[name], dtype=np.float32) for i, name in enumerate(self._agent_names)}
        
    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[str, Any]
    ]:
        action_dict = {name: actions[i] for i, name in enumerate(self._agent_names)}
        
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self._env.step(action_dict)
        self.step_count += 1
        
        observations = {i: np.array(obs_dict[name], dtype=np.float32) for i, name in enumerate(self._agent_names)}
        rewards = {i: float(reward_dict[name]) for i, name in enumerate(self._agent_names)}
        
        # MPE simple spread uses terminations (done) and truncations (horizon limit)
        dones = {i: bool(terminated_dict[name] or truncated_dict[name]) for i, name in enumerate(self._agent_names)}
        
        info = { 'step_count': self.step_count }
        
        return observations, rewards, dones, info

    def get_state(self) -> np.ndarray:
        # PettingZoo MPE provides a state() method
        return np.array(self._env.state(), dtype=np.float32)

    def get_avail_actions(self) -> List[List[int]]:
        # All actions available at all times for MPE
        return [[1] * self.action_dim for _ in range(self.num_agents)]

    def get_avail_agent_actions(self, agent_id: int) -> List[int]:
        return [1] * self.action_dim

    def get_env_info(self) -> Dict[str, Any]:
        return {
            "state_shape": self.get_state().shape[0],
            "obs_shape": self.obs_dim,
            "n_actions": self.action_dim,
            "n_agents": self.num_agents,
            "episode_limit": self.max_cycles,
        }

    def get_obs_dim(self) -> int:
        return self.obs_dim
        
    def get_action_dim(self) -> int:
        return self.action_dim

    def close(self):
        self._env.close()

class VMASWrapper(MultiAgentEnv):
    """
    Wrapper for VMAS scenarios using VMAS native vectorization when n_envs > 1.
    """

    def __init__(
        self,
        scenario_name: str,
        num_agents: int = 4,
        n_envs: int = 1,
        max_steps: int = 100,
        seed: int = None,
        device: str = "cpu",
        continuous_actions: bool = False,
        **scenario_kwargs,
    ):
        try:
            import vmas
            from gymnasium import spaces
        except ImportError as exc:
            raise ImportError(
                "VMAS is not installed. Install it with `pip install vmas` before "
                "using VMAS benchmarks."
            ) from exc

        if continuous_actions:
            raise ValueError(
                "This repository currently expects discrete-action environments. "
                "Use `continuous_actions: false` for VMAS benchmarks."
            )

        self._spaces = spaces
        self.scenario_name = scenario_name
        self.max_steps = max_steps
        self.device = device
        self.continuous_actions = continuous_actions
        self.scenario_kwargs = dict(scenario_kwargs)
        self.n_envs = n_envs
        self.n_rollout_threads = n_envs

        self._env = vmas.make_env(
            scenario=scenario_name,
            num_envs=n_envs,
            device=device,
            continuous_actions=continuous_actions,
            wrapper=None,
            max_steps=max_steps,
            seed=seed,
            dict_spaces=True,
            terminated_truncated=True,
            n_agents=num_agents,
            **self.scenario_kwargs,
        )

        self._agent_refs = list(self._env.agents)
        self._agent_names = self._infer_agent_names()
        self.num_agents = len(self._agent_names)
        self.agents = list(range(self.num_agents))

        obs_space = self._extract_agent_space(self._env.observation_space, self._agent_names[0])
        if isinstance(obs_space, self._spaces.Dict):
            raise ValueError(
                f"Scenario `{scenario_name}` returned nested dict observations per agent. "
                "This wrapper currently expects flat Box observations."
            )
        self.obs_dim = int(np.prod(obs_space.shape))

        action_space = self._extract_agent_space(self._env.action_space, self._agent_names[0])
        if not hasattr(action_space, "n"):
            raise ValueError(
                f"Scenario `{scenario_name}` did not expose a discrete action space."
            )
        self.action_dim = int(action_space.n)

        super().__init__(num_agents=self.num_agents, seed=seed)
        self.step_count = 0

    @property
    def is_batched(self) -> bool:
        return self.n_envs > 1

    def _extract_agent_space(self, spaces_obj, agent_name: str):
        if isinstance(spaces_obj, self._spaces.Dict):
            return spaces_obj[agent_name]
        if isinstance(spaces_obj, self._spaces.Tuple):
            agent_index = self._agent_names.index(agent_name)
            return spaces_obj[agent_index]
        if hasattr(spaces_obj, "spaces"):
            spaces_attr = spaces_obj.spaces
            if isinstance(spaces_attr, dict):
                return spaces_attr[agent_name]
            if isinstance(spaces_attr, (list, tuple)):
                agent_index = self._agent_names.index(agent_name)
                return spaces_attr[agent_index]
        return spaces_obj

    def _infer_agent_names(self) -> List[str]:
        for spaces_obj in (self._env.observation_space, self._env.action_space):
            if isinstance(spaces_obj, self._spaces.Dict):
                return list(spaces_obj.spaces.keys())
            if hasattr(spaces_obj, "spaces") and isinstance(spaces_obj.spaces, dict):
                return list(spaces_obj.spaces.keys())
        return [f"agent_{idx}" for idx in range(len(self._agent_refs))]

    @staticmethod
    def _to_numpy(value: Any, flatten: bool = False) -> np.ndarray:
        if torch.is_tensor(value):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == 1:
            array = array[0]
        if flatten:
            array = np.asarray(array, dtype=np.float32).reshape(-1)
        return array

    def _extract_single_env_value(self, value: Any, env_idx: int, flatten: bool = False) -> np.ndarray:
        array = self._to_numpy(value, flatten=False)
        if self.is_batched and array.ndim > 0 and array.shape[0] == self.n_envs:
            array = array[env_idx]
        if flatten:
            array = np.asarray(array, dtype=np.float32).reshape(-1)
        return array

    def _split_obs_batch(self, obs_data: Any) -> List[Dict[int, np.ndarray]]:
        if not isinstance(obs_data, dict):
            raise TypeError(f"Unsupported VMAS observation format: {type(obs_data)}")
        return [
            {
                idx: self._extract_single_env_value(obs_data[name], env_idx, flatten=True).astype(np.float32)
                for idx, name in enumerate(self._agent_names)
            }
            for env_idx in range(self.n_envs)
        ]

    def _split_rewards_batch(self, reward_data: Any) -> List[Dict[int, float]]:
        if isinstance(reward_data, dict):
            return [
                {
                    idx: float(np.asarray(self._extract_single_env_value(reward_data[name], env_idx)).reshape(-1)[0])
                    for idx, name in enumerate(self._agent_names)
                }
                for env_idx in range(self.n_envs)
            ]
        raise TypeError(f"Unsupported VMAS reward format for batched env: {type(reward_data)}")

    def _split_done_flags(self, done_data: Any) -> List[bool]:
        if isinstance(done_data, dict):
            first_value = next(iter(done_data.values()))
            array = np.asarray(self._to_numpy(first_value)).reshape(-1)
        else:
            array = np.asarray(self._to_numpy(done_data)).reshape(-1)
        if array.shape[0] == 1 and self.is_batched:
            return [bool(array[0]) for _ in range(self.n_envs)]
        return [bool(array[min(i, array.shape[0] - 1)]) for i in range(self.n_envs)]

    def _split_info_batch(self, info_data: Any, done_flags: List[bool]) -> List[Dict[str, Any]]:
        info_list = []
        for env_idx in range(self.n_envs):
            env_info = {
                "step_count": self.step_count,
                "scenario_name": self.scenario_name,
                "episode_done": bool(done_flags[env_idx]),
            }
            if isinstance(info_data, dict):
                env_raw_info = {}
                for key, value in info_data.items():
                    if isinstance(value, dict):
                        env_raw_info[key] = {
                            sub_key: self._extract_single_env_value(sub_value, env_idx, flatten=False)
                            for sub_key, sub_value in value.items()
                        }
                    else:
                        env_raw_info[key] = self._extract_single_env_value(value, env_idx, flatten=False)
                env_info["raw_info"] = env_raw_info
            else:
                env_info["raw_info"] = info_data
            info_list.append(env_info)
        return info_list

    def _reset_done_envs_in_place(self, obs_data: Dict[str, Any], done_flags: List[bool]):
        if not any(done_flags):
            return
        for env_idx, done in enumerate(done_flags):
            if not done:
                continue
            reset_obs = self._env.reset_at(env_idx)
            for agent_name in self._agent_names:
                current_value = obs_data[agent_name]
                reset_value = reset_obs[agent_name]
                if torch.is_tensor(current_value):
                    if not getattr(current_value, "_base", None) is None:
                        current_value = current_value.clone()
                        obs_data[agent_name] = current_value
                    reset_tensor = torch.as_tensor(
                        self._to_numpy(reset_value, flatten=False),
                        dtype=current_value.dtype,
                        device=current_value.device,
                    )
                    current_value[env_idx] = torch.as_tensor(
                        reset_tensor[env_idx],
                        dtype=current_value.dtype,
                        device=current_value.device,
                    )
                else:
                    current_value = np.asarray(current_value).copy()
                    reset_array = np.asarray(self._to_numpy(reset_value, flatten=False))
                    current_value[env_idx] = reset_array[env_idx]
                    obs_data[agent_name] = current_value

    def _convert_obs(self, obs_data: Any) -> Dict[int, np.ndarray]:
        if isinstance(obs_data, dict):
            return {
                idx: self._to_numpy(obs_data[name], flatten=True).astype(np.float32)
                for idx, name in enumerate(self._agent_names)
            }
        if isinstance(obs_data, (list, tuple)):
            return {
                idx: self._to_numpy(obs, flatten=True).astype(np.float32)
                for idx, obs in enumerate(obs_data)
            }
        raise TypeError(f"Unsupported VMAS observation format: {type(obs_data)}")

    def _convert_rewards(self, reward_data: Any) -> Dict[int, float]:
        if isinstance(reward_data, dict):
            return {
                idx: float(np.asarray(self._to_numpy(reward_data[name])).reshape(-1)[0])
                for idx, name in enumerate(self._agent_names)
            }
        if isinstance(reward_data, (list, tuple)):
            return {
                idx: float(np.asarray(self._to_numpy(rew)).reshape(-1)[0])
                for idx, rew in enumerate(reward_data)
            }
        reward_value = float(np.asarray(self._to_numpy(reward_data)).reshape(-1)[0])
        return {idx: reward_value for idx in range(self.num_agents)}

    def _convert_done_flag(self, done_data: Any) -> bool:
        if isinstance(done_data, dict):
            first_value = next(iter(done_data.values()))
            return bool(np.asarray(self._to_numpy(first_value)).reshape(-1)[0])
        return bool(np.asarray(self._to_numpy(done_data)).reshape(-1)[0])

    def _build_single_action_dict(self, actions: Dict[int, int]) -> Dict[str, torch.Tensor]:
        return {
            name: torch.tensor([int(actions[idx])], dtype=torch.long, device=self.device)
            for idx, name in enumerate(self._agent_names)
        }

    def _build_batched_action_dict(self, actions_batch: List[Dict[int, int]]) -> Dict[str, torch.Tensor]:
        if len(actions_batch) != self.n_envs:
            raise ValueError(f"Expected {self.n_envs} action dicts, got {len(actions_batch)}")
        return {
            name: torch.tensor(
                [int(actions_batch[env_idx][agent_idx]) for env_idx in range(self.n_envs)],
                dtype=torch.long,
                device=self.device,
            )
            for agent_idx, name in enumerate(self._agent_names)
        }

    def _get_batched_avail_actions(self) -> List[Dict[int, List[int]]]:
        avail = {agent_id: [1] * self.action_dim for agent_id in range(self.num_agents)}
        return [{agent_id: actions[:] for agent_id, actions in avail.items()} for _ in range(self.n_envs)]

    def reset(self):
        reset_output = self._env.reset(seed=self._seed)
        obs_data = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        self.step_count = 0
        if self.is_batched:
            return self._split_obs_batch(obs_data), self._get_batched_avail_actions()
        return self._convert_obs(obs_data)

    def step(self, actions):
        action_dict = self._build_batched_action_dict(actions) if self.is_batched else self._build_single_action_dict(actions)

        step_output = self._env.step(action_dict)
        self.step_count += 1

        if len(step_output) == 5:
            obs_data, reward_data, terminated_data, truncated_data, info_data = step_output
            done_flag = self._convert_done_flag(terminated_data) or self._convert_done_flag(truncated_data)
        elif len(step_output) == 4:
            obs_data, reward_data, done_data, info_data = step_output
            done_flag = self._convert_done_flag(done_data)
        else:
            raise ValueError(f"Unexpected VMAS step output length: {len(step_output)}")

        if self.is_batched:
            done_flags = [t or tr for t, tr in zip(self._split_done_flags(terminated_data), self._split_done_flags(truncated_data))] if len(step_output) == 5 else self._split_done_flags(done_data)
            self._reset_done_envs_in_place(obs_data, done_flags)
            observations_batch = self._split_obs_batch(obs_data)
            rewards_batch = self._split_rewards_batch(reward_data)
            dones_batch = [
                {agent_id: done_flags[env_idx] for agent_id in range(self.num_agents)}
                for env_idx in range(self.n_envs)
            ]
            infos_batch = self._split_info_batch(info_data, done_flags)
            return observations_batch, rewards_batch, dones_batch, infos_batch, self._get_batched_avail_actions()

        observations = self._convert_obs(obs_data)
        rewards = self._convert_rewards(reward_data)
        dones = {idx: done_flag for idx in range(self.num_agents)}
        info = {
            "step_count": self.step_count,
            "scenario_name": self.scenario_name,
            "raw_info": info_data,
        }
        return observations, rewards, dones, info

    def get_avail_actions(self) -> List[List[int]]:
        return [[1] * self.action_dim for _ in range(self.num_agents)]

    def get_avail_agent_actions(self, agent_id: int) -> List[int]:
        return [1] * self.action_dim

    def get_env_info(self) -> Dict[str, Any]:
        return {
            "obs_shape": self.obs_dim,
            "n_actions": self.action_dim,
            "n_agents": self.num_agents,
            "episode_limit": self.max_steps,
        }

    def get_obs_dim(self) -> int:
        return self.obs_dim

    def get_action_dim(self) -> int:
        return self.action_dim

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

class VMASNavigationEnv(VMASWrapper):
    """VMAS navigation benchmark."""

    def __init__(
        self,
        num_agents: int = 4,
        n_envs: int = 1,
        max_steps: int = 100,
        seed: int = None,
        device: str = "cpu",
        continuous_actions: bool = False,
        **scenario_kwargs,
    ):
        super().__init__(
            scenario_name="navigation",
            num_agents=num_agents,
            n_envs=n_envs,
            max_steps=max_steps,
            seed=seed,
            device=device,
            continuous_actions=continuous_actions,
            **scenario_kwargs,
        )

class VMASDispersionEnv(VMASWrapper):
    """VMAS dispersion benchmark."""

    def __init__(
        self,
        num_agents: int = 4,
        n_envs: int = 1,
        max_steps: int = 100,
        seed: int = None,
        device: str = "cpu",
        continuous_actions: bool = False,
        **scenario_kwargs,
    ):
        super().__init__(
            scenario_name="dispersion",
            num_agents=num_agents,
            n_envs=n_envs,
            max_steps=max_steps,
            seed=seed,
            device=device,
            continuous_actions=continuous_actions,
            **scenario_kwargs,
        )

class SMAXWrapper(MultiAgentEnv):
    """
    JaxMARL SMAX(HeuristicEnemySMAX) Wrapper.
    
    Available maps:
        "3m"               : 3 Marine vs 3 Marine
        "8m"               : 8 Marine vs 8 Marine
        "2s3z"             : 2 Stalker + 3 Zealot vs 같음
        "smacv2_5_units"   : 5 units (SMACv2 위치/타입 생성)
        "smacv2_10_units"  : 10 units
        None               : num_allies/num_enemies 직접 지정
    """

    @staticmethod
    def _build_smax_env_kwargs(
        map_name: Optional[str],
        num_allies: int,
        num_enemies: int,
        max_steps: int,
        map_width: int,
        map_height: int,
        world_steps_per_env_step: int,
        time_per_step: float,
        see_enemy_actions: bool,
        won_battle_bonus: float,
        walls_cause_death: bool,
        observation_type: str,
        smacv2_position_generation: bool,
        smacv2_unit_type_generation: bool,
    ) -> Dict[str, Any]:
        if observation_type not in {"unit_list", "conic"}:
            raise ValueError(
                f"Unsupported SMAX observation_type: {observation_type}. "
                "Expected 'unit_list' or 'conic'."
            )

        env_kwargs: Dict[str, Any] = {}
        if map_name is not None:
            try:
                from jaxmarl.environments.smax.smax_env import map_name_to_scenario

                scenario = map_name_to_scenario(map_name)
                env_kwargs["scenario"] = scenario
                env_kwargs["num_allies"] = scenario.num_allies
                env_kwargs["num_enemies"] = scenario.num_enemies
                env_kwargs["smacv2_position_generation"] = scenario.smacv2_position_generation
                env_kwargs["smacv2_unit_type_generation"] = scenario.smacv2_unit_type_generation
                print(
                    f"[SMAXWrapper] 시나리오 '{map_name}': "
                    f"{scenario.num_allies} allies vs {scenario.num_enemies} enemies"
                )
            except KeyError:
                print(
                    f"[SMAXWrapper] 경고: 알 수 없는 map_name '{map_name}', "
                    f"num_allies={num_allies}, num_enemies={num_enemies} 사용"
                )
                env_kwargs["num_allies"] = num_allies
                env_kwargs["num_enemies"] = num_enemies
                env_kwargs["smacv2_position_generation"] = smacv2_position_generation
                env_kwargs["smacv2_unit_type_generation"] = smacv2_unit_type_generation
        else:
            env_kwargs["num_allies"] = num_allies
            env_kwargs["num_enemies"] = num_enemies
            env_kwargs["smacv2_position_generation"] = smacv2_position_generation
            env_kwargs["smacv2_unit_type_generation"] = smacv2_unit_type_generation

        env_kwargs["max_steps"] = max_steps
        env_kwargs["map_width"] = map_width
        env_kwargs["map_height"] = map_height
        env_kwargs["world_steps_per_env_step"] = world_steps_per_env_step
        env_kwargs["time_per_step"] = time_per_step
        env_kwargs["see_enemy_actions"] = see_enemy_actions
        env_kwargs["won_battle_bonus"] = won_battle_bonus
        env_kwargs["walls_cause_death"] = walls_cause_death
        env_kwargs["observation_type"] = observation_type
        return env_kwargs

    @staticmethod
    def _build_default_allowed_targets(
        num_allies: int,
        num_enemies: int,
        allowed_target_window: Optional[int] = None,
    ) -> Dict[int, List[int]]:
        import math

        step = num_enemies / max(1, num_allies)
        if allowed_target_window is None:
            window = min(num_enemies, max(2, math.ceil(step) + 1))
        else:
            window = min(num_enemies, max(1, int(allowed_target_window)))
        allowed_targets: Dict[int, List[int]] = {}
        for ally_id in range(num_allies):
            start = round(ally_id * step)
            allowed_targets[ally_id] = [
                (start + j) % num_enemies for j in range(window)
            ]
        return allowed_targets

    @staticmethod
    def _build_cooperative_kill_assignments(
        num_allies: int,
        num_enemies: int,
        allowed_targets: Optional[Dict[int, List[int]]] = None,
        cooperative_size: int = 2,
    ) -> Dict[int, List[int]]:
        required_agents: Dict[int, List[int]] = {}
        cooperative_size = max(1, int(cooperative_size))

        for enemy_id in range(num_enemies):
            allowed_agents = []
            if allowed_targets is not None:
                allowed_agents = [
                    agent_id
                    for agent_id in range(num_allies)
                    if enemy_id in allowed_targets.get(agent_id, [])
                ]

            preferred_owner = enemy_id % max(1, num_allies)
            primary_pool = allowed_agents if allowed_agents else list(range(num_allies))
            owner = preferred_owner if preferred_owner in primary_pool else primary_pool[0]

            coalition: List[int] = [owner]
            seen = {owner}
            target_size = min(cooperative_size, num_allies)

            for offset in range(1, num_allies):
                if len(coalition) >= target_size:
                    break
                candidate = (owner + offset) % num_allies
                if candidate in primary_pool and candidate not in seen:
                    coalition.append(candidate)
                    seen.add(candidate)

            if len(coalition) < target_size:
                for offset in range(1, num_allies):
                    if len(coalition) >= target_size:
                        break
                    candidate = (owner + offset) % num_allies
                    if candidate not in seen:
                        coalition.append(candidate)
                        seen.add(candidate)

            required_agents[enemy_id] = coalition

        return required_agents

    @staticmethod
    def _augment_allowed_targets_with_required_agents(
        allowed_targets: Optional[Dict[int, List[int]]],
        required_agents: Dict[int, List[int]],
        num_allies: int,
    ) -> Optional[Dict[int, List[int]]]:
        if allowed_targets is None:
            return None

        merged = {
            agent_id: list(targets)
            for agent_id, targets in allowed_targets.items()
        }
        for agent_id in range(num_allies):
            merged.setdefault(agent_id, [])

        for enemy_id, assigned_agents in required_agents.items():
            for agent_id in assigned_agents:
                if enemy_id not in merged[agent_id]:
                    merged[agent_id].append(enemy_id)
                    merged[agent_id].sort()

        return merged

    @staticmethod
    def _build_allowed_target_matrix(
        num_allies: int,
        num_enemies: int,
        allowed_targets: Optional[Dict[int, List[int]]],
    ) -> np.ndarray:
        matrix = np.ones((num_allies, num_enemies), dtype=np.float32)
        if allowed_targets is None:
            return matrix
        matrix.fill(0.0)
        for agent_id in range(num_allies):
            for enemy_id in allowed_targets.get(agent_id, []):
                if 0 <= enemy_id < num_enemies:
                    matrix[agent_id, enemy_id] = 1.0
        return matrix

    @staticmethod
    def _build_cooperative_kill_reward_tables(
        required_agents: Dict[int, List[int]],
        num_allies: int,
        num_enemies: int,
        owner_ratio: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        required_mask = np.zeros((num_enemies, num_allies), dtype=np.bool_)
        reward_weights = np.zeros((num_enemies, num_allies), dtype=np.float32)

        owner_ratio = float(np.clip(owner_ratio, 0.0, 1.0))
        for enemy_id in range(num_enemies):
            assigned_agents = list(required_agents.get(enemy_id, []))
            if not assigned_agents:
                continue

            for agent_id in assigned_agents:
                required_mask[enemy_id, agent_id] = True

            if len(assigned_agents) == 1:
                reward_weights[enemy_id, assigned_agents[0]] = 1.0
                continue

            owner = assigned_agents[0]
            partners = assigned_agents[1:]
            reward_weights[enemy_id, owner] = owner_ratio
            partner_share = (1.0 - owner_ratio) / max(1, len(partners))
            for agent_id in partners:
                reward_weights[enemy_id, agent_id] = partner_share

        return required_mask, reward_weights

    def __init__(
        self,
        num_allies: int = 5,
        num_enemies: int = 5,
        map_name: str = None,
        seed: int = None,
        max_steps: int = 100,
        use_individual_rewards: bool = False,
        enemy_shoots: bool = True,
        attack_mode: str = "closest",
        map_width: int = 32,
        map_height: int = 32,
        world_steps_per_env_step: int = 8,
        time_per_step: float = 1.0 / 16,
        see_enemy_actions: bool = True,
        won_battle_bonus: float = 1.0,
        walls_cause_death: bool = True,
        observation_type: str = "unit_list",
        smacv2_position_generation: bool = False,
        smacv2_unit_type_generation: bool = False,
        use_allowed_targets: bool = False,
        allowed_target_window: Optional[int] = None,
        allowed_targets: Dict[int, List[int]] = None,
        use_partial_damage_restore: bool = False,
        use_cooperative_kill_bonus: bool = False,
        cooperative_bonus_mode: str = "kill",
        cooperative_kill_window: int = 2,
        cooperative_kill_bonus: float = 2.0,
        cooperative_kill_owner_ratio: float = 0.7,
        cooperative_size: int = 2,
        **kwargs
    ):
        try:
            import jax
            import jax.numpy as jnp
            from jaxmarl.environments.smax import HeuristicEnemySMAX
        except ImportError:
            raise ImportError(
                "JaxMARL이 설치되지 않았습니다. 다음 명령으로 설치하세요:\n"
                "  pip install jaxmarl"
            )

        self._jax = jax
        self._jnp = jnp

        env_kwargs = self._build_smax_env_kwargs(
            map_name=map_name,
            num_allies=num_allies,
            num_enemies=num_enemies,
            max_steps=max_steps,
            map_width=map_width,
            map_height=map_height,
            world_steps_per_env_step=world_steps_per_env_step,
            time_per_step=time_per_step,
            see_enemy_actions=see_enemy_actions,
            won_battle_bonus=won_battle_bonus,
            walls_cause_death=walls_cause_death,
            observation_type=observation_type,
            smacv2_position_generation=smacv2_position_generation,
            smacv2_unit_type_generation=smacv2_unit_type_generation,
        )

        # HeuristicEnemySMAX 생성
        self._env = HeuristicEnemySMAX(
            enemy_shoots=enemy_shoots,
            attack_mode=attack_mode,
            **env_kwargs
        )

        # 에이전트 수 설정
        _num_allies = self._env.num_allies
        _num_enemies = self._env.num_enemies

        super().__init__(num_agents=_num_allies, seed=seed)

        self.num_enemies = _num_enemies
        self.use_individual_rewards = use_individual_rewards
        self.map_name = map_name
        self.max_steps = max_steps
        self.use_cooperative_kill_bonus = use_cooperative_kill_bonus
        self.cooperative_bonus_mode = str(cooperative_bonus_mode).lower()
        if self.cooperative_bonus_mode not in ("kill", "damage"):
            raise ValueError(
                f"Unsupported cooperative_bonus_mode: {cooperative_bonus_mode}. "
                "Use 'kill' or 'damage'."
            )
        self.cooperative_size = max(1, int(cooperative_size))
        self.cooperative_kill_window = max(1, int(cooperative_kill_window))
        self.cooperative_kill_bonus = float(cooperative_kill_bonus)
        self.cooperative_kill_owner_ratio = float(cooperative_kill_owner_ratio)

        # 에이전트 키 매핑: int id <-> str "ally_{i}"
        self._agent_keys = [f"ally_{i}" for i in range(_num_allies)]
        self._id_to_key = {i: f"ally_{i}" for i in range(_num_allies)}
        self._key_to_id = {f"ally_{i}": i for i in range(_num_allies)}

        # obs_dim, action_dim 계산
        self.obs_dim = self._env.obs_size
        # ally의 action 수 = num_movement_actions + num_enemies
        self.action_dim = self._env.num_ally_actions
        self._unit_type_health_np = np.array(self._env.unit_type_health, dtype=np.float32)
        self._unit_type_health_jax = self._jnp.array(self._env.unit_type_health)
        self._use_self_play_reward = bool(self._env.use_self_play_reward)
        self._won_battle_bonus = float(self._env.won_battle_bonus)

        # JAX PRNGKey 초기화
        _seed_val = seed if seed is not None else 0
        self._rng = jax.random.PRNGKey(_seed_val)
        self._state = None

        # allowed_targets 설정 (StagHuntEnv의 required_agents 패턴과 동일)
        self.allowed_target_window = None if allowed_target_window is None else max(1, int(allowed_target_window))
        target_window_covers_all = (
            allowed_targets is None
            and self.allowed_target_window is not None
            and self.allowed_target_window >= _num_enemies
        )
        self.use_allowed_targets = bool(use_allowed_targets) and not target_window_covers_all
        self._num_movement_actions = self._env.num_movement_actions  # 통상 5
        if self.use_allowed_targets:
            if allowed_targets is not None:
                self.allowed_targets = allowed_targets
            else:
                self.allowed_targets = self._build_default_allowed_targets(
                    _num_allies,
                    _num_enemies,
                    self.allowed_target_window,
                )
        else:
            self.allowed_targets = None
        self.use_partial_damage_restore = bool(use_partial_damage_restore)

        self.cooperative_kill_required_agents = self._build_cooperative_kill_assignments(
            _num_allies,
            _num_enemies,
            self.allowed_targets,
            self.cooperative_size,
        )
        if self.use_allowed_targets and self.use_cooperative_kill_bonus:
            self.allowed_targets = self._augment_allowed_targets_with_required_agents(
                self.allowed_targets,
                self.cooperative_kill_required_agents,
                _num_allies,
            )
        self._allowed_target_matrix_np = self._build_allowed_target_matrix(
            _num_allies,
            _num_enemies,
            self.allowed_targets,
        )
        self._allowed_target_matrix_jax = self._jnp.array(self._allowed_target_matrix_np)
        if self.use_allowed_targets:
            self._env = swap_to_target_aware_inner_smax(
                self._env,
                enemy_shoots=enemy_shoots,
                attack_mode=attack_mode,
                env_kwargs=env_kwargs,
                allowed_target_matrix=self._allowed_target_matrix_np,
                use_partial_damage_restore=self.use_partial_damage_restore,
            )
        self._coop_required_mask_np, self._coop_reward_weights_np = (
            self._build_cooperative_kill_reward_tables(
                self.cooperative_kill_required_agents,
                _num_allies,
                _num_enemies,
                self.cooperative_kill_owner_ratio,
            )
        )
        self._coop_attack_memory = np.zeros((_num_allies, _num_enemies), dtype=np.int32)

        if self.use_allowed_targets:
            mode = "partial" if self.use_partial_damage_restore else "full"
            print(
                f"[SMAXWrapper] allowed_targets ({mode} source-level damage mask, "
                f"window={self.allowed_target_window}): {self.allowed_targets}"
            )
        if self.use_cooperative_kill_bonus:
            print(
                "[SMAXWrapper] cooperative_kill_bonus: "
                f"mode={self.cooperative_bonus_mode}, "
                f"cooperative_size={self.cooperative_size}, "
                f"window={self.cooperative_kill_window}, bonus={self.cooperative_kill_bonus}, "
                f"required_agents={self.cooperative_kill_required_agents}"
            )

        print(f"[SMAXWrapper] obs_dim={self.obs_dim}, action_dim={self.action_dim}")
        print(f"[SMAXWrapper] {_num_allies} allies vs {_num_enemies} enemies (heuristic)")

        # JAX JIT 컴파일 사전 수행 (warm-up)
        # 학습 루프 진입 전에 미리 컴파일해야 첫 iteration에서 멈추지 않음
        self._warmup()

    def _warmup(self):
        """JAX JIT 함수들을 미리 컴파일합니다.
        
        SMAX의 reset/step/get_avail_actions는 @jax.jit으로 감싸져 있어
        첫 호출 시 XLA 컴파일이 발생합니다 (수십 초 소요).
        
        중요: 단순히 한 번 호출하는 것으로는 부족합니다.
        HeuristicEnemySMAX의 heuristic enemy action, jax.lax.scan 내부의
        여러 코드 경로까지 컴파일하려면 실제 에피소드를 끝까지 실행해야 합니다.
        에피소드 2~3회를 실행하면 모든 JAX 경로가 컴파일됩니다.
        """
        import time
        import numpy as np

        print("[SMAXWrapper] JAX JIT warm-up 시작...")
        print("[SMAXWrapper]   (JIT 컴파일을 위해 더미 에피소드를 실행합니다)")
        t0 = time.time()

        for episode_idx in range(3):
            print(f"[SMAXWrapper]   에피소드 {episode_idx+1}/3 실행 중... ({time.time()-t0:.1f}s 경과)")
            self._rng, reset_key = self._jax.random.split(self._rng)
            obs_dict, warmup_state = self._env.reset(reset_key)
            self._jax.block_until_ready(list(obs_dict.values()))

            for step_idx in range(self._env.max_steps + 5):
                # avail_actions 컴파일
                avail = self._env.get_avail_actions(warmup_state)
                self._jax.block_until_ready(list(avail.values()))

                # 유효한 액션 선택 (stop=4 대신 실제 avail 사용)
                dummy_actions = {}
                for key in self._agent_keys:
                    avail_arr = np.array(avail[key])
                    valid = np.where(avail_arr > 0)[0]
                    chosen = int(valid[0]) if len(valid) > 0 else 4
                    dummy_actions[key] = self._jnp.int32(chosen)

                self._rng, step_key = self._jax.random.split(self._rng)
                obs2, warmup_state2, rew2, done2, _ = self._env.step_env(
                    step_key, warmup_state, dummy_actions
                )
                self._jax.block_until_ready(list(obs2.values()))
                warmup_state = warmup_state2

                # 에피소드 종료 체크
                all_done = bool(np.array(done2.get("__all__", False)))
                if all_done:
                    break

        print(f"[SMAXWrapper] JAX JIT warm-up 완료! (총 {time.time()-t0:.1f}초)")
        # warmup 후 state는 None으로 초기화 (reset()이 시작점)
        self._state = None

    def _apply_cooperative_kill_bonus_numpy(
        self,
        enemy_unit_types: np.ndarray,
        prev_enemy_health: np.ndarray,
        next_enemy_health: np.ndarray,
        prev_enemy_alive: np.ndarray,
        next_enemy_alive: np.ndarray,
        actions_by_agent: Dict[int, int],
    ) -> Tuple[np.ndarray, List[int]]:
        if not self.use_cooperative_kill_bonus:
            return np.zeros(self.num_agents, dtype=np.float32), []

        attack_memory = np.maximum(self._coop_attack_memory - 1, 0)
        for agent_id in range(self.num_agents):
            action = int(actions_by_agent.get(agent_id, 0))
            if action >= self._num_movement_actions:
                enemy_id = action - self._num_movement_actions
                if 0 <= enemy_id < self.num_enemies:
                    attack_memory[agent_id, enemy_id] = self.cooperative_kill_window

        killed_mask = prev_enemy_alive & (~next_enemy_alive)
        damage = np.maximum(prev_enemy_health - next_enemy_health, 0.0)
        damage_ratio = damage / np.maximum(
            self._unit_type_health_np[np.array(enemy_unit_types, dtype=np.int32)],
            1.0e-8,
        )
        damaged_mask = damage > 1.0e-8
        recent_attacks = attack_memory > 0
        coop_ready = np.all(
            np.where(self._coop_required_mask_np, recent_attacks.T, True),
            axis=-1,
        )
        if self.cooperative_bonus_mode == "kill":
            coop_success = killed_mask & coop_ready
            event_scale = coop_success.astype(np.float32) * self.cooperative_kill_bonus
        else:
            coop_success = damaged_mask & coop_ready
            event_scale = damage_ratio.astype(np.float32) * coop_success.astype(np.float32) * self.cooperative_kill_bonus

        bonus = event_scale @ self._coop_reward_weights_np
        self._coop_attack_memory = attack_memory
        return bonus.astype(np.float32), np.where(coop_success)[0].tolist()

    def _compute_damage_masked_enemy_mask_numpy(
        self,
        actions_by_agent: Dict[int, int],
    ) -> np.ndarray:
        masked_enemies = np.zeros(self.num_enemies, dtype=np.bool_)
        if not self.use_allowed_targets or self.allowed_targets is None:
            return masked_enemies

        for agent_id in range(self.num_agents):
            action = int(actions_by_agent.get(agent_id, 0))
            if action < self._num_movement_actions:
                continue
            enemy_id = action - self._num_movement_actions
            if enemy_id < 0 or enemy_id >= self.num_enemies:
                continue
            if self._allowed_target_matrix_np[agent_id, enemy_id] <= 0.0:
                masked_enemies[enemy_id] = True

        return masked_enemies

    def _compute_team_reward_numpy(self, prev_state, next_state) -> float:
        health_before = np.array(prev_state.state.unit_health, dtype=np.float32)
        health_after = np.array(next_state.state.unit_health, dtype=np.float32)
        unit_types = np.array(next_state.state.unit_types, dtype=np.int32)
        normalized_delta = (health_after - health_before) / self._unit_type_health_np[unit_types]
        enemy_health_decrease = np.sum(normalized_delta[self._env.num_allies:])
        enemy_health_decrease_reward = abs(enemy_health_decrease) / float(self.num_enemies)
        if self._use_self_play_reward:
            enemy_health_decrease_reward = 0.0

        alive = np.array(next_state.state.unit_alive, dtype=np.bool_)
        won_battle = bool(np.all(~alive[self._env.num_allies:]))
        lost_battle = bool(np.all(~alive[:self._env.num_allies]))
        lost_battle_bonus = (
            -self._won_battle_bonus
            if lost_battle and self._use_self_play_reward and not won_battle
            else 0.0
        )
        won_battle_bonus = self._won_battle_bonus if won_battle and not lost_battle else 0.0
        return float(enemy_health_decrease_reward + won_battle_bonus + lost_battle_bonus)

    def reset(self) -> Dict[int, np.ndarray]:
        """환경을 리셋하고 초기 관찰을 반환합니다."""
        self._rng, reset_key = self._jax.random.split(self._rng)
        obs_dict, self._state = self._env.reset(reset_key)
        self._coop_attack_memory = np.zeros((self.num_agents, self.num_enemies), dtype=np.int32)

        # Dict[str, jax.Array] → Dict[int, np.ndarray]
        observations = {}
        for i, key in enumerate(self._agent_keys):
            observations[i] = np.array(obs_dict[key], dtype=np.float32)

        return observations

    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[str, Any]
    ]:
        """환경에서 한 스텝을 진행합니다."""
        assert self._state is not None, "step() 전에 reset()을 호출하세요."
        prev_state = self._state
        prev_enemy_alive = np.array(
            prev_state.state.unit_alive[self._env.num_allies:],
            dtype=np.bool_,
        )
        prev_enemy_health = np.array(
            prev_state.state.unit_health[self._env.num_allies:],
            dtype=np.float32,
        )
        enemy_unit_types = np.array(
            prev_state.state.unit_types[self._env.num_allies:],
            dtype=np.int32,
        )

        # Dict[int, int] → Dict[str, jnp.int32]
        jax_actions = {
            self._id_to_key[i]: self._jnp.int32(a)
            for i, a in actions.items()
        }
        
        self._rng, step_key = self._jax.random.split(self._rng)
        _, next_state, _, _, _ = \
            self._env.step_env(step_key, self._state, jax_actions)
        self._state = next_state
        damage_masked_enemies = np.where(
            self._compute_damage_masked_enemy_mask_numpy(actions)
        )[0].tolist()
        obs_dict = self._env.get_obs(self._state)
        observations = {}
        for i, key in enumerate(self._agent_keys):
            observations[i] = np.array(obs_dict[key], dtype=np.float32)

        team_reward = self._compute_team_reward_numpy(prev_state, self._state)
        rewards = {i: team_reward for i in range(self.num_agents)}

        next_enemy_alive = np.array(
            self._state.state.unit_alive[self._env.num_allies:],
            dtype=np.bool_,
        )
        next_enemy_health = np.array(
            self._state.state.unit_health[self._env.num_allies:],
            dtype=np.float32,
        )
        coop_bonus, cooperative_kills = self._apply_cooperative_kill_bonus_numpy(
            enemy_unit_types,
            prev_enemy_health,
            next_enemy_health,
            prev_enemy_alive,
            next_enemy_alive,
            actions,
        )
        for agent_id in range(self.num_agents):
            rewards[agent_id] += float(coop_bonus[agent_id])

        # done 변환
        # SMAX (SMAC과 동일): 개별 에이전트가 죽어도 에피소드는 계속됨.
        # 오직 __all__ 신호만 에피소드 종료로 사용 (모든 ally/enemy가 죽거나 시간 초과).
        # 개별 agent_done을 에피소드 done으로 전파하면 첫 사망 시 에피소드가 1스텝에 끝남.
        all_done = bool(np.array(self._env.is_terminal(self._state)).item())

        dones = {i: all_done for i in range(self.num_agents)}

        # battle_won 계산: 모든 enemy가 죽으면 승리
        enemies_alive = self._state.state.unit_alive[self._env.num_allies:]
        battle_won = bool(np.array(~self._jnp.any(enemies_alive)).item())

        info = {
            'battle_won': battle_won,
            'episode_done': all_done,
            'cooperative_kills': cooperative_kills,
            'damage_masked_enemies': damage_masked_enemies,
        }

        return observations, rewards, dones, info

    def get_avail_actions(self) -> List[List[int]]:
        """
        각 에이전트의 사용 가능한 액션 마스크를 반환합니다.

        Returns:
            List[List[int]]: shape (num_agents, action_dim), 값은 0 또는 1
        """
        assert self._state is not None, "get_avail_actions() 전에 reset()을 호출하세요."

        avail_dict = self._env.get_avail_actions(self._state)

        avail_list = []
        for i, key in enumerate(self._agent_keys):
            mask = np.array(avail_dict[key], dtype=np.int32)
            avail_list.append(mask.tolist())

        return avail_list

    def get_obs_dim(self) -> int:
        """관찰 차원을 반환합니다."""
        return self.obs_dim

    def get_action_dim(self) -> int:
        """액션 차원을 반환합니다."""
        return self.action_dim

    def get_env_info(self) -> Dict[str, Any]:
        """환경 정보를 반환합니다 (SMACv2Wrapper 호환)."""
        return {
            'obs_shape': self.obs_dim,
            'n_actions': self.action_dim,
            'n_agents': self.num_agents,
            'episode_limit': self.max_steps,
        }

    def close(self):
        """환경을 닫습니다."""
        pass


# ============================================================================
# SMAXBatchWrapper: jax.vmap 기반 병렬 SMAX 환경
# ============================================================================

class SMAXBatchWrapper(MultiAgentEnv):
    """
    JAX vmap을 사용해 N개의 SMAX 환경을 GPU/CPU에서 병렬 실행합니다.

    fork 기반 multiprocessing 없이 ParallelEnvWrapper와 동일한 인터페이스를 제공합니다:
      - reset() → (obs_list, avail_list)
      - step(actions_batch) → (obs_list, rewards_list, dones_list, infos_list, avail_list)
      - 에피소드 종료 시 자동 리셋 (jnp.where 사용)

    Args:
        n_envs: 병렬 환경 수 (vmap batchsize)
        map_name: SMAX 시나리오 이름 ('smacv2_5_units', '3m' 등)
        seed: 랜덤 시드
        ... (SMAXWrapper와 동일한 나머지 파라미터)
    """

    def __init__(
        self,
        n_envs: int = 4,
        map_name: Optional[str] = None,
        num_allies: int = 5,
        num_enemies: int = 5,
        seed: Optional[int] = None,
        max_steps: int = 100,
        use_individual_rewards: bool = False,
        enemy_shoots: bool = True,
        attack_mode: str = 'closest',
        map_width: int = 32,
        map_height: int = 32,
        world_steps_per_env_step: int = 8,
        time_per_step: float = 1.0 / 16,
        see_enemy_actions: bool = True,
        won_battle_bonus: float = 1.0,
        walls_cause_death: bool = True,
        observation_type: str = "unit_list",
        smacv2_position_generation: bool = False,
        smacv2_unit_type_generation: bool = False,
        use_allowed_targets: bool = True,
        allowed_target_window: Optional[int] = None,
        allowed_targets: Dict[int, List[int]] = None,
        use_partial_damage_restore: bool = False,
        use_cooperative_kill_bonus: bool = True,
        cooperative_bonus_mode: str = "kill",
        cooperative_size: int = 2,
        cooperative_kill_window: int = 2,
        cooperative_kill_bonus: float = 2.0,
        cooperative_kill_owner_ratio: float = 0.7,
        warmup_episodes: int = 3,
    ):
        try:
            import jax
            import jax.numpy as jnp
            from jaxmarl.environments.smax import HeuristicEnemySMAX
        except ImportError as e:
            raise ImportError(f"jax/jaxmarl 설치 필요: {e}")

        self._jax = jax
        self._jnp = jnp

        # n_envs 저장 (ParallelEnvWrapper 호환)
        self.n_envs = n_envs
        self.n_rollout_threads = n_envs

        env_kwargs = SMAXWrapper._build_smax_env_kwargs(
            map_name=map_name,
            num_allies=num_allies,
            num_enemies=num_enemies,
            max_steps=max_steps,
            map_width=map_width,
            map_height=map_height,
            world_steps_per_env_step=world_steps_per_env_step,
            time_per_step=time_per_step,
            see_enemy_actions=see_enemy_actions,
            won_battle_bonus=won_battle_bonus,
            walls_cause_death=walls_cause_death,
            observation_type=observation_type,
            smacv2_position_generation=smacv2_position_generation,
            smacv2_unit_type_generation=smacv2_unit_type_generation,
        )
        if map_name is not None and "scenario" in env_kwargs:
            print(
                f"[SMAXBatchWrapper] '{map_name}': "
                f"{env_kwargs['num_allies']}v{env_kwargs['num_enemies']}, n_envs={n_envs}"
            )

        self._env = HeuristicEnemySMAX(
            enemy_shoots=enemy_shoots,
            attack_mode=attack_mode,
            **env_kwargs
        )

        _num_allies = self._env.num_allies
        _num_enemies = self._env.num_enemies
        super().__init__(num_agents=_num_allies, seed=seed)

        self.num_enemies = _num_enemies
        self.map_name = map_name
        self.max_steps = max_steps
        self.use_individual_rewards = use_individual_rewards
        self.use_cooperative_kill_bonus = use_cooperative_kill_bonus
        self.cooperative_bonus_mode = str(cooperative_bonus_mode).lower()
        if self.cooperative_bonus_mode not in ("kill", "damage"):
            raise ValueError(
                f"Unsupported cooperative_bonus_mode: {cooperative_bonus_mode}. "
                "Use 'kill' or 'damage'."
            )
        self.cooperative_size = max(1, int(cooperative_size))
        self.cooperative_kill_window = max(1, int(cooperative_kill_window))
        self.cooperative_kill_bonus = float(cooperative_kill_bonus)
        self.cooperative_kill_owner_ratio = float(cooperative_kill_owner_ratio)
        self._agent_keys = [f"ally_{i}" for i in range(_num_allies)]
        self._id_to_key = {i: f"ally_{i}" for i in range(_num_allies)}

        self.obs_dim = self._env.obs_size
        self.action_dim = self._env.num_ally_actions
        self._num_movement_actions = self._env.num_movement_actions
        self.use_partial_damage_restore = bool(use_partial_damage_restore)
        self.allowed_target_window = None if allowed_target_window is None else max(1, int(allowed_target_window))
        self._unit_type_health_jax = jnp.array(self._env.unit_type_health)
        self._unit_type_health_np = np.array(self._env.unit_type_health, dtype=np.float32)
        self._use_self_play_reward = bool(self._env.use_self_play_reward)
        self._won_battle_bonus = float(self._env.won_battle_bonus)

        # allowed_targets 설정 (SMAXWrapper와 동일한 로직)
        target_window_covers_all = (
            allowed_targets is None
            and self.allowed_target_window is not None
            and self.allowed_target_window >= _num_enemies
        )
        self.use_allowed_targets = bool(use_allowed_targets) and not target_window_covers_all
        if self.use_allowed_targets:
            if allowed_targets is not None:
                self.allowed_targets = allowed_targets
            else:
                self.allowed_targets = SMAXWrapper._build_default_allowed_targets(
                    _num_allies,
                    _num_enemies,
                    self.allowed_target_window,
                )
        else:
            self.allowed_targets = None
        self.cooperative_kill_required_agents = SMAXWrapper._build_cooperative_kill_assignments(
            _num_allies,
            _num_enemies,
            self.allowed_targets,
            self.cooperative_size,
        )
        if self.use_allowed_targets and self.use_cooperative_kill_bonus:
            self.allowed_targets = SMAXWrapper._augment_allowed_targets_with_required_agents(
                self.allowed_targets,
                self.cooperative_kill_required_agents,
                _num_allies,
            )
        self._allowed_target_matrix_np = SMAXWrapper._build_allowed_target_matrix(
            _num_allies,
            _num_enemies,
            self.allowed_targets,
        )
        self._allowed_target_matrix_jax = jnp.array(self._allowed_target_matrix_np)
        if self.use_allowed_targets:
            self._env = swap_to_target_aware_inner_smax(
                self._env,
                enemy_shoots=enemy_shoots,
                attack_mode=attack_mode,
                env_kwargs=env_kwargs,
                allowed_target_matrix=self._allowed_target_matrix_np,
                use_partial_damage_restore=self.use_partial_damage_restore,
            )

        self._coop_required_mask_np, self._coop_reward_weights_np = (
            SMAXWrapper._build_cooperative_kill_reward_tables(
                self.cooperative_kill_required_agents,
                _num_allies,
                _num_enemies,
                self.cooperative_kill_owner_ratio,
            )
        )
        self._coop_required_mask_jax = jnp.array(self._coop_required_mask_np)
        self._coop_reward_weights_jax = jnp.array(self._coop_reward_weights_np)

        _seed_val = seed if seed is not None else 0
        self._rng = jax.random.PRNGKey(_seed_val)
        self._warmup_episodes = max(0, int(warmup_episodes))
        self._batch_states = None  # 배치 상태 (n_envs, ...)
        self._coop_attack_memory_np = np.zeros((n_envs, _num_allies, _num_enemies), dtype=np.int32)
        self._coop_attack_memory_jax = jnp.zeros((n_envs, _num_allies, _num_enemies), dtype=jnp.int32)

        # vmap + JIT 컴파일된 함수
        self._batch_reset = jax.jit(jax.vmap(self._env.reset))
        self._batch_step = jax.jit(jax.vmap(self._env.step_env))
        self._batch_obs = jax.jit(jax.vmap(self._env.get_obs))
        self._batch_avail = jax.jit(jax.vmap(self._env.get_avail_actions))
        self._batch_terminal = jax.jit(jax.vmap(self._env.is_terminal))

        if self.use_allowed_targets:
            mode = "partial" if self.use_partial_damage_restore else "full"
            print(
                f"[SMAXBatchWrapper] allowed_targets ({mode} source-level damage mask, "
                f"window={self.allowed_target_window}): {self.allowed_targets}"
            )
        if self.use_cooperative_kill_bonus:
            print(
                "[SMAXBatchWrapper] cooperative_kill_bonus: "
                f"mode={self.cooperative_bonus_mode}, "
                f"cooperative_size={self.cooperative_size}, "
                f"window={self.cooperative_kill_window}, bonus={self.cooperative_kill_bonus}, "
                f"required_agents={self.cooperative_kill_required_agents}"
            )
        print(f"[SMAXBatchWrapper] obs_dim={self.obs_dim}, action_dim={self.action_dim}")
        print(f"[SMAXBatchWrapper] {_num_allies}v{_num_enemies} × {n_envs} 환경 (vmap 병렬)")

        if self._warmup_episodes > 0:
            self._warmup()
        else:
            print("[SMAXBatchWrapper] 배치 JIT warm-up 생략")

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    def _warmup(self):
        """배치 vmap 함수들을 사전 JIT 컴파일합니다."""
        import time
        import numpy as np

        print("[SMAXBatchWrapper] 배치 JIT warm-up 시작...")
        t0 = time.time()

        for ep in range(self._warmup_episodes):
            print(
                f"[SMAXBatchWrapper]   에피소드 {ep+1}/{self._warmup_episodes} "
                f"({time.time()-t0:.1f}s)"
            )
            self._rng, *env_keys = self._jax.random.split(self._rng, self.n_envs + 1)
            keys_arr = self._jnp.stack(env_keys)

            obs_b, batch_st = self._batch_reset(keys_arr)
            self._jax.block_until_ready(obs_b)

            for _ in range(self._env.max_steps + 5):
                avail_b = self._batch_avail(batch_st)
                self._jax.block_until_ready(avail_b)

                # 유효한 첫 번째 액션 선택
                dummy_actions = {}
                for key in self._agent_keys:
                    avail_np = np.array(avail_b[key])          # (n_envs, action_dim)
                    chosen = np.argmax(avail_np, axis=1)        # (n_envs,)
                    dummy_actions[key] = self._jnp.array(chosen, dtype=self._jnp.int32)

                self._rng, *step_keys = self._jax.random.split(self._rng, self.n_envs + 1)
                step_keys_arr = self._jnp.stack(step_keys)

                obs_b2, batch_st2, _, dones_b, _ = self._batch_step(
                    step_keys_arr, batch_st, dummy_actions
                )
                self._jax.block_until_ready(obs_b2)
                batch_st = batch_st2

                if np.all(np.array(dones_b["__all__"])):
                    break

        print(f"[SMAXBatchWrapper] warm-up 완료! (총 {time.time()-t0:.1f}초)")
        self._batch_states = None

    def _pack_batch_state(self, env_state, coop_attack_memory):
        return {
            "env_state": env_state,
            "coop_attack_memory": coop_attack_memory,
        }

    def _unpack_batch_state(self, batch_state):
        if isinstance(batch_state, dict) and "env_state" in batch_state:
            return batch_state["env_state"], batch_state["coop_attack_memory"]
        zero_memory = self._jnp.zeros(
            (self.n_envs, self.num_agents, self.num_enemies),
            dtype=self._jnp.int32,
        )
        return batch_state, zero_memory

    def _apply_cooperative_kill_bonus_numpy_batch(
        self,
        attack_memory: np.ndarray,
        actions: np.ndarray,
        enemy_unit_types: np.ndarray,
        prev_enemy_health: np.ndarray,
        next_enemy_health: np.ndarray,
        prev_enemy_alive: np.ndarray,
        next_enemy_alive: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.use_cooperative_kill_bonus:
            zero_bonus = np.zeros((actions.shape[0], self.num_agents), dtype=np.float32)
            zero_success = np.zeros((actions.shape[0], self.num_enemies), dtype=np.bool_)
            return zero_bonus, attack_memory, zero_success

        updated_memory = np.maximum(attack_memory - 1, 0)
        attack_targets = np.clip(actions - self._num_movement_actions, 0, self.num_enemies - 1)
        attack_mask = actions >= self._num_movement_actions
        for enemy_id in range(self.num_enemies):
            hit_enemy = attack_mask & (attack_targets == enemy_id)
            updated_memory[:, :, enemy_id] = np.where(
                hit_enemy,
                self.cooperative_kill_window,
                updated_memory[:, :, enemy_id],
            )

        killed_mask = prev_enemy_alive & (~next_enemy_alive)
        enemy_max_health = np.maximum(self._unit_type_health_np[enemy_unit_types], 1.0e-8)
        damage = np.maximum(prev_enemy_health - next_enemy_health, 0.0)
        damage_ratio = damage / enemy_max_health
        damaged_mask = damage > 1.0e-8
        recent_attacks = np.transpose(updated_memory > 0, (0, 2, 1))
        coop_ready = np.all(
            np.where(self._coop_required_mask_np[None, :, :], recent_attacks, True),
            axis=-1,
        )
        if self.cooperative_bonus_mode == "kill":
            coop_success = killed_mask & coop_ready
            event_scale = coop_success.astype(np.float32) * self.cooperative_kill_bonus
        else:
            coop_success = damaged_mask & coop_ready
            event_scale = damage_ratio.astype(np.float32) * coop_success.astype(np.float32) * self.cooperative_kill_bonus
        bonus = event_scale @ self._coop_reward_weights_np
        return bonus.astype(np.float32), updated_memory, coop_success

    def _apply_cooperative_kill_bonus_jax(
        self,
        attack_memory,
        actions,
        enemy_unit_types,
        prev_enemy_health,
        next_enemy_health,
        prev_enemy_alive,
        next_enemy_alive,
    ):
        if not self.use_cooperative_kill_bonus:
            zero_bonus = self._jnp.zeros((actions.shape[0], self.num_agents), dtype=self._jnp.float32)
            zero_success = self._jnp.zeros((actions.shape[0], self.num_enemies), dtype=bool)
            return zero_bonus, attack_memory, zero_success

        updated_memory = self._jnp.maximum(attack_memory - 1, 0)
        attack_mask = actions >= self._num_movement_actions
        attack_targets = self._jnp.clip(actions - self._num_movement_actions, 0, self.num_enemies - 1)
        current_targets = self._jax.nn.one_hot(
            attack_targets,
            self.num_enemies,
            dtype=self._jnp.int32,
        )
        current_targets = (
            current_targets
            * attack_mask[..., None].astype(self._jnp.int32)
            * self.cooperative_kill_window
        )
        updated_memory = self._jnp.maximum(updated_memory, current_targets)

        killed_mask = prev_enemy_alive & (~next_enemy_alive)
        enemy_max_health = self._jnp.maximum(self._unit_type_health_jax[enemy_unit_types], 1.0e-8)
        damage = self._jnp.maximum(prev_enemy_health - next_enemy_health, 0.0)
        damage_ratio = damage / enemy_max_health
        damaged_mask = damage > 1.0e-8
        recent_attacks = self._jnp.transpose(updated_memory > 0, (0, 2, 1))
        coop_ready = self._jnp.all(
            self._jnp.where(self._coop_required_mask_jax[None, :, :], recent_attacks, True),
            axis=-1,
        )
        if self.cooperative_bonus_mode == "kill":
            coop_success = killed_mask & coop_ready
            event_scale = coop_success.astype(self._jnp.float32) * self.cooperative_kill_bonus
        else:
            coop_success = damaged_mask & coop_ready
            event_scale = damage_ratio.astype(self._jnp.float32) * coop_success.astype(self._jnp.float32) * self.cooperative_kill_bonus
        bonus = event_scale @ self._coop_reward_weights_jax
        return bonus, updated_memory, coop_success

    def _compute_damage_masked_enemy_mask_jax(self, actions):
        if not self.use_allowed_targets or self.allowed_targets is None:
            return self._jnp.zeros((actions.shape[0], self.num_enemies), dtype=bool)

        attack_mask = actions >= self._num_movement_actions
        attack_targets = self._jnp.clip(
            actions - self._num_movement_actions,
            0,
            self.num_enemies - 1,
        )
        attack_one_hot = self._jax.nn.one_hot(
            attack_targets,
            self.num_enemies,
            dtype=self._jnp.float32,
        )
        attack_one_hot = attack_one_hot * attack_mask[..., None].astype(self._jnp.float32)
        off_target_counts = self._jnp.sum(
            attack_one_hot * (1.0 - self._allowed_target_matrix_jax[None, :, :]),
            axis=1,
        )
        return off_target_counts > 0.0

    def _compute_team_rewards_jax(self, prev_env_state, next_env_state):
        health_before = prev_env_state.state.unit_health
        health_after = next_env_state.state.unit_health
        unit_types = next_env_state.state.unit_types
        normalized_delta = (health_after - health_before) / self._unit_type_health_jax[unit_types]
        enemy_health_decrease = self._jnp.sum(
            normalized_delta[:, self._env.num_allies:],
            axis=1,
        )
        enemy_health_decrease_reward = self._jnp.abs(enemy_health_decrease) / float(self.num_enemies)
        if self._use_self_play_reward:
            enemy_health_decrease_reward = self._jnp.zeros_like(enemy_health_decrease_reward)

        alive = next_env_state.state.unit_alive
        won_battle = self._jnp.all(~alive[:, self._env.num_allies:], axis=1)
        lost_battle = self._jnp.all(~alive[:, :self._env.num_allies], axis=1)
        lost_battle_bonus = self._jnp.where(
            lost_battle & self._use_self_play_reward & ~won_battle,
            -self._won_battle_bonus,
            0.0,
        )
        won_battle_bonus = self._jnp.where(
            won_battle & ~lost_battle,
            self._won_battle_bonus,
            0.0,
        )
        team_rewards = enemy_health_decrease_reward + won_battle_bonus + lost_battle_bonus
        return self._jnp.broadcast_to(team_rewards[:, None], (team_rewards.shape[0], self.num_agents))

    # ------------------------------------------------------------------
    # ParallelEnvWrapper 인터페이스
    # ------------------------------------------------------------------

    def reset(self):
        """
        모든 환경을 리셋합니다.
        Returns:
            obs_list:   List[Dict[int, np.ndarray]]   길이 n_envs
            avail_list: List[Dict[int, List[int]]]    길이 n_envs
        """
        self._rng, *env_keys = self._jax.random.split(self._rng, self.n_envs + 1)
        keys_arr = self._jnp.stack(env_keys)

        obs_b, self._batch_states = self._batch_reset(keys_arr)
        self._jax.block_until_ready(obs_b)
        self._coop_attack_memory_np = np.zeros(
            (self.n_envs, self.num_agents, self.num_enemies),
            dtype=np.int32,
        )

        avail_b = self._batch_avail(self._batch_states)
        return self._obs_to_list(obs_b), self._avail_to_list(avail_b)

    def step(self, actions_batch):
        """
        모든 환경을 한 스텝 진행합니다. 에피소드 종료 시 자동 리셋.

        Args:
            actions_batch: List[Dict[int, int]]    길이 n_envs

        Returns: (obs, rewards, dones, infos, avail) 각각 길이 n_envs인 리스트
            obs: 에피소드 종료 환경은 새 에피소드의 첫 관찰 반환
        """
        import numpy as np

        jax_actions = self._actions_to_jax(actions_batch)
        actions_np = np.stack(
            [np.array(jax_actions[self._id_to_key[agent_id]], dtype=np.int32) for agent_id in range(self.num_agents)],
            axis=1,
        )
        prev_enemy_alive = np.array(
            self._batch_states.state.unit_alive[:, self._env.num_allies:],
            dtype=np.bool_,
        )
        prev_enemy_health = np.array(
            self._batch_states.state.unit_health[:, self._env.num_allies:],
            dtype=np.float32,
        )
        enemy_unit_types = np.array(
            self._batch_states.state.unit_types[:, self._env.num_allies:],
            dtype=np.int32,
        )

        self._rng, *step_keys = self._jax.random.split(self._rng, self.n_envs + 1)
        step_keys_arr = self._jnp.stack(step_keys)

        _, new_states, _, _, _ = self._batch_step(
            step_keys_arr, self._batch_states, jax_actions
        )
        corrected_states = new_states
        damage_masked = self._compute_damage_masked_enemy_mask_jax(
            self._jnp.array(actions_np, dtype=self._jnp.int32)
        )
        obs_b = self._batch_obs(corrected_states)
        self._jax.block_until_ready(obs_b)
        next_enemy_alive = np.array(
            corrected_states.state.unit_alive[:, self._env.num_allies:],
            dtype=np.bool_,
        )
        next_enemy_health = np.array(
            corrected_states.state.unit_health[:, self._env.num_allies:],
            dtype=np.float32,
        )
        rew_stacked = np.array(
            self._compute_team_rewards_jax(self._batch_states, corrected_states),
            dtype=np.float32,
        )
        coop_bonus, self._coop_attack_memory_np, _ = self._apply_cooperative_kill_bonus_numpy_batch(
            self._coop_attack_memory_np,
            actions_np,
            enemy_unit_types,
            prev_enemy_health,
            next_enemy_health,
            prev_enemy_alive,
            next_enemy_alive,
        )
        rew_stacked = rew_stacked + coop_bonus

        all_done_np = np.array(self._batch_terminal(corrected_states))   # (n_envs,) bool
        if np.any(all_done_np):
            self._coop_attack_memory_np[all_done_np] = 0

        # 에피소드 종료 환경 자동 리셋 ─────────────────────────────
        if np.any(all_done_np):
            self._rng, *reset_keys = self._jax.random.split(self._rng, self.n_envs + 1)
            reset_keys_arr = self._jnp.stack(reset_keys)
            reset_obs_b, reset_states = self._batch_reset(reset_keys_arr)
            self._jax.block_until_ready(reset_obs_b)

            done_flags = self._jnp.array(all_done_np)   # (n_envs,)

            def _select(new, reset):
                # done=True → reset 선택, done=False → new 유지
                cond = done_flags.reshape((-1,) + (1,) * (new.ndim - 1))
                return self._jnp.where(cond, reset, new)

            self._batch_states = self._jax.tree_map(_select, corrected_states, reset_states)
            final_obs_b = _select(
                self._stack_agent_dict(obs_b),
                self._stack_agent_dict(reset_obs_b),
            )
        else:
            self._batch_states = corrected_states
            final_obs_b = self._stack_agent_dict(obs_b)
        # ──────────────────────────────────────────────────────────

        avail_b = self._batch_avail(self._batch_states)
        damage_masked_np = np.array(damage_masked)
        damage_masked_lists = [np.where(damage_masked_np[e])[0].tolist() for e in range(self.n_envs)]

        agent_dones_np = np.logical_not(
            np.array(corrected_states.state.unit_alive[:, : self.num_agents])
        )

        return (
            self._obs_to_list(final_obs_b),
            [
                {agent_id: float(rew_stacked[e, agent_id]) for agent_id in range(self.num_agents)}
                for e in range(self.n_envs)
            ],
            self._dones_to_list(agent_dones_np),
            self._infos_to_list(corrected_states, all_done_np, damage_masked_lists),
            self._avail_to_list(avail_b),
        )

    def get_avail_actions_batch(self):
        """사용 가능한 액션 마스크를 모든 환경에서 가져옵니다."""
        avail_b = self._batch_avail(self._batch_states)
        return self._avail_to_list(avail_b)

    def _stack_agent_dict(self, data_b):
        return self._jnp.stack([data_b[key] for key in self._agent_keys], axis=1)

    def _mask_avail_jax(self, avail_stacked):
        """Damage masking은 action availability를 직접 바꾸지 않습니다."""
        return avail_stacked

    def reset_jax(self):
        self._rng, batch_state, obs, avail = self.reset_jax_pure(self._rng)
        self._batch_states = batch_state["env_state"]
        self._coop_attack_memory_jax = batch_state["coop_attack_memory"]
        return obs, avail

    def reset_jax_pure(self, rng):
        rng, *env_keys = self._jax.random.split(rng, self.n_envs + 1)
        keys_arr = self._jnp.stack(env_keys)
        return rng, *self.reset_jax_pure_from_keys(keys_arr)

    def reset_jax_pure_from_keys(self, keys_arr):
        if keys_arr.shape[0] != self.n_envs:
            raise ValueError(f"Expected {self.n_envs} reset keys, got {keys_arr.shape[0]}")
        obs_b, env_state = self._batch_reset(keys_arr)
        coop_attack_memory = self._jnp.zeros(
            (self.n_envs, self.num_agents, self.num_enemies),
            dtype=self._jnp.int32,
        )
        avail_b = self._batch_avail(env_state)
        return self._pack_batch_state(env_state, coop_attack_memory), self._stack_agent_dict(obs_b), self._mask_avail_jax(self._stack_agent_dict(avail_b))

    def step_jax(self, actions):
        batch_state = self._pack_batch_state(self._batch_states, self._coop_attack_memory_jax)
        (
            self._rng,
            next_batch_state,
            obs,
            avail,
            rewards,
            dones,
            dones_all,
            battle_won,
        ) = self.step_jax_pure(self._rng, batch_state, actions)
        self._batch_states = next_batch_state["env_state"]
        self._coop_attack_memory_jax = next_batch_state["coop_attack_memory"]
        return obs, avail, rewards, dones, dones_all, battle_won

    def step_jax_pure(self, rng, batch_states, actions):
        env_state, coop_attack_memory = self._unpack_batch_state(batch_states)
        jax_actions = {
            self._id_to_key[i]: actions[:, i].astype(self._jnp.int32)
            for i in range(self.num_agents)
        }

        rng, *step_keys = self._jax.random.split(rng, self.n_envs + 1)
        step_keys_arr = self._jnp.stack(step_keys)
        prev_enemy_alive = env_state.state.unit_alive[:, self._env.num_allies:]
        enemy_unit_types = env_state.state.unit_types[:, self._env.num_allies:]
        prev_enemy_health = env_state.state.unit_health[:, self._env.num_allies:]
        _, next_env_state, _, _, _ = self._batch_step(
            step_keys_arr,
            env_state,
            jax_actions,
        )
        obs_b = self._batch_obs(next_env_state)
        agent_dones = ~next_env_state.state.unit_alive[:, : self.num_agents]
        next_enemy_health = next_env_state.state.unit_health[:, self._env.num_allies:]
        next_enemy_alive = next_env_state.state.unit_alive[:, self._env.num_allies:]
        battle_won = ~self._jnp.any(next_enemy_alive, axis=-1)
        rewards_stacked = self._compute_team_rewards_jax(env_state, next_env_state)
        coop_bonus, coop_attack_memory, _ = self._apply_cooperative_kill_bonus_jax(
            coop_attack_memory,
            actions,
            enemy_unit_types,
            prev_enemy_health,
            next_enemy_health,
            prev_enemy_alive,
            next_enemy_alive,
        )
        rewards_stacked = rewards_stacked + coop_bonus

        rng, *reset_keys = self._jax.random.split(rng, self.n_envs + 1)
        reset_keys_arr = self._jnp.stack(reset_keys)
        reset_obs_b, reset_states = self._batch_reset(reset_keys_arr)
        done_flags = self._batch_terminal(next_env_state)

        def _select(new, reset):
            cond = done_flags.reshape((-1,) + (1,) * (new.ndim - 1))
            return self._jnp.where(cond, reset, new)

        final_env_state = self._jax.tree_map(_select, next_env_state, reset_states)
        final_obs_b = _select(
            self._stack_agent_dict(obs_b),
            self._stack_agent_dict(reset_obs_b),
        )
        avail_b = self._batch_avail(final_env_state)
        coop_attack_memory = self._jnp.where(
            done_flags[:, None, None],
            self._jnp.zeros_like(coop_attack_memory),
            coop_attack_memory,
        )

        return (
            rng,
            self._pack_batch_state(final_env_state, coop_attack_memory),
            final_obs_b,
            self._mask_avail_jax(self._stack_agent_dict(avail_b)),
            rewards_stacked,
            agent_dones,
            done_flags,
            battle_won,
        )

    def step_jax_pure_from_keys(self, rng_keys, batch_states, actions):
        if rng_keys.shape[0] != self.n_envs:
            raise ValueError(f"Expected {self.n_envs} RNG keys, got {rng_keys.shape[0]}")

        env_state, coop_attack_memory = self._unpack_batch_state(batch_states)
        jax_actions = {
            self._id_to_key[i]: actions[:, i].astype(self._jnp.int32)
            for i in range(self.num_agents)
        }

        step_split = self._jax.vmap(lambda key: self._jax.random.split(key, 2))(rng_keys)
        next_rng_keys = step_split[:, 0]
        step_keys_arr = step_split[:, 1]
        prev_enemy_alive = env_state.state.unit_alive[:, self._env.num_allies:]
        enemy_unit_types = env_state.state.unit_types[:, self._env.num_allies:]
        prev_enemy_health = env_state.state.unit_health[:, self._env.num_allies:]
        _, next_env_state, _, _, _ = self._batch_step(
            step_keys_arr,
            env_state,
            jax_actions,
        )
        obs_b = self._batch_obs(next_env_state)
        agent_dones = ~next_env_state.state.unit_alive[:, : self.num_agents]
        next_enemy_health = next_env_state.state.unit_health[:, self._env.num_allies:]
        next_enemy_alive = next_env_state.state.unit_alive[:, self._env.num_allies:]
        battle_won = ~self._jnp.any(next_enemy_alive, axis=-1)
        rewards_stacked = self._compute_team_rewards_jax(env_state, next_env_state)
        coop_bonus, coop_attack_memory, _ = self._apply_cooperative_kill_bonus_jax(
            coop_attack_memory,
            actions,
            enemy_unit_types,
            prev_enemy_health,
            next_enemy_health,
            prev_enemy_alive,
            next_enemy_alive,
        )
        rewards_stacked = rewards_stacked + coop_bonus

        reset_split = self._jax.vmap(lambda key: self._jax.random.split(key, 2))(next_rng_keys)
        final_rng_keys = reset_split[:, 0]
        reset_keys_arr = reset_split[:, 1]
        reset_obs_b, reset_states = self._batch_reset(reset_keys_arr)
        done_flags = self._batch_terminal(next_env_state)

        def _select(new, reset):
            cond = done_flags.reshape((-1,) + (1,) * (new.ndim - 1))
            return self._jnp.where(cond, reset, new)

        final_env_state = self._jax.tree_map(_select, next_env_state, reset_states)
        final_obs_b = _select(
            self._stack_agent_dict(obs_b),
            self._stack_agent_dict(reset_obs_b),
        )
        avail_b = self._batch_avail(final_env_state)
        coop_attack_memory = self._jnp.where(
            done_flags[:, None, None],
            self._jnp.zeros_like(coop_attack_memory),
            coop_attack_memory,
        )

        return (
            final_rng_keys,
            self._pack_batch_state(final_env_state, coop_attack_memory),
            final_obs_b,
            self._mask_avail_jax(self._stack_agent_dict(avail_b)),
            rewards_stacked,
            agent_dones,
            done_flags,
            battle_won,
        )

    # ------------------------------------------------------------------
    # 변환 헬퍼
    # ------------------------------------------------------------------

    def _obs_to_list(self, obs_b) -> List:
        import numpy as np
        if isinstance(obs_b, dict):
            obs_np = {k: np.array(obs_b[k]) for k in self._agent_keys}
            return [
                {i: obs_np[k][e].astype(np.float32) for i, k in enumerate(self._agent_keys)}
                for e in range(self.n_envs)
            ]

        obs_np = np.array(obs_b)
        return [
            {i: obs_np[e, i].astype(np.float32) for i in range(self.num_agents)}
            for e in range(self.n_envs)
        ]

    def _dones_to_list(self, done_np) -> List:
        return [{i: bool(done_np[e, i]) for i in range(self.num_agents)}
                for e in range(self.n_envs)]

    def _infos_to_list(self, new_states, all_done_np, damage_masked_lists=None) -> List:
        import numpy as np
        enemies_alive = np.array(
            new_states.state.unit_alive[:, self._env.num_allies:]  # (n_envs, n_enemies)
        )
        battle_won = ~np.any(enemies_alive, axis=1)   # (n_envs,)
        info_list = []
        for e in range(self.n_envs):
            info = {'battle_won': bool(battle_won[e]), 'episode_done': bool(all_done_np[e])}
            if damage_masked_lists is not None:
                info['damage_masked_enemies'] = damage_masked_lists[e]
            info_list.append(info)
        return info_list

    def _avail_to_list(self, avail_b) -> List:
        import numpy as np
        avail_np = {k: np.array(avail_b[k]) for k in self._agent_keys}
        return [
            {i: avail_np[k][e].tolist() for i, k in enumerate(self._agent_keys)}
            for e in range(self.n_envs)
        ]

    def _actions_to_jax(self, actions_batch):
        """List[Dict[int, int]] → Dict[str, jnp.ndarray (n_envs,)]"""
        return {
            key: self._jnp.array(
                [int(actions_batch[e][agent_id]) for e in range(self.n_envs)],
                dtype=self._jnp.int32,
            )
            for agent_id, key in self._id_to_key.items()
        }

    # ------------------------------------------------------------------
    # MultiAgentEnv 인터페이스
    # ------------------------------------------------------------------

    def get_obs_dim(self) -> int:
        return self.obs_dim

    def get_action_dim(self) -> int:
        return self.action_dim

    def close(self):
        pass
