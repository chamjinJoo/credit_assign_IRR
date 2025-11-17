import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import seaborn as sns
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import os
import json
from collections import defaultdict

if TYPE_CHECKING:
    from ..algorithms.pci_mappo import PCIMAPPOAlgorithm
    from ..envs.multi_agent_env import StagHuntEnv


class AdvancedCausalEvaluator:
    """
    PCI-MAPPO의 causal effect 학습을 심층 분석하는 고급 평가 클래스.
    
    Features:
    1. Agent별, Stag별 세분화된 causal effect 분석
    2. 협력 파트너별 causal effect 비교
    3. Policy bias 분석 (object 주변에서의 행동 경향)
    """
    
    def __init__(
        self,
        algorithm: "PCIMAPPOAlgorithm",
        env: "StagHuntEnv",
        num_eval_steps: int = 5000,
        device: str = "cpu"
    ):
        self.algorithm = algorithm
        self.env = env
        self.num_eval_steps = num_eval_steps
        self.device = device
        
        # Stag-hunt specific parameters
        self.num_agents = env.num_agents
        self.stag_reward = env.stag_reward
        self.forage_reward = env.forage_reward
        self.mauling_punishment = env.mauling_punishment
        
        # Required agents for each stag (from environment)
        self.required_agents = env.required_agents
        
        # Results storage
        self.detailed_causal_effects = defaultdict(lambda: defaultdict(list))
        self.policy_bias_data = defaultdict(lambda: defaultdict(list))
        self.object_choice_data = defaultdict(lambda: defaultdict(list))  # 새로운 데이터 구조
        
        self.direction_names = ['towards', 'away', 'stay', 'neutral']

        
    def evaluate_detailed_causal_effects(self) -> Dict:
        """
        지정된 수의 환경 상호작용을 실행하여 Agent별, Stag별로 세분화된 causal effect를 평가합니다.
        
        Returns:
            Dict containing detailed statistics for each agent and scenario
        """
        print(f"상세 Causal Effect 평가 시작: {self.num_eval_steps} 스텝")
        
        self.algorithm.outcome_bridge_net.eval()
        self.algorithm.action_bridge_net.eval()
        
        # 환경 초기화
        observations = self.env.reset()
        prev_observations = {agent_id: np.zeros_like(obs) for agent_id, obs in observations.items()}
        
        step_count = 0
        episode_count = 0
        
        while step_count < self.num_eval_steps:
            if step_count % 1000 == 0:
                print(f"스텝 {step_count}/{self.num_eval_steps} 진행 중... (에피소드 {episode_count})")
            
            # 관찰을 torch tensor로 변환
            torch_observations = {}
            for agent_id, obs in observations.items():
                torch_observations[agent_id] = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                )
            
            # 액션 선택
            actions = self.algorithm.get_actions(torch_observations)
            action_dict = {agent_id: action for agent_id, (action, _) in actions.items()}
            
            # 환경 스텝
            next_observations, rewards, dones, info = self.env.step(action_dict)
            
            # 상세한 causal effect와 policy bias 수집
            self._collect_detailed_data(prev_observations, observations, action_dict, rewards, info)
            
            # 상태 업데이트
            prev_observations = observations
            observations = next_observations
            step_count += 1
            
            # 에피소드가 끝났으면 새 에피소드 시작
            if all(dones.values()):
                observations = self.env.reset()
                prev_observations = {agent_id: np.zeros_like(obs) for agent_id, obs in observations.items()}
                episode_count += 1
        
        statistics = self._compute_detailed_statistics()
        
        print(f"상세 Causal Effect 평가 완료! 총 {step_count} 스텝, {episode_count} 에피소드 실행")
        return statistics
    
    def _collect_detailed_data(
        self,
        prev_observations: Dict[int, np.ndarray],
        observations: Dict[int, np.ndarray],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        info: Dict
    ):
        """단일 스텝에서 상세한 causal effect와 policy bias를 수집합니다."""
        # Agent 위치 업데이트 (policy bias 분석용)
        if 'agent_positions' in info:
            self.agent_positions = info['agent_positions'].copy()
        
        # Causal effect 추정
        causal_effects = self._estimate_causal_effects(
            prev_observations, observations, actions, rewards, info
        )
        
        # 상세 시나리오별로 분류하여 저장
        self._classify_and_store_detailed_effects(
            causal_effects, rewards, info, actions, observations
        )
        
        # Policy bias 데이터 수집
        self._collect_policy_bias_data(observations, actions, info)
    
    def _estimate_causal_effects(
        self,
        prev_observations: Dict[int, np.ndarray],
        observations: Dict[int, np.ndarray],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        info: Dict
    ) -> Dict[int, float]:
        """각 agent의 causal effect를 추정합니다."""
        causal_effects = {}
        
        with torch.no_grad():
            torch_prev_observations = {}
            torch_observations = {}
            for agent_id in range(self.num_agents):
                torch_prev_observations[agent_id] = torch.tensor(
                    prev_observations[agent_id], dtype=torch.float32, device=self.device
                )
                torch_observations[agent_id] = torch.tensor(
                    observations[agent_id], dtype=torch.float32, device=self.device
                )
            
            z_t = self.algorithm._concatenate_observations(torch_prev_observations)
            w_t = self.algorithm._concatenate_observations(torch_observations)
            
            for agent_id in range(self.num_agents):
                delta_t, _ = self.algorithm.causal_intrinsic_reward(
                    agent_id=agent_id,
                    w_t=w_t,
                    z_t=z_t,
                    o_t=torch_observations[agent_id],
                    current_actions=actions,
                    current_rewards=rewards
                )
                causal_effects[agent_id] = delta_t
        
        return causal_effects
    
    def _classify_and_store_detailed_effects(
        self,
        causal_effects: Dict[int, float],
        rewards: Dict[int, float],
        info: Dict,
        actions: Dict[int, int],
        observations: Dict[int, np.ndarray]
    ):
        """
        Causal effect를 상세하게 분류하여 저장합니다.
        
        분류 기준:
        - Agent별
        - 자신과 관련된 stag의 보상 vs 관련 없는 agent의 보상
        - 협력 파트너의 보상
        - Hare 사냥 보상
        """
        stag_hunts = info.get('stag_hunts', {})
        hare_hunts = info.get('hare_hunts', [])
        
        for agent_id in range(self.num_agents):
            delta = causal_effects[agent_id]
            
            # 1. 자신이 협력 사냥에 참여했는지
            participated_in_stag = False
            participated_stag_id = None
            for stag_id, participants in stag_hunts.items():
                if agent_id in participants:
                    participated_in_stag = True
                    participated_stag_id = stag_id
                    break
            
            # 2. 각 다른 agent의 보상 상황 분석
            for other_agent_id in range(self.num_agents):
                if other_agent_id == agent_id:
                    continue
                
                other_reward = rewards[other_agent_id]
                
                # 협력 관계 파악
                is_cooperation_partner = False
                related_stag_id = None
                
                # agent_id가 참여할 수 있는 stag 찾기
                for stag_id, required_agents in self.required_agents.items():
                    if agent_id in required_agents and other_agent_id in required_agents:
                        is_cooperation_partner = True
                        related_stag_id = stag_id
                        break
                
                # 시나리오 분류
                if other_reward == self.stag_reward:
                    if is_cooperation_partner and participated_in_stag:
                        # 협력 파트너가 stag reward를 받았고, 내가 협력에 참여함
                        scenario = f'agent_{agent_id}_partner_stag_cooperative'
                    else:
                        # 협력 관계가 아닌 다른 agent의 stag reward
                        scenario = f'agent_{agent_id}_nonpartner_stag'
                elif other_reward == self.forage_reward:
                    if is_cooperation_partner:
                        # 협력 파트너가 hare를 사냥함 (배신)
                        scenario = f'agent_{agent_id}_partner_hare_defection'
                    else:
                        # 관련 없는 agent의 hare 사냥
                        scenario = f'agent_{agent_id}_nonpartner_hare'
                else:
                    # 다른 agent가 보상을 받지 못함
                    if is_cooperation_partner:
                        scenario = f'agent_{agent_id}_partner_no_reward'
                    else:
                        scenario = f'agent_{agent_id}_nonpartner_no_reward'
                
                self.detailed_causal_effects[agent_id][scenario].append(delta)
    
    def _get_action_direction(self, action: int) -> Tuple[int, int]:
        """
        행동에 따른 이동 방향 벡터를 반환합니다.
        
        Returns:
            (dx, dy) 이동 벡터
        """
        # StagHunt 환경의 action: 0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
        action_to_direction = {
            0: (0, 0),   # STAY
            1: (0, 1),   # UP
            2: (0, -1),  # DOWN
            3: (-1, 0),  # LEFT
            4: (1, 0)    # RIGHT
        }
        return action_to_direction.get(action, (0, 0))
    
    def _euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """유클리드 거리 계산"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
    
    def _is_action_towards_object(
        self, 
        agent_pos: Tuple[int, int], 
        object_pos: Tuple[int, int], 
        action: int
    ) -> str:
        """
        행동이 object를 향하는지, 멀어지는지, 중립적인지 판단
        
        Returns:
            'towards': object에 가까워지는 행동 또는 같은 위치에서 STAY (object 유지)
            'away': object에서 멀어지는 행동
            'stay': 다른 위치에서 STAY 행동
            'neutral': 거리 변화가 없음 (수학적으로 같은 거리)
        """
        dx, dy = self._get_action_direction(action)
        
        # STAY인 경우
        if dx == 0 and dy == 0:
            # Agent와 object가 같은 위치에 있으면 towards (object를 유지)
            if agent_pos == object_pos:
                return 'towards'
            # 다른 위치에 있으면 stay
            return 'stay'
        
        # Agent와 object가 이미 같은 위치에 있는 경우
        if agent_pos == object_pos:
            # 같은 위치에서 이동하면 무조건 멀어짐
            return 'away'
        
        # 행동 후 예상 위치
        new_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
        
        # 현재 거리와 행동 후 거리 비교 (Euclidean distance)
        current_dist = self._euclidean_distance(agent_pos, object_pos)
        new_dist = self._euclidean_distance(new_pos, object_pos)
        
        # 거리 차이 계산
        dist_change = new_dist - current_dist
    
        if dist_change < 0:
            return 'towards'
        elif dist_change > 0:
            return 'away'
        else:
            # 거리 변화가 거의 없음 (수학적으로 동일한 거리)
            return 'neutral'
    
    def _parse_observation(self, obs: np.ndarray, agent_id: int) -> Dict:
        """
        Agent의 observation을 파싱하여 3x3 grid 정보를 추출합니다.
        
        Observation 구조:
        - coord_obs: [agent_x, agent_y] (2차원)
        - grid_obs: 3x3 grid의 각 cell에 (agent0, ..., stag0, ..., hare) 정보 (9 * (2*num_agents+1)차원)
        - relative_positions: 다른 agent들의 상대적 위치 (2*(num_agents-1)차원)
        
        Returns:
            Dict containing:
            - 'visible_stags': List[(stag_id, relative_pos), ...]
            - 'visible_hares': List[relative_pos, ...]
            - 'partner_in_vision': Dict[partner_id: relative_pos]
        """
        # Observation 파싱
        coord_size = 2
        cell_size = 2 * self.num_agents + 1  # (agents + stags + hare)
        grid_size = 9 * cell_size
        
        # coord_obs = obs[:coord_size]
        grid_obs = obs[coord_size:coord_size + grid_size]
        # relative_positions = obs[coord_size + grid_size:]
        
        # 3x3 grid 파싱 (cell order: [-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1])
        grid_positions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        visible_stags = []
        visible_hares = []
        partner_in_vision = {}
        
        for idx, (dx, dy) in enumerate(grid_positions):
            cell_start = idx * cell_size
            cell_info = grid_obs[cell_start:cell_start + cell_size]
            
            # 경계 밖인 경우 건너뛰기
            if cell_info[0] == -1.0:
                continue
            
            # Agent 정보 (first num_agents elements)
            for other_agent_id in range(self.num_agents):
                if cell_info[other_agent_id] == 1.0 and other_agent_id != agent_id:
                    # 협력 파트너인지 확인
                    for stag_id, required_agents in self.required_agents.items():
                        if agent_id in required_agents and other_agent_id in required_agents:
                            partner_in_vision[other_agent_id] = (dx, dy)
                            break
            
            # Stag 정보 (next num_agents elements)
            for stag_id in range(self.num_agents):
                stag_idx = self.num_agents + stag_id
                if cell_info[stag_idx] == 1.0:
                    visible_stags.append((stag_id, (dx, dy)))
            
            # Hare 정보 (last element)
            hare_idx = 2 * self.num_agents
            if cell_info[hare_idx] == 1.0:
                visible_hares.append((dx, dy))
        
        return {
            'visible_stags': visible_stags,
            'visible_hares': visible_hares,
            'partner_in_vision': partner_in_vision
        }
    
    def _collect_policy_bias_data(
        self,
        observations: Dict[int, np.ndarray],
        actions: Dict[int, int],
        info: Dict
    ):
        """
        Policy bias 데이터를 수집합니다 (huntable_objects 기반).
        
        수집 기준:
        - info의 huntable_objects를 사용하여 사냥 가능한 objects만 분석
        - 실제 사냥 가능한 상황에서의 선택 패턴 분석
        
        Object 분류 (huntable_objects 기반):
        - my_stag: 자신이 보상을 받는 stag (stag_id == agent_id)
        - coop_stag: 협력에 참여하는 다른 stag (agent_id in required_agents but stag_id != agent_id)
        - hare: 사냥 가능한 hare
        
        조합 분석:
        - huntable_objects의 실제 조합에 따라 case 구분
        - attempts 기반으로 어떤 object를 시도했는지 기록 (my/coop/hare/none)
        """
        
        # Huntable objects 정보 추출 (사냥 전 시점 기준, env에서 보장)
        huntable_objects = info.get('huntable_objects', {})
        huntable_stags = huntable_objects.get('huntable_stags', {})
        huntable_hares = huntable_objects.get('huntable_hares', [])

        # 시도 정보 추출 (케이스별 검증용)
        stag_attempts = info.get('stag_attempts', {})  # {stag_id: [attempting_agent_ids]}
        hare_attempts = set(info.get('hare_attempts', []))  # [agent_ids]

        # 케이스별 상세 기록 저장소 초기화 (처음 호출 시)
        if not hasattr(self, 'object_choice_cases'):
            self.object_choice_cases = []
        # 케이스별 attempt 집계 저장소 초기화
        if not hasattr(self, 'case_attempt_data'):
            self.case_attempt_data = defaultdict(lambda: defaultdict(list))
        
        for agent_id in range(self.num_agents):
            if agent_id not in observations:
                continue
            
            agent_pos = info['agent_positions'][agent_id]
            
            # 이 agent가 사냥 가능한 objects 식별
            my_stag_id = None
            coop_stag_ids = []
            other_stag_ids = []
            hare_ids = []
            agents_with_hare_opportunity = set()
            
            # My stag (agent_id == stag_id)
            if agent_id in huntable_stags:
                my_stag_id = agent_id
            
            # Coop stags (agent_id in required_agents but stag_id != agent_id)
            for stag_id, stag_info in huntable_stags.items():
                if stag_id != agent_id and agent_id in stag_info['required_agents']:
                    coop_stag_ids.append(stag_id)
            
            # Other stags (required_agents에 포함되지 않은 stag 중 이 agent에게 huntable로 기록된 것)
            other_stags_by_agent = huntable_objects.get('other_stags_by_agent', {})
            if agent_id in other_stags_by_agent:
                other_stag_ids = list(other_stags_by_agent[agent_id].keys())
            
            # Hares (agent가 사냥 가능한 hare들)
            for hare_info in huntable_hares:
                hunting_agents = hare_info.get('hunting_agents', [])
                for a_id in hunting_agents:
                    agents_with_hare_opportunity.add(a_id)
                if agent_id in hunting_agents:
                    hare_ids.append(hare_info['hare_id'])
            
            # Object 조합 key 생성 (my_stag, coop_stag, other_stag, hare)
            objects_present = []
            if my_stag_id is not None:
                objects_present.append('my_stag')
            if coop_stag_ids:
                objects_present.append('coop_stag')
            if other_stag_ids:
                objects_present.append('other_stag')
            if hare_ids:
                objects_present.append('hare')
            
            if not objects_present:
                continue
            
            # 일관된 순서로 정렬하여 case key 생성
            priority_order = ['my_stag', 'coop_stag', 'other_stag', 'hare']
            sorted_objects = [obj for obj in priority_order if obj in objects_present]
            case_key = '+'.join(sorted_objects)
            
            attempted_my_stag = False
            attempted_coop_stag = False 
            attempted_other_stag = False
            attempted_hare = False
            
            # Stag attempts: 해당 stag가 huntable이고 agent가 required_agents에 포함 + 실제 시도 목록에 있을 때
            if stag_attempts and huntable_stags:
                for s_id, attempting_agents in stag_attempts.items():
                    if attempting_agents and s_id in huntable_stags:
                        req_agents = huntable_stags[s_id].get('required_agents', [])
                        if agent_id in attempting_agents and agent_id in req_agents:
                            if s_id == agent_id:
                                attempted_my_stag = True
                                break  # 한 agent는 한 스텝에 하나의 stag만 시도 가능
                            else:
                                attempted_coop_stag = True
                                break  # 한 agent는 한 스텝에 하나의 stag만 시도 가능
            # Other stag attempts: other_stags_by_agent에 기록된 해당 stag를 시도했는지 확인
            if not (attempted_my_stag or attempted_coop_stag):
                if 'other_stags_by_agent' in huntable_objects:
                    agent_other = huntable_objects['other_stags_by_agent'].get(agent_id, {})
                    for s_id, attempting_agents in stag_attempts.items():
                        if s_id in agent_other and agent_id in attempting_agents:
                            attempted_other_stag = True
                            break
            # Hare attempts
            if agent_id in hare_attempts and agent_id in agents_with_hare_opportunity:
                attempted_hare = True

            # Attempt 유형 코드화: 0=my_stag, 1=coop_stag, 2=other_stag, 3=hare, 4=none
            # 우선순위: my_stag > coop_stag > other_stag > hare > none
            if attempted_my_stag:
                attempt_type = 'my_stag'
                attempt_code = 0
            elif attempted_coop_stag:
                attempt_type = 'coop_stag'
                attempt_code = 1
            elif attempted_other_stag:
                attempt_type = 'other_stag'
                attempt_code = 2
            elif attempted_hare:
                attempt_type = 'hare'
                attempt_code = 3
            else:
                attempt_type = 'none'
                attempt_code = 4

            # Case별 attempt 분포 집계 (agent, case_key)
            self.case_attempt_data[agent_id][case_key].append(attempt_code)
            self.object_choice_data[agent_id][case_key].append(attempt_code)

            self.object_choice_cases.append({
                'step_count': info.get('step_count'),
                'agent_id': agent_id,
                'agent_pos': agent_pos,
                'present': {
                    'my_stag_id': my_stag_id,
                    'coop_stag_ids': coop_stag_ids.copy(),
                    'other_stag_ids': other_stag_ids.copy(),
                    'hare_ids': hare_ids.copy()
                },
                'chosen_object': attempt_type,
                'choice_code': attempt_code,
                'attempt_type': attempt_type,
                'attempts': {
                    'attempted_my_stag': attempted_my_stag,
                    'attempted_coop_stag': attempted_coop_stag,
                    'attempted_other_stag': attempted_other_stag,
                    'attempted_hare': attempted_hare
                }
            })
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """맨하탄 거리 계산"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _chebyshev_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """체비셰프 거리 계산 (3x3 시야 범위 판단용)"""
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
    
    def _is_in_observation_range(self, agent_pos: Tuple[int, int], object_pos: Tuple[int, int]) -> bool:
        """
        Object가 agent의 3x3 관찰 범위 내에 있는지 확인합니다.
        3x3 범위 = 체비셰프 거리 1 이하
        """
        return self._chebyshev_distance(agent_pos, object_pos) <= 1
    
    def _compute_detailed_statistics(self) -> Dict:
        """수집된 데이터의 통계를 계산합니다."""
        statistics = {
            'causal_effects': {},
            'policy_bias': {},
            'attempt_bias': {}
        }
        
        for agent_id, scenarios in self.detailed_causal_effects.items():
            statistics['causal_effects'][f'agent_{agent_id}'] = {}
            for scenario, effects in scenarios.items():
                if len(effects) > 0:
                    effects_array = np.array(effects)
                    statistics['causal_effects'][f'agent_{agent_id}'][scenario] = {
                        'mean': float(np.mean(effects_array)),
                        'std': float(np.std(effects_array)),
                        'median': float(np.median(effects_array)),
                        'count': len(effects_array),
                        'min': float(np.min(effects_array)),
                        'max': float(np.max(effects_array))
                    }
        
        for agent_id, cases in self.object_choice_data.items():
            statistics['policy_bias'][f'agent_{agent_id}'] = {}
            for case_key, choice_codes in cases.items():
                if len(choice_codes) > 0:
                    # 선택 코드: 0=my_stag, 1=coop_stag, 2=other_stag, 3=hare, 4=none
                    choice_counts = np.bincount(choice_codes, minlength=5)
                    choice_probs = choice_counts / len(choice_codes)
                    statistics['policy_bias'][f'agent_{agent_id}'][case_key] = {
                        'choice_distribution': choice_probs.tolist(),  # [my, coop, other, hare, none]
                        'choice_counts': choice_counts.tolist(),
                        'total_count': len(choice_codes),
                        'most_frequent_choice': int(np.argmax(choice_counts)),
                        'my_stag_ratio': float(choice_probs[0]),
                        'coop_stag_ratio': float(choice_probs[1]),
                        'other_stag_ratio': float(choice_probs[2]),
                        'hare_ratio': float(choice_probs[3]),
                        'none_ratio': float(choice_probs[4])
                    }
        
        # Attempt 분포 집계 추가: case별 어떤 시도를 했는지 (0=my_stag, 1=coop_stag, 2=other_stag, 3=hare, 4=none)
        for agent_id, cases in getattr(self, 'case_attempt_data', {}).items():
            statistics['attempt_bias'][f'agent_{agent_id}'] = {}
            for case_key, attempt_codes in cases.items():
                if len(attempt_codes) > 0:
                    attempt_counts = np.bincount(attempt_codes, minlength=5)
                    attempt_probs = attempt_counts / len(attempt_codes)
                    statistics['attempt_bias'][f'agent_{agent_id}'][case_key] = {
                        'attempt_distribution': attempt_probs.tolist(),  # [my, coop, other, hare, none]
                        'attempt_counts': attempt_counts.tolist(),
                        'total_count': len(attempt_codes),
                        'most_frequent_attempt': int(np.argmax(attempt_counts)),
                        'my_stag_ratio': float(attempt_probs[0]),
                        'coop_stag_ratio': float(attempt_probs[1]),
                        'other_stag_ratio': float(attempt_probs[2]),
                        'hare_ratio': float(attempt_probs[3]),
                        'none_ratio': float(attempt_probs[4])
                    }

        return statistics
    
    def plot_detailed_causal_effects(
        self,
        statistics: Dict,
        save_path: Optional[str] = None,
        return_figure: bool = False
    ) -> Optional[plt.Figure]:
        """
        Agent별, 시나리오별 causal effect를 시각화합니다.
        """
        causal_stats = statistics.get('causal_effects', {})
        
        num_agents = len(causal_stats)
        if num_agents == 0:
            print("시각화할 causal effect 데이터가 없습니다.")
            return None
        
        fig, axes = plt.subplots(1, num_agents, figsize=(6 * num_agents, 5))
        if num_agents == 1:
            axes = [axes]
        
        for idx, (agent_key, agent_data) in enumerate(sorted(causal_stats.items())):
            ax = axes[idx]
            
            cooperative_scenarios = []
            defection_scenarios = []
            other_scenarios = []
            
            for scenario, stats in agent_data.items():
                if 'cooperative' in scenario:
                    cooperative_scenarios.append((scenario, stats))
                elif 'defection' in scenario or 'hare' in scenario:
                    defection_scenarios.append((scenario, stats))
                else:
                    other_scenarios.append((scenario, stats))

            all_scenarios = cooperative_scenarios + defection_scenarios + other_scenarios
            
            if not all_scenarios:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'{agent_key}')
                continue

            scenario_labels = []
            means = []
            stds = []
            colors = []
            
            for scenario, stats in all_scenarios:
                if 'partner_stag_cooperative' in scenario:
                    label = 'Partner Stag\n(Cooperative)'
                    color = 'green'
                elif 'partner_hare' in scenario or 'defection' in scenario:
                    label = 'Partner Hare\n(Defection)'
                    color = 'orange'
                elif 'nonpartner_stag' in scenario:
                    label = 'Other Agent\nStag'
                    color = 'blue'
                elif 'nonpartner_hare' in scenario:
                    label = 'Other Agent\nHare'
                    color = 'purple'
                elif 'partner_no_reward' in scenario:
                    label = 'Partner\nNo Reward'
                    color = 'gray'
                else:
                    label = 'Other\nNo Reward'
                    color = 'lightgray'
                
                scenario_labels.append(label)
                means.append(stats['mean'])
                stds.append(stats['std'])
                colors.append(color)
            
            # Bar plot
            x_pos = np.arange(len(scenario_labels))
            ax.bar(x_pos, means, yerr=stds, capsize=2.5, alpha=0.7, color=colors)
            
            ax.set_xlabel('Other Agent Reward Scenario', fontsize=10)
            ax.set_ylabel('Causal Effect (Δ)', fontsize=10)
            ax.set_title(f'{agent_key.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(scenario_labels, fontsize=8, rotation=0)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"상세 Causal Effect plot 저장: {save_path}")
        
        if return_figure:
            return fig
        else:
            plt.show()
            plt.close()
            return None
    
    def plot_policy_bias(
        self,
        statistics: Dict,
        save_path: Optional[str] = None,
        return_figure: bool = False
    ) -> Optional[plt.Figure]:
        """
        Object 조합별로 어떤 object를 선택하는지 시각화합니다.
        huntable_objects 기반의 실제 사냥 가능한 상황에서의 선택 패턴을 표시합니다.
        
        Object Types (huntable_objects 기반):
        - my_stag: 자신이 보상을 받는 stag (stag_id == agent_id)
        - coop_stag: 협력에 참여하는 다른 stag (agent_id in required_agents but stag_id != agent_id)
        - hare: 사냥 가능한 hare
        """
        policy_stats = statistics.get('policy_bias', {})
        
        num_agents = len(policy_stats)
        if num_agents == 0:
            print("시각화할 policy bias 데이터가 없습니다.")
            return None
        
        fig, axes = plt.subplots(num_agents, 1, figsize=(14, 5 * num_agents))
        if num_agents == 1:
            axes = [axes]
        
        for idx, (agent_key, agent_data) in enumerate(sorted(policy_stats.items())):
            ax = axes[idx]
            
            case_priority = {
                # Single objects
                'my_stag': 0,
                'coop_stag': 1,
                'other_stag': 2,
                'hare': 3,
                # Two objects
                'my_stag+coop_stag': 4,
                'my_stag+other_stag': 5,
                'my_stag+hare': 6,
                'coop_stag+other_stag': 7,
                'coop_stag+hare': 8,
                'other_stag+hare': 9,
                # Three objects
                'my_stag+coop_stag+other_stag': 10,
                'my_stag+coop_stag+hare': 11,
                'my_stag+other_stag+hare': 12,
                'coop_stag+other_stag+hare': 13,
                # All four (이론상 드물지만 안전)
                'my_stag+coop_stag+other_stag+hare': 14
            }
            
            def get_case_priority(case_key):
                """Case 키의 우선순위 반환"""
                return case_priority.get(case_key, 999)

            sorted_items = sorted(agent_data.items(), key=lambda x: get_case_priority(x[0]))
            
            cases = []
            choice_distributions = []
            
            for case_key, stats in sorted_items:
                label = case_key.replace('my_stag', 'M').replace('coop_stag', 'C').replace('other_stag', 'O').replace('hare', 'H')
                label = label.replace('+', '+')

                total_count = stats['total_count']
                label_with_count = f'{label}\n(n={total_count})'
                
                cases.append(label_with_count)
                choice_distributions.append(stats['choice_distribution'])
            
            if not cases:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'{agent_key} Policy Bias')
                continue
            
            # Stacked bar plot
            choice_distributions = np.array(choice_distributions)
            x_pos = np.arange(len(cases))
            
            choice_colors = ['#27AE60', '#82E0AA', '#F39C12', '#E74C3C', '#95A5A6']
            choice_names = ['→ My Stag', '→ Coop Stag', '→ Other Stag', '→ Hare', '→ None']
            bottom = np.zeros(len(cases))
            
            for choice_idx in range(5):
                ax.bar(
                    x_pos,
                    choice_distributions[:, choice_idx],
                    bottom=bottom,
                    label=choice_names[choice_idx],
                    alpha=0.85,
                    color=choice_colors[choice_idx],
                    edgecolor='white',
                    linewidth=0.5
                )
                bottom += choice_distributions[:, choice_idx]
            
            ax.set_xlabel('Object Combination in 3x3 Vision (M=My, C=Coop, O=Other, H=Hare)', fontsize=10)
            ax.set_ylabel('Object Choice Ratio', fontsize=11)
            ax.set_title(f'{agent_key.replace("_", " ").title()} - Object Selection Policy',
                        fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(cases, fontsize=8, rotation=45, ha='right')
            ax.legend(loc='upper right', fontsize=9, framealpha=0.95, ncol=2)
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"save policy bias plot: {save_path}")
        
        if return_figure:
            return fig
        else:
            plt.show()
            plt.close()
            return None
    
    def save_results(
        self,
        statistics: Dict,
        save_dir: str
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        
        # 통계 데이터 저장
        stats_path = os.path.join(save_dir, 'advanced_causal_statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        print(f"save results: {save_dir}")

