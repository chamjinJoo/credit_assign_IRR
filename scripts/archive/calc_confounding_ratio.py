
import numpy as np
import os
import sys
from collections import Counter

# Add project root to path
# Assuming this script is in scripts/ directory, so project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.multi_agent_env import StagHuntEnv, SMACv2Wrapper

class RandomPolicy:
    def __init__(self, env):
        self.env = env
    
    def get_action(self, state): 
        actions = {}
        for i in range(self.env.num_agents):
            actions[i] = np.random.randint(0, self.env.action_dim)
        return actions

import numpy as np

class HeuristicPolicy:
    def __init__(self, env):
        self.env = env
        # StagHuntEnv의 액션 매핑 (보통 0:Stay, 1:Up, 2:Down, 3:Left, 4:Right)
        self.actions = {
            'STAY': 0, 'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4
        }

    def _get_distance(self, pos1, pos2):
        """맨해튼 거리 계산"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _move_toward(self, current_pos, target_pos):
        """대상 방향으로 이동하기 위한 액션 결정"""
        y_dist = target_pos[0] - current_pos[0]
        x_dist = target_pos[1] - current_pos[1]

        if abs(y_dist) > abs(x_dist):
            return self.actions['UP'] if y_dist < 0 else self.actions['DOWN']
        elif abs(x_dist) > 0:
            return self.actions['LEFT'] if x_dist < 0 else self.actions['RIGHT']
        return self.actions['STAY']

    def get_action(self, state): 
        actions = {}
        # state[0]: agent_pos (dict {id: [y, x]})
        # state[1]: stag_pos (dict {id: [y, x]})
        # state[2]: hare_pos (dict {id: [y, x]})
        agent_positions = state[0]
        stag_positions = state[1]
        hare_positions = state[2]

        for i in range(self.env.num_agents):
            my_pos = agent_positions[i]
            
            # 1. 협력 가능한 스테그 탐색 (가장 가까운 스테그 우선)
            best_stag = None
            min_stag_dist = float('inf')
            
            for s_id, s_pos in stag_positions.items():
                dist = self._get_distance(my_pos, s_pos)
                if dist < min_stag_dist:
                    min_stag_dist = dist
                    best_stag = s_pos

            # 스테그가 가깝고(예: 거리 4 이내), 다른 에이전트도 근처에 있다면 스테그 추적
            # (학술적 교란을 위해 상태에 강하게 반응하도록 설정)
            if best_stag is not None and min_stag_dist <= 4:
                actions[i] = self._move_toward(my_pos, best_stag)
                continue

            # 2. 스테그가 멀다면 가장 가까운 토끼(Hare) 탐색
            best_hare = None
            min_hare_dist = float('inf')
            for h_id, h_pos in hare_positions.items():
                dist = self._get_distance(my_pos, h_pos)
                if dist < min_hare_dist:
                    min_hare_dist = dist
                    best_hare = h_pos

            if best_hare is not None:
                actions[i] = self._move_toward(my_pos, best_hare)
            else:
                # 3. 주변에 아무것도 없으면 무작위 이동
                actions[i] = np.random.randint(0, self.env.action_dim)

        return actions
    
class RolloutwithPolicy:
    def __init__(self, env, policy, max_steps=100):
        self.env = env
        self.policy = policy
        self.max_steps = max_steps
    
    def run(self):
        obs = self.env.reset()
        done = False
        rollout = []
        for _ in range(self.max_steps):
            # Capture state BEFORE step (S_t)
            # Use deepcopy or recreate dicts to avoid reference issues
            current_state = [
                self.env.agent_positions.copy(), 
                self.env.stag_positions.copy(), 
                self.env.hare_positions.copy()
            ]
            
            action = self.policy.get_action(current_state)

            next_obs, reward, done, info = self.env.step(action)
            
            # Use current_state instead of info (which is S_{t+1})
            if sum([r for r in reward.values()]) != 0.0:
                rollout.append((current_state, obs, action, reward, next_obs, done))
            obs = next_obs
            
            # FIXED: Check if all agents are done
            if all(done.values()):
                break
        return rollout

def get_probabilities(data):
    """데이터 리스트를 받아 확률 분포(dict) 반환"""
    if not data: return {}
    count = Counter(data)
    total = sum(count.values())
    return {k: v / total for k, v in count.items()}

def compute_entropy(data):
    """샤논 엔트로피 H(X) 계산"""
    p = get_probabilities(data)
    ent = 0
    for val in p.values():
        if val > 0:
            ent -= val * np.log2(val)
    return ent

def compute_conditional_mi(data_a, data_y, data_s):
    """
    CMI 계산: I(A; Y | S) = H(A, S) + H(S, Y) - H(A, S, Y) - H(S)
    """
    # 각 변수 조합의 엔트로피 계산
    h_s = compute_entropy(data_s)
    h_as = compute_entropy(list(zip(data_a, data_s)))
    h_sy = compute_entropy(list(zip(data_s, data_y)))
    h_asy = compute_entropy(list(zip(data_a, data_s, data_y)))
    
    cmi = h_as + h_sy - h_asy - h_s
    return max(0, cmi)

def to_one_hot(action, dim=5):
    vec = [0] * dim
    vec[action] = 1
    return tuple(vec)

def extract_data_from_rollout(rollout, agent_id):
    """rollout에서 MI/CMI 계산에 필요한 A, Y, S 데이터를 추출"""
    actions = [to_one_hot(step[2][agent_id]) for step in rollout]
    
    peer_rewards = []
    for step in rollout:
        rewards = step[3]
        p_rewards = [rewards[aid] for aid in sorted(rewards.keys()) if aid != agent_id]
        peer_rewards.append(tuple(p_rewards))
        
    states = []
    for step in rollout:
        flat_state = []
        # agent_positions, stag_positions, hare_positions를 하나의 튜플로 평탄화
        for pos_dict in step[0]:
            for k in sorted(pos_dict.keys()):
                flat_state.extend(pos_dict[k])
        states.append(tuple(flat_state))
        
    return actions, peer_rewards, states

def compute_mutual_info_onehot(rollout, agent_id):
    """I(A; Y) 계산"""
    actions, peer_rewards, _ = extract_data_from_rollout(rollout, agent_id)
    
    h_a = compute_entropy(actions)
    h_y = compute_entropy(peer_rewards)
    h_ay = compute_entropy(list(zip(actions, peer_rewards)))
    
    return max(0, h_a + h_y - h_ay)

def compute_red_onehot(rollout, agent_id):
    """
    CMI 기반 Refined Redundancy 계산
    Red(A, S; Y) = I(A; Y) - I(A; Y | S)
    """
    actions, peer_rewards, states = extract_data_from_rollout(rollout, agent_id)
    
    # 1. I(A; Y) 계산
    h_a = compute_entropy(actions)
    h_y = compute_entropy(peer_rewards)
    h_ay = compute_entropy(list(zip(actions, peer_rewards)))
    mi_a_y = max(0, h_a + h_y - h_ay)
    
    # 2. I(A; Y | S) 계산
    cmi_a_y_s = compute_conditional_mi(actions, peer_rewards, states)
    
    # 3. Redundancy = I(A; Y) - I(A; Y | S)
    # 행동이 상태와 독립적(Random)일수록 CMI가 MI와 비슷해져 Red는 0에 가까워짐
    red = max(0, mi_a_y - cmi_a_y_s)
    print(mi_a_y, cmi_a_y_s)
    # 비교를 위해 I(S; Y)도 함께 반환
    h_s = compute_entropy(states)
    mi_s_y = max(0, h_s + h_y - compute_entropy(list(zip(states, peer_rewards))))
    
    return red, mi_s_y

def get_confounding_ratio(env, policy, max_steps=100, num_episodes=100):
    all_rollout = []
    
    print(f"Collecting data for {num_episodes} episodes...")
    for _ in range(num_episodes):
        rollout_runner = RolloutwithPolicy(env, policy, max_steps)
        episode_rollout = rollout_runner.run()
        all_rollout.extend(episode_rollout)
        
    print(f"Total collected data points: {len(all_rollout)}")
    
    total_cr = 0
    num_agents = env.num_agents
    print(f"Number of agents: {num_agents}")
    
    for i in range(num_agents):
        mi_oh = compute_mutual_info_onehot(all_rollout, i)
        red_oh, mi_s_oh = compute_red_onehot(all_rollout, i)
        
        if mi_oh > 1e-10:
            cr_oh = red_oh / mi_oh
        else:
            cr_oh = 0.0

        total_cr += cr_oh
        print(f"Agent {i}: Red={red_oh:.4f}, MI(A;Y)={mi_oh:.4f}, MI(S;Y)={mi_s_oh:.4f}, CR={cr_oh:.4f}")

    return total_cr / num_agents

if __name__ == "__main__":
    # 환경 설정 (비대칭 보상 시뮬레이션을 위해 coop_bonus 설정 가능)
    env = StagHuntEnv(num_agents=3, coop_bonus=0.0)

    # Random Policy는 상태와 무관하게 행동하므로 CR이 낮게 나와야 정상임
    policy = RandomPolicy(env)

    # Confounding Ratio 측정
    confounding_ratio = get_confounding_ratio(env, policy, max_steps=50, num_episodes=2000)

    print("-" * 30)
    print(f"Average Confounding Ratio: {confounding_ratio:.4f}")