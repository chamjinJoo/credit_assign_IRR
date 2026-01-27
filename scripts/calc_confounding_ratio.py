
import numpy as np
import os
import sys
from collections import Counter

# Add project root to path
# Assuming this script is in scripts/ directory, so project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.multi_agent_env import StagHuntEnv

class RandomPolicy:
    def __init__(self, env):
        self.env = env
    
    def get_action(self, obs): 
        actions = {}
        for i in range(self.env.num_agents):
            actions[i] = np.random.randint(0, self.env.action_dim)
        return actions

class RolloutwithRandomPolicy:
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
            
            action = self.policy.get_action(obs)
            next_obs, reward, done, info = self.env.step(action)
            
            # Use current_state instead of info (which is S_{t+1})
            rollout.append((current_state, obs, action, reward, next_obs, done))
            obs = next_obs
            
            # FIXED: Check if all agents are done
            if all(done.values()):
                break
        return rollout

def get_probabilities(data):
    """데이터 리스트를 받아 확률 분포(dict) 반환"""
    count = Counter(data)
    total = sum(count.values())
    return {k: v / total for k, v in count.items()}

def get_joint_probabilities(data1, data2):
    """두 데이터 리스트의 결합 확률 분포 반환"""
    joint_data = list(zip(data1, data2))
    return get_probabilities(joint_data)

def compute_mutual_info(rollout, agent_id):
    """
    I(A; Y) 계산
    A: agent_id의 개별 action
    Y: 해당 agent를 제외한 나머지 agent들의 reward vector (tuple로 변환하여 처리)
    """
    # 1. 데이터 추출
    actions = [step[2][agent_id] for step in rollout] # step[2] is action (dict)
    
    # peer_rewards를 hashable한 tuple로 변환
    # step[3] is reward (dict)
    peer_rewards = []
    for step in rollout:
        rewards = step[3]
        # Sort keys to ensure consistent order
        p_rewards = []
        for aid in sorted(rewards.keys()):
            if aid != agent_id:
                p_rewards.append(rewards[aid])
        peer_rewards.append(tuple(p_rewards))
    
    # 2. 확률 분포 계산
    p_a = get_probabilities(actions)
    p_y = get_probabilities(peer_rewards)
    p_ay = get_joint_probabilities(actions, peer_rewards)
    
    # 3. MI 계산
    mi = 0
    for (a, y), p_joint in p_ay.items():
        mi += p_joint * np.log2(p_joint / (p_a[a] * p_y[y]) + 1e-10)
    
    return max(0, mi)

def compute_specific_mi(x_data, y_data):
    """특정 y값에 대한 Specific MI: I(X; Y=y) 리스트 반환"""
    p_x = get_probabilities(x_data)
    p_y = get_probabilities(y_data)
    p_xy = get_joint_probabilities(x_data, y_data)
    
    y_values = list(p_y.keys())
    specific_mi_dict = {}
    
    for y in y_values:
        val = 0
        for x in p_x.keys():
            p_joint = p_xy.get((x, y), 0)
            if p_joint > 0:
                p_cond = p_joint / p_y[y]
                val += p_cond * np.log2(p_cond / p_x[x] + 1e-10)
        specific_mi_dict[y] = val
        
    return specific_mi_dict

def compute_red(rollout, agent_id):
    """
    Red(A, S; Y) 계산
    A: agent_id의 action
    S: global state (info['agent_positions'] 등)
    Y: peer rewards
    """
    actions = [step[2][agent_id] for step in rollout]
    
    peer_rewards = []
    for step in rollout:
        rewards = step[3]
        p_rewards = []
        for aid in sorted(rewards.keys()):
            if aid != agent_id:
                p_rewards.append(rewards[aid])
        peer_rewards.append(tuple(p_rewards))
    
    states = []
    for step in rollout:
        # step[0] is state: [agent_pos(dict), stag_pos(dict), hare_pos(dict)]
        flat_state = []
        # agent_positions
        for k in sorted(step[0][0].keys()):
            flat_state.extend(step[0][0][k])
        # stag_positions
        for k in sorted(step[0][1].keys()):
            flat_state.extend(step[0][1][k])
        # hare_positions
        for k in sorted(step[0][2].keys()):
            flat_state.extend(step[0][2][k])
            
        states.append(tuple(flat_state))
    
    # 2. 각 변수에 대한 Specific MI 계산
    spec_mi_a = compute_specific_mi(actions, peer_rewards)
    spec_mi_s = compute_specific_mi(states, peer_rewards)
    
    p_y = get_probabilities(peer_rewards)
    
    # 3. Redundancy = sum_y p(y) * min(I(A; Y=y), I(S; Y=y))
    red = 0
    mi_s = 0
    for y, prob in p_y.items():
        i_a_y = spec_mi_a.get(y, 0)
        i_s_y = spec_mi_s.get(y, 0)
        red += prob * min(i_a_y, i_s_y)
        mi_s += prob * i_s_y
        
    return max(0, red), mi_s


def to_one_hot(action, dim=5):
    vec = [0] * dim
    vec[action] = 1
    return tuple(vec)

def compute_red_onehot(rollout, agent_id):
    """
    Red(A, S; Y) 계산 (One-hot action 사용)
    """
    actions = [to_one_hot(step[2][agent_id]) for step in rollout] # Convert to one-hot tuple
    
    peer_rewards = []
    for step in rollout:
        rewards = step[3]
        p_rewards = []
        for aid in sorted(rewards.keys()):
            if aid != agent_id:
                p_rewards.append(rewards[aid])
        peer_rewards.append(tuple(p_rewards))
    
    states = []
    for step in rollout:
        flat_state = []
        for k in sorted(step[0][0].keys()): flat_state.extend(step[0][0][k])
        for k in sorted(step[0][1].keys()): flat_state.extend(step[0][1][k])
        for k in sorted(step[0][2].keys()): flat_state.extend(step[0][2][k])
        states.append(tuple(flat_state))
    
    spec_mi_a = compute_specific_mi(actions, peer_rewards)
    spec_mi_s = compute_specific_mi(states, peer_rewards)
    p_y = get_probabilities(peer_rewards)
    
    red = 0
    mi_s = 0
    for y, prob in p_y.items():
        i_a_y = spec_mi_a.get(y, 0)
        i_s_y = spec_mi_s.get(y, 0)
        red += prob * min(i_a_y, i_s_y)
        mi_s += prob * i_s_y
        
    return max(0, red), mi_s

def compute_mutual_info_onehot(rollout, agent_id):
    """
    I(A; Y) 계산 (One-hot action 사용)
    """
    actions = [to_one_hot(step[2][agent_id]) for step in rollout] # Convert to one-hot tuple
    
    peer_rewards = []
    for step in rollout:
        rewards = step[3]
        p_rewards = []
        for aid in sorted(rewards.keys()):
            if aid != agent_id:
                p_rewards.append(rewards[aid])
        peer_rewards.append(tuple(p_rewards))
    
    p_a = get_probabilities(actions)
    p_y = get_probabilities(peer_rewards)
    p_ay = get_joint_probabilities(actions, peer_rewards)
    
    mi = 0
    for (a, y), p_joint in p_ay.items():
        mi += p_joint * np.log2(p_joint / (p_a[a] * p_y[y]) + 1e-10)
    
    return max(0, mi)

def get_confounding_ratio(env, policy, max_steps=100, num_episodes=100):
    all_rollout = []
    
    print(f"Collecting data for {num_episodes} episodes...")
    for _ in range(num_episodes):
        rollout_runner = RolloutwithRandomPolicy(env, policy, max_steps)
        episode_rollout = rollout_runner.run()
        all_rollout.extend(episode_rollout)
        
    print('Total collected data points: ', len(all_rollout))
    
    cr = 0
    print('N: ', env.num_agents)
    for i in range(env.num_agents):
        # compute confounding ratio for each agent i
        # # 1. Scalar Action
        # red_i, mi_s_i = compute_red(all_rollout, i)
        # mutual_info_i = compute_mutual_info(all_rollout, i)
        # if mutual_info_i > 1e-10:
        #     cr_i = red_i / mutual_info_i
        # else:
        #     cr_i = 0.0
            
        # 2. One-hot Action
        red_oh, mi_s_oh = compute_red_onehot(all_rollout, i)
        mi_oh = compute_mutual_info_onehot(all_rollout, i)
        if mi_oh > 1e-10:
            cr_oh = red_oh / mi_oh
        else:
            cr_oh = 0.0

        cr += cr_oh
        
        print(f"Agent {i} [OneHot]: Red={red_oh:.4f}, MI(A;Y)={mi_oh:.4f}, MI(S;Y)={mi_s_oh:.4f}, CR={cr_oh:.4f}")

    confounding_ratio = cr / env.num_agents

    return confounding_ratio

if __name__ == "__main__":
    # Bring environment
    env = StagHuntEnv(num_agents=3)

    # Get random policy
    policy = RandomPolicy(env)

    # Get confounding ratio with 100 episodes
    confounding_ratio = get_confounding_ratio(env, policy, max_steps=50, num_episodes=100) # StagHuntEnv default max_steps is 50

    # Print confounding ratio
    print(f"Average Confounding ratio: {confounding_ratio}")
