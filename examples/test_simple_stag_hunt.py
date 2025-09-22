#!/usr/bin/env python3
"""
Simple Stag Hunt 환경 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pci_mappo.envs.multi_agent_env import SimpleStagHuntEnv

def test_simple_stag_hunt():
    """Simple Stag Hunt 환경을 테스트합니다."""
    print("=== Simple Stag Hunt Environment Test ===")
    
    # 환경 생성
    env = SimpleStagHuntEnv(
        cooperation_reward=5,
        defect_alone_reward=1,
        defect_together_reward=1,
        failed_cooperation_punishment=-5,
        eps_per_game=3
    )
    
    print(f"Environment created successfully!")
    print(f"Observation dimension: {env.get_obs_dim()}")
    print(f"Action dimension: {env.get_action_dim()}")
    print(f"Number of agents: {env.num_agents}")
    print()
    
    # 여러 에피소드 테스트
    for episode in range(3):
        print(f"--- Episode {episode + 1} ---")
        
        # 환경 리셋
        observations = env.reset()
        print(f"Initial observations: {observations}")
        
        # 모든 가능한 액션 조합 테스트
        action_combinations = [
            (0, 0),  # 둘 다 협력
            (0, 1),  # A 협력, B 배신
            (1, 0),  # A 배신, B 협력
            (1, 1),  # 둘 다 배신
        ]
        
        for i, (action_0, action_1) in enumerate(action_combinations):
            print(f"\nTest {i+1}: Agent 0={action_0} (COOPERATE)" if action_0 == 0 else f"\nTest {i+1}: Agent 0={action_0} (DEFECT)")
            print(f"         Agent 1={action_1} (COOPERATE)" if action_1 == 0 else f"         Agent 1={action_1} (DEFECT)")
            
            # 액션 실행
            actions = {0: action_0, 1: action_1}
            observations, rewards, dones, info = env.step(actions)
            
            print(f"Rewards: {rewards}")
            print(f"Observations: {observations}")
            print(f"Dones: {dones}")
            
            # 렌더링
            env.render()
        
        print(f"Episode {episode + 1} completed!")
        print()
    
    print("=== Test completed successfully! ===")

def test_reward_matrix():
    """보상 매트릭스를 테스트합니다."""
    print("=== Reward Matrix Test ===")
    
    env = SimpleStagHuntEnv()
    
    # 보상 매트릭스 출력
    print("Reward Matrix (Agent 0, Agent 1):")
    print("           Agent 1")
    print("         COOPERATE  DEFECT")
    print("Agent 0")
    
    action_names = ["COOPERATE", "DEFECT"]
    
    for i, action_0 in enumerate([0, 1]):
        print(f"{action_names[i]:<10}", end="")
        for action_1 in [0, 1]:
            rewards = env._compute_rewards(action_0, action_1)
            print(f"({rewards[0]:>3}, {rewards[1]:>3})", end="  ")
        print()
    
    print()
    print("Expected behavior:")
    print("- Both cooperate: High reward (5, 5)")
    print("- One cooperates, one defects: Defector gets reward (1), cooperator gets punished (-5)")
    print("- Both defect: Low reward (1, 1)")
    print()

if __name__ == "__main__":
    test_reward_matrix()
    test_simple_stag_hunt()
