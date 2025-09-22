#!/usr/bin/env python3
"""
Stag-Hunt Environment Test

이 파일은 StagHuntEnv의 기능을 테스트하는 예제입니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pci_mappo.envs import StagHuntEnv


def test_stag_hunt_basic():
    """Stag-Hunt 환경의 기본 기능을 테스트합니다."""
    print("=== Stag-Hunt 환경 기본 테스트 ===\n")
    
    # 환경 생성
    env = StagHuntEnv(max_steps=10, stag_move_prob=0.5)
    
    print(f"에이전트 수: {env.num_agents}")
    print(f"관찰 차원: {env.get_obs_dim()}")
    print(f"액션 차원: {env.get_action_dim()}")
    print(f"그리드 크기: {env.grid_size_x} x {env.grid_size_y}")
    print()
    
    # 환경 리셋
    observations = env.reset()
    print("초기 관찰:")
    for agent_id, obs in observations.items():
        print(f"  Agent {agent_id}: {obs}")
    print()
    
    # 환경 렌더링
    print("초기 환경 상태:")
    env.render()
    
    # 몇 스텝 실행
    for step in range(5):
        print(f"\n--- 스텝 {step + 1} ---")
        
        # 랜덤 액션 선택
        actions = {
            0: np.random.randint(0, env.get_action_dim()),
            1: np.random.randint(0, env.get_action_dim())
        }
        
        print(f"액션: {actions}")
        
        # 환경 스텝
        next_observations, rewards, dones, info = env.step(actions)
        
        print(f"보상: {rewards}")
        print(f"종료: {dones}")
        print(f"정보: {info}")
        
        # 환경 렌더링
        env.render()
        
        # 다음 관찰로 업데이트
        observations = next_observations
        
        # 에피소드 종료 확인
        if any(dones.values()):
            print("에피소드 종료!")
            break
    
    print("\n기본 테스트 완료!\n")


def test_stag_hunt_hunt_actions():
    """Hunt 액션의 동작을 테스트합니다."""
    print("=== Hunt 액션 테스트 ===\n")
    
    env = StagHuntEnv(max_steps=20, stag_move_prob=0.0)  # Stag가 움직이지 않도록 설정
    
    observations = env.reset()
    env.render()
    
    # Hare 사냥 테스트
    print("\n--- Hare 사냥 테스트 ---")
    
    # Agent 0을 Hare 근처로 이동 (Agent spawn 위치에서 Hare로 이동)
    # Agent spawn 위치: (2,1), (6,1), (2,5), (6,5)
    # Hare 위치: (1,3), (7,3)
    # Agent가 (2,1)에 있다면 (1,3)으로 이동
    for step in range(3):
        actions = {0: 1, 1: 0}  # Agent 0: 아래로, Agent 1: 제자리
        observations, rewards, dones, info = env.step(actions)
        env.render()
        
        if any(dones.values()):
            break
    
    # Hare 사냥 시도
    actions = {0: 4, 1: 0}  # Agent 0: hunt, Agent 1: 제자리
    observations, rewards, dones, info = env.step(actions)
    
    print(f"Hare 사냥 결과:")
    print(f"  보상: {rewards}")
    print(f"  종료: {dones}")
    env.render()
    
    print("\nHunt 액션 테스트 완료!\n")


def test_stag_hunt_cooperation():
    """협력 사냥을 테스트합니다."""
    print("=== 협력 사냥 테스트 ===\n")
    
    env = StagHuntEnv(max_steps=20, stag_move_prob=0.0)
    
    observations = env.reset()
    env.render()
    
    # 두 에이전트를 Stag 근처로 이동
    # Stag 위치: (4,3)
    # Agent spawn 위치: (2,1), (6,1), (2,5), (6,5)
    print("\n--- Stag 근처로 이동 ---")
    for step in range(3):
        actions = {0: 1, 1: 1}  # 둘 다 아래로 (또는 위로, 위치에 따라)
        observations, rewards, dones, info = env.step(actions)
        env.render()
        
        if any(dones.values()):
            break
    
    # 협력 사냥 시도
    print("\n--- 협력 사냥 시도 ---")
    actions = {0: 4, 1: 4}  # 둘 다 hunt
    observations, rewards, dones, info = env.step(actions)
    
    print(f"협력 사냥 결과:")
    print(f"  보상: {rewards}")
    print(f"  종료: {dones}")
    env.render()
    
    print("\n협력 사냥 테스트 완료!\n")


def test_stag_hunt_individual_hunt():
    """개별 Stag 사냥을 테스트합니다."""
    print("=== 개별 Stag 사냥 테스트 ===\n")
    
    env = StagHuntEnv(max_steps=20, stag_move_prob=0.0)
    
    observations = env.reset()
    env.render()
    
    # Agent 0만 Stag 근처로 이동
    print("\n--- Agent 0만 Stag 근처로 이동 ---")
    for step in range(3):
        actions = {0: 1, 1: 0}  # Agent 0: 아래, Agent 1: 제자리
        observations, rewards, dones, info = env.step(actions)
        env.render()
        
        if any(dones.values()):
            break
    
    # 개별 사냥 시도
    print("\n--- 개별 사냥 시도 ---")
    actions = {0: 4, 1: 0}  # Agent 0만 hunt
    observations, rewards, dones, info = env.step(actions)
    
    print(f"개별 사냥 결과:")
    print(f"  보상: {rewards}")
    print(f"  종료: {dones}")
    env.render()
    
    print("\n개별 사냥 테스트 완료!\n")


def test_observation_format():
    """관찰 형식을 테스트합니다."""
    print("=== 관찰 형식 테스트 ===\n")
    
    env = StagHuntEnv(max_steps=10, stag_move_prob=0.0)
    observations = env.reset()
    
    print("관찰 형식:")
    for agent_id, obs in observations.items():
        print(f"Agent {agent_id}:")
        print(f"  관찰 벡터: {obs}")
        print(f"  관찰 형태: {obs.shape}")
        print(f"  관찰 타입: {obs.dtype}")
        
        # 3x3 그리드로 재구성
        grid = obs.reshape(3, 3)
        print(f"  3x3 그리드:")
        for row in reversed(grid):  # y축을 위에서부터 표시
            print(f"    {row}")
        print()
    
    print("관찰 형식 테스트 완료!\n")


def test_layout_verification():
    """레이아웃 검증을 테스트합니다."""
    print("=== 레이아웃 검증 테스트 ===\n")
    
    env = StagHuntEnv(max_steps=5, stag_move_prob=0.0)
    observations = env.reset()
    
    print("레이아웃 정보:")
    print(f"  그리드 크기: {env.grid_size_x} x {env.grid_size_y}")
    print(f"  Stag 위치: {env.stag_position}")
    print(f"  Hare 위치: {env.hare_positions}")
    print(f"  Agent spawn 위치: {env.spawn_positions}")
    print(f"  현재 Agent 위치: {env.agent_positions}")
    print(f"  벽 위치: {env.wall_positions}")
    
    print("\n레이아웃 시각화:")
    env.render()
    
    # 거리 계산
    stag_x, stag_y = env.stag_position
    for i, hare_pos in enumerate(env.hare_positions):
        hare_x, hare_y = hare_pos
        distance = abs(hare_x - stag_x) + abs(hare_y - stag_y)
        print(f"  Hare {i}와 Stag 사이 맨해튼 거리: {distance}칸")
    
    print("\n레이아웃 검증 완료!\n")


if __name__ == "__main__":
    # 각 테스트 실행
    test_layout_verification()  # 먼저 레이아웃 검증
    test_stag_hunt_basic()
    test_stag_hunt_hunt_actions()
    test_stag_hunt_cooperation()
    test_stag_hunt_individual_hunt()
    test_observation_format()
    
    print("모든 테스트 완료!")
