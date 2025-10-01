#!/usr/bin/env python3
"""
StagHunt 환경의 렌더링 기능을 테스트하는 스크립트입니다.
"""

import numpy as np
import time
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pci_mappo.envs.multi_agent_env import StagHuntEnv

def test_staghunt_rendering():
    """StagHunt 환경의 렌더링 기능을 테스트합니다."""
    print("StagHunt 환경 렌더링 테스트를 시작합니다...")
    
    # 환경 생성
    env = StagHuntEnv(
        max_steps=20,
        stag_follows=False,
        forage_quantity=3,
        forage_reward=2.0,
        stag_reward=5.0,
        seed=42
    )
    
    print(f"환경 정보:")
    print(f"- Agent 수: {env.num_agents}")
    print(f"- Grid 크기: {env.grid_size_x}x{env.grid_size_y}")
    print(f"- 최대 스텝: {env.max_steps}")
    print(f"- Plant 수: {env.forage_quantity}")
    print()
    
    # 환경 리셋
    observations = env.reset()
    print("환경이 리셋되었습니다.")
    print(f"Agent 위치: {env.agent_positions}")
    print(f"Stag 위치: {env.stag_positions}")
    print(f"Plant 위치: {env.plant_positions}")
    print()
    
    # 렌더링 테스트
    print("렌더링을 시작합니다...")
    env.render(mode='rgb_array')
    
    # 몇 스텝 실행하면서 렌더링
    for step in range(10):
        print(f"\n--- Step {step + 1} ---")
        
        # 랜덤 액션 생성
        actions = {}
        for agent_id in env.agents:
            actions[agent_id] = np.random.randint(0, env.get_action_dim())
        
        print(f"Actions: {actions}")
        
        # 환경 스텝 실행
        observations, rewards, dones, info = env.step(actions)
        
        print(f"Rewards: {rewards}")
        print(f"Agent 위치: {env.agent_positions}")
        print(f"Stag 위치: {env.stag_positions}")
        print(f"Plant 위치: {env.plant_positions}")
        
        # 렌더링 업데이트
        env.render(mode='rgb_array')
        
        # 잠시 대기 (렌더링 확인용)
        time.sleep(0.5)
        
        # 에피소드 종료 확인
        if any(dones.values()):
            print("에피소드가 종료되었습니다.")
            break
    
    # 렌더링 창 닫기
    env.close_render()
    print("렌더링 테스트가 완료되었습니다.")

def test_animation():
    """애니메이션 기능을 테스트합니다."""
    print("\n애니메이션 테스트를 시작합니다...")
    
    env = StagHuntEnv(
        max_steps=15,
        forage_quantity=2,
        seed=123
    )
    
    env.reset()
    
    print("애니메이션을 시작합니다. 창을 닫으면 애니메이션이 중지됩니다.")
    animation = env.start_animation(max_frames=50, interval=200)
    
    if animation is not None:
        try:
            # 애니메이션 표시
            import matplotlib.pyplot as plt
            plt.show()
        except KeyboardInterrupt:
            print("애니메이션이 중단되었습니다.")
        finally:
            env.stop_animation()
            env.close_render()
    
    print("애니메이션 테스트가 완료되었습니다.")

def test_rgb_array_rendering():
    """RGB 배열 렌더링을 테스트합니다."""
    print("\nRGB 배열 렌더링 테스트를 시작합니다...")
    
    env = StagHuntEnv(seed=123)
    env.reset()
    
    # RGB 배열 렌더링
    rgb_array = env.render(mode='rgb_array')
    
    if rgb_array is not None:
        print(f"RGB 배열 크기: {rgb_array.shape}")
        print(f"RGB 배열 타입: {rgb_array.dtype}")
        print(f"RGB 값 범위: {rgb_array.min()} ~ {rgb_array.max()}")
        
        # RGB 배열을 파일로 저장 (선택사항)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 8))
            plt.imshow(rgb_array)
            plt.title('StagHunt Environment (RGB Array)')
            plt.axis('off')
            plt.savefig('staghunt_render_test.png', dpi=100, bbox_inches='tight')
            print("렌더링 이미지가 'staghunt_render_test.png'로 저장되었습니다.")
            plt.close()
        except Exception as e:
            print(f"이미지 저장 중 오류: {e}")
    else:
        print("RGB 배열 렌더링에 실패했습니다.")
    
    env.close_render()

if __name__ == "__main__":
    try:
        print("=== StagHunt 환경 렌더링 테스트 ===\n")
        
        # 기본 렌더링 테스트
        test_staghunt_rendering()
        
        # RGB 배열 렌더링 테스트
        test_rgb_array_rendering()
        
        # 애니메이션 테스트 (선택사항)
        user_input = input("\n애니메이션 테스트를 실행하시겠습니까? (y/n): ").lower().strip()
        if user_input in ['y', 'yes', '예']:
            test_animation()
        
        print("\n모든 테스트가 완료되었습니다!")
        
    except KeyboardInterrupt:
        print("\n테스트가 중단되었습니다.")
    except Exception as e:
        print(f"테스트 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
