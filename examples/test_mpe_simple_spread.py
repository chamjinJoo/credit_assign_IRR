#!/usr/bin/env python3
"""
MPE Simple Spread 환경에서 PCI-MAPPO 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

from pci_mappo.utils.config import Config
from pci_mappo.training.trainer import PCIMAPPOTrainer
from pci_mappo.envs import create_environment, get_available_environments

def test_mpe_simple_spread():
    """MPE Simple Spread 환경에서 PCI-MAPPO 기본 기능 테스트"""
    print("=== MPE Simple Spread + PCI-MAPPO 테스트 ===")
    
    try:
        # 설정 파일 로드
        config_path = Path(__file__).parent.parent / "configs" / "mpe_simple_spread.yaml"
        print(f"설정 파일 로드: {config_path}")
        
        config = Config.from_yaml(str(config_path))
        print(f"✓ 설정 로드 성공")
        print(f"  - 환경: {config.environment.env_name}")
        print(f"  - 에이전트 수: {config.environment.num_agents}")
        print(f"  - 최대 사이클: {config.environment.max_cycles}")
        
        # 환경 생성
        env = create_environment(config.environment)
        print(f"✓ 환경 생성 성공")
        print(f"  - 관찰 차원: {env.get_obs_dim()}")
        print(f"  - 액션 차원: {env.get_action_dim()}")
        
        # PCI-MAPPO 알고리즘 생성
        from pci_mappo.algorithms import PCIMAPPOAlgorithm
        
        algorithm = PCIMAPPOAlgorithm(
            num_agents=env.num_agents,
            obs_dim=env.get_obs_dim(),
            action_dim=env.get_action_dim(),
            w_dim=env.get_obs_dim(),  # observation dim only (excluding previous reward)
            hidden_dim=config.network.hidden_dim,
            rnn_hidden_dim=config.network.rnn_hidden_dim,
            lr=config.network.lr,
            lr_bridge=config.network.lr_bridge,
            lr_critic=config.network.lr_critic,
            causal_lambda=config.algorithm.causal_lambda,
            gamma=config.algorithm.gamma,
            gae_lambda=config.algorithm.gae_lambda,
            clip_epsilon=config.algorithm.clip_epsilon,
            entropy_coef=config.algorithm.entropy_coef,
            value_coef=config.algorithm.value_coef,
            max_grad_norm=config.algorithm.max_grad_norm,
            ppo_epochs=config.algorithm.ppo_epochs,
            batch_size=config.algorithm.batch_size,
            num_bridge_samples=config.algorithm.num_bridge_samples,
            num_batches=config.algorithm.num_batches
        )
        print(f"✓ PCI-MAPPO 알고리즘 생성 성공")
        
        # PCI-MAPPO 트레이너 생성
        trainer = PCIMAPPOTrainer(
            env=env,
            algorithm=algorithm,
            max_iterations=config.training.max_iterations,
            max_steps_per_iteration=config.training.max_steps_per_iteration,
            log_interval=config.training.log_interval,
            save_interval=config.training.save_interval,
            eval_interval=config.training.eval_interval,
            eval_episodes=config.training.eval_episodes,
            save_dir=config.training.save_dir,
            use_wandb=config.wandb.use_wandb,
            wandb_project=config.wandb.project,
            wandb_run_name=config.wandb.run_name,
            device=config.training.device
        )
        print(f"✓ PCI-MAPPO 트레이너 생성 성공")
        
        # 짧은 훈련 실행 (5 iteration만)
        print(f"\n=== 짧은 훈련 테스트 (5 iterations) ===")
        
        # 원래 설정 백업
        original_max_iterations = config.training.max_iterations
        original_max_steps = config.training.max_steps_per_iteration
        original_log_interval = config.training.log_interval
        
        # 테스트용 설정으로 변경
        config.training.max_iterations = 5
        config.training.max_steps_per_iteration = 64
        config.training.log_interval = 1
        
        try:
            # 짧은 훈련 실행
            trainer.train()
            print(f"✓ 훈련 완료")
            
            # 평가 실행
            print(f"\n=== 평가 테스트 ===")
            eval_metrics = trainer.evaluate(num_episodes=3)
            print(f"✓ 평가 완료")
            print(f"  - 평균 보상: {eval_metrics.get('mean_reward', 'N/A')}")
            print(f"  - 평균 에피소드 길이: {eval_metrics.get('mean_episode_length', 'N/A')}")
            
        finally:
            # 원래 설정 복구
            config.training.max_iterations = original_max_iterations
            config.training.max_steps_per_iteration = original_max_steps
            config.training.log_interval = original_log_interval
        
        print(f"\n=== 테스트 완료: MPE Simple Spread + PCI-MAPPO 통합 성공! ===")
        return True
        
    except ImportError as e:
        print(f"❌ Import 오류: {e}")
        print("JAX 또는 JaxMARL이 설치되지 않았습니다.")
        print("설치 명령어:")
        print("  pip install jax jaxlib")
        print("  pip install git+https://github.com/instadeepai/jaxmarl.git")
        return False
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_environment_only():
    """환경만 테스트 (PCI-MAPPO 없이)"""
    print("\n=== MPE Simple Spread 환경 단독 테스트 ===")
    
    try:
        # 직접 환경 생성
        from pci_mappo.envs.multi_agent_env import MPESimpleSpreadEnv
        
        env = MPESimpleSpreadEnv(
            n_agents=2,
            local_ratio=0.5,
            max_cycles=25,
            continuous_actions=False,
            seed=42
        )
        
        print(f"✓ 환경 생성 성공")
        print(f"  - 에이전트 수: {env.num_agents}")
        print(f"  - 관찰 차원: {env.get_obs_dim()}")
        print(f"  - 액션 차원: {env.get_action_dim()}")
        
        # 에피소드 실행
        obs = env.reset()
        total_rewards = {i: 0.0 for i in range(env.num_agents)}
        
        for step in range(10):
            # 랜덤 액션
            actions = {}
            for agent_id in range(env.num_agents):
                actions[agent_id] = np.random.randint(0, env.get_action_dim())
            
            obs, rewards, dones, info = env.step(actions)
            
            for agent_id in range(env.num_agents):
                total_rewards[agent_id] += rewards[agent_id]
            
            print(f"Step {step + 1}: rewards={rewards}, done={any(dones.values())}")
            
            if any(dones.values()):
                break
        
        print(f"✓ 환경 테스트 완료")
        print(f"  - 총 보상: {total_rewards}")
        
        return True
        
    except Exception as e:
        print(f"❌ 환경 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_available_environments():
    """사용 가능한 환경 목록 표시"""
    print("\n=== 사용 가능한 환경 목록 ===")
    envs = get_available_environments()
    for env_name, description in envs.items():
        print(f"  - {env_name}: {description}")


if __name__ == "__main__":
    print("MPE Simple Spread + PCI-MAPPO 통합 테스트")
    print("=" * 50)
    
    # 사용 가능한 환경 목록 표시
    list_available_environments()
    
    # 환경 단독 테스트
    env_success = demo_environment_only()
    
    # PCI-MAPPO 통합 테스트
    if env_success:
        full_success = test_mpe_simple_spread()
    else:
        full_success = False
    
    print("\n" + "=" * 50)
    if full_success:
        print("🎉 모든 테스트 성공!")
    else:
        print("❌ 일부 테스트 실패")
    
    sys.exit(0 if full_success else 1)
