#!/usr/bin/env python3
"""
Policy Validation Test for Stag-Hunt Environment

이 스크립트는 학습된 policy의 품질을 검증하는 예제입니다.
단순한 exploration이 아닌 전략적 판단 능력을 측정합니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from pci_mappo.training.trainer import PCIMAPPOTrainer
from pci_mappo.algorithms import PCIMAPPOAlgorithm
from pci_mappo.envs.env_utils import create_environment
from pci_mappo.utils import load_config


def test_policy_validation():
    """Policy 품질 검증 테스트를 실행합니다."""
    print("="*80)
    print("Policy Quality Validation Test for Stag-Hunt Environment")
    print("="*80)
    
    # 설정 로드
    config_path = "configs/stag_hunt.yaml"
    config = load_config(config_path)
    
    # 환경 생성
    print("Creating Stag-Hunt environment...")
    env = create_environment(config['environment'])
    print(f"Environment: {env.__class__.__name__}")
    print(f"Number of agents: {env.num_agents}")
    print(f"Grid size: {env.grid_size_x}x{env.grid_size_y}")
    print(f"Stag reward: {env.stag_reward}, Forage reward: {env.forage_reward}")
    print()
    
    # 알고리즘 생성
    print("Creating PCI-MAPPO algorithm...")
    algorithm = PCIMAPPOAlgorithm(
        env=env,
        config=config['algorithm'],
        network_config=config['network'],
        device=config['training']['device']
    )
    print("Algorithm created successfully!")
    print()
    
    # 트레이너 생성
    print("Creating trainer...")
    trainer = PCIMAPPOTrainer(
        env=env,
        algorithm=algorithm,
        max_iterations=config['training']['max_iterations'],
        max_steps_per_iteration=config['training']['max_steps_per_iteration'],
        log_interval=config['training']['log_interval'],
        save_interval=config['training']['save_interval'],
        eval_interval=config['training']['eval_interval'],
        eval_episodes=config['training']['eval_episodes'],
        save_dir=config['training']['save_dir'],
        use_wandb=config.get('wandb', {}).get('use_wandb', False),
        device=config['training']['device']
    )
    print("Trainer created successfully!")
    print()
    
    # Policy 품질 검증 실행
    print("Running policy quality validation...")
    print("This will test the policy's ability to make strategic decisions")
    print("rather than just random exploration.")
    print()
    
    # 테스트 실행 (policy 품질 검증 활성화)
    test_results = trainer.test_policy(
        num_episodes=20,  # 더 많은 에피소드로 정확한 평가
        render=False,  # 시각화 비활성화 (빠른 실행)
        save_animation=False,  # 애니메이션 저장 비활성화
        evaluate_advanced_causal=False,  # 고급 인과관계 분석 비활성화
        evaluate_policy_quality=True  # Policy 품질 검증 활성화
    )
    
    # 결과 요약
    print("\n" + "="*80)
    print("Policy Validation Results Summary")
    print("="*80)
    
    if test_results['policy_quality_metrics']:
        metrics = test_results['policy_quality_metrics']
        
        print(f"📊 협력 사냥 성공률: {metrics['cooperation_success_rate']:.3f}")
        print(f"🎯 사냥 기회 활용률: {metrics['hunting_opportunity_utilization']:.3f}")
        print(f"⚡ 효율성 지수: {metrics['efficiency_index']:.3f}")
        print(f"⚖️  협력-경쟁 균형: {metrics['cooperation_competition_balance']:.3f}")
        print()
        
        print("📈 상세 통계:")
        print(f"  • Stag 사냥 시도: {metrics['stag_hunt_attempts']} (성공: {metrics['stag_hunt_successes']})")
        print(f"  • Hare 사냥 시도: {metrics['hare_hunt_attempts']} (성공: {metrics['hare_hunt_successes']})")
        print(f"  • 사냥 가능한 Stag 기회: {metrics['huntable_stag_opportunities']}")
        print(f"  • 사냥 가능한 Hare 기회: {metrics['huntable_hare_opportunities']}")
        print(f"  • 총 에피소드: {metrics['total_episodes']}")
        print()
        
        # Policy 품질 해석
        print("🔍 Policy 품질 해석:")
        
        # 협력 능력 평가
        coop_rate = metrics['cooperation_success_rate']
        if coop_rate > 0.7:
            print("  ✅ 우수한 협력 능력: Policy가 효과적으로 협력하여 Stag를 사냥합니다.")
        elif coop_rate > 0.4:
            print("  ⚠️  보통 수준의 협력 능력: Policy가 때때로 협력하지만 일관성이 부족합니다.")
        else:
            print("  ❌ 낮은 협력 능력: Policy가 협력보다 개별 행동을 선호합니다.")
        
        # 사냥 기회 활용 평가
        hunt_util = metrics['hunting_opportunity_utilization']
        if hunt_util > 0.6:
            print("  ✅ 높은 사냥 기회 활용: Policy가 사냥 기회를 잘 인식하고 활용합니다.")
        elif hunt_util > 0.3:
            print("  ⚠️  보통 수준의 사냥 기회 활용: Policy가 일부 사냥 기회를 놓칩니다.")
        else:
            print("  ❌ 낮은 사냥 기회 활용: Policy가 사냥 기회를 제대로 인식하지 못합니다.")
        
        # 효율성 평가
        efficiency = metrics['efficiency_index']
        if efficiency > 0.5:
            print("  ✅ 높은 효율성: Policy가 전략적으로 최적의 행동을 선택합니다.")
        elif efficiency > 0.3:
            print("  ⚠️  보통 수준의 효율성: Policy가 어느 정도 전략적이지만 개선 여지가 있습니다.")
        else:
            print("  ❌ 낮은 효율성: Policy가 비효율적인 행동을 자주 선택합니다.")
        
        # 전반적 평가
        print("\n🎯 전반적 Policy 품질 평가:")
        overall_score = (coop_rate + hunt_util + efficiency) / 3
        
        if overall_score > 0.6:
            print(f"  🏆 우수한 Policy (종합 점수: {overall_score:.3f})")
            print("     → 학습된 policy가 전략적 사고와 협력 능력을 보여줍니다.")
        elif overall_score > 0.4:
            print(f"  📈 개선 가능한 Policy (종합 점수: {overall_score:.3f})")
            print("     → Policy가 기본적인 전략을 학습했지만 더 개선이 필요합니다.")
        else:
            print(f"  🔄 재학습 권장 Policy (종합 점수: {overall_score:.3f})")
            print("     → Policy가 단순한 exploration 수준에 머물러 있습니다.")
    
    else:
        print("Policy 품질 지표를 사용할 수 없습니다.")
        print("Stag-Hunt 환경에서만 이 기능을 사용할 수 있습니다.")
    
    print("\n" + "="*80)
    print("Policy Validation Test Complete!")
    print("="*80)


if __name__ == "__main__":
    test_policy_validation()
