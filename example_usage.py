#!/usr/bin/env python3
"""
PCI-MAPPO Example Usage

이 파일은 PCI-MAPPO를 사용하는 방법을 보여주는 예제입니다.
"""

import torch
import numpy as np
from pci_mappo.envs import SimpleCooperativeEnv, StagHuntEnv, dict_to_tensor
from pci_mappo.algorithms import PCIMAPPOAlgorithm
from pci_mappo.training import PCIMAPPOTrainer
from pci_mappo.utils import Config, Logger


def basic_usage_example():
    """기본 사용법 예제"""
    print("=== PCI-MAPPO 기본 사용법 예제 ===\n")
    
    # 로거 설정
    logger = Logger()
    
    # 환경 생성
    logger.info("환경 생성 중...")
    env = SimpleCooperativeEnv(
        num_agents=3,
        grid_size=5,
        max_steps=50,
        cooperation_bonus=2.0
    )
    
    # 알고리즘 생성
    logger.info("PCI-MAPPO 알고리즘 생성 중...")
    algorithm = PCIMAPPOAlgorithm(
        num_agents=env.num_agents,
        obs_dim=env.get_obs_dim(),
        action_dim=env.get_action_dim(),
        w_dim=env.get_obs_dim(),
        hidden_dim=64,
        rnn_hidden_dim=32,
        causal_lambda=0.1
    )
    # 간단한 에피소드 실행
    logger.info("샘플 에피소드 실행 중...")
    observations = env.reset()
    algorithm.reset_agents()
    observations = dict_to_tensor(observations)
    
    total_rewards = {i: 0.0 for i in range(env.num_agents)}
    
    for step in range(20):
        # 액션 선택
        actions = algorithm.get_actions(observations)
        
        # 환경에서 스텝 실행
        next_observations, rewards, dones, info = env.step(actions)
        next_observations = dict_to_tensor(next_observations)
        
        # 보상 누적
        for agent_id in range(env.num_agents):
            total_rewards[agent_id] += rewards[agent_id]
        
        # 다음 관찰로 업데이트
        observations = next_observations
        
        # 에피소드 종료 확인
        if any(dones.values()):
            break
    
    logger.info(f"에피소드 완료! 총 보상: {total_rewards}")
    logger.info("기본 사용법 예제 완료\n")


def stag_hunt_example():
    """Stag-Hunt 환경 사용 예제"""
    print("=== Stag-Hunt 환경 사용 예제 ===\n")
    
    # 로거 설정
    logger = Logger()
    
    # Stag-Hunt 환경 생성
    logger.info("Stag-Hunt 환경 생성 중...")
    env = StagHuntEnv(
        max_steps=50,
        stag_move_prob=0.7
    )
    
    # 환경 정보 출력
    logger.info(f"에이전트 수: {env.num_agents}")
    logger.info(f"관찰 차원: {env.get_obs_dim()}")
    logger.info(f"액션 차원: {env.get_action_dim()}")
    
    # 알고리즘 생성
    logger.info("PCI-MAPPO 알고리즘 생성 중...")
    algorithm = PCIMAPPOAlgorithm(
        num_agents=env.num_agents,
        obs_dim=env.get_obs_dim(),
        action_dim=env.get_action_dim(),
        w_dim=env.get_obs_dim(),
        hidden_dim=64,
        rnn_hidden_dim=32,
        causal_lambda=0.1,
        causal_lambda_schedule=True,
        causal_lambda_start=0.0,
        causal_lambda_final=0.2,
        causal_lambda_warmup_steps=10000
    )
    
    # 샘플 에피소드 실행
    logger.info("Stag-Hunt 샘플 에피소드 실행 중...")
    observations = env.reset()
    algorithm.reset_agents()
    observations = dict_to_tensor(observations)
    
    total_rewards = {i: 0.0 for i in range(env.num_agents)}
    
    for step in range(10):  # 짧은 에피소드로 테스트
        # 환경 렌더링 (선택사항)
        if step % 3 == 0:
            env.render()
        
        # 액션 선택
        actions = algorithm.get_actions(observations)
        
        # 환경에서 스텝 실행
        next_observations, rewards, dones, info = env.step(actions)
        next_observations = dict_to_tensor(next_observations)
        
        # 보상 누적
        for agent_id in range(env.num_agents):
            total_rewards[agent_id] += rewards[agent_id]
        
        # 다음 관찰로 업데이트
        observations = next_observations
        
        # 에피소드 종료 확인
        if any(dones.values()):
            logger.info(f"에피소드 종료 (스텝 {step})")
            break
    
    logger.info(f"Stag-Hunt 에피소드 완료! 총 보상: {total_rewards}")
    logger.info("Stag-Hunt 사용법 예제 완료\n")


def config_usage_example():
    """설정 파일 사용 예제"""
    print("\n=== 설정 파일 사용 예제 ===\n")
    
    logger = Logger()
    
    # 기본 설정 생성
    config = Config()
    
    # 설정 수정
    config.environment.num_agents = 4
    config.algorithm.causal_lambda = 0.2
    config.training.max_iterations = 500
    
    logger.info("수정된 설정:")
    for section_name, section in config.to_dict().items():
        logger.info(f"  {section_name}: {section}")
    
    # 설정을 YAML로 저장
    config.to_yaml("example_config.yaml")
    logger.info("설정이 example_config.yaml에 저장되었습니다.")
    
    # YAML에서 설정 로드
    loaded_config = Config.from_yaml("example_config.yaml")
    logger.info("YAML에서 설정을 성공적으로 로드했습니다.")


def training_example():
    """트레이닝 예제"""
    print("\n=== 트레이닝 예제 ===\n")
    
    logger = Logger()
    
    # 환경과 알고리즘 생성
    env = SimpleCooperativeEnv(num_agents=2, grid_size=3, max_steps=25)
    algorithm = PCIMAPPOAlgorithm(
        num_agents=env.num_agents,
        obs_dim=env.get_obs_dim(),
        action_dim=env.get_action_dim(),
        w_dim=env.get_obs_dim(),
        causal_lambda=0.05,
        causal_lambda_schedule=True,
        causal_lambda_start=0.0,
        causal_lambda_final=0.2,
        causal_lambda_warmup_steps=10000
    )
    
    # 트레이너 생성 (짧은 트레이닝을 위한 설정)
    trainer = PCIMAPPOTrainer(
        env=env,
        algorithm=algorithm,
        max_iterations=20,  # 짧은 예제를 위해 적게 설정
        max_steps_per_iteration=256,
        log_interval=5,
        save_interval=10,
        eval_interval=10,
        eval_episodes=3,
        save_dir="./example_checkpoints"
    )
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("트레이닝이 중단되었습니다.")


def network_analysis_example():
    """네트워크 분석 예제"""
    print("\n=== 네트워크 분석 예제 ===\n")
    
    logger = Logger()
    
    # 알고리즘 생성
    algorithm = PCIMAPPOAlgorithm(
        num_agents=3,
        obs_dim=10,
        action_dim=5,
        w_dim=10
    )
    
    # 더미 데이터 생성
    batch_size = 32
    obs = torch.randn(batch_size, 10)
    actions = torch.randint(0, 5, (batch_size,))
    context = torch.randn(batch_size, 32)  # RNN hidden state 크기
    
    logger.info("네트워크 순전파 테스트...")
    
    # Policy network 테스트
    agent = algorithm.agents[0]
    action_logits, hidden = agent.policy_net(obs)
    logger.info(f"Policy network 출력 형태: {action_logits.shape}")
    
    # Value network 테스트
    values = agent.value_net(obs)
    logger.info(f"Value network 출력 형태: {values.shape}")
    
    # Bridge network 테스트
    actions_one_hot = torch.zeros(batch_size, 5)
    actions_one_hot.scatter_(1, actions.unsqueeze(1), 1)
    confounders = torch.randn(batch_size, 10)  # obs_dim only
    bridge_output = algorithm.bridge_net(actions_one_hot, confounders)
    logger.info(f"Bridge network 출력 형태: {bridge_output.shape}")
    
    # Critic network 테스트
    critic_output = algorithm.critic_net(context, actions_one_hot)
    logger.info(f"Critic network 출력 형태: {critic_output.shape}")
    
    # Causal reward shaping 테스트
    # Use actual data: context[0] as z_i_t, obs[0] as o_i_t, context[0] as h_i_t_minus_1
    causal_shaping = algorithm.causal_intrinsic_reward(0, context[0], obs[0], context[0].unsqueeze(0).unsqueeze(0))
    logger.info(f"Causal reward shaping: {causal_shaping:.4f}")


def environment_comparison_example():
    """다양한 환경 비교 예제"""
    print("\n=== 환경 비교 예제 ===\n")
    
    logger = Logger()
    
    # 다양한 설정의 환경들
    envs = {
        "작은 환경": SimpleCooperativeEnv(num_agents=2, grid_size=3, cooperation_bonus=1.0),
        "중간 환경": SimpleCooperativeEnv(num_agents=3, grid_size=5, cooperation_bonus=2.0),
        "큰 환경": SimpleCooperativeEnv(num_agents=4, grid_size=7, cooperation_bonus=3.0)
    }
    
    for env_name, env in envs.items():
        logger.info(f"\n{env_name}:")
        logger.info(f"  에이전트 수: {env.num_agents}")
        logger.info(f"  관찰 차원: {env.get_obs_dim()}")
        logger.info(f"  액션 차원: {env.get_action_dim()}")
        
        # 샘플 에피소드 실행
        obs = env.reset()
        total_reward = 0
        cooperation_events = 0
        
        for step in range(20):
            # 랜덤 액션
            actions = {i: np.random.randint(0, env.get_action_dim()) for i in range(env.num_agents)}
            next_obs, rewards, dones, info = env.step(actions)
            
            total_reward += sum(rewards.values())
            cooperation_events += info.get('cooperation_events', 0)
            
            if any(dones.values()):
                break
        
        logger.info(f"  샘플 에피소드 총 리워드: {total_reward:.2f}")
        logger.info(f"  협력 이벤트 수: {cooperation_events}")


def main():
    """메인 함수"""
    print("PCI-MAPPO 예제 실행\n")
    
    try:
        # 각 예제 실행
        basic_usage_example()
        stag_hunt_example()
        config_usage_example()
        network_analysis_example()
        environment_comparison_example()
        
        # 트레이닝 예제는 선택적으로 실행
        run_training = input("\n트레이닝 예제를 실행하시겠습니까? (y/n): ").lower().strip()
        if run_training == 'y':
            training_example()
        
        print("\n모든 예제가 완료되었습니다!")
        
    except Exception as e:
        print(f"예제 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
