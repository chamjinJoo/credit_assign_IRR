#!/usr/bin/env python3
"""
PCI-MAPPO Training Script

Example usage:
    python main.py --config configs/default.yaml
    python main.py --env simple_cooperative --num-agents 3 --use-wandb
    python main.py --config configs/small_scale.yaml --device cuda
"""

import os
import torch
from utils import parse_args, create_config_from_args, Logger, get_causal_evaluator
from utils.seed_utils import set_global_seed, ensure_reproducible_environment, log_reproducibility_info
from envs import create_environment, get_available_environments
from algorithms import PCIMAPPOAlgorithm, MAPPOAlgorithm, IPPOAlgorithm
from runner import Trainer


def create_algorithm(env, config):
    """Create algorithm based on config.algorithm.type."""
    network_config = config.network
    algorithm_config = config.algorithm
    
    # Get required_agents from environment if available (for StagHunt)
    env_required_agents = None
    if hasattr(env, 'required_agents'):
        env_required_agents = env.required_agents
    
    algorithm_type = getattr(algorithm_config, 'type', 'pci_mappo')
    
    if algorithm_type == 'pci_mappo':
        algorithm = PCIMAPPOAlgorithm(
            num_agents=env.num_agents,
            obs_dim=env.get_obs_dim(),
            action_dim=env.get_action_dim(),
            w_dim=env.get_obs_dim(), # observation dim only (excluding previous reward)
            hidden_dim=network_config.hidden_dim,
            lr=network_config.lr,
            lr_bridge=network_config.lr_bridge,
            lr_critic=network_config.lr_critic,
            causal_lambda=algorithm_config.causal_lambda,
            causal_lambda_schedule=getattr(algorithm_config, 'causal_lambda_schedule', False),
            causal_lambda_start=getattr(algorithm_config, 'causal_lambda_start', 0.0),
            causal_lambda_final=getattr(algorithm_config, 'causal_lambda_final', 0.2),
            causal_lambda_warmup_steps=getattr(algorithm_config, 'causal_lambda_warmup_steps', 10000),
            causal_lambda_startingpoint=getattr(algorithm_config, 'causal_lambda_startingpoint', 0),
            gamma=algorithm_config.gamma,
            gae_lambda=algorithm_config.gae_lambda,
            clip_epsilon=algorithm_config.clip_epsilon,
            entropy_coef=algorithm_config.entropy_coef,
            value_coef=algorithm_config.value_coef,
            max_grad_norm=algorithm_config.max_grad_norm,
            ppo_epochs=algorithm_config.ppo_epochs,
            batch_size=algorithm_config.batch_size,
            num_batches=algorithm_config.num_batches,
            # Buffer size limits for FIFO queue behavior
            max_buffer_size=getattr(algorithm_config, 'max_buffer_size', 10000),
            max_causal_buffer_size=getattr(algorithm_config, 'max_causal_buffer_size', 2000),
            device=config.training.device,
            # RKHS Kernel parameters for analytical loss
            rkhs_gamma=getattr(network_config, 'rkhs_gamma', 1.0),
            # Environment required_agents for accurate scenario classification
            env_required_agents=env_required_agents,
        )
    elif algorithm_type == 'mappo':
        algorithm = MAPPOAlgorithm(
            num_agents=env.num_agents,
            obs_dim=env.get_obs_dim(),
            action_dim=env.get_action_dim(),
            hidden_dim=network_config.hidden_dim,
            lr=network_config.lr,
            lr_critic=network_config.lr_critic,
            gamma=algorithm_config.gamma,
            gae_lambda=algorithm_config.gae_lambda,
            clip_epsilon=algorithm_config.clip_epsilon,
            entropy_coef=algorithm_config.entropy_coef,
            value_coef=algorithm_config.value_coef,
            max_grad_norm=algorithm_config.max_grad_norm,
            ppo_epochs=algorithm_config.ppo_epochs,
            batch_size=algorithm_config.batch_size,
            num_batches=algorithm_config.num_batches,
            epsilon_start=getattr(algorithm_config, 'epsilon_start', 0.6),
            epsilon_end=getattr(algorithm_config, 'epsilon_end', 0.0),
            epsilon_decay_steps=getattr(algorithm_config, 'epsilon_decay_steps', 1000),
            device=config.training.device,
        )
    elif algorithm_type == 'ippo':
        algorithm = IPPOAlgorithm(
            num_agents=env.num_agents,
            obs_dim=env.get_obs_dim(),
            action_dim=env.get_action_dim(),
            hidden_dim=network_config.hidden_dim,
            lr=network_config.lr,
            lr_critic=network_config.lr_critic,
            gamma=algorithm_config.gamma,
            gae_lambda=algorithm_config.gae_lambda,
            clip_epsilon=algorithm_config.clip_epsilon,
            entropy_coef=algorithm_config.entropy_coef,
            value_coef=algorithm_config.value_coef,
            max_grad_norm=algorithm_config.max_grad_norm,
            ppo_epochs=algorithm_config.ppo_epochs,
            batch_size=algorithm_config.batch_size,
            num_batches=algorithm_config.num_batches,
            epsilon_start=getattr(algorithm_config, 'epsilon_start', 0.6),
            epsilon_end=getattr(algorithm_config, 'epsilon_end', 0.0),
            epsilon_decay_steps=getattr(algorithm_config, 'epsilon_decay_steps', 1000),
            device=config.training.device,
        )
    else:
        raise ValueError(f"Invalid algorithm type: {algorithm_type}. Must be 'pci_mappo', 'mappo', or 'ippo'")
    
    return algorithm


def run_causal_evaluation(env, algorithm, args, logger):
    """
    Causal effect 평가를 실행합니다.
    
    Args:
        env: 환경 객체
        algorithm: PCI-MAPPO 알고리즘 객체
        args: 명령행 인자
        logger: 로거 객체
    """
    # StagHunt 환경인지 확인
    if not hasattr(env, '__class__') or 'StagHunt' not in env.__class__.__name__:
        logger.error("Causal effect evaluation은 StagHunt 환경에서만 지원됩니다.")
        return
    
    # Causal effect 평가자 생성
    CausalEffectEvaluator = get_causal_evaluator()
    evaluator = CausalEffectEvaluator(
        algorithm=algorithm,
        env=env,
        num_eval_steps=1000,
        device=algorithm.device
    )
    
    # Causal effect 평가 실행
    statistics = evaluator.evaluate_causal_effects()
    
    # 결과 저장 - training_summary.json과 같은 폴더에 저장
    checkpoint_dir = os.path.dirname(args.checkpoint)
    results_dir = checkpoint_dir  # training_summary.json과 같은 폴더
    
    # 통계 저장
    evaluator.save_results(statistics, results_dir)
    
    # 플롯 생성 및 저장
    plot_path = os.path.join(results_dir, 'causal_effect_plot.png')
    evaluator.plot_causal_effects(statistics, save_path=plot_path)
    
    # t-SNE plot 생성
    tsne_path = os.path.join(results_dir, 'observation_tsne.png')
    logger.info("Observation t-SNE 분석 시작...")
    evaluator.plot_tsne_observations(
        save_path=tsne_path,
        perplexity=30
    )
    
    logger.info(f"Causal effect 평가 완료. 결과가 저장되었습니다: {results_dir}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Create config
    config = create_config_from_args(args)
    
    # Set up reproducibility BEFORE any other operations
    ensure_reproducible_environment()
    actual_seed = set_global_seed(config.training.seed, config.training.deterministic)
    
    # Set device
    device = config.training.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
        config.training.device = device
    
    # Get algorithm type for logger name
    algorithm_type = getattr(config.algorithm, 'type', 'pci_mappo').upper()
    
    # Create logger
    logger = Logger(name=algorithm_type)
    logger.info(f"Starting {algorithm_type} training")
    
    # Log reproducibility information
    log_reproducibility_info(actual_seed, logger)
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Create environment
    logger.info("Creating environment...")
    # Add seed to environment config
    env_config = config.environment
    if hasattr(env_config, '__dict__'):
        env_config.seed = actual_seed
    else:
        env_config['seed'] = actual_seed
    env = create_environment(env_config)
    logger.info(f"Environment created: {env.__class__.__name__}")
    logger.info(f"Number of agents: {env.num_agents}")
    logger.info(f"Observation dimension: {env.get_obs_dim()}")
    logger.info(f"Action dimension: {env.get_action_dim()}")
    
    # Auto-generate wandb run_name based on environment if not explicitly set
    algorithm_type = getattr(config.algorithm, 'type', 'pci_mappo')
    if not config.wandb.run_name:
        env_class_name = env.__class__.__name__
        # For LBForagingWrapper, use env_id
        if env_class_name == 'LBForagingWrapper' and hasattr(env, 'env_id'):
            env_id = env.env_id
            # Remove -v3 suffix from env_id if present
            if env_id.endswith('-v3'):
                env_id = env_id[:-3]
            config.wandb.run_name = env_id
        else:
            # For other environments, use environment class name
            env_name = env_class_name.lower()
            config.wandb.run_name = env_name

    # Log available environments
    available_envs = get_available_environments()
    logger.info(f"Available environments: {list(available_envs.keys())}")
    
    # Create algorithm
    logger.info("Creating algorithm...")
    algorithm = create_algorithm(env, config)
    logger.info("Algorithm created")
    
    # Create trainer
    logger.info("Creating trainer...")
    
    # Test 모드인 경우 checkpoint 저장 비활성화
    is_test_mode = hasattr(args, 'test') and args.test
    trainer_save_dir = None if is_test_mode else config.training.save_dir
    
    if is_test_mode:
        logger.info("Test mode: Model checkpoint saving disabled")
    
    # wandb run name에 알고리즘 타입 추가
    algorithm_type = getattr(config.algorithm, 'type', 'pci_mappo')
    wandb_run_name = config.wandb.run_name
    if wandb_run_name:
        # run_name이 있으면 앞에 알고리즘 타입 추가
        wandb_run_name = f"{algorithm_type}_{wandb_run_name}"
    else:
        # run_name이 없으면 알고리즘 타입만 사용
        wandb_run_name = algorithm_type
    
    trainer = Trainer(
        env=env,
        algorithm=algorithm,
        max_iterations=config.training.max_iterations,
        max_steps_per_iteration=config.training.max_steps_per_iteration,
        log_interval=config.training.log_interval,
        save_interval=config.training.save_interval,
        eval_interval=config.training.eval_interval,
        eval_episodes=config.training.eval_episodes,
        save_dir=trainer_save_dir,
        use_wandb=config.wandb.use_wandb,
        wandb_project=config.wandb.project,
        wandb_run_name=wandb_run_name,
        device=device
    )
    # Check for evaluation or test mode
    if hasattr(args, 'eval') and args.eval:
        if args.checkpoint:
            logger.info(f"Loading checkpoint: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)
        
        logger.info("Running evaluation...")
        eval_results = trainer.evaluate_policy()
        logger.info(f"Evaluation results: {eval_results}")
        return
    
    # Always run causal effect evaluation after training
    causal_eval_enabled = True
    
    if hasattr(args, 'test') and args.test:
        if args.checkpoint:
            logger.info(f"Loading checkpoint: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)
        else:
            logger.info("No checkpoint provided, using random policy")
        
        logger.info("Running test...")
        
        # Ask user about animation saving
        save_animation = input("Test episode들을 animation으로 저장하시겠습니까? (y/n): ").lower().strip()
        save_animation = save_animation in ['y', 'yes', '예', 'ㅇ']
        
        if save_animation:
            animation_dir = input("Animation 저장 디렉토리를 입력하세요 (기본값: ./animations): ").strip()
            if not animation_dir:
                animation_dir = "./animations"
            logger.info(f"Animation 저장 디렉토리: {animation_dir}")
        
        # Test 모드에서 advanced causal evaluation 결과를 저장하기 위해 임시 save_dir 설정
        if args.checkpoint:
            # checkpoint 파일이 있으면 그 디렉토리를 사용
            checkpoint_dir = os.path.dirname(args.checkpoint)
            temp_save_dir = checkpoint_dir
        else:
            # checkpoint가 없으면 현재 디렉토리에 test_results 폴더 생성
            temp_save_dir = "./test_results"
            os.makedirs(temp_save_dir, exist_ok=True)
        
        # Temporarily set save_dir for evaluation results only
        original_save_dir = trainer.save_dir
        trainer.save_dir = temp_save_dir
        
        test_results = trainer.test_policy(
            num_episodes=10,
            render=getattr(args, 'render', False),
            save_animation=save_animation,
            animation_dir=animation_dir if save_animation else "./animations",
            evaluate_advanced_causal=True,
            advanced_causal_steps=5000,
        )
        
        # Restore original save_dir (None for test mode)
        trainer.save_dir = original_save_dir
        
        logger.info(f"Test results: {test_results}")
        logger.info(f"Advanced causal evaluation results saved to: {temp_save_dir}/advanced_causal_evaluation")
        return
    
    # Resume from checkpoint if provided
    if hasattr(args, 'checkpoint') and args.checkpoint:
        logger.info(f"Resuming from checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Run causal effect evaluation after training completion
        if causal_eval_enabled:
            logger.info("Running causal effect evaluation after training...")
            # Use the latest checkpoint for evaluation
            latest_checkpoint = trainer.get_latest_checkpoint_path()
            if latest_checkpoint:
                # Load checkpoint before evaluation
                logger.info(f"Loading checkpoint for causal evaluation: {latest_checkpoint}")
                trainer.load_checkpoint(latest_checkpoint)

                # Create a mock args object for causal evaluation
                class MockArgs:
                    def __init__(self, checkpoint_path, eval_episodes=50):
                        self.checkpoint = checkpoint_path
                        self.eval_episodes = eval_episodes
                
                mock_args = MockArgs(latest_checkpoint, eval_episodes=50)
                run_causal_evaluation(env, algorithm, mock_args, logger)
            else:
                logger.warning("No checkpoint found for causal effect evaluation")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint()
        logger.info("Checkpoint saved before exit")
        
        # Run causal effect evaluation even after interruption if checkpoint exists
        if causal_eval_enabled:
            logger.info("Running causal effect evaluation after interruption...")
            latest_checkpoint = trainer.get_latest_checkpoint_path()
            if latest_checkpoint:
                logger.info(f"Loading checkpoint for causal evaluation: {latest_checkpoint}")
                trainer.load_checkpoint(latest_checkpoint)

                class MockArgs:
                    def __init__(self, checkpoint_path, eval_episodes=50):
                        self.checkpoint = checkpoint_path
                        self.eval_episodes = eval_episodes
                
                mock_args = MockArgs(latest_checkpoint, eval_episodes=50)
                run_causal_evaluation(env, algorithm, mock_args, logger)
            else:
                logger.warning("No checkpoint found for causal effect evaluation")
                
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
