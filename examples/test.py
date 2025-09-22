#!/usr/bin/env python3
"""
PCI-MAPPO Testing Script

Example usage:
    python test.py --checkpoint checkpoints/best_model.pt --render
    python test.py --config configs/default.yaml --checkpoint checkpoints/checkpoint_500.pt
    python test.py --env simple_cooperative --num-agents 3 --checkpoint checkpoints/final_checkpoint.pt
    python test.py --env toy_causal --checkpoint checkpoints/best_model.pt
    python test.py --env dec_tiger --checkpoint checkpoints/best_model.pt
    python test.py --env stag_hunt --checkpoint checkpoints/best_model.pt
    python test.py --env causal_testing --checkpoint checkpoints/best_model.pt
"""

import sys
import os

# Add the parent directory to Python path to find pci_mappo module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from pci_mappo.utils import create_config_from_args, Logger
from pci_mappo.envs import SimpleCooperativeEnv, PettingZooWrapper, ToyCausalEnv, DecTigerEnv, StagHuntEnv, CausalTestingEnv
from pci_mappo.algorithms import PCIMAPPOAlgorithm
from pci_mappo.training import PCIMAPPOTrainer


def parse_test_args():
    """Parse test-specific command line arguments."""
    parser = argparse.ArgumentParser(description="PCI-MAPPO Testing")
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    
    # Config file
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config YAML file')
    
    # Environment
    parser.add_argument('--env', type=str, default='simple_cooperative',
                        choices=['simple_cooperative', 'toy_causal', 'dec_tiger', 'stag_hunt', 'causal_testing', 'pettingzoo'],
                        help='Environment name')
    parser.add_argument('--num-agents', type=int, default=3,
                        help='Number of agents (for simple_cooperative)')
    
    # Testing
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Number of test episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during testing')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    
    # Environment specific parameters
    parser.add_argument('--grid-size', type=int, default=5,
                        help='Grid size for simple_cooperative environment')
    parser.add_argument('--max-steps', type=int, default=50,
                        help='Max steps per episode')
    parser.add_argument('--episode-length', type=int, default=5,
                        help='Episode length for toy_causal environment')
    parser.add_argument('--stag-max-steps', type=int, default=15,
                        help='Max steps for stag_hunt environment')
    parser.add_argument('--dec-tiger-max-steps', type=int, default=4,
                        help='Max steps for dec_tiger environment')
    parser.add_argument('--failure-obs-prob', type=float, default=0.15,
                        help='Failure observation probability for dec_tiger')
    parser.add_argument('--fully-observable', action='store_true',
                        help='Make dec_tiger environment fully observable')
    parser.add_argument('--stag-move-prob', type=float, default=0.7,
                        help='Stag movement probability for stag_hunt')
    
    # CausalTestingEnv specific parameters
    parser.add_argument('--action-dim', type=int, default=3,
                        help='Action dimension for causal_testing environment')
    parser.add_argument('--self-effect', type=float, default=2.0,
                        help='Self effect coefficient for causal_testing environment')
    parser.add_argument('--cross-effect', type=float, default=0.5,
                        help='Cross effect coefficient for causal_testing environment')
    parser.add_argument('--confounder-strength', type=float, default=1.0,
                        help='Confounder strength for causal_testing environment')
    parser.add_argument('--obs-noise-std', type=float, default=0.1,
                        help='Observation noise standard deviation for causal_testing environment')
    
    return parser.parse_args()


def create_environment_from_args(args):
    """Create environment from test arguments."""
    if args.env == "simple_cooperative":
        env = SimpleCooperativeEnv(
            num_agents=args.num_agents,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            cooperation_bonus=2.0
        )
    elif args.env == "toy_causal":
        env = ToyCausalEnv(
            num_agents=2,  # ToyCausalEnv는 항상 2개 에이전트
            episode_length=args.episode_length
        )
    elif args.env == "dec_tiger":
        env = DecTigerEnv(
            max_steps=args.dec_tiger_max_steps,
            failure_obs_prob=args.failure_obs_prob,
            fully_observable=args.fully_observable
        )
    elif args.env == "stag_hunt":
        env = StagHuntEnv(
            max_steps=args.stag_max_steps,
            stag_move_prob=args.stag_move_prob
        )
    elif args.env == "causal_testing":
        env = CausalTestingEnv(
            num_agents=2,  # CausalTestingEnv는 항상 2개 에이전트
            action_dim=args.action_dim,
            obs_dim=8,  # MODIFIED: Updated observation dimension
            max_steps=args.max_steps,
            self_effect=args.self_effect,
            cross_effect=args.cross_effect,
            confounder_strength=args.confounder_strength,
            obs_noise_std=args.obs_noise_std
        )
    elif args.env.startswith("pettingzoo"):
        env_module = args.env.replace("pettingzoo.", "")
        env = PettingZooWrapper(env_module)
    else:
        raise ValueError(f"Unknown environment: {args.env}")
    
    return env


def create_algorithm_from_checkpoint(env, checkpoint_path, device, config=None):
    """Create algorithm and load from checkpoint."""
    # Load checkpoint with weights_only=False for compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get network dimensions from config or use defaults that match the checkpoint
    if config and hasattr(config, 'network'):
        hidden_dim = config.network.hidden_dim
        rnn_hidden_dim = config.network.rnn_hidden_dim
    else:
        # Default values that match the checkpoint (32 hidden_dim, 16 rnn_hidden_dim)
        hidden_dim = 32
        rnn_hidden_dim = 16
    
    # Create algorithm with environment-specific parameters
    algorithm = PCIMAPPOAlgorithm(
        num_agents=env.num_agents,
        obs_dim=env.get_obs_dim(),
        action_dim=env.get_action_dim(),
        w_dim=env.get_obs_dim(),
        hidden_dim=hidden_dim,
        rnn_hidden_dim=rnn_hidden_dim
    )
    
    # Load algorithm state
    algorithm_state = checkpoint['algorithm_state']
    
    # Load network states
    algorithm.bridge_net.load_state_dict(algorithm_state['bridge_net'])
    algorithm.critic_net.load_state_dict(algorithm_state['critic_net'])
    
    # Load agent states
    for agent_id, agent_state in algorithm_state['agents'].items():
        agent = algorithm.agents[int(agent_id)]
        agent.policy_net.load_state_dict(agent_state['policy_net'])
    
    return algorithm


def test_interactive_demo(env, algorithm, logger):
    """Run an interactive demo showing agent decisions."""
    logger.info("Starting interactive demo...")
    
    observations = env.reset()
    algorithm.reset_agents()
    
    from pci_mappo.envs import dict_to_tensor
    observations = dict_to_tensor(observations)
    
    episode_reward = {agent_id: 0.0 for agent_id in range(env.num_agents)}
    step = 0
    done = False
    
    if hasattr(env, 'render'):
        env.render()
    
    while not done and step < 100:
        # Get actions
        actions = {}
        with torch.no_grad():
            for agent_id, obs in observations.items():
                action, _, _ = algorithm.agents[agent_id].get_action(obs)
                actions[agent_id] = action
        
        # Show actions with environment-specific action names
        action_names = get_action_names(env)

        logger.info(f"Step {step + 1}:")
        for agent_id, action in actions.items():
            action_name = action_names.get(action, f"action_{action}")
            logger.info(f"  Agent {agent_id}: {action_name}")
        
        # Take step
        next_observations, rewards, dones, info = env.step(actions)
        next_observations = dict_to_tensor(next_observations)
        
        # Update rewards
        for agent_id, reward in rewards.items():
            episode_reward[agent_id] += reward
        
        logger.info(f"  Rewards: {rewards}")
        logger.info(f"  Cumulative: {episode_reward}")
        
        if hasattr(env, 'render'):
            env.render()
        
        observations = next_observations
        step += 1
        done = any(dones.values())
        
        # Wait for user input
        input("Press Enter to continue...")
    
    logger.info(f"Demo completed! Final rewards: {episode_reward}")


def get_action_names(env):
    """Get action names for different environments."""
    if isinstance(env, SimpleCooperativeEnv):
        return {0: "stay", 1: "up", 2: "down", 3: "left", 4: "right"}
    elif isinstance(env, ToyCausalEnv):
        return {0: "left", 1: "right"}
    elif isinstance(env, DecTigerEnv):
        return {0: "open-left", 1: "open-right", 2: "listen"}
    elif isinstance(env, StagHuntEnv):
        return {0: "up", 1: "down", 2: "left", 3: "right", 4: "hunt"}
    elif isinstance(env, CausalTestingEnv):
        return {i: f"action_{i}" for i in range(env.get_action_dim())}
    else:
        # Default action names for unknown environments
        return {i: f"action_{i}" for i in range(env.get_action_dim())}


def main():
    """Main testing function."""
    args = parse_test_args()
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Create logger
    logger = Logger()
    logger.info("Starting PCI-MAPPO testing")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Number of episodes: {args.num_episodes}")
    
    # Create environment
    logger.info("Creating environment...")
    env = create_environment_from_args(args)
    logger.info(f"Environment: {env.__class__.__name__}")
    logger.info(f"Agents: {env.num_agents}, Obs dim: {env.get_obs_dim()}, Action dim: {env.get_action_dim()}")
    
    # Print environment-specific information
    if isinstance(env, SimpleCooperativeEnv):
        logger.info(f"Grid size: {env.grid_size}, Max steps: {env.max_steps}")
    elif isinstance(env, ToyCausalEnv):
        logger.info(f"Episode length: {env.episode_length}")
    elif isinstance(env, DecTigerEnv):
        logger.info(f"Max steps: {env.maxtimestep}, Failure obs prob: {env.failure_obs_prob}")
    elif isinstance(env, StagHuntEnv):
        logger.info(f"Max steps: {env.max_steps}, Stag move prob: {env.stag_move_prob}")
    elif isinstance(env, CausalTestingEnv):
        logger.info(f"Max steps: {env.max_steps}, Action dim: {env.action_dim}")
        logger.info(f"Self effect: {env.self_effect}, Cross effect: {env.cross_effect}")
        logger.info(f"Confounder strength: {env.confounder_strength}, Obs noise std: {env.obs_noise_std}")
    
    # Load config if provided
    config = None
    if args.config:
        logger.info(f"Loading config from: {args.config}")
        config = create_config_from_args(args)
    
    # Create algorithm and load checkpoint
    logger.info("Loading checkpoint...")
    algorithm = create_algorithm_from_checkpoint(env, args.checkpoint, device, config)
    logger.info("Checkpoint loaded successfully")
    
    # Create trainer for testing (disable checkpoint saving)
    trainer = PCIMAPPOTrainer(
        env=env,
        algorithm=algorithm,
        device=device,
        save_dir=None,  # Disable checkpoint saving during testing
        use_wandb=False  # Disable wandb during testing
    )
    
    # Run tests
    logger.info(f"Running {args.num_episodes} test episodes...")
    test_results = trainer.test_policy(
        num_episodes=args.num_episodes,
        render=args.render
    )
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("TEST RESULTS")
    logger.info("="*50)
    logger.info(f"Mean Reward: {test_results['mean_reward']:.3f} ± {test_results['std_reward']:.3f}")
    logger.info(f"Mean Episode Length: {test_results['mean_length']:.1f}")
    logger.info("\nAgent-specific results:")
    for agent_id, reward in test_results['agent_rewards'].items():
        logger.info(f"  Agent {agent_id}: {reward:.3f}")
    
    # Ask if user wants interactive demo
    try:
        demo_choice = input("\nRun interactive demo? (y/n): ").lower().strip()
        if demo_choice == 'y':
            test_interactive_demo(env, algorithm, logger)
    except KeyboardInterrupt:
        logger.info("\nTesting interrupted by user")
    
    logger.info("Testing completed!")


if __name__ == "__main__":
    main()
