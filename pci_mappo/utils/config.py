import yaml
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class NetworkConfig:
    """Neural network configuration."""
    hidden_dim: int = 64
    lr: float = 3e-4
    lr_bridge: float = 3e-4
    lr_critic: float = 3e-4
    # RKHS Kernel parameters for analytical loss
    rkhs_gamma: float = 1.0  # Gamma parameter for RBF kernel


@dataclass
class AlgorithmConfig:
    """Algorithm configuration."""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    causal_lambda: float = 0.1
    causal_lambda_schedule: bool = False
    causal_lambda_start: float = 0.0
    causal_lambda_final: float = 0.2
    causal_lambda_warmup_steps: int = 10000
    causal_lambda_startingpoint: int = 0  # Starting iteration for causal lambda scheduling
    num_batches: int = 4  # Number of batches K for training
    # Epsilon greedy parameters for causal data collection
    epsilon_start: float = 0.6
    epsilon_end: float = 0.0
    epsilon_decay_steps: int = 1000
    # Buffer size limits for FIFO queue behavior
    max_buffer_size: int = 10000
    max_causal_buffer_size: int = 2000


@dataclass
class TrainingConfig:
    """Training configuration."""
    max_iterations: int = 1000
    max_steps_per_iteration: int = 2048
    log_interval: int = 1
    save_interval: int = 100
    eval_interval: int = 50
    eval_episodes: int = 10
    save_dir: str = "./checkpoints"
    device: str = "cpu"
    # Reproducibility settings
    seed: int = 42
    deterministic: bool = True


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    env_name: str = "simple_cooperative"
    name: str = "simple_cooperative"  # alias for env_name
    num_agents: int = 3
    grid_size: int = 5
    max_steps: int = 50
    cooperation_bonus: float = 2.0
    # For toy causal env
    episode_length: int = 5
    
    # For Dec-Tiger environment
    seed: int = 0
    failure_obs_prob: float = 0.15
    fully_observable: bool = False
    # For Stag-Hunt environment
    stag_move_prob: float = 0.7
    
    # For MPE Simple Spread environment
    local_ratio: float = 0.5
    max_cycles: int = 25
    continuous_actions: bool = False
    
    # For Simple Stag Hunt environment
    cooperation_reward: int = 5
    defect_alone_reward: int = 1
    defect_together_reward: int = 1
    failed_cooperation_punishment: int = -5
    eps_per_game: int = 1

    # For visualization
    use_visualization: bool = True
    visualize_freq: int = 1
    save_animations: bool = True
    animation_interval: int = 100

@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    use_wandb: bool = False
    project: str = "pci-mappo"
    run_name: Optional[str] = None
    tags: Optional[list] = None


@dataclass
class Config:
    """Main configuration class."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        config = cls()
        
        if 'network' in config_dict:
            config.network = NetworkConfig(**config_dict['network'])
        
        if 'algorithm' in config_dict:
            config.algorithm = AlgorithmConfig(**config_dict['algorithm'])
        
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        
        # 'env' 또는 'environment' 키 지원
        env_config = config_dict.get('environment', config_dict.get('env', {}))
        if env_config:
            config.environment = EnvironmentConfig(**env_config)
        
        if 'wandb' in config_dict:
            config.wandb = WandbConfig(**config_dict['wandb'])
        
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load config from YAML file."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PCI-MAPPO Training")
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    
    # Environment
    parser.add_argument('--env', type=str, default='simple_cooperative',
                        help='Environment name')
    parser.add_argument('--num-agents', type=int, default=3,
                        help='Number of agents')
    parser.add_argument('--episode-length', type=int, default=None,
                        help='Episode length (for toy_causal)')
    
    # Training
    parser.add_argument('--max-iterations', type=int, default=1000,
                        help='Maximum number of training iterations')
    parser.add_argument('--max-steps', type=int, default=2048,
                        help='Maximum steps per iteration')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Training device (cpu/cuda)')
    
    # Algorithm
    parser.add_argument('--causal-lambda', type=float, default=None,
                        help='Causal reward shaping coefficient')
    parser.add_argument('--causal-lambda-schedule', action='store_true', default=False,
                        help='Enable causal lambda scheduling')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    
    # Logging
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Save directory for checkpoints')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Logging interval')
    
    # Wandb
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='pci-mappo',
                        help='Wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Wandb run name')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic algorithms (slower but reproducible)')
    parser.add_argument('--non-deterministic', action='store_true', default=False,
                        help='Disable deterministic mode for faster training')
    
    # Evaluation and testing
    parser.add_argument('--eval', action='store_true',
                        help='Evaluation mode')
    parser.add_argument('--test', action='store_true',
                        help='Test mode')
    parser.add_argument('--causal-eval', action='store_true',
                        help='Causal effect evaluation mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for evaluation/testing')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during testing')
    parser.add_argument('--eval-episodes', type=int, default=50,
                        help='Number of episodes for evaluation')
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> Config:
    """Create config from command line arguments."""
    # Load from YAML if provided
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override with command line arguments
    if hasattr(args, 'env') and args.env:
        config.environment.env_name = args.env
    if hasattr(args, 'num_agents') and args.num_agents:
        config.environment.num_agents = args.num_agents
    if hasattr(args, 'episode_length') and args.episode_length:
        config.environment.episode_length = args.episode_length
    
    if hasattr(args, 'max_iterations') and args.max_iterations:
        config.training.max_iterations = args.max_iterations
    if hasattr(args, 'max_steps') and args.max_steps:
        config.training.max_steps_per_iteration = args.max_steps
    if hasattr(args, 'device') and args.device:
        config.training.device = args.device
    if hasattr(args, 'save_dir') and args.save_dir:
        config.training.save_dir = args.save_dir
    if hasattr(args, 'log_interval') and args.log_interval:
        config.training.log_interval = args.log_interval
    
    if hasattr(args, 'causal_lambda') and args.causal_lambda is not None:
        config.algorithm.causal_lambda = args.causal_lambda
    if hasattr(args, 'lr') and args.lr:
        config.network.lr = args.lr
        config.network.lr_bridge = args.lr
        config.network.lr_critic = args.lr
    
    if hasattr(args, 'use_wandb') and args.use_wandb:
        config.wandb.use_wandb = args.use_wandb
    if hasattr(args, 'wandb_project') and args.wandb_project:
        config.wandb.project = args.wandb_project
    if hasattr(args, 'wandb_run_name') and args.wandb_run_name:
        config.wandb.run_name = args.wandb_run_name
    
    # Reproducibility settings
    if hasattr(args, 'seed') and args.seed:
        config.training.seed = args.seed
    if hasattr(args, 'deterministic') and args.deterministic:
        config.training.deterministic = not args.non_deterministic
    
    return config
