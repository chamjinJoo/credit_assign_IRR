import yaml
import argparse
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field


@dataclass
class NetworkConfig:
    """Neural network configuration."""
    hidden_dim: int = 64
    lr_actor: float = 3e-4
    lr_bridge: float = 3e-4
    lr_critic: float = 3e-4
    rkhs_gamma: float = 1.0  # Gamma parameter for RBF kernel
    recurrent_N: int = 1  # Number of recurrent layers


@dataclass
class AlgorithmConfig:
    """Algorithm configuration."""
    type: str = "r_pci" 
    gamma: float = 0.99
    gae_lambda: float = 0.95
    rnn_chunk_length: int = 50
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64

    clip_delta: float = 0.2
    causal_lambda: float = 0.1
    causal_lambda_schedule: bool = False
    causal_lambda_adaptive_per_agent: bool = False
    causal_lambda_adaptive_alpha: float = 0.9
    causal_lambda_adaptive_min: float = 0.1
    causal_lambda_adaptive_max: float = 1.0
    causal_lambda_adaptive_error_threshold: float = 0.5
    causal_lambda_start: float = 0.0
    causal_lambda_final: float = 0.2
    causal_lambda_warmup_steps: int = 10000
    causal_lambda_startingpoint: int = 0  # Starting iteration for causal lambda scheduling
    num_batches: int = 4  # Number of batches K for training
    causal_mini_batch_size: int = 50  # Chunk size for RKHS solver
    use_value_parameter_sharing: bool = False  # If True, use single shared value network for all agents
    use_centralized_critic: bool = False  # If True, use MAPPO-style centralized critic (obs_dim * num_agents input)
    
    use_episode_buffer: bool = False
    # Buffer size limits for FIFO queue behavior
    max_buffer_size: int = 10000

@dataclass
class TrainingConfig:
    """Training configuration."""
    max_iterations: int = 1000
    max_steps_per_iteration: int = 2048
    log_interval: int = 1
    save_interval: int = 100
    eval_interval: int = 50
    eval_episodes: int = 10
    save_dir: str = "./save_checkpoints"
    device: str = "cpu"
    n_rollout_threads: int = 1  # [ADDED] Number of parallel environments for rollout collection
    episodes_per_iteration: Optional[int] = None  # [ADDED] Number of episodes to collect per iteration
    # Reproducibility settings
    seed: int = 42
    deterministic: bool = True


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    env_name: str = "simple_cooperative"
    name: str = "simple_cooperative"  # alias for env_name
    num_agents: int = 2
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
    coop_bonus: float = 1.0
    forage_reward: float = 3.0
    stag_reward: float = 5.0
    mauling_punishment: float = -1.0
    grid_observation: bool = False
    forage_quantity: Optional[int] = None
    obs_noise_std: float = 0.0

    # For MPE Simple Spread environment
    local_ratio: float = 0.5
    max_cycles: int = 25
    continuous_actions: bool = False

    # For LBForaging environment
    env_id: str = "Foraging-8x8-2p-1f-v3"
    field_size: int = 7
    max_food: int = 2
    player_level: Tuple[int, int] = (1, 1)
    food_level: Tuple[int, int] = (1, 3)
    sight: int = 7
    max_episode_steps: int = 50
    force_coop: bool = False
    grid_observation: bool = True

    # For Simple Stag Hunt environment
    cooperation_reward: int = 5
    defect_alone_reward: int = 1
    defect_together_reward: int = 1
    failed_cooperation_punishment: int = -5
    eps_per_game: int = 1

    # For SMAC environment
    map_name: str = "3m"
    
    # For SMACv2 environment
    capability_config: Optional[Dict[str, Any]] = None   

    # For visualization
    use_visualization: bool = True
    visualize_freq: int = 1
    save_animations: bool = True
    animation_interval: int = 100

    # Generic args for any environment
    env_args: Dict[str, Any] = field(default_factory=dict)

    def get_relevant_config(self) -> Dict[str, Any]:
        """Get configuration dictionary with only relevant fields for the current environment."""
        base_config = asdict(self)
        relevant_config = {}
        
        # Common fields for all environments
        common_fields = {
            'env_name', 'name', 'num_agents', 'seed', 'use_visualization', 
            'visualize_freq', 'save_animations', 'animation_interval', 'env_args'
        }
        
        # Environment specific fields map
        env_specific_fields = {
            'simple_cooperative': {'max_steps', 'cooperation_bonus', 'episode_length'},
            'toy_causal': {'episode_length'},
            'dec_tiger': {'failure_obs_prob', 'fully_observable'},
            'staghunt': {
                'stag_move_prob', 'coop_bonus', 'forage_reward', 'stag_reward', 
                'mauling_punishment', 'grid_observation', 'forage_quantity', 'max_steps'
            },
            'mpe_simple_spread': {'local_ratio', 'max_cycles', 'continuous_actions'},
            'lbforaging': {
                'env_id', 'field_size', 'max_food', 'player_level', 'food_level', 
                'sight', 'max_episode_steps', 'force_coop', 'grid_observation'
            },
            'simple_stag_hunt': {
                'cooperation_reward', 'defect_alone_reward', 'defect_together_reward',
                'failed_cooperation_punishment', 'eps_per_game'
            },
            'smac': {'map_name'},
            'smacv2': {'map_name', 'capability_config'}
        }
        
        # Determine current env key
        current_env = self.env_name.lower()
        # Handle aliases or specific checks if needed
        if 'stag' in current_env and 'hunt' in current_env and 'simple' not in current_env:
            current_env = 'staghunt'
        elif 'mpe' in current_env or 'spread' in current_env:
            current_env = 'mpe_simple_spread'
        elif 'foraging' in current_env:
            current_env = 'lbforaging'
            
        # Select relevant fields
        fields_to_keep = common_fields.union(env_specific_fields.get(current_env, set()))
        
        # SMAC/SMACv2 special handling
        if 'smac' in current_env:
             fields_to_keep.add('map_name')
             if 'v2' in current_env:
                  fields_to_keep.add('capability_config')

        for key, value in base_config.items():
            if key in fields_to_keep:
                relevant_config[key] = value
                
        return relevant_config
        
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
        
        # 외부 config 파일 import 지원
        if env_config and 'import_config' in env_config:
            import_path = env_config.pop('import_config')
            try:
                with open(import_path, 'r', encoding='utf-8') as f:
                    imported_dict = yaml.safe_load(f)
                    
                # imported_dict가 environment 섹션을 포함하고 있으면 그 내부를, 아니면 dict 전체를 사용
                imported_env_config = imported_dict.get('environment', imported_dict.get('env', imported_dict))
                # 병합: 현재 설정(env_config)이 imported 설정보다 우선순위를 가집니다. imported_env_config를 기본으로 하고, env_config로 덮어씁니다.
                merged_config = imported_env_config.copy()
                merged_config.update(env_config)
                env_config = merged_config

            except Exception as e:
                print(f"Warning: Failed to import config from {import_path}: {e}")
                
        if env_config:
            # dataclass 필드인 것과 아닌 것을 분리
            valid_keys = set(EnvironmentConfig.__dataclass_fields__.keys())
            known_args = {k: v for k, v in env_config.items() if k in valid_keys}
            unknown_args = {k: v for k, v in env_config.items() if k not in valid_keys}
            
            # known_args로 객체 생성
            env_conf_obj = EnvironmentConfig(**known_args)
            
            # unknown_args는 env_args 필드에 추가 (기존 env_args가 있다면 병합)
            if 'env_args' in known_args:
                env_conf_obj.env_args.update(unknown_args)
            else:
                env_conf_obj.env_args = unknown_args
            
            config.environment = env_conf_obj
        
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
    parser.add_argument('--num-agents', type=int, default=None,
                        help='Number of agents')
    parser.add_argument('--episode-length', type=int, default=None,
                        help='Episode length (for toy_causal)')
    parser.add_argument('--grid-observation', type=str, default=None,
                        help='Use grid observation (true/false)')
    parser.add_argument('--forage-quantity', type=int, default=None,
                        help='Number of forage items (StagHunt)')
    parser.add_argument('--map-name', type=str, default=None,
                        help='Map name for SMAC environment')
    
    # Training
    parser.add_argument('--max-iterations', type=int, default=1000,
                        help='Maximum number of training iterations')
    parser.add_argument('--max-steps', type=int, default=2048,
                        help='Maximum steps per iteration')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Training device (cpu/cuda)')
    parser.add_argument('--n-rollout-threads', type=int, default=1,
                        help='Number of parallel environments for rollout collection')  # [ADDED]
    parser.add_argument('--episodes-per-iteration', type=int, default=None,
                        help='Number of episodes to collect per iteration')  # [ADDED]

    
    # Algorithm
    parser.add_argument('--algorithm-type', '--model', type=str, default=None,
                        help='Algorithm type: pci_mappo, mappo, or ippo')
    parser.add_argument('--clip-delta', type=float, default=None,
                        help='Clip threshold for delta (causal intrinsic reward)')
    parser.add_argument('--causal-lambda', type=float, default=None,
                        help='Causal reward shaping coefficient')
    parser.add_argument('--causal-lambda-schedule', action='store_true', default=False,
                        help='Enable causal lambda scheduling')
    parser.add_argument('--causal-lambda-adaptive-per-agent', type=str, default=None,
                        help='Enable adaptive causal lambda per agent (true/false)')
    parser.add_argument('--use-value-parameter-sharing', type=str, default=None,
                        help='Use shared value network for all agents (true/false)')
    parser.add_argument('--use-centralized-critic', type=str, default=None,
                        help='Use MAPPO-style centralized critic with global observations (true/false)')
    parser.add_argument('--lr-actor', type=float, default=None,
                        help='Learning rate for actor/policy')
    parser.add_argument('--lr-bridge', type=float, default=None,
                        help='Learning rate for bridge networks')
    parser.add_argument('--lr-critic', type=float, default=None,
                        help='Learning rate for critic')
    parser.add_argument('--recurrent-N', type=int, default=1,
                        help='Number of recurrent layers for RNN')
    parser.add_argument('--global-reward', type=str, default=None,
                        help='Use global reward (true/false)')
    
    # Logging
    parser.add_argument('--save-dir', type=str, default='./save_checkpoints',
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
    if hasattr(args, 'grid_observation') and args.grid_observation is not None:
        # Convert string to boolean
        is_grid = args.grid_observation.lower() in ('true', '1', 'yes', 'on')
        config.environment.grid_observation = is_grid
    if hasattr(args, 'forage_quantity') and args.forage_quantity is not None:
        config.environment.forage_quantity = args.forage_quantity
    if hasattr(args, 'map_name') and args.map_name:
        config.environment.map_name = args.map_name
        print(config.environment.map_name)
        
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
    if hasattr(args, 'n_rollout_threads') and args.n_rollout_threads:  # [ADDED]
        config.training.n_rollout_threads = args.n_rollout_threads
    if hasattr(args, 'episodes_per_iteration') and args.episodes_per_iteration:  # [ADDED]
        config.training.episodes_per_iteration = args.episodes_per_iteration
        # Auto-enable EpisodeBuffer if episodes_per_iteration is set
        if hasattr(config.algorithm, 'use_episode_buffer'):
             config.algorithm.use_episode_buffer = True
             print(f"Auto-enabling EpisodeBuffer because episodes_per_iteration is set to {args.episodes_per_iteration}")

    
    # Support both 'model' and 'algorithm_type' for backward compatibility
    algorithm_type = getattr(args, 'algorithm_type', None) or getattr(args, 'model', None)
    if algorithm_type:
        # Convert hyphen to underscore for convenience (e.g., pci-mappo -> pci_mappo)
        algorithm_type = algorithm_type.replace('-', '_')
        config.algorithm.type = algorithm_type
    if hasattr(args, 'clip_delta') and args.clip_delta is not None:
        config.algorithm.clip_delta = args.clip_delta
    if hasattr(args, 'causal_lambda') and args.causal_lambda is not None:
        config.algorithm.causal_lambda = args.causal_lambda
    if hasattr(args, 'use_value_parameter_sharing') and args.use_value_parameter_sharing is not None:
        # Convert string to boolean
        use_sharing = args.use_value_parameter_sharing.lower() in ('true', '1', 'yes', 'on')
        config.algorithm.use_value_parameter_sharing = use_sharing
    if hasattr(args, 'use_centralized_critic') and args.use_centralized_critic is not None:
        # Convert string to boolean
        use_central = args.use_centralized_critic.lower() in ('true', '1', 'yes', 'on')
        config.algorithm.use_centralized_critic = use_central
    if hasattr(args, 'causal_lambda_adaptive_per_agent') and args.causal_lambda_adaptive_per_agent is not None:
        # Convert string to boolean
        use_adaptive = args.causal_lambda_adaptive_per_agent.lower() in ('true', '1', 'yes', 'on')
        config.algorithm.causal_lambda_adaptive_per_agent = use_adaptive
    if hasattr(args, 'lr_actor') and args.lr_actor:
        config.network.lr_actor = args.lr_actor
    if hasattr(args, 'lr_bridge') and args.lr_bridge:
        config.network.lr_bridge = args.lr_bridge
    if hasattr(args, 'lr_critic') and args.lr_critic:
        config.network.lr_critic = args.lr_critic
    if hasattr(args, 'recurrent_N') and args.recurrent_N is not None:
        config.network.recurrent_N = args.recurrent_N
    if hasattr(args, 'global_reward') and args.global_reward is not None:
        # Convert string to boolean
        use_global = args.global_reward.lower() in ('true', '1', 'yes', 'on')
        config.algorithm.global_reward = use_global
    
    if hasattr(args, 'use_wandb') and args.use_wandb:
        config.wandb.use_wandb = args.use_wandb
    if hasattr(args, 'wandb_project') and args.wandb_project:
        config.wandb.project = args.wandb_project
    if hasattr(args, 'wandb_run_name') and args.wandb_run_name:
        config.wandb.run_name = args.wandb_run_name
    # Reproducibility settings
    if hasattr(args, 'seed') and args.seed:
        config.training.seed = args.seed
    if hasattr(args, 'deterministic'):
        config.training.deterministic = args.deterministic
    
    return config
