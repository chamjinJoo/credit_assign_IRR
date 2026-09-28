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
    lr_delta_critic: Optional[float] = None  # None: follow lr_critic
    lr: float = 5e-4  # General learning rate (used by QMIX)
    rkhs_gamma: float = 1.0  # Gamma parameter for RBF kernel
    recurrent_N: int = 1  # Number of recurrent layers
    use_recurrent_policy: bool = True  # If False, use feedforward actor/critic


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

    clip_delta: float = 0.0
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
    bridge_epoch: int = 4  # Number of epochs for bridge network training
    bridge_update_interval: int = 1  # Update bridge network every N QMIX updates
    num_batches: int = 4  # Number of batches K for training
    causal_mini_batch_size: Optional[int] = None  # None/0: use the full causal batch for r_pci bridge training
    tikhonov_gamma: float = 1e-4  # Tikhonov regularization for RKHS solver
    stabilizer_lambda: float = 1.0  # Gram matrix scaling in RKHS loss
    use_value_parameter_sharing: bool = True  # If True, use single shared value network for all agents
    use_centralized_critic: bool = False  # If True, use MAPPO-style centralized critic (obs_dim * num_agents input)
    # Legacy identifiers retained for checkpoint compatibility. The paper refers
    # to this MSE-trained model as the observational regressor.
    train_naivepredictor: bool = False
    use_naive_predictor_for_delta: bool = False
    print_bridge_predictions: bool = False  # Print console-only bridge predictions during training
    bridge_prediction_print_samples: int = 10  # Number of rollout states to average per log event
    bridge_prediction_print_interval: int = 50  # Print bridge predictions every N iterations
    bridge_prediction_batch_size: int = 2048

    use_episode_buffer: bool = False
    # Buffer size limits for FIFO queue behavior
    max_buffer_size: int = 10000

    global_reward: bool = True  # If True, use team/shared rewards when supported
    lse_temperature: float = 1.0  # Temperature for LSE-style bridge potential shaping
    normalize_returns: bool = True  # Normalize return targets when supported


    normalize_pci_signal: bool = False  # True: per-agent normalize PCI signal before combining with env_adv
    anneal_lr: bool = False             # True: iteration-based linear LR annealing from init_value to 0

    # Agent ID feature
    add_agent_id: bool = False  # If True, concatenate agent_id one-hot vector to observations

    # Proxy shuffle ablation for Torch r_pci
    proxy_shuffle_mode: str = "none"  # "none", "w", "z", or "wz"
    proxy_shuffle_prob: float = 0.0
    proxy_shuffle_strategy: str = "episode_axis"

    # QMIX-specific parameters
    target_update_interval: int = 50  # Update target network every N iterations
    epsilon_start: float = 1.0  # Initial exploration rate
    epsilon_finish: float = 0.05  # Final exploration rate
    epsilon_anneal_time: Optional[int] = None  # Steps to anneal epsilon (None = auto-calculate)
    double_q: bool = True  # Use double Q-learning
    mixing_embed_dim: int = 32  # Mixing network embedding dimension
    hypernet_layers: int = 2  # Number of hypernetwork layers
    hypernet_embed: int = 64  # Hypernetwork embedding dimension
    mixer_init_scale: float = 1.0e-3  # QMIX mixer/hypernetwork orthogonal init scale
    reward_scale: float = 1.0  # QMIX replay reward scale before TD targets
    replay_buffer_size: int = 5000  # Maximum number of episodes in replay buffer
    train_batch_size: int = 40  # Number of episodes to sample for training
    qmix_update_epochs: Optional[int] = None  # None: legacy episode_count-based QMIX updates
    learning_starts: int = 0  # Environment steps before starting QMIX TD updates

@dataclass
class TrainingConfig:
    """Training configuration."""
    max_iterations: int = 1000
    max_steps_per_iteration: int = 2048
    log_interval: int = 1
    save_interval: int = 100
    eval_interval: int = 50
    eval_episodes: int = 10
    save_dir: str = "./runs/checkpoints"
    checkpoint_keep_last: int = 1  # Keep only the most recent iteration checkpoint on disk
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
    # For Simple Cooperative environment
    grid_size: int = 5
    
    # For toy causal env
    episode_length: int = 5

    # For Dec-Tiger environment
    seed: int = 0
    failure_obs_prob: float = 0.15
    fully_observable: bool = False

    # For Stag-Hunt environment
    stag_follows: bool = False
    run_away_after_maul: bool = False
    stag_move_prob: float = 0.7
    coop_bonus: float = 1.0
    forage_reward: float = 3.0
    stag_reward: float = 5.0
    mauling_punishment: float = -1.0
    grid_observation: bool = False
    forage_quantity: Optional[int] = None
    vision: int = 2 # (2*vsision + 1) x (2*vsision + 1) grid
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
    use_individual_rewards: bool = False

    # For SMAX environment
    enemy_shoots: bool = True
    attack_mode: str = "closest"
    use_allowed_targets: bool = False
    allowed_target_window: Optional[int] = None
    allowed_targets: Optional[Dict[int, Any]] = None
    use_partial_damage_restore: bool = False
    use_cooperative_kill_bonus: bool = False
    cooperative_bonus_mode: str = "kill"
    coop_size: int = 2
    cooperative_kill_window: int = 2
    cooperative_kill_bonus: float = 2.0
    cooperative_kill_owner_ratio: float = 0.7
    map_width: int = 32
    map_height: int = 32
    world_steps_per_env_step: int = 8
    time_per_step: float = 1.0 / 16
    see_enemy_actions: bool = True
    won_battle_bonus: float = 1.0
    walls_cause_death: bool = True
    observation_type: str = "unit_list"
    smacv2_position_generation: bool = False
    smacv2_unit_type_generation: bool = False

    # For visualization
    use_visualization: bool = True
    visualize_freq: int = 1
    save_animations: bool = True
    animation_interval: int = 100

    # Generic args for any environment
    env_args: Dict[str, Any] = field(default_factory=dict)

    def get_canonical_env_name(self) -> str:
        """Return the effective environment name, respecting the legacy `name` alias."""
        env_name = (self.env_name or "").strip()
        alias_name = (self.name or "").strip()
        default_env_name = EnvironmentConfig.__dataclass_fields__['env_name'].default

        if alias_name and (not env_name or env_name == default_env_name) and alias_name != default_env_name:
            return alias_name
        return env_name or alias_name or default_env_name

    def get_env_family_key(self) -> str:
        """Normalize environment aliases to the canonical family key used for filtering."""
        current_env = self.get_canonical_env_name().lower().replace('-', '_').replace(' ', '_')
        compact_env = current_env.replace('_', '')

        if compact_env in {'simplecooperative', 'simplecooperativeenv'}:
            return 'simple_cooperative'
        if compact_env in {'toy', 'toycausal', 'toycausalenv'}:
            return 'toy_causal'
        if compact_env in {'dectiger', 'dec_tiger'}:
            return 'dec_tiger'
        if compact_env in {'simplestaghunt', 'simplestaghuntenv'}:
            return 'simple_stag_hunt'
        if 'stag' in compact_env and 'hunt' in compact_env:
            return 'staghunt'
        if 'mpe' in compact_env or 'spread' in compact_env:
            return 'mpe_simple_spread'
        if compact_env in {'vmasnavigation', 'navigation'}:
            return 'vmas_navigation'
        if compact_env in {'vmasdispersion', 'dispersion'}:
            return 'vmas_dispersion'
        if 'foraging' in compact_env or compact_env == 'lbf':
            return 'lbforaging'
        if compact_env == 'smacv2':
            return 'smacv2'
        if compact_env in {'smac', 'starcraft2'}:
            return 'smac'
        if compact_env == 'smax':
            return 'smax'
        if 'foraging' in compact_env or compact_env in ('lbf', 'jaxlbf', 'jumanjilbf', 'jax_lbf', 'jumanji_lbf'):
            return 'lbforaging'

        return current_env

    def get_relevant_config(self) -> Dict[str, Any]:
        """Get configuration dictionary with only relevant fields for the current environment."""
        base_config = asdict(self)
        env_args = dict(base_config.get('env_args') or {})
        relevant_config = {}
        canonical_env_name = self.get_canonical_env_name()
        base_config['env_name'] = canonical_env_name
        base_config['name'] = canonical_env_name
        
        # Common fields for all environments
        common_fields = {
            'env_name', 'seed', 'use_visualization',
            'visualize_freq', 'save_animations', 'animation_interval'
        }
        
        # Environment specific fields map
        env_specific_fields = {
            'simple_cooperative': {'num_agents', 'grid_size', 'max_steps', 'cooperation_bonus'},
            'toy_causal': {'num_agents', 'episode_length'},
            'dec_tiger': {'max_steps', 'failure_obs_prob', 'fully_observable'},
            'staghunt': {
                'num_agents', 'max_steps', 'stag_follows', 'run_away_after_maul',
                'stag_move_prob', 'coop_bonus', 'forage_reward', 'stag_reward',
                'mauling_punishment', 'grid_observation', 'forage_quantity', 'obs_noise_std'
            },
            'mpe_simple_spread': {'num_agents', 'local_ratio', 'max_cycles', 'continuous_actions'},
            'vmas_navigation': {'num_agents', 'max_steps', 'continuous_actions'},
            'vmas_dispersion': {'num_agents', 'max_steps', 'continuous_actions'},
            'lbforaging': {
                'num_agents', 'env_id', 'field_size', 'max_food', 'player_level', 'food_level',
                'sight', 'max_episode_steps', 'force_coop', 'grid_observation',
                'normalize_reward', 'penalty'
            },
            'simple_stag_hunt': {
                'cooperation_reward', 'defect_alone_reward', 'defect_together_reward',
                'failed_cooperation_punishment', 'eps_per_game'
            },
            'smac': {'map_name'},
            'smacv2': {'map_name', 'capability_config', 'use_individual_rewards'},
            'smax': {
                'map_name', 'max_steps',
                'use_individual_rewards', 'enemy_shoots', 'attack_mode',
                'use_allowed_targets', 'allowed_target_window', 'allowed_targets',
                'use_partial_damage_restore',
                'use_cooperative_kill_bonus',
                'cooperative_bonus_mode',
                'coop_size',
                'cooperative_kill_window', 'cooperative_kill_bonus', 'cooperative_kill_owner_ratio',
                'map_width', 'map_height', 'world_steps_per_env_step',
                'time_per_step', 'see_enemy_actions', 'won_battle_bonus',
                'walls_cause_death', 'observation_type',
                'smacv2_position_generation', 'smacv2_unit_type_generation'
            }
        }
        
        # Determine current env key
        current_env = self.get_env_family_key()
            
        # Select relevant fields
        fields_to_keep = common_fields.union(env_specific_fields.get(current_env, set()))
        
        # SMAC/SMACv2 special handling
        if 'smac' in current_env:
             fields_to_keep.add('map_name')
             if 'v2' in current_env:
                  fields_to_keep.add('capability_config')

        for key in sorted(fields_to_keep):
            if key == 'env_args':
                continue
            if key in base_config:
                value = base_config[key]
            else:
                value = env_args.pop(key, None)

            if value is None:
                continue
            relevant_config[key] = value

        # Preserve truly generic extra arguments after hoisting any env-specific keys.
        if env_args:
            relevant_config['env_args'] = env_args

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
            algorithm_dict = dict(config_dict['algorithm'])
            legacy_use_recurrent = algorithm_dict.pop('use_recurrent_policy', None)
            algorithm_dict.pop('use_delta_critic', None)
            config.algorithm = AlgorithmConfig(**algorithm_dict)
            if legacy_use_recurrent is not None and 'network' not in config_dict:
                config.network.use_recurrent_policy = legacy_use_recurrent
        
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

            # Keep legacy `name` and canonical `env_name` synchronized.
            if 'env_name' in known_args and 'name' not in known_args:
                env_conf_obj.name = env_conf_obj.env_name
            elif 'name' in known_args and 'env_name' not in known_args:
                env_conf_obj.env_name = env_conf_obj.name
            elif 'env_name' in known_args and 'name' in known_args and env_conf_obj.name != env_conf_obj.env_name:
                env_conf_obj.name = env_conf_obj.env_name
            
            # unknown_args는 env_args 필드에 추가 (기존 env_args가 있다면 병합)
            if 'env_args' in known_args:
                env_conf_obj.env_args.update(unknown_args)
            else:
                env_conf_obj.env_args = unknown_args
            
            config.environment = env_conf_obj
        
        if 'wandb' in config_dict:
            wandb_dict = dict(config_dict['wandb'])
            wandb_dict.pop('run_name_postfix', None)
            wandb_dict.pop('group', None)
            config.wandb = WandbConfig(**wandb_dict)
        
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load config from YAML file."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self, filter_environment: bool = False) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = asdict(self)
        if filter_environment and hasattr(self.environment, 'get_relevant_config'):
            config_dict['environment'] = self.environment.get_relevant_config()
        return config_dict
    
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
    parser.add_argument('--env', type=str, default=None,
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
    parser.add_argument('--allowed-target-window', type=int, default=None,
                        help='Allowed target window for SMAX target allocation')
    parser.add_argument('--field-size', type=int, default=None,
                        help='LBForaging field size')
    parser.add_argument('--max-food', type=int, default=None,
                        help='LBForaging maximum number of food items')
    parser.add_argument('--sight', type=int, default=None,
                        help='LBForaging agent sight range')
    parser.add_argument('--max-episode-steps', type=int, default=None,
                        help='LBForaging maximum episode steps')
    parser.add_argument('--force-coop', type=str, default=None,
                        help='LBForaging force cooperation mode (true/false)')
    
    # Training
    parser.add_argument('--max-iterations', type=int, default=None,
                        help='Maximum number of training iterations')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum steps per iteration')
    parser.add_argument('--device', type=str, default=None,
                        help='Training device (cpu/cuda)')
    parser.add_argument('--n-rollout-threads', type=int, default=None,
                        help='Number of parallel environments for rollout collection')  # [ADDED]
    parser.add_argument('--episodes-per-iteration', type=int, default=None,
                        help='Number of episodes to collect per iteration')  # [ADDED]

    
    # Algorithm
    parser.add_argument('--algorithm-type', '--model', type=str, default=None,
                        help='Algorithm type: pci_mappo, mappo, ippo, qmix, jax_qmix, jax_ippo, jax_mappo, jax_rpci, etc.')
    parser.add_argument('--clip-delta', type=float, default=None,
                        help='Clip threshold for delta (causal intrinsic reward)')
    parser.add_argument('--causal-lambda', type=float, default=None,
                        help='Causal reward shaping coefficient')
    parser.add_argument('--causal-lambda-schedule', action='store_true', default=False,
                        help='Enable causal lambda scheduling')
    parser.add_argument('--causal-lambda-adaptive-per-agent', type=str, default=None,
                        help='Enable adaptive causal lambda per agent (true/false)')
    parser.add_argument('--bridge-epoch', type=int, default=None,
                        help='Number of epochs for bridge network training')
    parser.add_argument('--bridge-update-interval', type=int, default=None,
                        help='Update bridge network every N QMIX updates')
    parser.add_argument('--causal-mini-batch-size', type=int, default=None,
                        help='Causal bridge mini-batch size; use 0 for full-batch r_pci training')
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
    parser.add_argument('--lr', type=float, default=None,
                        help='General learning rate for value-based algorithms such as QMIX')
    parser.add_argument('--lr-delta-critic', type=float, default=None,
                        help='Learning rate for delta critic; defaults to lr_critic when unset')
    parser.add_argument('--max-grad-norm', type=float, default=None,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--recurrent-N', type=int, default=1,
                        help='Number of recurrent layers for RNN')
    parser.add_argument('--use-recurrent-policy', type=str, default=None,
                        help='Use recurrent actor/critic instead of feedforward (true/false)')
    parser.add_argument('--global-reward', type=str, default=None,
                        help='Use global reward (true/false)')
    parser.add_argument('--train-naivepredictor', '--train-naive-predictor', dest='train_naivepredictor', type=str, default=None,
                        help='Train the MSE observational regressor ablation (true/false)')
    parser.add_argument('--use-naive-predictor-for-delta', type=str, default=None,
                        help='Use the observational regressor instead of the outcome bridge for delta computation (true/false)')
    parser.add_argument('--proxy-shuffle-mode', type=str, default=None,
                        help="Proxy shuffle mode for r_pci: none, w, z, or wz")
    parser.add_argument('--proxy-shuffle-prob', type=float, default=None,
                        help='Probability of applying proxy shuffle')
    parser.add_argument('--proxy-shuffle-strategy', type=str, default=None,
                        help='Proxy shuffle strategy')

    # QMIX-specific parameters
    parser.add_argument('--target-update-interval', type=int, default=None,
                        help='Update target network every N training steps')
    parser.add_argument('--epsilon-start', type=float, default=None,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon-finish', type=float, default=None,
                        help='Final exploration rate')
    parser.add_argument('--anneal-lr', type=str, default=None,
                        help='Enable linear LR annealing for actor/critic (true/false)')
    parser.add_argument('--epsilon-anneal-time', type=int, default=None,
                        help='Steps to anneal epsilon over')
    parser.add_argument('--replay-buffer-size', type=int, default=None,
                        help='Maximum number of episodes in replay buffer')
    parser.add_argument('--mixer-init-scale', type=float, default=None,
                        help='QMIX mixer/hypernetwork orthogonal initialization scale')
    parser.add_argument('--reward-scale', type=float, default=None,
                        help='QMIX reward scale applied before replay TD targets')
    parser.add_argument('--train-batch-size', type=int, default=None,
                        help='Number of episodes to sample for training')
    parser.add_argument('--qmix-update-epochs', type=int, default=None,
                        help='Fixed number of QMIX replay updates per training iteration')
    parser.add_argument('--learning-starts', type=int, default=None,
                        help='Environment steps before starting QMIX replay TD updates')

    # Logging
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Save directory for checkpoints')
    parser.add_argument('--log-interval', type=int, default=None,
                        help='Logging interval')
    
    # Wandb
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='Wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Wandb run name')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=None,
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
    parser.add_argument('--save-replay', action='store_true',
                        help='Save SMAX episode replay as GIF (JAX SMAX only)')
    parser.add_argument('--replay-dir', type=str, default='./smax_replays',
                        help='Directory to save replay GIFs')
    parser.add_argument('--replay-episodes', type=int, default=3,
                        help='Number of episodes to record as replay')
    parser.add_argument('--eval-episodes', type=int, default=None,
                        help='Number of episodes for evaluation (defaults to the config value)')
    
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
        config.environment.name = args.env
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
    if hasattr(args, 'allowed_target_window') and args.allowed_target_window is not None:
        config.environment.allowed_target_window = args.allowed_target_window
    if hasattr(args, 'field_size') and args.field_size is not None:
        config.environment.field_size = args.field_size
    if hasattr(args, 'max_food') and args.max_food is not None:
        config.environment.max_food = args.max_food
    if hasattr(args, 'sight') and args.sight is not None:
        config.environment.sight = args.sight
    if hasattr(args, 'max_episode_steps') and args.max_episode_steps is not None:
        config.environment.max_episode_steps = args.max_episode_steps
    if hasattr(args, 'force_coop') and args.force_coop is not None:
        config.environment.force_coop = args.force_coop.lower() in ('true', '1', 'yes', 'on')
        
    if hasattr(args, 'max_grad_norm') and args.max_grad_norm is not None:
        config.algorithm.max_grad_norm = args.max_grad_norm
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
    if hasattr(args, 'eval_episodes') and args.eval_episodes is not None:
        config.training.eval_episodes = args.eval_episodes
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
    if hasattr(args, 'bridge_epoch') and args.bridge_epoch is not None:
        config.algorithm.bridge_epoch = args.bridge_epoch
    if hasattr(args, 'bridge_update_interval') and args.bridge_update_interval is not None:
        config.algorithm.bridge_update_interval = args.bridge_update_interval
    if hasattr(args, 'causal_mini_batch_size') and args.causal_mini_batch_size is not None:
        config.algorithm.causal_mini_batch_size = None if args.causal_mini_batch_size <= 0 else args.causal_mini_batch_size
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
    if hasattr(args, 'anneal_lr') and args.anneal_lr is not None:
        config.algorithm.anneal_lr = args.anneal_lr.lower() in ('true', '1', 'yes', 'on')
    if hasattr(args, 'lr_actor') and args.lr_actor:
        config.network.lr_actor = args.lr_actor
    if hasattr(args, 'lr_bridge') and args.lr_bridge:
        config.network.lr_bridge = args.lr_bridge
    if hasattr(args, 'lr_critic') and args.lr_critic:
        config.network.lr_critic = args.lr_critic
    if hasattr(args, 'lr') and args.lr is not None:
        config.network.lr = args.lr
    if hasattr(args, 'lr_delta_critic') and args.lr_delta_critic is not None:
        config.network.lr_delta_critic = args.lr_delta_critic
    if hasattr(args, 'recurrent_N') and args.recurrent_N is not None:
        config.network.recurrent_N = args.recurrent_N
    if hasattr(args, 'use_recurrent_policy') and args.use_recurrent_policy is not None:
        use_recurrent = args.use_recurrent_policy.lower() in ('true', '1', 'yes', 'on')
        config.network.use_recurrent_policy = use_recurrent
    if hasattr(args, 'global_reward') and args.global_reward is not None:
        # Convert string to boolean
        use_global = args.global_reward.lower() in ('true', '1', 'yes', 'on')
        config.algorithm.global_reward = use_global
    if hasattr(args, 'train_naivepredictor') and args.train_naivepredictor is not None:
        # Convert string to boolean
        use_naive = args.train_naivepredictor.lower() in ('true', '1', 'yes', 'on')
        config.algorithm.train_naivepredictor = use_naive
    if hasattr(args, 'use_naive_predictor_for_delta') and args.use_naive_predictor_for_delta is not None:
        config.algorithm.use_naive_predictor_for_delta = args.use_naive_predictor_for_delta.lower() in ('true', '1', 'yes', 'on')
    if hasattr(args, 'proxy_shuffle_mode') and args.proxy_shuffle_mode is not None:
        config.algorithm.proxy_shuffle_mode = args.proxy_shuffle_mode
    if hasattr(args, 'proxy_shuffle_prob') and args.proxy_shuffle_prob is not None:
        config.algorithm.proxy_shuffle_prob = args.proxy_shuffle_prob
    if hasattr(args, 'proxy_shuffle_strategy') and args.proxy_shuffle_strategy is not None:
        config.algorithm.proxy_shuffle_strategy = args.proxy_shuffle_strategy

    # QMIX-specific parameters
    if hasattr(args, 'target_update_interval') and args.target_update_interval is not None:
        config.algorithm.target_update_interval = args.target_update_interval
    if hasattr(args, 'epsilon_start') and args.epsilon_start is not None:
        config.algorithm.epsilon_start = args.epsilon_start
    if hasattr(args, 'epsilon_finish') and args.epsilon_finish is not None:
        config.algorithm.epsilon_finish = args.epsilon_finish
    if hasattr(args, 'epsilon_anneal_time') and args.epsilon_anneal_time is not None:
        config.algorithm.epsilon_anneal_time = args.epsilon_anneal_time
    if hasattr(args, 'replay_buffer_size') and args.replay_buffer_size is not None:
        config.algorithm.replay_buffer_size = args.replay_buffer_size
    if hasattr(args, 'mixer_init_scale') and args.mixer_init_scale is not None:
        config.algorithm.mixer_init_scale = args.mixer_init_scale
    if hasattr(args, 'reward_scale') and args.reward_scale is not None:
        config.algorithm.reward_scale = args.reward_scale
    if hasattr(args, 'train_batch_size') and args.train_batch_size is not None:
        config.algorithm.train_batch_size = args.train_batch_size
    if hasattr(args, 'qmix_update_epochs') and args.qmix_update_epochs is not None:
        config.algorithm.qmix_update_epochs = None if args.qmix_update_epochs <= 0 else args.qmix_update_epochs
    if hasattr(args, 'learning_starts') and args.learning_starts is not None:
        config.algorithm.learning_starts = max(0, args.learning_starts)

    if hasattr(args, 'use_wandb') and args.use_wandb:
        config.wandb.use_wandb = args.use_wandb
    if hasattr(args, 'wandb_project') and args.wandb_project:
        config.wandb.project = args.wandb_project
    if hasattr(args, 'wandb_run_name') and args.wandb_run_name:
        config.wandb.run_name = args.wandb_run_name
    # Reproducibility settings
    if hasattr(args, 'seed') and args.seed is not None:
        config.training.seed = args.seed
    if hasattr(args, 'deterministic'):
        config.training.deterministic = args.deterministic
    
    return config
