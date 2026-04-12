import numpy as np
import torch
from typing import Dict, Union
from .multi_agent_env import MultiAgentEnv


def create_environment(env_config) -> MultiAgentEnv:
    if hasattr(env_config, '__dict__'):
        from dataclasses import asdict
        config_dict = asdict(env_config) if hasattr(env_config, '__dataclass_fields__') else env_config.__dict__
    else:
        config_dict = env_config
    
    env_name = config_dict.get('env_name', 'null')
    
    if env_name == 'Toy':
        from .multi_agent_env import ToyCausalEnv
        return ToyCausalEnv(
            num_agents=config_dict.get('num_agents', 2),
            episode_length=config_dict.get('episode_length', 5),
            seed=config_dict.get('seed', None)
        )
    
    elif env_name == 'simple_cooperative' or env_name == 'SimpleCooperativeEnv':
        from .multi_agent_env import SimpleCooperativeEnv
        return SimpleCooperativeEnv(
            num_agents=config_dict.get('num_agents', 3),
            grid_size=config_dict.get('grid_size', 5),
            max_steps=config_dict.get('max_steps', 50),
            cooperation_bonus=config_dict.get('cooperation_bonus', 2.0),
            seed=config_dict.get('seed', None)
        )
    
    elif env_name == 'DecTiger':
        from .multi_agent_env import DecTigerEnv
        return DecTigerEnv(
            seed=config_dict.get('seed', 0),
            max_steps=config_dict.get('max_steps', 4),
            failure_obs_prob=config_dict.get('failure_obs_prob', 0.15),
            fully_observable=config_dict.get('fully_observable', False)
        )
    
    elif env_name == 'StagHunt' or env_name == 'StagHuntEnv':
        from .multi_agent_env import StagHuntEnv
        return StagHuntEnv(
            num_agents=config_dict.get('num_agents', 2),
            max_steps=config_dict.get('max_steps', 50),
            stag_follows=config_dict.get('stag_follows', False),
            run_away_after_maul=config_dict.get('run_away_after_maul', False),
            forage_quantity=config_dict.get('forage_quantity', None),
            forage_reward=config_dict.get('forage_reward', 3.0),
            stag_reward=config_dict.get('stag_reward', 5.0),
            mauling_punishment=config_dict.get('mauling_punishment', -1.0),
            coop_bonus=config_dict.get('coop_bonus', 1.0),
            seed=config_dict.get('seed', 0),
            grid_observation=config_dict.get('grid_observation', False),
            obs_noise_std=config_dict.get('obs_noise_std', 0.0)
        )
        
    elif env_name == 'LBForaging' or env_name == 'LBF' or env_name == 'lbforaging':
        from .multi_agent_env import LBForagingWrapper
        return LBForagingWrapper(
            field_size=config_dict.get('field_size', 7),
            num_agents=config_dict.get('num_agents', 2),
            max_food=config_dict.get('max_food', 1),
            player_level=config_dict.get('player_level', (3, 3)),
            food_level=config_dict.get('food_level', (3, 3)),
            sight=config_dict.get('sight', 2),
            max_episode_steps=config_dict.get('max_episode_steps', 50),
            force_coop=config_dict.get('force_coop', False),
            seed=config_dict.get('seed', None),
            grid_observation=config_dict.get('grid_observation', True)
        )

    elif env_name == 'SimpleSpread' or env_name == 'simple_spread':
        from .multi_agent_env import MPESimpleSpreadEnv
        return MPESimpleSpreadEnv(
            n_agents=config_dict.get('num_agents', 2),
            local_ratio=config_dict.get('local_ratio', 0.5),
            max_cycles=config_dict.get('max_cycles', 25),
            continuous_actions=config_dict.get('continuous_actions', False),
            seed=config_dict.get('seed', 0)
        )

    elif env_name in ['VMASNavigation', 'vmas_navigation', 'Navigation']:
        from .multi_agent_env import VMASNavigationEnv
        return VMASNavigationEnv(
            num_agents=config_dict.get('num_agents', 4),
            max_steps=config_dict.get('max_steps', 100),
            seed=config_dict.get('seed', 0),
            continuous_actions=config_dict.get('continuous_actions', False),
            **config_dict.get('env_args', {})
        )

    elif env_name in ['VMASDispersion', 'vmas_dispersion', 'Dispersion']:
        from .multi_agent_env import VMASDispersionEnv
        return VMASDispersionEnv(
            num_agents=config_dict.get('num_agents', 4),
            max_steps=config_dict.get('max_steps', 100),
            seed=config_dict.get('seed', 0),
            continuous_actions=config_dict.get('continuous_actions', False),
            **config_dict.get('env_args', {})
        )
    
    elif env_name == 'SimpleStagHunt' or env_name == 'SimpleStagHuntEnv':
        from .multi_agent_env import SimpleStagHuntEnv
        return SimpleStagHuntEnv(
            cooperation_reward=config_dict.get('cooperation_reward', 5),
            defect_alone_reward=config_dict.get('defect_alone_reward', 1),
            defect_together_reward=config_dict.get('defect_together_reward', 1),
            failed_cooperation_punishment=config_dict.get('failed_cooperation_punishment', -5),
            eps_per_game=config_dict.get('eps_per_game', 1),
            seed=config_dict.get('seed', None)
        )
    
    elif env_name == 'SMAC' or env_name == 'StarCraft2':
        from .multi_agent_env import SMACWrapper
        return SMACWrapper(
            map_name=config_dict.get('map_name', '3m'),
            seed=config_dict.get('seed', None),
            **config_dict.get('env_args', {})
        )

    elif env_name == 'SMACv2':
        from .multi_agent_env import SMACv2Wrapper
        return SMACv2Wrapper(
            map_name=config_dict.get('map_name'),
            seed=config_dict.get('seed', None),
            use_individual_rewards=config_dict.get('use_individual_rewards', False),
            capability_config=config_dict.get('capability_config', None),
            **config_dict.get('env_args', {})
        )
    
    elif env_name == 'SMAX':
        from .multi_agent_env import SMAXWrapper
        return SMAXWrapper(
            map_name=config_dict.get('map_name', None),
            seed=config_dict.get('seed', None),
            max_steps=config_dict.get('max_steps', 100),
            use_individual_rewards=config_dict.get('use_individual_rewards', False),
            enemy_shoots=config_dict.get('enemy_shoots', True),
            attack_mode=config_dict.get('attack_mode', 'closest'),
            map_width=config_dict.get('map_width', 32),
            map_height=config_dict.get('map_height', 32),
            world_steps_per_env_step=config_dict.get('world_steps_per_env_step', 8),
            time_per_step=config_dict.get('time_per_step', 1.0 / 16),
            see_enemy_actions=config_dict.get('see_enemy_actions', True),
            won_battle_bonus=config_dict.get('won_battle_bonus', 1.0),
            walls_cause_death=config_dict.get('walls_cause_death', True),
            observation_type=config_dict.get('observation_type', 'unit_list'),
            smacv2_position_generation=config_dict.get('smacv2_position_generation', False),
            smacv2_unit_type_generation=config_dict.get('smacv2_unit_type_generation', False),
            use_allowed_targets=config_dict.get('use_allowed_targets', False),
            allowed_targets=config_dict.get('allowed_targets', None),
            use_partial_damage_restore=config_dict.get('use_partial_damage_restore', False),
            use_cooperative_kill_bonus=config_dict.get('use_cooperative_kill_bonus', False),
            cooperative_bonus_mode=config_dict.get('cooperative_bonus_mode', 'kill'),
            cooperative_size=config_dict.get('cooperative_size', 2),
            cooperative_kill_window=config_dict.get('cooperative_kill_window', 2),
            cooperative_kill_bonus=config_dict.get('cooperative_kill_bonus', 2.0),
            cooperative_kill_owner_ratio=config_dict.get('cooperative_kill_owner_ratio', 0.7),
        )
    
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def get_available_environments() -> Dict[str, str]:
    """사용 가능한 환경 목록을 반환합니다."""
    return {
        'ToyCausalEnv': '인과 추론을 위한 토이 환경 (2 에이전트)',
        'simple_cooperative': '간단한 협력 환경 (3 에이전트)',
        'SimpleCooperativeEnv': '간단한 협력 환경 (3 에이전트)',
        'DecTiger': 'Dec-Tiger 협력 게임 환경 (2 에이전트)',
        'StagHunt': 'Stag-Hunt 협력 게임 환경 (2 에이전트)',
        'StagHuntEnv': 'Stag-Hunt 협력 게임 환경 (2 에이전트)',
        'SimpleStagHunt': 'Simple Stag Hunt 게임 이론 환경 (2 에이전트)',
        'SimpleStagHuntEnv': 'Simple Stag Hunt 게임 이론 환경 (2 에이전트)',
        'MPESimpleSpread': 'MPE Simple Spread 환경 (N 에이전트)',
        'MPE': 'MPE Simple Spread 환경 (N 에이전트)',
        'mpe_simple_spread': 'MPE Simple Spread 환경 (N 에이전트)',
        'VMASNavigation': 'VMAS Navigation benchmark (N 에이전트)',
        'vmas_navigation': 'VMAS Navigation benchmark (N 에이전트)',
        'VMASDispersion': 'VMAS Dispersion benchmark (N 에이전트)',
        'vmas_dispersion': 'VMAS Dispersion benchmark (N 에이전트)',
        'SMAC': 'StarCraftII 환경 (N 에이전트)',
        'StarCraft2': 'StarCraftII 환경 (N 에이전트)',
        'SMACv2': 'StarCraftII 환경 (N 에이전트)',
        'SMAX': 'JaxMARL SMAX 환경 - SC2 바이너리 불필요, JAX 구현 (N 에이전트)',
    }


def dict_to_tensor(data_dict: Dict[int, Union[np.ndarray, float, int]], device: str = "cpu") -> Dict[int, torch.Tensor]:
    """Convert dictionary of numpy arrays/scalars to tensors."""
    tensor_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            tensor_dict[key] = torch.from_numpy(value).float().to(device)
        elif isinstance(value, (int, float)):
            tensor_dict[key] = torch.tensor(value, dtype=torch.float32, device=device)
        else:
            tensor_dict[key] = torch.tensor(value, device=device)
    return tensor_dict


def tensor_to_dict(tensor_dict: Dict[int, torch.Tensor]) -> Dict[int, np.ndarray]:
    """Convert dictionary of tensors to numpy arrays."""
    array_dict = {}
    for key, tensor in tensor_dict.items():
        array_dict[key] = tensor.detach().cpu().numpy()
    return array_dict


def normalize_observations(
    observations: Dict[int, torch.Tensor], 
    obs_mean: Dict[int, torch.Tensor] = None,
    obs_std: Dict[int, torch.Tensor] = None
) -> Dict[int, torch.Tensor]:
    """Normalize observations using running mean and std."""
    if obs_mean is None or obs_std is None:
        return observations
    
    normalized_obs = {}
    for agent_id, obs in observations.items():
        if agent_id in obs_mean and agent_id in obs_std:
            normalized_obs[agent_id] = (obs - obs_mean[agent_id]) / (obs_std[agent_id] + 1e-8)
        else:
            normalized_obs[agent_id] = obs
    
    return normalized_obs


def compute_running_stats(
    observations: Dict[int, torch.Tensor],
    obs_mean: Dict[int, torch.Tensor],
    obs_std: Dict[int, torch.Tensor],
    alpha: float = 0.01
) -> tuple:
    """Update running mean and std statistics."""
    new_obs_mean = {}
    new_obs_std = {}
    
    for agent_id, obs in observations.items():
        if agent_id not in obs_mean:
            new_obs_mean[agent_id] = obs.mean(dim=0)
            new_obs_std[agent_id] = obs.std(dim=0)
        else:
            new_obs_mean[agent_id] = (1 - alpha) * obs_mean[agent_id] + alpha * obs.mean(dim=0)
            new_obs_std[agent_id] = (1 - alpha) * obs_std[agent_id] + alpha * obs.std(dim=0)
    
    return new_obs_mean, new_obs_std


class RunningMeanStd:
    """Running mean and standard deviation tracker."""
    
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
