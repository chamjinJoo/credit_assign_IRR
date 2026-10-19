import numpy as np
import torch
from typing import Dict, Union
import gymnasium as gym
from .base import MultiAgentEnv

def create_environment(env_config) -> MultiAgentEnv:
    if hasattr(env_config, '__dict__'):
        from dataclasses import asdict
        config_dict = asdict(env_config) if hasattr(env_config, '__dataclass_fields__') else env_config.__dict__
    else:
        config_dict = env_config
    
    env_name = config_dict.get('env_name', 'null')
    normalized_env_name = str(env_name).lower().replace('-', '').replace('_', '')

    if normalized_env_name in ('vote', 'voteenv'):
        from .vote import VoteEnv
        env_args = config_dict.get('env_args') or {}
        return VoteEnv(
            num_agents=env_args.get('n_agents', config_dict.get('num_agents', 4)),
            episode_limit=env_args.get('episode_limit', config_dict.get('episode_limit', 3)),
            observation_correct_prob=env_args.get(
                'observation_correct_prob', config_dict.get('observation_correct_prob', 0.7)
            ),
            seed=env_args.get('seed', config_dict.get('seed', 0)),
        )
    
    elif env_name == 'StagHunt' or env_name == 'StagHuntEnv':
        from .stag_hunt import StagHuntEnv
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
            vision=config_dict.get('vision', 2),
            obs_noise_std=config_dict.get('obs_noise_std', 0.0),
            global_reward=config_dict.get('global_reward', False),
        )

    elif env_name == 'SMAX':
        from .smax import SMAXWrapper
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
            allowed_target_window=config_dict.get('allowed_target_window', None),
            allowed_targets=config_dict.get('allowed_targets', None),
            use_partial_damage_restore=config_dict.get('use_partial_damage_restore', False),
            use_cooperative_kill_bonus=config_dict.get('use_cooperative_kill_bonus', False),
            cooperative_bonus_mode=config_dict.get('cooperative_bonus_mode', 'kill'),
            cooperative_size=config_dict.get('cooperative_size', 2),
            cooperative_kill_window=config_dict.get('cooperative_kill_window', 2),
            cooperative_kill_bonus=config_dict.get('cooperative_kill_bonus', 2.0),
            cooperative_kill_owner_ratio=config_dict.get('cooperative_kill_owner_ratio', 0.7),
        )

    elif env_name == 'LBForaging' or env_name == 'LBF' or env_name == 'lbforaging':
        from .lbf import LBForagingWrapper
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

    elif env_name in ('JumanjiLBF', 'JaxLBF', 'JAXLBF', 'jumanji_lbf', 'jax_lbf'):
        from .lbf import JumanjiLBFBatchWrapper
        return JumanjiLBFBatchWrapper(
            n_envs=config_dict.get('n_envs', 1),
            field_size=config_dict.get('field_size', 10),
            num_agents=config_dict.get('num_agents', 3),
            max_food=config_dict.get('max_food', 3),
            player_level=config_dict.get('player_level', (1, 2)),
            sight=config_dict.get('sight', 2),
            max_episode_steps=config_dict.get('max_episode_steps', 100),
            force_coop=config_dict.get('force_coop', False),
            seed=config_dict.get('seed', None),
            grid_observation=config_dict.get('grid_observation', True),
            normalize_reward=config_dict.get('normalize_reward', True),
            penalty=config_dict.get('penalty', 0.0),
        )

    else:
        raise ValueError(f"Unknown environment: {env_name}")


def get_available_environments() -> Dict[str, str]:
    return {
        'StagHuntEnv': 'Stag-Hunt',
        'SMAX': 'JaxMARL SMAX',
        'Vote': 'Cooperative Vote Dec-POMDP',
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
