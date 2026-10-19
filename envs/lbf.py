from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from .base import MultiAgentEnv


class LBForagingWrapper(MultiAgentEnv):
    """
    LBForaging environment wrapper
    """
    def __init__(
        self,
        seed: int = None,
        field_size: int = 7,
        num_agents: int = 2,
        max_food: int = 2,
        sight: int = 2,
        max_episode_steps: int = 50,
        force_coop: bool = False,
        player_level: Tuple[int, int] = (1, 1),
        food_level: Tuple[int, int] = (1, 3),
        grid_observation: bool = True,
        obs_noise_std: float = 0.0
    ):
        super().__init__(num_agents=num_agents, seed=seed)
        
        self.field_size = field_size
        self.max_food = max_food
        self.sight = sight
        self.player_level = player_level
        self.food_level = food_level
        self.max_episode_steps = max_episode_steps
        self.force_coop = force_coop
        self.grid_observation = grid_observation
        self.obs_noise_std = obs_noise_std

        from gymnasium.envs.registration import register
        env_id = f"Foraging-{field_size}x{field_size}-{num_agents}p-{max_food}f"
        if force_coop:
            env_id += "-coop"
        env_id += "-v3"
        self.env_id = env_id
        try:
            env_registered = env_id in gym.envs.registry
        except:
            env_registered = False

        self.kwargs = {"players": num_agents,
                    "min_player_level": player_level[0],
                    "max_player_level": player_level[1],
                    "min_food_level": food_level[0],
                    "max_food_level": food_level[1],
                    "field_size": (field_size, field_size),
                    "max_num_food": max_food,
                    "sight": sight,
                    "max_episode_steps": max_episode_steps,
                    "force_coop": force_coop,
                    "grid_observation": grid_observation
        }
        if env_registered:
            try:
                from gymnasium.envs.registration import registry
                del registry[env_id]
            except:
                pass
        
        # 항상 새로 등록하여 올바른 설정으로 환경 생성
        register(
            id=env_id,
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs=self.kwargs,
        )
        self.env = gym.make(env_id)
        self.env_id = env_id
        
        # print(f"LBF env: {self.env_id}")
        
        self.agent_names = [f"agent_{i}" for i in range(self.num_agents)]
        self.agent_mapping = {i: f"agent_{i}" for i in range(self.num_agents)}
        self.reverse_mapping = {f"agent_{i}": i for i in range(self.num_agents)}

        obs_array, _ = self.env.reset(seed=self._seed) 
        # LBForaging observation is array of shape (N, obs_dim)

        if self.kwargs.get("grid_observation", False):
            self.obs_dim = int(np.prod(obs_array[0].shape))
        else:
            self.obs_dim = len(obs_array[0]) 
        if hasattr(self.env.action_space, '__getitem__'):
            self.action_dim = self.env.action_space[0].n
        else:
            self.action_dim = self.env.action_space.n

    def reset(self) -> Dict[int, np.ndarray]:
        obs_array, _ = self.env.reset() # (observation_array, info_dict)
        observations = {i: obs_array[i].astype(np.float32).flatten() for i in range(self.num_agents)}
        
        # add noise to observations if std > 0
        if self.obs_noise_std > 0:
            for agent_id in observations:
                observations[agent_id] = observations[agent_id] + np.random.normal(0, self.obs_noise_std, size=observations[agent_id].shape)
        return observations

    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[str, Any]
    ]:
        gym_actions = np.array([actions[i] for i in range(self.num_agents)], dtype=np.int64)
        
        step_result = self.env.step(gym_actions)
        obs_array, reward_array, terminated, truncated, info = step_result
        
        observations = {}
        rewards = {}
        dones = {}
        
        for i in range(self.num_agents):
            observations[i] = obs_array[i].astype(np.float32).flatten()
            # add noise to observations if std > 0
            if self.obs_noise_std > 0:
                observations[i] = observations[i] + np.random.normal(0, self.obs_noise_std, size=observations[i].shape)
            # reward: each agent's maximum reward in one episode is 1.0
            rewards[i] = float(reward_array[i])
     
        episode_done = bool(terminated) or bool(truncated)
        
        for agent_id in self.agents:
            dones[agent_id] = episode_done

        # Info
        info_dict = {
            # 'step_count': self.env._elapsed_steps,
            'max_cycles': self.max_episode_steps,
            'episode_done': episode_done,
            'raw_info': info
        }
        return observations, rewards, dones, info_dict

    def get_obs_dim(self) -> int:
        return self.obs_dim
    
    def get_action_dim(self) -> int:
        return self.action_dim
    
    def render(self, mode='human'):
        if hasattr(self.env, 'render'):
            return self.env.render(mode)
        return None
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()


class JumanjiLBFBatchWrapper(MultiAgentEnv):
    """
    JAX-native Level-Based Foraging wrapper backed by Jumanji.

    The interface mirrors SMAXBatchWrapper's pure JAX API so the existing
    JAXTrainer can collect fully jitted batched rollouts.
    """

    def __init__(
        self,
        n_envs: int = 1,
        seed: int = None,
        field_size: int = 10,
        num_agents: int = 3,
        max_food: int = 3,
        sight: Optional[int] = 2,
        max_episode_steps: int = 100,
        force_coop: bool = False,
        player_level: Tuple[int, int] = (1, 2),
        grid_observation: bool = True,
        normalize_reward: bool = True,
        penalty: float = 0.0,
        **kwargs,
    ):
        super().__init__(num_agents=num_agents, seed=seed)

        try:
            import jax
            import jax.numpy as jnp
            from jumanji.environments.routing.lbf.env import LevelBasedForaging
            from jumanji.environments.routing.lbf.generator import RandomGenerator
        except ImportError as exc:
            raise ImportError(
                "JumanjiLBFBatchWrapper requires `jumanji`. Install it in the "
                "pcimarl environment with `python -m pip install jumanji`."
            ) from exc

        self._jax = jax
        self._jnp = jnp
        self.n_envs = int(n_envs)
        self.n_rollout_threads = self.n_envs
        self.field_size = int(field_size)
        self.max_food = int(max_food)
        self.sight = None if sight is None else int(sight)
        self.max_episode_steps = int(max_episode_steps)
        self.force_coop = bool(force_coop)
        self.player_level = tuple(player_level)
        self.grid_observation = bool(grid_observation)
        self.normalize_reward = bool(normalize_reward)
        self.penalty = float(penalty)
        self.env_id = self._build_env_id()

        max_agent_level = int(self.player_level[1])
        generator = RandomGenerator(
            grid_size=self.field_size,
            num_agents=self.num_agents,
            num_food=self.max_food,
            fov=self.sight,
            max_agent_level=max_agent_level,
            force_coop=self.force_coop,
        )
        self._env = LevelBasedForaging(
            generator=generator,
            time_limit=self.max_episode_steps,
            grid_observation=self.grid_observation,
            normalize_reward=self.normalize_reward,
            penalty=self.penalty,
        )

        self._batch_reset = jax.jit(jax.vmap(self._env.reset))
        self._batch_step = jax.jit(jax.vmap(self._env.step))
        self._rng = jax.random.PRNGKey(int(seed or 0))
        self._batch_states = None

        warm_key = jax.random.PRNGKey(int(seed or 0))
        warm_state, warm_obs, warm_avail = self.reset_jax_pure_from_keys(
            jax.random.split(warm_key, self.n_envs)
        )
        jax.block_until_ready(warm_obs)
        self._batch_states = warm_state
        self.obs_dim = int(warm_obs.shape[-1])
        self.action_dim = int(warm_avail.shape[-1])

    def _build_env_id(self) -> str:
        prefix = f"{self.sight}s-" if self.sight is not None and self.sight < self.field_size else ""
        env_id = f"JumanjiLBF-{prefix}{self.field_size}x{self.field_size}-{self.num_agents}p-{self.max_food}f"
        if self.force_coop:
            env_id += "-coop"
        return env_id

    def _flatten_observation(self, observation):
        if self.grid_observation:
            obs = observation.agents_view.astype(self._jnp.float32)
            return obs.reshape(obs.shape[0], obs.shape[1], -1)

        obs = observation.agents_view.astype(self._jnp.float32)
        return obs.reshape(obs.shape[0], obs.shape[1], -1)

    def _action_mask(self, observation):
        return observation.action_mask.astype(self._jnp.float32)

    def reset_jax(self):
        self._rng, self._batch_states, obs, avail = self.reset_jax_pure(self._rng)
        return obs, avail

    def reset_jax_pure(self, rng):
        rng, *env_keys = self._jax.random.split(rng, self.n_envs + 1)
        keys_arr = self._jnp.stack(env_keys)
        return rng, *self.reset_jax_pure_from_keys(keys_arr)

    def reset_jax_pure_from_keys(self, keys_arr):
        if keys_arr.shape[0] != self.n_envs:
            raise ValueError(f"Expected {self.n_envs} reset keys, got {keys_arr.shape[0]}")
        states, timesteps = self._batch_reset(keys_arr)
        return states, self._flatten_observation(timesteps.observation), self._action_mask(timesteps.observation)

    def step_jax(self, actions):
        self._rng, self._batch_states, obs, avail, rewards, dones, dones_all, success = (
            self.step_jax_pure(self._rng, self._batch_states, actions)
        )
        return obs, avail, rewards, dones, dones_all, success

    def step_jax_pure(self, rng, batch_states, actions):
        rng, *reset_keys = self._jax.random.split(rng, self.n_envs + 1)
        reset_keys_arr = self._jnp.stack(reset_keys)
        next_states, timesteps = self._batch_step(batch_states, actions.astype(self._jnp.int32))
        return rng, *self._finalize_step(next_states, timesteps, reset_keys_arr)

    def step_jax_pure_from_keys(self, rng_keys, batch_states, actions):
        if rng_keys.shape[0] != self.n_envs:
            raise ValueError(f"Expected {self.n_envs} RNG keys, got {rng_keys.shape[0]}")
        split_keys = self._jax.vmap(lambda key: self._jax.random.split(key, 2))(rng_keys)
        final_rng_keys = split_keys[:, 0]
        reset_keys_arr = split_keys[:, 1]
        next_states, timesteps = self._batch_step(batch_states, actions.astype(self._jnp.int32))
        return final_rng_keys, *self._finalize_step(next_states, timesteps, reset_keys_arr)

    def _finalize_step(self, next_states, timesteps, reset_keys_arr):
        reset_states, reset_timesteps = self._batch_reset(reset_keys_arr)
        done_flags = timesteps.last()
        success = self._jnp.all(next_states.food_items.eaten, axis=-1)

        def _select(new, reset):
            cond = done_flags.reshape((-1,) + (1,) * (new.ndim - 1))
            return self._jnp.where(cond, reset, new)

        final_states = self._jax.tree_util.tree_map(_select, next_states, reset_states)
        next_obs = self._flatten_observation(timesteps.observation)
        reset_obs = self._flatten_observation(reset_timesteps.observation)
        next_avail = self._action_mask(timesteps.observation)
        reset_avail = self._action_mask(reset_timesteps.observation)
        agent_dones = self._jnp.broadcast_to(done_flags[:, None], (self.n_envs, self.num_agents))
        return (
            final_states,
            _select(next_obs, reset_obs),
            _select(next_avail, reset_avail),
            timesteps.reward.astype(self._jnp.float32),
            agent_dones,
            done_flags,
            success,
        )

    def reset(self):
        self._rng, self._batch_states, obs, _ = self.reset_jax_pure(self._rng)
        obs_np = np.array(obs)
        return [
            {agent_id: obs_np[e, agent_id].astype(np.float32) for agent_id in range(self.num_agents)}
            for e in range(self.n_envs)
        ]

    def step(self, actions_batch):
        actions = self._jnp.array(
            [
                [int(actions_batch[e][agent_id]) for agent_id in range(self.num_agents)]
                for e in range(self.n_envs)
            ],
            dtype=self._jnp.int32,
        )
        self._rng, self._batch_states, obs, avail, rewards, dones, dones_all, success = (
            self.step_jax_pure(self._rng, self._batch_states, actions)
        )
        obs_np = np.array(obs)
        rewards_np = np.array(rewards)
        dones_np = np.array(dones)
        dones_all_np = np.array(dones_all)
        success_np = np.array(success)
        avail_np = np.array(avail)
        return (
            [
                {agent_id: obs_np[e, agent_id].astype(np.float32) for agent_id in range(self.num_agents)}
                for e in range(self.n_envs)
            ],
            [
                {agent_id: float(rewards_np[e, agent_id]) for agent_id in range(self.num_agents)}
                for e in range(self.n_envs)
            ],
            [
                {agent_id: bool(dones_np[e, agent_id]) for agent_id in range(self.num_agents)}
                for e in range(self.n_envs)
            ],
            [
                {'episode_done': bool(dones_all_np[e]), 'battle_won': bool(success_np[e])}
                for e in range(self.n_envs)
            ],
            [
                {agent_id: avail_np[e, agent_id].astype(np.int32).tolist() for agent_id in range(self.num_agents)}
                for e in range(self.n_envs)
            ],
        )

    def get_obs_dim(self) -> int:
        return self.obs_dim

    def get_action_dim(self) -> int:
        return self.action_dim

    def close(self):
        pass
