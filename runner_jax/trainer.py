from __future__ import annotations

import math
import os
import pickle
from collections import defaultdict
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from runner.trainer import Trainer


class JAXTrainer(Trainer):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if not hasattr(self.env, "reset_jax_pure") or not hasattr(self.env, "step_jax_pure"):
            raise ValueError("JAXTrainer requires an environment exposing reset_jax_pure/step_jax_pure.")

        self.rollout_envs = getattr(self.env, "n_envs", self.n_rollout_threads)
        self.rollout_horizon = max(1, math.ceil(self.max_steps_per_iteration / max(1, self.rollout_envs)))

        self._env_rng = jax.random.PRNGKey(self.seed or 0)
        (
            self._env_rng,
            self._batch_env_state,
            self._current_obs,
            self._current_avail,
        ) = self.env.reset_jax_pure(self._env_rng)
        self._runner_state = self.algorithm.initialize_runner_state(self.rollout_envs)

        self._running_episode_rewards = np.zeros((self.rollout_envs, self.env.num_agents), dtype=np.float32)
        self._running_episode_lengths = np.zeros((self.rollout_envs,), dtype=np.int32)
        self._last_eval_is_best = False

    def collect_rollouts(self) -> Dict[str, Any]:
        (
            self._env_rng,
            self._batch_env_state,
            self._current_obs,
            self._current_avail,
            self._runner_state,
            rollout,
            last_value,
        ) = self.algorithm.collect_rollout(
            self.env,
            self._env_rng,
            self._batch_env_state,
            self._current_obs,
            self._current_avail,
            self._runner_state,
            self.rollout_horizon,
        )

        rollout = dict(rollout)
        rollout["last_value"] = last_value
        self.algorithm.latest_rollout = rollout

        rewards = np.asarray(jax.device_get(rollout["rewards"]))
        dones = np.asarray(jax.device_get(rollout["dones"])).astype(bool)
        battle_won = np.asarray(jax.device_get(rollout["battle_won"]))

        episode_rewards = defaultdict(list)
        episode_lengths = []
        episode_successes = []
        episode_count = 0

        for t in range(rewards.shape[0]):
            self._running_episode_rewards += rewards[t]
            self._running_episode_lengths += 1

            for env_idx in range(self.rollout_envs):
                if dones[t, env_idx]:
                    total_reward = self._running_episode_rewards[env_idx].copy()
                    for agent_id, reward in enumerate(total_reward):
                        episode_rewards[agent_id].append(float(reward))

                    mean_episode_reward = float(np.mean(total_reward))
                    self.recent_rewards.append(mean_episode_reward)

                    episode_lengths.append(int(self._running_episode_lengths[env_idx]))
                    episode_successes.append(float(battle_won[t, env_idx] > 0.5))
                    episode_count += 1

                    self._running_episode_rewards[env_idx] = 0.0
                    self._running_episode_lengths[env_idx] = 0

        rollout_data = {
            "step_count": int(rewards.shape[0] * rewards.shape[1]),
            "episode_count": episode_count,
            "episode_rewards": dict(episode_rewards),
            "episode_lengths": episode_lengths,
            "mean_episode_reward": float(np.mean([np.mean(v) for v in episode_rewards.values()])) if episode_rewards else 0.0,
            "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "mean_success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
        }
        return rollout_data

    def evaluate_policy(self) -> Dict[str, float]:
        eval_rng = jax.random.PRNGKey((self.seed or 0) + 10_000 + self.iteration)
        eval_envs = getattr(self.eval_env, "n_envs", self.rollout_envs)
        runner_state = self.algorithm.initialize_runner_state(eval_envs)
        eval_rng, batch_states, obs, avail = self.eval_env.reset_jax_pure(eval_rng)

        running_rewards = np.zeros((eval_envs, self.eval_env.num_agents), dtype=np.float32)
        running_lengths = np.zeros((eval_envs,), dtype=np.int32)
        episode_rewards = defaultdict(list)
        episode_lengths = []
        episode_successes = []
        collected = 0

        while collected < self.eval_episodes:
            actions, runner_state = self.algorithm.greedy_actions(obs, avail, runner_state)
            (
                eval_rng,
                batch_states,
                obs,
                avail,
                rewards,
                _,
                dones_all,
                battle_won,
            ) = self.eval_env.step_jax_pure(eval_rng, batch_states, actions)

            rewards_np = np.asarray(jax.device_get(rewards))
            dones_np = np.asarray(jax.device_get(dones_all)).astype(bool)
            battle_won_np = np.asarray(jax.device_get(battle_won))

            running_rewards += rewards_np
            running_lengths += 1

            next_prev_actions = jax.nn.one_hot(actions, self.algorithm.action_dim, dtype=jnp.float32)
            next_prev_actions = next_prev_actions * (1.0 - jnp.asarray(dones_all, dtype=jnp.float32))[:, None, None]
            runner_state["prev_actions"] = next_prev_actions
            runner_state["done_prev"] = jnp.asarray(dones_all, dtype=jnp.float32)

            for env_idx in range(eval_envs):
                if dones_np[env_idx] and collected < self.eval_episodes:
                    total_reward = running_rewards[env_idx].copy()
                    for agent_id, reward in enumerate(total_reward):
                        episode_rewards[agent_id].append(float(reward))
                    episode_lengths.append(int(running_lengths[env_idx]))
                    episode_successes.append(float(battle_won_np[env_idx] > 0.5))
                    collected += 1
                    running_rewards[env_idx] = 0.0
                    running_lengths[env_idx] = 0

        eval_info = {
            "eval_success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
            "eval_mean_reward": float(np.mean([np.mean(v) for v in episode_rewards.values()])) if episode_rewards else 0.0,
            "eval_std_reward": float(np.std([np.mean(v) for v in episode_rewards.values()])) if episode_rewards else 0.0,
            "eval_mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "eval_min_reward": float(np.min([np.mean(v) for v in episode_rewards.values()])) if episode_rewards else 0.0,
            "eval_max_reward": float(np.max([np.mean(v) for v in episode_rewards.values()])) if episode_rewards else 0.0,
            "causal_effects": None,
        }

        if eval_info["eval_mean_reward"] > self.best_eval_reward:
            self.best_eval_reward = eval_info["eval_mean_reward"]
            eval_info["is_best"] = True
        else:
            eval_info["is_best"] = False
        self._last_eval_is_best = eval_info["is_best"]
        return eval_info

    def save_checkpoint(self, is_final: bool = False):
        if self.save_dir is None:
            return

        os.makedirs(self.save_dir, exist_ok=True)
        checkpoint_name = f"jax_checkpoint_iter_{self.iteration}.pkl"
        if is_final:
            checkpoint_name = "jax_checkpoint_final.pkl"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

        checkpoint = {
            "iteration": self.iteration,
            "total_steps": self.total_steps,
            "best_eval_reward": self.best_eval_reward,
            "algorithm_state": self.algorithm.state_dict(),
            "env_rng": jax.device_get(self._env_rng),
            "batch_env_state": jax.device_get(self._batch_env_state),
            "current_obs": jax.device_get(self._current_obs),
            "current_avail": jax.device_get(self._current_avail),
            "runner_state": jax.device_get(self._runner_state),
            "running_episode_rewards": self._running_episode_rewards,
            "running_episode_lengths": self._running_episode_lengths,
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

        if self._last_eval_is_best:
            best_path = os.path.join(self.save_dir, "jax_checkpoint_best.pkl")
            with open(best_path, "wb") as f:
                pickle.dump(checkpoint, f)

    def load_checkpoint(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        self.iteration = int(checkpoint.get("iteration", 0))
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.best_eval_reward = float(checkpoint.get("best_eval_reward", float("-inf")))
        self.algorithm.load_state_dict(checkpoint["algorithm_state"])
        self._env_rng = jnp.asarray(checkpoint["env_rng"])
        self._batch_env_state = checkpoint["batch_env_state"]
        self._current_obs = jnp.asarray(checkpoint["current_obs"])
        self._current_avail = jnp.asarray(checkpoint["current_avail"])
        self._runner_state = jax.tree_util.tree_map(jnp.asarray, checkpoint["runner_state"])
        self._running_episode_rewards = np.asarray(checkpoint["running_episode_rewards"])
        self._running_episode_lengths = np.asarray(checkpoint["running_episode_lengths"])
