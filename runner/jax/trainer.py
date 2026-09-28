from __future__ import annotations

import math
import os
import pickle
from collections import defaultdict
from typing import Any, Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from runner.torch.trainer import Trainer


class JAXTrainer(Trainer):
    @staticmethod
    def _summarize_eval_metrics(
        episode_rewards: Dict[int, list[float]],
        episode_lengths: list[int],
        episode_successes: list[float],
    ) -> Dict[str, float]:
        mean_per_agent = [float(np.mean(rewards)) for rewards in episode_rewards.values()] if episode_rewards else []
        return {
            "eval_success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
            "eval_mean_reward": float(np.mean(mean_per_agent)) if mean_per_agent else 0.0,
            "eval_std_reward": float(np.std(mean_per_agent)) if mean_per_agent else 0.0,
            "eval_mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "eval_min_reward": float(np.min(mean_per_agent)) if mean_per_agent else 0.0,
            "eval_max_reward": float(np.max(mean_per_agent)) if mean_per_agent else 0.0,
        }

    def _build_eval_env(self, env):
        """Keep JAX-native SMAX evaluation in the JAX trainer path."""
        if env.__class__.__name__ == "SMAXBatchWrapper":
            return env
        return super()._build_eval_env(env)

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if not hasattr(self.env, "reset_jax_pure") or not hasattr(self.env, "step_jax_pure"):
            raise ValueError("JAXTrainer requires an environment exposing reset_jax_pure/step_jax_pure.")

        self.rollout_envs = getattr(self.env, "n_envs", self.n_rollout_threads)
        self.rollout_horizon = max(1, math.ceil(self.max_steps_per_iteration / max(1, self.rollout_envs)))

        base_rng = jax.random.PRNGKey(int(self.seed or 0))
        self._env_rng = jax.random.fold_in(base_rng, 0)
        self._action_rng = jax.random.fold_in(base_rng, 1)
        self._eval_rng = jax.random.fold_in(base_rng, 2)
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
        self._latest_rollout_for_update_metrics = None
        self._latest_rollout_start_env_rng = None
        self._latest_rollout_start_env_state = None
        self._latest_causal_delta_for_update_metrics = None
        self._latest_causal_delta_rollout = None
        self._latest_eval_diagnostic_rollout = None
        self._wrap_causal_reward_cache()

    def _wrap_causal_reward_cache(self):
        compute_causal_rewards = getattr(self.algorithm, "compute_causal_rewards", None)
        if not callable(compute_causal_rewards):
            return
        if getattr(self.algorithm, "_jax_trainer_delta_cache_wrapped", False):
            return

        original_compute_causal_rewards = compute_causal_rewards

        def cached_compute_causal_rewards(rollout, *args, **kwargs):
            c_matrix, info = original_compute_causal_rewards(rollout, *args, **kwargs)
            try:
                self._latest_causal_delta_for_update_metrics = c_matrix.sum(axis=-1)
                self._latest_causal_delta_rollout = rollout
            except Exception:
                self._latest_causal_delta_for_update_metrics = None
                self._latest_causal_delta_rollout = None
            return c_matrix, info

        self.algorithm.compute_causal_rewards = cached_compute_causal_rewards
        self.algorithm._jax_trainer_delta_cache_wrapped = True

    def _can_collect_eval_diagnostics(self) -> bool:
        return all(
            hasattr(self.algorithm, name)
            for name in ("_augment_obs", "_policy_step", "_value_step")
        ) and (
            hasattr(self.algorithm, "compute_causal_rewards")
            or hasattr(self.algorithm, "compute_gram_spectrum_metrics")
        )

    def _run_jax_policy_evaluation(
        self,
        num_episodes: int,
        deterministic: bool = True,
        capture_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        """Run JAX evaluation using the native pure JAX environment API."""
        eval_rng = jax.random.fold_in(self._eval_rng, self.iteration)
        eval_envs = getattr(self.eval_env, "n_envs", self.rollout_envs)
        episode_limit = self._get_eval_episode_limit()
        max_eval_steps = max(
            episode_limit,
            episode_limit * math.ceil(num_episodes / max(1, eval_envs)) * 2,
        )
        runner_state = self.algorithm.initialize_runner_state(eval_envs)
        eval_rng, batch_states, obs, avail = self.eval_env.reset_jax_pure(eval_rng)

        collect_diagnostics = bool(capture_diagnostics and deterministic and self._can_collect_eval_diagnostics())
        diagnostic_rollout_lists = defaultdict(list)
        diagnostic_episode_ids = []
        diagnostic_counted_episode_ids = set()
        diagnostic_episode_id_by_env = np.arange(eval_envs, dtype=np.int64)
        diagnostic_next_episode_id = int(eval_envs)
        self._latest_eval_diagnostic_rollout = None

        running_rewards = np.zeros((eval_envs, self.eval_env.num_agents), dtype=np.float32)
        running_lengths = np.zeros((eval_envs,), dtype=np.int32)
        episode_rewards = defaultdict(list)
        episode_lengths = []
        episode_successes = []
        collected = 0
        eval_steps = 0

        while collected < num_episodes and eval_steps < max_eval_steps:
            if not deterministic:
                raise NotImplementedError("JAX evaluation currently supports deterministic greedy rollouts only.")

            if collect_diagnostics:
                policy_rng = jax.random.fold_in(eval_rng, eval_steps)
                obs_aug = self.algorithm._augment_obs(obs)
                actor_hidden_prev = runner_state["actor_hidden"]
                critic_hidden_prev = runner_state["critic_hidden"]
                delta_critic_hidden_prev = runner_state["delta_critic_hidden"]
                raw_prev_actions_prev = runner_state["prev_actions"]
                prev_obs_prev = runner_state["prev_obs"]
                done_prev = runner_state["done_prev"]
                agent_done_prev = runner_state["agent_done_prev"]
                actor_reset_masks = agent_done_prev
                use_centralized_critic = bool(getattr(self.algorithm, "use_centralized_critic", False))
                critic_reset_masks = done_prev if use_centralized_critic else agent_done_prev

                actor_hidden_next, actions, log_probs, probs = self.algorithm._policy_step(
                    self.algorithm.actor_state.params,
                    obs_aug,
                    avail,
                    actor_hidden_prev,
                    raw_prev_actions_prev,
                    actor_reset_masks,
                    policy_rng,
                    True,
                )
                critic_hidden_next, values = self.algorithm._value_step(
                    self.algorithm.critic_state.params,
                    obs_aug,
                    critic_hidden_prev,
                    raw_prev_actions_prev,
                    critic_reset_masks,
                )
                delta_critic_state = getattr(self.algorithm, "delta_critic_state", None)
                if (
                    delta_critic_state is not None
                    and hasattr(self.algorithm, "_delta_value_step")
                ):
                    delta_critic_hidden_next, delta_values = self.algorithm._delta_value_step(
                        delta_critic_state.params,
                        obs_aug,
                        delta_critic_hidden_prev,
                        raw_prev_actions_prev,
                        critic_reset_masks,
                    )
                else:
                    delta_critic_hidden_next = delta_critic_hidden_prev
                    delta_values = jnp.zeros((obs_aug.shape[0], self.algorithm.num_agents), dtype=jnp.float32)
            else:
                actions, runner_state = self.algorithm.greedy_actions(obs, avail, runner_state)

            (
                eval_rng,
                batch_states,
                next_obs,
                next_avail,
                rewards,
                agent_dones,
                dones_all,
                battle_won,
            ) = self.eval_env.step_jax_pure(eval_rng, batch_states, actions)

            if collect_diagnostics:
                diagnostic_rollout_lists["prev_obs"].append(prev_obs_prev.astype(jnp.float32))
                diagnostic_rollout_lists["obs"].append(obs_aug.astype(jnp.float32))
                diagnostic_rollout_lists["avail_actions"].append(avail.astype(jnp.float32))
                diagnostic_rollout_lists["agent_dones"].append(agent_done_prev.astype(jnp.float32))
                diagnostic_rollout_lists["actions"].append(actions.astype(jnp.int32))
                diagnostic_rollout_lists["log_probs"].append(log_probs.astype(jnp.float32))
                diagnostic_rollout_lists["probs"].append(probs.astype(jnp.float32))
                diagnostic_rollout_lists["rewards"].append(rewards.astype(jnp.float32))
                diagnostic_rollout_lists["dones"].append(dones_all.astype(jnp.float32))
                diagnostic_rollout_lists["masks"].append((1.0 - done_prev).astype(jnp.float32))
                diagnostic_rollout_lists["values"].append(values.astype(jnp.float32))
                diagnostic_rollout_lists["delta_values"].append(delta_values.astype(jnp.float32))
                diagnostic_rollout_lists["actor_hidden"].append(actor_hidden_prev.astype(jnp.float32))
                diagnostic_rollout_lists["critic_hidden"].append(critic_hidden_prev.astype(jnp.float32))
                diagnostic_rollout_lists["delta_critic_hidden"].append(delta_critic_hidden_prev.astype(jnp.float32))
                diagnostic_rollout_lists["prev_actions"].append(raw_prev_actions_prev.astype(jnp.float32))
                diagnostic_rollout_lists["battle_won"].append(battle_won.astype(jnp.float32))
                diagnostic_episode_ids.append(diagnostic_episode_id_by_env.copy())

            rewards_np = np.asarray(jax.device_get(rewards))
            dones_np = np.asarray(jax.device_get(dones_all)).astype(bool)
            battle_won_np = np.asarray(jax.device_get(battle_won))

            running_rewards += rewards_np
            running_lengths += 1
            eval_steps += 1

            next_prev_actions = jax.nn.one_hot(
                actions,
                self.algorithm.action_dim,
                dtype=jnp.float32,
            )
            agent_done_prev = jnp.maximum(
                jnp.asarray(agent_dones, dtype=jnp.float32),
                jnp.asarray(dones_all, dtype=jnp.float32)[:, None],
            )
            next_prev_actions = next_prev_actions * (1.0 - agent_done_prev[..., None])

            if collect_diagnostics:
                runner_state["actor_hidden"] = actor_hidden_next
                runner_state["critic_hidden"] = critic_hidden_next
                runner_state["delta_critic_hidden"] = delta_critic_hidden_next
            runner_state["prev_actions"] = next_prev_actions
            if collect_diagnostics:
                runner_state["prev_obs"] = self.algorithm._augment_obs(next_obs)
            runner_state["done_prev"] = jnp.asarray(dones_all, dtype=jnp.float32)
            runner_state["agent_done_prev"] = agent_done_prev

            obs = next_obs
            avail = next_avail

            for env_idx in range(eval_envs):
                if dones_np[env_idx] and collected < num_episodes:
                    if collect_diagnostics:
                        diagnostic_counted_episode_ids.add(int(diagnostic_episode_id_by_env[env_idx]))
                    total_reward = running_rewards[env_idx].copy()
                    for agent_id, reward in enumerate(total_reward):
                        episode_rewards[agent_id].append(float(reward))
                    episode_lengths.append(int(running_lengths[env_idx]))
                    episode_successes.append(float(battle_won_np[env_idx] > 0.5))
                    collected += 1
                    running_rewards[env_idx] = 0.0
                    running_lengths[env_idx] = 0
                    if collect_diagnostics:
                        diagnostic_episode_id_by_env[env_idx] = diagnostic_next_episode_id
                        diagnostic_next_episode_id += 1
                elif dones_np[env_idx] and collect_diagnostics:
                    diagnostic_episode_id_by_env[env_idx] = diagnostic_next_episode_id
                    diagnostic_next_episode_id += 1

        if collect_diagnostics and diagnostic_rollout_lists and diagnostic_counted_episode_ids:
            diagnostic_rollout = {
                key: jnp.stack(values, axis=0)
                for key, values in diagnostic_rollout_lists.items()
            }
            episode_ids = np.stack(diagnostic_episode_ids, axis=0)
            counted_mask = np.isin(
                episode_ids,
                np.asarray(sorted(diagnostic_counted_episode_ids), dtype=np.int64),
            ).astype(np.float32)
            diagnostic_rollout["masks"] = diagnostic_rollout["masks"] * jnp.asarray(counted_mask, dtype=jnp.float32)

            self._latest_eval_diagnostic_rollout = diagnostic_rollout

        return {
            "episode_rewards": dict(episode_rewards),
            "episode_lengths": episode_lengths,
            "episode_successes": episode_successes,
            "collected_episodes": collected,
            "eval_steps": eval_steps,
            "episode_limit": episode_limit,
            "requested_episodes": num_episodes,
        }

    def _run_jax_policy_evaluation_by_seed_batch(
        self,
        eval_seeds: Sequence[int],
        num_episodes: int,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        eval_seeds = [int(seed) for seed in eval_seeds]
        if not eval_seeds:
            raise ValueError("eval_seeds must contain at least one seed")

        eval_envs = getattr(self.eval_env, "n_envs", self.rollout_envs)
        if len(eval_seeds) != eval_envs:
            raise ValueError(f"Expected {eval_envs} eval seeds, got {len(eval_seeds)}")
        if not hasattr(self.eval_env, "reset_jax_pure_from_keys") or not hasattr(self.eval_env, "step_jax_pure_from_keys"):
            raise ValueError("Batched explicit-seed JAX evaluation requires reset_jax_pure_from_keys/step_jax_pure_from_keys.")

        episode_limit = self._get_eval_episode_limit()
        max_eval_steps = max(episode_limit, episode_limit * max(1, num_episodes) * 2)
        runner_state = self.algorithm.initialize_runner_state(eval_envs)
        seed_keys = jnp.stack([jax.random.PRNGKey(seed) for seed in eval_seeds], axis=0)
        batch_states, obs, avail = self.eval_env.reset_jax_pure_from_keys(seed_keys)
        env_rng_keys = seed_keys

        running_rewards = np.zeros((eval_envs, self.eval_env.num_agents), dtype=np.float32)
        running_lengths = np.zeros((eval_envs,), dtype=np.int32)
        episode_rewards = [defaultdict(list) for _ in range(eval_envs)]
        episode_lengths = [[] for _ in range(eval_envs)]
        episode_successes = [[] for _ in range(eval_envs)]
        collected = np.zeros((eval_envs,), dtype=np.int32)
        eval_steps = 0

        while np.any(collected < num_episodes) and eval_steps < max_eval_steps:
            if not deterministic:
                raise NotImplementedError("JAX SMAX evaluation currently supports deterministic greedy rollouts only.")
            actions, runner_state = self.algorithm.greedy_actions(obs, avail, runner_state)
            (
                env_rng_keys,
                batch_states,
                obs,
                avail,
                rewards,
                agent_dones,
                dones_all,
                battle_won,
            ) = self.eval_env.step_jax_pure_from_keys(env_rng_keys, batch_states, actions)

            rewards_np = np.asarray(jax.device_get(rewards))
            dones_np = np.asarray(jax.device_get(dones_all)).astype(bool)
            battle_won_np = np.asarray(jax.device_get(battle_won))

            running_rewards += rewards_np
            running_lengths += 1
            eval_steps += 1

            next_prev_actions = jax.nn.one_hot(
                actions,
                self.algorithm.action_dim,
                dtype=jnp.float32,
            )
            agent_done_prev = jnp.maximum(
                jnp.asarray(agent_dones, dtype=jnp.float32),
                jnp.asarray(dones_all, dtype=jnp.float32)[:, None],
            )
            next_prev_actions = next_prev_actions * (1.0 - agent_done_prev[..., None])
            runner_state["prev_actions"] = next_prev_actions
            runner_state["done_prev"] = jnp.asarray(dones_all, dtype=jnp.float32)
            runner_state["agent_done_prev"] = agent_done_prev

            for env_idx in range(eval_envs):
                if not dones_np[env_idx] or collected[env_idx] >= num_episodes:
                    continue
                total_reward = running_rewards[env_idx].copy()
                for agent_id, reward in enumerate(total_reward):
                    episode_rewards[env_idx][agent_id].append(float(reward))
                episode_lengths[env_idx].append(int(running_lengths[env_idx]))
                episode_successes[env_idx].append(float(battle_won_np[env_idx] > 0.5))
                collected[env_idx] += 1
                running_rewards[env_idx] = 0.0
                running_lengths[env_idx] = 0

        return {
            "eval_seeds": eval_seeds,
            "episode_rewards": [dict(agent_rewards) for agent_rewards in episode_rewards],
            "episode_lengths": episode_lengths,
            "episode_successes": episode_successes,
            "collected_episodes": collected.tolist(),
            "eval_steps": eval_steps,
            "episode_limit": episode_limit,
            "requested_episodes": int(num_episodes),
        }

    def collect_rollouts(self) -> Dict[str, Any]:
        self._latest_causal_delta_for_update_metrics = None
        self._latest_causal_delta_rollout = None
        rollout_start_env_rng = self._env_rng
        rollout_start_env_state = self._batch_env_state
        (
            self._env_rng,
            self._action_rng,
            self._batch_env_state,
            self._current_obs,
            self._current_avail,
            self._runner_state,
            rollout,
            last_value,
        ) = self.algorithm.collect_rollout(
            self.env,
            self._env_rng,
            self._action_rng,
            self._batch_env_state,
            self._current_obs,
            self._current_avail,
            self._runner_state,
            self.rollout_horizon,
        )

        rollout = dict(rollout)
        rollout["last_value"] = last_value
        self.algorithm.latest_rollout = rollout
        self._latest_rollout_for_update_metrics = rollout
        self._latest_rollout_start_env_rng = rollout_start_env_rng
        self._latest_rollout_start_env_state = rollout_start_env_state

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

    def log_training_metrics(self, rollout_data: Dict[str, Any], update_info: Dict[str, float]):
        super().log_training_metrics(rollout_data, update_info)

    def log_evaluation_metrics(
        self,
        eval_info: Dict[str, float],
        update_info: Dict[str, float] | None = None,
        title: str = "Evaluation Results:",
    ):
        old_latest_rollout = getattr(self.algorithm, "latest_rollout", None)
        if self._latest_eval_diagnostic_rollout is not None:
            self.algorithm.latest_rollout = self._latest_eval_diagnostic_rollout
        try:
            super().log_evaluation_metrics(eval_info, update_info=update_info, title=title)
        finally:
            self.algorithm.latest_rollout = old_latest_rollout

    def evaluate_policy(self, num_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        eval_data = self._run_jax_policy_evaluation(
            num_episodes,
            deterministic=deterministic,
            capture_diagnostics=True,
        )
        episode_rewards = eval_data["episode_rewards"]
        episode_lengths = eval_data["episode_lengths"]
        episode_successes = eval_data["episode_successes"]

        if eval_data["collected_episodes"] < num_episodes:
            self.logger.warning(
                f"JAX evaluation stopped after {eval_data['eval_steps']} steps with "
                f"{eval_data['collected_episodes']}/{num_episodes} episodes collected. "
                f"Episode limit={eval_data['episode_limit']}."
            )

        eval_info = self._summarize_eval_metrics(episode_rewards, episode_lengths, episode_successes)
        eval_info["causal_effects"] = None

        if eval_info["eval_mean_reward"] > self.best_eval_reward:
            self.best_eval_reward = eval_info["eval_mean_reward"]
            eval_info["is_best"] = True
        else:
            eval_info["is_best"] = False
        self._last_eval_is_best = eval_info["is_best"]
        return eval_info

    def evaluate_policy_by_seeds(
        self,
        eval_seeds: Sequence[int],
        num_episodes: int = 1,
        deterministic: bool = True,
    ) -> list[Dict[str, float]]:
        eval_data = self._run_jax_policy_evaluation_by_seed_batch(
            eval_seeds,
            num_episodes=num_episodes,
            deterministic=deterministic,
        )
        results = []
        for env_idx, eval_seed in enumerate(eval_data["eval_seeds"]):
            episode_rewards = eval_data["episode_rewards"][env_idx]
            episode_lengths = eval_data["episode_lengths"][env_idx]
            episode_successes = eval_data["episode_successes"][env_idx]
            collected = int(eval_data["collected_episodes"][env_idx])
            if collected < num_episodes:
                self.logger.warning(
                    f"JAX evaluation for eval_seed={eval_seed} stopped after {eval_data['eval_steps']} steps with "
                    f"{collected}/{num_episodes} episodes collected. Episode limit={eval_data['episode_limit']}."
                )
            eval_info = self._summarize_eval_metrics(episode_rewards, episode_lengths, episode_successes)
            eval_info["eval_seed"] = int(eval_seed)
            results.append(eval_info)
        return results

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

            # --- RNG states ---
            "env_rng": jax.device_get(self._env_rng),
            "action_rng": jax.device_get(self._action_rng),
            "eval_rng": jax.device_get(self._eval_rng),

            # --- env / rollout state ---
            "batch_env_state": jax.device_get(self._batch_env_state),
            "current_obs": jax.device_get(self._current_obs),
            "current_avail": jax.device_get(self._current_avail),
            "runner_state": jax.device_get(self._runner_state),

            # --- stats ---
            "running_episode_rewards": self._running_episode_rewards,
            "running_episode_lengths": self._running_episode_lengths,
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

        if self._last_eval_is_best:
            best_path = os.path.join(self.save_dir, "jax_checkpoint_best.pkl")
            with open(best_path, "wb") as f:
                pickle.dump(checkpoint, f)

    def load_checkpoint(self, checkpoint_path: str, actor_only: bool = False):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        if actor_only or checkpoint.get("format") == "jax_eval_actor_only_v1":
            self.algorithm.load_actor_state_dict(checkpoint["algorithm_state"])
            self.iteration = int(checkpoint.get("iteration", 0))
            self.total_steps = int(checkpoint.get("total_steps", 0))
            self.best_eval_reward = float(checkpoint.get("best_eval_reward", float("-inf")))
            return

        self.iteration = int(checkpoint.get("iteration", 0))
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.best_eval_reward = float(checkpoint.get("best_eval_reward", float("-inf")))

        self.algorithm.load_state_dict(checkpoint["algorithm_state"])

        # --- RNG restore (with fallback for old checkpoints) ---
        self._env_rng = jnp.asarray(checkpoint["env_rng"])

        if "action_rng" in checkpoint:
            self._action_rng = jnp.asarray(checkpoint["action_rng"])
        else:
            # fallback: split from env_rng (best-effort)
            self._env_rng, self._action_rng = jax.random.split(self._env_rng)

        if "eval_rng" in checkpoint:
            self._eval_rng = jnp.asarray(checkpoint["eval_rng"])
        else:
            # fallback: fold_in to keep deterministic but distinct stream
            self._eval_rng = jax.random.fold_in(self._env_rng, 999)

        # --- env / rollout state ---
        self._batch_env_state = checkpoint["batch_env_state"]
        self._current_obs = jnp.asarray(checkpoint["current_obs"])
        self._current_avail = jnp.asarray(checkpoint["current_avail"])
        self._runner_state = jax.tree_util.tree_map(jnp.asarray, checkpoint["runner_state"])

        # --- stats ---
        self._running_episode_rewards = np.asarray(checkpoint["running_episode_rewards"])
        self._running_episode_lengths = np.asarray(checkpoint["running_episode_lengths"])
