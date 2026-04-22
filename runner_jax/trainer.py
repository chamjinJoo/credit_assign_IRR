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

    def _run_jax_policy_evaluation(self, num_episodes: int) -> Dict[str, Any]:
        """Run SMAX/JAX evaluation using the native pure JAX environment API."""
        eval_rng = jax.random.fold_in(self._eval_rng, self.iteration)
        eval_envs = getattr(self.eval_env, "n_envs", self.rollout_envs)
        episode_limit = self._get_eval_episode_limit()
        max_eval_steps = max(
            episode_limit,
            episode_limit * math.ceil(num_episodes / max(1, eval_envs)) * 2,
        )
        runner_state = self.algorithm.initialize_runner_state(eval_envs)
        eval_rng, batch_states, obs, avail = self.eval_env.reset_jax_pure(eval_rng)

        running_rewards = np.zeros((eval_envs, self.eval_env.num_agents), dtype=np.float32)
        running_lengths = np.zeros((eval_envs,), dtype=np.int32)
        episode_rewards = defaultdict(list)
        episode_lengths = []
        episode_successes = []
        collected = 0
        eval_steps = 0

        while collected < num_episodes and eval_steps < max_eval_steps:
            actions, runner_state = self.algorithm.greedy_actions(obs, avail, runner_state)
            (
                eval_rng,
                batch_states,
                obs,
                avail,
                rewards,
                agent_dones,
                dones_all,
                battle_won,
            ) = self.eval_env.step_jax_pure(eval_rng, batch_states, actions)

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
            next_prev_actions = next_prev_actions * (
                1.0 - agent_done_prev[..., None]
            )
            runner_state["prev_actions"] = next_prev_actions
            runner_state["done_prev"] = jnp.asarray(dones_all, dtype=jnp.float32)
            runner_state["agent_done_prev"] = agent_done_prev

            for env_idx in range(eval_envs):
                if dones_np[env_idx] and collected < num_episodes:
                    total_reward = running_rewards[env_idx].copy()
                    for agent_id, reward in enumerate(total_reward):
                        episode_rewards[agent_id].append(float(reward))
                    episode_lengths.append(int(running_lengths[env_idx]))
                    episode_successes.append(float(battle_won_np[env_idx] > 0.5))
                    collected += 1
                    running_rewards[env_idx] = 0.0
                    running_lengths[env_idx] = 0

        return {
            "episode_rewards": dict(episode_rewards),
            "episode_lengths": episode_lengths,
            "episode_successes": episode_successes,
            "collected_episodes": collected,
            "eval_steps": eval_steps,
            "episode_limit": episode_limit,
            "requested_episodes": num_episodes,
        }

    def collect_rollouts(self) -> Dict[str, Any]:
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

    def evaluate_policy(self, num_episodes: int = 10) -> Dict[str, float]:
        eval_data = self._run_jax_policy_evaluation(num_episodes)
        episode_rewards = eval_data["episode_rewards"]
        episode_lengths = eval_data["episode_lengths"]
        episode_successes = eval_data["episode_successes"]

        if eval_data["collected_episodes"] < num_episodes:
            self.logger.warning(
                f"JAX evaluation stopped after {eval_data['eval_steps']} steps with "
                f"{eval_data['collected_episodes']}/{num_episodes} episodes collected. "
                f"Episode limit={eval_data['episode_limit']}."
            )

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

    def test_policy(
        self,
        num_episodes: int = 10,
        render: bool = False,
        save_animation: bool = False,
        animation_dir: str = "./save_animations",
        save_replay: bool = False,
        evaluate_advanced_causal: bool = False,
        advanced_causal_steps: int = 5000,
        evaluate_policy_quality: bool = False,
    ):
        """JAX 환경 전용 test_policy.

        베이스 Trainer.test_policy는 env.reset()이 Dict를 반환한다고 가정하지만
        SMAXBatchWrapper.reset()은 (obs_list, avail_list) 튜플을 반환하므로
        JAX 경로(reset_jax_pure / step_jax_pure / greedy_actions)를 사용한다.
        """
        self.logger.info(f"Testing policy for {num_episodes} episodes (JAX path)...")

        eval_data = self._run_jax_policy_evaluation(num_episodes)
        episode_rewards = eval_data["episode_rewards"]
        episode_lengths = eval_data["episode_lengths"]
        episode_successes = eval_data["episode_successes"]

        if eval_data["collected_episodes"] < num_episodes:
            self.logger.warning(
                f"JAX test_policy stopped after {eval_data['eval_steps']} steps with "
                f"{eval_data['collected_episodes']}/{num_episodes} episodes collected. "
                f"Episode limit={eval_data['episode_limit']}."
            )

        mean_reward = float(np.mean([np.mean(v) for v in episode_rewards.values()])) if episode_rewards else 0.0
        success_rate = float(np.mean(episode_successes)) if episode_successes else 0.0
        mean_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0

        self.logger.info(f"Test  success_rate={success_rate:.3f}  mean_reward={mean_reward:.4f}  mean_length={mean_length:.1f}")

        return {
            "test_success_rate": success_rate,
            "test_mean_reward": mean_reward,
            "test_mean_length": mean_length,
            "test_episode_rewards": dict(episode_rewards),
        }

    def save_smax_replay(self, save_dir: str, num_episodes: int = 3) -> None:
        """학습된 policy를 실행하고 에피소드를 GIF로 저장합니다.

        SMAXVisualizer.expand_state_seq는 (key, state, actions_dict) 튜플 리스트를
        기대하므로, inner env(HeuristicEnemySMAX)를 직접 step하며 수집합니다.

        Args:
            save_dir: GIF를 저장할 디렉토리
            num_episodes: 저장할 에피소드 수
        """
        try:
            from jaxmarl.viz.visualizer import SMAXVisualizer
        except ImportError:
            self.logger.error("jaxmarl.viz를 불러올 수 없습니다. replay 저장을 건너뜁니다.")
            return

        if not hasattr(self.env, "_env"):
            self.logger.error("env._env (HeuristicEnemySMAX)가 없습니다. SMAXBatchWrapper가 필요합니다.")
            return

        os.makedirs(save_dir, exist_ok=True)
        inner_env = self.env._env          # HeuristicEnemySMAX
        agent_keys = self.env._agent_keys  # ["ally_0", ...]
        num_allies = self.env.num_agents
        action_dim = self.env.action_dim
        world_steps = inner_env.world_steps_per_env_step  # 보통 8

        # ── JIT 사전 컴파일 ──────────────────────────────────────────────────
        # SMAXBatchWrapper warmup은 vmap 버전만 컴파일하므로,
        # 단일 env용 step_env (get_state_sequence=False / True) 를 별도로 warmup
        self.logger.info("SMAX replay: env JIT compiling...")
        _wup_rng = jax.random.PRNGKey(0)
        _wup_rng, _wk = jax.random.split(_wup_rng)
        _, _wup_state = inner_env.reset(_wk)
        _wup_actions = {a: jnp.int32(4) for a in inner_env.agents}
        _wup_rng, _wk = jax.random.split(_wup_rng)
        # get_state_sequence=False
        _r = inner_env.step_env(_wk, _wup_state, _wup_actions, False)
        jax.block_until_ready(_r)
        # get_state_sequence=True  (expand_state_seq 내부에서 사용)
        _r = inner_env.step_env(_wk, _wup_state, _wup_actions, True)
        jax.block_until_ready(_r)
        self.logger.info("SMAX replay: JIT compile is done")
        # ──────────────────────────────────────────────────────────────────────

        self.logger.info(f"SMAX replay logging: {num_episodes} episodes → {save_dir}")

        for ep in range(num_episodes):
            rng = jax.random.PRNGKey((self.seed or 0) + ep + 77_777)
            # n_envs=1 로 runner state 초기화
            runner_state = self.algorithm.initialize_runner_state(1)

            # inner env 단일 에피소드 리셋
            rng, reset_key = jax.random.split(rng)
            obs_dict, state = inner_env.reset(reset_key)

            # state_seq: expand_state_seq가 기대하는 (key, state, actions_dict) 튜플 리스트
            state_seq = []
            battle_result = "timeout"

            for _ in range(self.env.max_steps):
                # obs/avail → (1, num_agents, dim) 배치로 변환
                obs_np = np.stack(
                    [np.array(obs_dict[k]) for k in agent_keys], axis=0
                )  # (num_agents, obs_dim)
                obs_batch = jnp.array(obs_np)[None]  # (1, num_agents, obs_dim)

                avail_dict = inner_env.get_avail_actions(state)
                avail_np = np.stack(
                    [np.array(avail_dict[k]) for k in agent_keys], axis=0
                )  # (num_agents, action_dim)
                avail_batch = jnp.array(avail_np)[None]  # (1, num_agents, action_dim)

                # 정책으로 행동 결정
                actions, runner_state = self.algorithm.greedy_actions(
                    obs_batch, avail_batch, runner_state
                )  # actions: (1, num_agents)

                # inner env용 actions dict 생성: {ally_key: scalar_action}
                actions_dict = {k: actions[0, i] for i, k in enumerate(agent_keys)}

                # (key, state, actions_dict) 수집 — visualizer가 재-step할 때 사용
                rng, step_key = jax.random.split(rng)
                state_seq.append((step_key, state, actions_dict))

                # inner env step
                obs_dict, state, _, dones, _ = inner_env.step_env(
                    step_key, state, actions_dict
                )

                # RNN 상태 갱신 (n_envs=1 기준)
                done_val = jnp.array([float(dones["__all__"])])  # (1,)
                agent_done_val = jnp.array(
                    [[float(dones[i] or dones["__all__"]) for i in range(self.env.num_agents)]],
                    dtype=jnp.float32,
                )
                next_prev = jax.nn.one_hot(actions, action_dim, dtype=jnp.float32)
                next_prev = next_prev * (1.0 - agent_done_val)[..., None]
                runner_state["prev_actions"] = next_prev
                runner_state["done_prev"] = done_val
                runner_state["agent_done_prev"] = agent_done_val

                if bool(dones["__all__"]):
                    alive_enemies = np.array(state.state.unit_alive[num_allies:])
                    battle_result = "won" if not np.any(alive_enemies) else "lost"
                    break

            save_fname = os.path.join(save_dir, f"smax_replay_ep{ep:02d}_{battle_result}.gif")
            n_steps = len(state_seq)
            self.logger.info(f"  ep{ep:02d}: {battle_result}, {n_steps} steps — GIF 렌더링 중...")

            viz = SMAXVisualizer(inner_env, state_seq)
            viz.expand_state_seq()
            # frame_interval_ms: 프레임 간격(ms). 기본 64ms → 전체 재생 속도 조절
            viz.interval = 64  # 64ms(기본)보다 느리게 → 전투 장면을 볼 수 있도록
            viz.animate(save_fname=save_fname, view=False)
            self.logger.info(
                f"  ep{ep:02d}: {n_steps} env-steps → {len(viz.state_seq)} frames "
                f"@ {viz.interval}ms/frame → {save_fname}"
            )

        self.logger.info(f"SMAX replay 저장 완료: {save_dir}")

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

    def load_checkpoint(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

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