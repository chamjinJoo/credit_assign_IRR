from __future__ import annotations

import copy
from functools import partial
from typing import Any, Dict, List, Optional

import flax.linen as nn
import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from .common import (
    VERY_NEGATIVE,
    action_one_hot,
    add_agent_id_features,
    global_norm,
    make_gru_cell,
    tree_to_host,
)
from .preprocessing import (
    qmix_pack_episode_arrays,
    qmix_pack_rollout_slice,
    qmix_pack_transition_episode,
    qmix_stack_episodes,
    qmix_transition_from_rollout,
)


class JAXQMIXRNNAgent(nn.Module):
    hidden_dim: int
    action_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        hidden: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        init = nn.initializers.orthogonal(self.init_scale)
        x = nn.Dense(self.hidden_dim, kernel_init=init, bias_init=nn.initializers.zeros)(inputs)
        x = nn.relu(x)
        gru = make_gru_cell(self.hidden_dim)
        new_hidden, core_out = gru(hidden, x)
        q_values = nn.Dense(self.action_dim, kernel_init=init, bias_init=nn.initializers.zeros)(core_out)
        return new_hidden, q_values


class JAXQMixer(nn.Module):
    n_agents: int
    state_dim: int
    mixing_embed_dim: int = 32
    hypernet_layers: int = 2
    hypernet_embed: int = 64
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, agent_qs: jnp.ndarray, states: jnp.ndarray) -> jnp.ndarray:
        has_time = agent_qs.ndim == 3
        if has_time:
            batch_size, time_steps, _ = agent_qs.shape
            flat_agent_qs = agent_qs.reshape(batch_size * time_steps, 1, self.n_agents)
            flat_states = states.reshape(batch_size * time_steps, self.state_dim)
        else:
            batch_size = agent_qs.shape[0]
            flat_agent_qs = agent_qs.reshape(batch_size, 1, self.n_agents)
            flat_states = states.reshape(batch_size, self.state_dim)

        init = nn.initializers.orthogonal(self.init_scale)
        if self.hypernet_layers == 1:
            w1 = nn.Dense(
                self.mixing_embed_dim * self.n_agents,
                kernel_init=init,
                bias_init=nn.initializers.zeros,
            )(flat_states)
            w_final = nn.Dense(
                self.mixing_embed_dim,
                kernel_init=init,
                bias_init=nn.initializers.zeros,
            )(flat_states)
        elif self.hypernet_layers == 2:
            w1 = nn.Dense(
                self.hypernet_embed,
                kernel_init=init,
                bias_init=nn.initializers.zeros,
            )(flat_states)
            w1 = nn.relu(w1)
            w1 = nn.Dense(
                self.mixing_embed_dim * self.n_agents,
                kernel_init=init,
                bias_init=nn.initializers.zeros,
            )(w1)

            w_final = nn.Dense(
                self.hypernet_embed,
                kernel_init=init,
                bias_init=nn.initializers.zeros,
            )(flat_states)
            w_final = nn.relu(w_final)
            w_final = nn.Dense(
                self.mixing_embed_dim,
                kernel_init=init,
                bias_init=nn.initializers.zeros,
            )(w_final)
        else:
            raise ValueError(f"hypernet_layers must be 1 or 2, got {self.hypernet_layers}")

        b1 = nn.Dense(
            self.mixing_embed_dim,
            kernel_init=init,
            bias_init=nn.initializers.zeros,
        )(flat_states)
        w1 = jnp.abs(w1).reshape(-1, self.n_agents, self.mixing_embed_dim)
        b1 = b1.reshape(-1, 1, self.mixing_embed_dim)
        hidden = nn.elu(jnp.matmul(flat_agent_qs, w1) + b1)

        w_final = jnp.abs(w_final).reshape(-1, self.mixing_embed_dim, 1)
        v = nn.Dense(
            self.mixing_embed_dim,
            kernel_init=init,
            bias_init=nn.initializers.zeros,
        )(flat_states)
        v = nn.relu(v)
        v = nn.Dense(1, kernel_init=init, bias_init=nn.initializers.zeros)(v).reshape(-1, 1, 1)

        y = jnp.matmul(hidden, w_final) + v
        if has_time:
            return y.reshape(batch_size, time_steps, 1)
        return y.reshape(batch_size, 1)


class JAXQMIXAlgorithm:
    requires_jax_trainer = True
    use_episode_buffer = True

    def __init__(
        self,
        env: Any,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 5.0e-4,
        gamma: float = 0.99,
        max_grad_norm: float = 10.0,
        target_update_interval: int = 200,
        epsilon_start: float = 1.0,
        epsilon_finish: float = 0.05,
        epsilon_anneal_time: int = 50000,
        double_q: bool = True,
        mixing_embed_dim: int = 32,
        hypernet_layers: int = 2,
        hypernet_embed: int = 64,
        mixer_init_scale: float = 1.0e-3,
        reward_scale: float = 1.0,
        replay_buffer_size: int = 5000,
        train_batch_size: int = 32,
        qmix_update_epochs: Optional[int] = None,
        learning_starts: int = 0,
        device: str = "cpu",
        add_agent_id: bool = True,
        global_reward: bool = True,
        seed: int = 42,
        **_: Any,
    ):
        self.env = env
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.policy_obs_dim = self.obs_dim + self.num_agents if add_agent_id else self.obs_dim
        self.state_dim = self.policy_obs_dim * self.num_agents
        self.device = device
        self.add_agent_id = bool(add_agent_id)
        self.global_reward = bool(global_reward)

        self.gamma = float(gamma)
        self.max_grad_norm = float(max_grad_norm)
        self.target_update_interval = max(1, int(target_update_interval))
        self.epsilon_start = float(epsilon_start)
        self.epsilon_finish = float(epsilon_finish)
        self.epsilon_anneal_time = max(1, int(epsilon_anneal_time))
        self.epsilon = self.epsilon_start
        self.double_q = bool(double_q)
        self.mixer_init_scale = float(mixer_init_scale)
        self.reward_scale = float(reward_scale)
        self.replay_buffer_size = max(1, int(replay_buffer_size))
        self.train_batch_size = max(1, int(train_batch_size))
        self.qmix_update_epochs = (
            None if qmix_update_epochs is None else max(1, int(qmix_update_epochs))
        )
        self.learning_starts = max(0, int(learning_starts))

        self.training_steps = 0
        self.total_env_steps = 0
        self.latest_rollout: Optional[Dict[str, jnp.ndarray]] = None
        self._rollout_fns: Dict[int, Any] = {}
        self._current_episodes: List[List[Dict[str, np.ndarray]]] = []
        self._completed_episodes: List[Dict[str, np.ndarray]] = []
        self._rng_np = np.random.default_rng(int(seed))
        self._episode_limit = self._infer_episode_limit()

        base_rng = jax.random.PRNGKey(int(seed))
        init_rng, agent_key, mixer_key = jax.random.split(base_rng, 3)
        del init_rng

        self.agent_model = JAXQMIXRNNAgent(
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
        )
        self.mixer_model = JAXQMixer(
            n_agents=self.num_agents,
            state_dim=self.state_dim,
            mixing_embed_dim=mixing_embed_dim,
            hypernet_layers=hypernet_layers,
            hypernet_embed=hypernet_embed,
            init_scale=self.mixer_init_scale,
        )

        agent_params = self.agent_model.init(
            agent_key,
            jnp.zeros((1, self.policy_obs_dim + self.action_dim), dtype=jnp.float32),
            jnp.zeros((1, self.hidden_dim), dtype=jnp.float32),
        )["params"]
        mixer_params = self.mixer_model.init(
            mixer_key,
            jnp.zeros((1, self.num_agents), dtype=jnp.float32),
            jnp.zeros((1, self.state_dim), dtype=jnp.float32),
        )["params"]

        tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.radam(learning_rate=float(lr)),
        )
        self.train_state = train_state.TrainState.create(
            apply_fn=lambda *args, **kwargs: None,
            params={"agent": agent_params, "mixer": mixer_params},
            tx=tx,
        )
        self.target_params = copy.deepcopy(self.train_state.params)

    def _infer_episode_limit(self) -> Optional[int]:
        for attr in ("max_steps", "episode_limit", "max_episode_steps"):
            value = getattr(self.env, attr, None)
            if value is not None:
                try:
                    return max(1, int(value))
                except (TypeError, ValueError):
                    pass
        env_info = getattr(self.env, "get_env_info", None)
        if callable(env_info):
            try:
                info = env_info()
                value = info.get("episode_limit")
                if value is not None:
                    return max(1, int(value))
            except Exception:
                return None
        return None

    def _augment_obs(self, obs: jnp.ndarray) -> jnp.ndarray:
        return add_agent_id_features(obs) if self.add_agent_id else obs

    def _get_epsilon(self) -> float:
        delta = (self.epsilon_start - self.epsilon_finish) / self.epsilon_anneal_time
        self.epsilon = max(
            self.epsilon_finish,
            self.epsilon_start - delta * self.total_env_steps,
        )
        return self.epsilon

    def initialize_runner_state(self, n_envs: int) -> Dict[str, jnp.ndarray]:
        return {
            "hidden": jnp.zeros((n_envs, self.num_agents, self.hidden_dim), dtype=jnp.float32),
            "prev_actions": jnp.zeros((n_envs, self.num_agents, self.action_dim), dtype=jnp.float32),
            "done_prev": jnp.zeros((n_envs,), dtype=jnp.float32),
            "agent_done_prev": jnp.zeros((n_envs, self.num_agents), dtype=jnp.float32),
        }

    def reset_agents(self):
        return None

    def reset_agents_batch(self, n_envs: int):
        return self.initialize_runner_state(n_envs)

    @partial(jax.jit, static_argnums=0)
    def _q_values_step(
        self,
        agent_params: Dict[str, Any],
        obs: jnp.ndarray,
        hidden: jnp.ndarray,
        prev_actions: jnp.ndarray,
        reset_masks: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        n_envs = obs.shape[0]
        masks = 1.0 - reset_masks.astype(jnp.float32)
        flat_obs = obs.reshape(n_envs * self.num_agents, obs.shape[-1])
        flat_prev_actions = prev_actions.reshape(n_envs * self.num_agents, self.action_dim)
        flat_hidden = hidden.reshape(n_envs * self.num_agents, self.hidden_dim)
        flat_hidden = flat_hidden * masks.reshape(-1, 1)
        inputs = jnp.concatenate([flat_obs, flat_prev_actions], axis=-1)
        next_hidden_flat, q_flat = self.agent_model.apply(
            {"params": agent_params},
            inputs,
            flat_hidden,
        )
        next_hidden = next_hidden_flat.reshape(n_envs, self.num_agents, self.hidden_dim)
        q_values = q_flat.reshape(n_envs, self.num_agents, self.action_dim)
        return next_hidden, q_values

    @staticmethod
    def _masked_q_values(q_values: jnp.ndarray, avail_actions: jnp.ndarray) -> jnp.ndarray:
        valid = avail_actions > 0
        masked = jnp.where(valid, q_values, VERY_NEGATIVE)
        fallback = jnp.zeros_like(masked)
        return jnp.where(jnp.all(~valid, axis=-1, keepdims=True), fallback, masked)

    @partial(jax.jit, static_argnums=0)
    def _policy_step(
        self,
        agent_params: Dict[str, Any],
        obs: jnp.ndarray,
        avail_actions: jnp.ndarray,
        hidden: jnp.ndarray,
        prev_actions: jnp.ndarray,
        reset_masks: jnp.ndarray,
        rng: jax.Array,
        epsilon: float,
        deterministic: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        next_hidden, q_values = self._q_values_step(
            agent_params,
            obs,
            hidden,
            prev_actions,
            reset_masks,
        )
        masked_q = self._masked_q_values(q_values, avail_actions)
        greedy_actions = jnp.argmax(masked_q, axis=-1).astype(jnp.int32)

        flat_avail = avail_actions.reshape(-1, self.action_dim)
        flat_ids = jnp.arange(flat_avail.shape[0], dtype=jnp.uint32)
        sample_rng, explore_rng = jax.random.split(rng)
        action_keys = jax.vmap(lambda idx: jax.random.fold_in(sample_rng, idx))(flat_ids)

        def sample_one(mask, key):
            logits = jnp.where(mask > 0, 0.0, VERY_NEGATIVE)
            logits = jnp.where(jnp.any(mask > 0), logits, jnp.zeros_like(logits))
            return jax.random.categorical(key, logits).astype(jnp.int32)

        random_actions = jax.vmap(sample_one)(flat_avail, action_keys)
        random_actions = random_actions.reshape(obs.shape[0], self.num_agents)
        explore = jax.random.uniform(explore_rng, greedy_actions.shape) < epsilon
        actions = jnp.where(explore, random_actions, greedy_actions)
        actions = jax.lax.cond(
            deterministic,
            lambda _: greedy_actions,
            lambda _: actions,
            operand=None,
        )
        return next_hidden, actions.astype(jnp.int32)

    def _get_rollout_fn(self, env: Any, horizon: int):
        if horizon in self._rollout_fns:
            return self._rollout_fns[horizon]

        def rollout_fn(
            env_rng: jax.Array,
            action_rng: jax.Array,
            batch_states: Any,
            obs: jnp.ndarray,
            avail_actions: jnp.ndarray,
            runner_state: Dict[str, jnp.ndarray],
            params: Dict[str, Any],
            epsilon: float,
        ):
            def body_fn(carry, _):
                rng, policy_rng, states, current_obs, current_avail, current_runner = carry
                policy_rng, step_action_rng = jax.random.split(policy_rng)
                obs_aug = self._augment_obs(current_obs)
                hidden_prev = current_runner["hidden"]
                prev_actions_prev = current_runner["prev_actions"]
                done_prev = current_runner["done_prev"]
                agent_done_prev = current_runner["agent_done_prev"]

                hidden_next, actions = self._policy_step(
                    params["agent"],
                    obs_aug,
                    current_avail,
                    hidden_prev,
                    prev_actions_prev,
                    agent_done_prev,
                    step_action_rng,
                    epsilon,
                    False,
                )

                (
                    rng,
                    next_states,
                    next_obs,
                    next_avail,
                    rewards,
                    agent_dones,
                    dones_all,
                    battle_won,
                ) = env.step_jax_pure(rng, states, actions)

                agent_done_next = jnp.maximum(
                    agent_dones.astype(jnp.float32),
                    dones_all.astype(jnp.float32)[:, None],
                )
                next_prev_actions = action_one_hot(actions, self.action_dim)
                next_prev_actions = next_prev_actions * (1.0 - dones_all)[:, None, None]
                next_runner = {
                    "hidden": hidden_next,
                    "prev_actions": next_prev_actions,
                    "done_prev": dones_all.astype(jnp.float32),
                    "agent_done_prev": agent_done_next,
                }

                transition = {
                    "obs": obs_aug.astype(jnp.float32),
                    "avail_actions": current_avail.astype(jnp.float32),
                    "actor_hidden": hidden_prev.astype(jnp.float32),
                    "prev_actions": prev_actions_prev.astype(jnp.float32),
                    "actions": actions.astype(jnp.int32),
                    "rewards": rewards.astype(jnp.float32),
                    "dones": dones_all.astype(jnp.float32),
                    "masks": (1.0 - done_prev).astype(jnp.float32),
                    "agent_dones": agent_done_prev.astype(jnp.float32),
                    "next_obs": self._augment_obs(next_obs).astype(jnp.float32),
                    "next_avail_actions": next_avail.astype(jnp.float32),
                    "battle_won": battle_won.astype(jnp.float32),
                }
                next_carry = (
                    rng,
                    policy_rng,
                    next_states,
                    next_obs.astype(jnp.float32),
                    next_avail.astype(jnp.float32),
                    next_runner,
                )
                return next_carry, transition

            final_carry, rollout = jax.lax.scan(
                body_fn,
                (env_rng, action_rng, batch_states, obs, avail_actions, runner_state),
                None,
                length=horizon,
            )
            rng, action_rng, states, next_obs, next_avail, next_runner = final_carry
            last_value = jnp.zeros((next_obs.shape[0],), dtype=jnp.float32)
            return rng, action_rng, states, next_obs, next_avail, next_runner, rollout, last_value

        self._rollout_fns[horizon] = jax.jit(rollout_fn)
        return self._rollout_fns[horizon]

    def collect_rollout(
        self,
        env: Any,
        env_rng: jax.Array,
        action_rng: Optional[jax.Array],
        batch_states: Any,
        obs: jnp.ndarray,
        avail_actions: jnp.ndarray,
        runner_state: Dict[str, jnp.ndarray],
        horizon: int,
    ):
        rollout_fn = self._get_rollout_fn(env, horizon)
        if action_rng is None:
            action_rng = env_rng
        epsilon = self._get_epsilon()
        result = rollout_fn(
            env_rng,
            action_rng,
            batch_states,
            obs.astype(jnp.float32),
            avail_actions.astype(jnp.float32),
            runner_state,
            self.train_state.params,
            float(epsilon),
        )
        _, _, _, _, _, _, rollout, _ = result
        self._ingest_rollout_into_replay(rollout)
        self.total_env_steps += int(rollout["actions"].shape[0] * rollout["actions"].shape[1])
        return result

    def _ensure_episode_slots(self, n_envs: int):
        while len(self._current_episodes) < n_envs:
            self._current_episodes.append([])

    def _append_completed_episode(self, episode: Dict[str, np.ndarray]):
        self._completed_episodes.append(episode)
        if len(self._completed_episodes) > self.replay_buffer_size:
            self._completed_episodes.pop(0)

    def _transition_from_rollout(
        self,
        rollout_np: Dict[str, np.ndarray],
        actor_hidden: np.ndarray,
        prev_actions: np.ndarray,
        agent_dones: np.ndarray,
        env_idx: int,
        t: int,
    ) -> Dict[str, np.ndarray]:
        return qmix_transition_from_rollout(
            rollout_np,
            actor_hidden,
            prev_actions,
            agent_dones,
            env_idx,
            t,
        )

    def _pack_rollout_slice(
        self,
        rollout_np: Dict[str, np.ndarray],
        actor_hidden: np.ndarray,
        prev_actions: np.ndarray,
        agent_dones: np.ndarray,
        env_idx: int,
        start: int,
        end: int,
    ) -> Dict[str, np.ndarray]:
        return qmix_pack_rollout_slice(
            rollout_np,
            actor_hidden,
            prev_actions,
            agent_dones,
            env_idx,
            start,
            end,
            **self._qmix_pack_kwargs(),
        )

    def _qmix_pack_kwargs(self) -> Dict[str, Any]:
        return {
            "episode_limit": self._episode_limit,
            "num_agents": self.num_agents,
            "policy_obs_dim": self.policy_obs_dim,
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
            "global_reward": self.global_reward,
            "reward_scale": self.reward_scale,
        }

    def _pack_episode_arrays(self, episode: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return qmix_pack_episode_arrays(episode, **self._qmix_pack_kwargs())

    def _ingest_rollout_into_replay(self, rollout: Dict[str, jnp.ndarray]):
        rollout_np = {
            key: np.asarray(jax.device_get(value))
            for key, value in rollout.items()
        }
        time_steps, n_envs = rollout_np["actions"].shape[:2]
        self._ensure_episode_slots(n_envs)
        actor_hidden = rollout_np.get(
            "actor_hidden",
            np.zeros((time_steps, n_envs, self.num_agents, self.hidden_dim), dtype=np.float32),
        )
        prev_actions = rollout_np.get(
            "prev_actions",
            np.zeros((time_steps, n_envs, self.num_agents, self.action_dim), dtype=np.float32),
        )
        agent_dones = rollout_np.get(
            "agent_dones",
            np.zeros((time_steps, n_envs, self.num_agents), dtype=np.float32),
        )

        for env_idx in range(n_envs):
            start = 0
            done_indices = np.nonzero(rollout_np["dones"][:, env_idx] > 0.5)[0]
            for done_t_raw in done_indices:
                done_t = int(done_t_raw)
                current_episode = self._current_episodes[env_idx]
                if current_episode:
                    for t in range(start, done_t + 1):
                        current_episode.append(
                            self._transition_from_rollout(
                                rollout_np,
                                actor_hidden,
                                prev_actions,
                                agent_dones,
                                env_idx,
                                t,
                            )
                        )
                    self._append_completed_episode(self._pack_episode(current_episode))
                    self._current_episodes[env_idx] = []
                else:
                    self._append_completed_episode(
                        self._pack_rollout_slice(
                            rollout_np,
                            actor_hidden,
                            prev_actions,
                            agent_dones,
                            env_idx,
                            start,
                            done_t + 1,
                        )
                    )
                start = done_t + 1

            if start < time_steps:
                current_episode = self._current_episodes[env_idx]
                for t in range(start, time_steps):
                    current_episode.append(
                        self._transition_from_rollout(
                            rollout_np,
                            actor_hidden,
                            prev_actions,
                            agent_dones,
                            env_idx,
                            t,
                        )
                    )

    def _pack_episode(self, episode: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return qmix_pack_transition_episode(episode, **self._qmix_pack_kwargs())

    def _pad_and_stack(self, episodes: List[Any]) -> Dict[str, jnp.ndarray]:
        return qmix_stack_episodes(
            episodes,
            num_agents=self.num_agents,
            policy_obs_dim=self.policy_obs_dim,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            global_reward=self.global_reward,
            reward_scale=self.reward_scale,
        )

    def _sample_batch(self, n_available: int) -> Dict[str, jnp.ndarray]:
        indices = self._rng_np.choice(n_available, self.train_batch_size, replace=False)
        sampled = [self._completed_episodes[int(idx)] for idx in indices]
        return self._pad_and_stack(sampled)

    def _sample_update_batches(self, n_available: int, num_updates: int) -> Dict[str, jnp.ndarray]:
        batches = [self._sample_batch(n_available) for _ in range(num_updates)]
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *batches)

    @partial(jax.jit, static_argnums=0)
    def _forward_agent_q_values_and_hidden(
        self,
        agent_params: Dict[str, Any],
        agent_obs: jnp.ndarray,
        actions: jnp.ndarray,
    ):
        batch_size, tp1, n_agents, _ = agent_obs.shape
        zero_prev = jnp.zeros(
            (batch_size, 1, n_agents, self.action_dim),
            dtype=jnp.float32,
        )
        prev_actions = jnp.concatenate(
            [zero_prev, action_one_hot(actions[:, :-1], self.action_dim)],
            axis=1,
        )

        def scan_fn(hidden, xs):
            obs_t, prev_actions_t = xs
            decision_hidden = hidden.reshape(batch_size, n_agents, self.hidden_dim)
            inputs = jnp.concatenate([obs_t, prev_actions_t], axis=-1)
            inputs = inputs.reshape(batch_size * n_agents, -1)
            next_hidden, q_flat = self.agent_model.apply(
                {"params": agent_params},
                inputs,
                hidden,
            )
            q_values = q_flat.reshape(batch_size, n_agents, self.action_dim)
            return next_hidden, (q_values, decision_hidden)

        init_hidden = jnp.zeros((batch_size * n_agents, self.hidden_dim), dtype=jnp.float32)
        _, (q_seq, hidden_seq) = jax.lax.scan(
            scan_fn,
            init_hidden,
            (
                jnp.swapaxes(agent_obs, 0, 1),
                jnp.swapaxes(prev_actions, 0, 1),
            ),
        )
        return jnp.swapaxes(q_seq, 0, 1), jnp.swapaxes(hidden_seq, 0, 1)

    @partial(jax.jit, static_argnums=0)
    def _forward_agent_q_values(
        self,
        agent_params: Dict[str, Any],
        agent_obs: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        q_values, _ = self._forward_agent_q_values_and_hidden(
            agent_params,
            agent_obs,
            actions,
        )
        return q_values

    @partial(jax.jit, static_argnums=0)
    def _train_step(
        self,
        train_state_value: train_state.TrainState,
        target_params: Dict[str, Any],
        batch: Dict[str, jnp.ndarray],
    ):
        def loss_fn(params):
            mac_out = self._forward_agent_q_values(
                params["agent"],
                batch["agent_obs"],
                batch["actions"],
            )
            chosen_q = jnp.take_along_axis(
                mac_out[:, :-1],
                batch["actions"][:, :-1, :, None],
                axis=-1,
            ).squeeze(-1)

            target_mac_out = self._forward_agent_q_values(
                target_params["agent"],
                batch["agent_obs"],
                batch["actions"],
            )
            target_future = target_mac_out[:, 1:]
            target_future = self._masked_q_values(
                target_future,
                batch["available_actions"][:, 1:],
            )

            if self.double_q:
                online_future = self._masked_q_values(
                    mac_out[:, 1:],
                    batch["available_actions"][:, 1:],
                )
                max_actions = jnp.argmax(online_future, axis=-1)
                target_max_q = jnp.take_along_axis(
                    target_future,
                    max_actions[..., None],
                    axis=-1,
                ).squeeze(-1)
            else:
                target_max_q = jnp.max(target_future, axis=-1)

            chosen_tot = self.mixer_model.apply(
                {"params": params["mixer"]},
                chosen_q,
                batch["joint_obs"][:, :-1],
            )
            target_tot = self.mixer_model.apply(
                {"params": target_params["mixer"]},
                target_max_q,
                batch["joint_obs"][:, 1:],
            )

            rewards = batch["rewards"][..., None]
            terminated = batch["terminated"][..., None]
            masks = batch["masks"][..., None]
            targets = rewards + self.gamma * (1.0 - terminated) * target_tot
            td_error = chosen_tot - jax.lax.stop_gradient(targets)
            masked_td_error = td_error * masks
            denom = jnp.maximum(jnp.sum(masks), 1.0)
            loss = jnp.sum(jnp.square(masked_td_error)) / denom
            metrics = {
                "loss": loss,
                "td_error_abs": jnp.sum(jnp.abs(masked_td_error)) / denom,
                "q_taken_mean": jnp.sum(chosen_tot * masks) / (denom * self.num_agents),
                "target_mean": jnp.sum(targets * masks) / (denom * self.num_agents),
            }
            return loss, metrics

        grads, metrics = jax.grad(loss_fn, has_aux=True)(train_state_value.params)
        new_train_state = train_state_value.apply_gradients(grads=grads)
        metrics["grad_norm"] = global_norm(grads)
        return new_train_state, metrics

    @partial(jax.jit, static_argnums=0)
    def _train_many_steps(
        self,
        train_state_value: train_state.TrainState,
        target_params: Dict[str, Any],
        training_steps: jnp.ndarray,
        batches: Dict[str, jnp.ndarray],
    ):
        def scan_fn(carry, batch):
            state, tgt_params, step_count = carry
            state, metrics = self._train_step(state, tgt_params, batch)
            step_count = step_count + jnp.asarray(1, dtype=jnp.int32)
            should_update_target = (step_count % self.target_update_interval) == 0
            tgt_params = jax.lax.cond(
                should_update_target,
                lambda _: state.params,
                lambda _: tgt_params,
                operand=None,
            )
            return (state, tgt_params, step_count), metrics

        (new_state, new_target_params, new_training_steps), metrics_seq = jax.lax.scan(
            scan_fn,
            (train_state_value, target_params, training_steps),
            batches,
        )
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics_seq)
        return new_state, new_target_params, new_training_steps, metrics

    def update(self, **kwargs: Any) -> Dict[str, float]:
        n_available = len(self._completed_episodes)
        num_episodes_collected = int(kwargs.get("num_episodes_collected", 1))
        metadata = {
            "epsilon": float(self.epsilon),
            "total_env_steps": float(self.total_env_steps),
            "replay_buffer_episodes": float(n_available),
            "qmix_update_epochs": float(self.qmix_update_epochs or 0),
            "num_episodes_collected": float(num_episodes_collected),
            "learning_starts": float(self.learning_starts),
            "learning_started": float(self.total_env_steps >= self.learning_starts),
            "reward_scale": float(self.reward_scale),
            "mixer_init_scale": float(self.mixer_init_scale),
        }
        if n_available < self.train_batch_size:
            return metadata
        if self.total_env_steps < self.learning_starts:
            metadata["num_updates"] = 0.0
            return metadata

        num_updates = (
            self.qmix_update_epochs
            if self.qmix_update_epochs is not None
            else max(1, num_episodes_collected)
        )
        batches = self._sample_update_batches(n_available, num_updates)
        self.train_state, self.target_params, training_steps, metrics = self._train_many_steps(
            self.train_state,
            self.target_params,
            jnp.asarray(self.training_steps, dtype=jnp.int32),
            batches,
        )
        self.training_steps = int(training_steps)
        averaged = {key: float(value) for key, value in tree_to_host(metrics).items()}
        averaged.update(metadata)
        averaged["num_updates"] = float(num_updates)
        return averaged

    def greedy_actions(
        self,
        obs: jnp.ndarray,
        avail_actions: jnp.ndarray,
        runner_state: Dict[str, jnp.ndarray],
    ) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        obs_aug = self._augment_obs(obs)
        hidden_next, actions = self._policy_step(
            self.train_state.params["agent"],
            obs_aug,
            avail_actions,
            runner_state["hidden"],
            runner_state["prev_actions"],
            runner_state["agent_done_prev"],
            jax.random.PRNGKey(0),
            0.0,
            True,
        )
        next_runner = dict(runner_state)
        next_runner["hidden"] = hidden_next
        return actions, next_runner

    def state_dict(self) -> Dict[str, Any]:
        return {
            "train_state": serialization.to_state_dict(self.train_state),
            "target_params": tree_to_host(self.target_params),
            "training_steps": int(self.training_steps),
            "total_env_steps": int(self.total_env_steps),
            "epsilon": float(self.epsilon),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.train_state = serialization.from_state_dict(
            self.train_state,
            state_dict["train_state"],
        )
        self.target_params = serialization.from_state_dict(
            self.train_state.params,
            state_dict["target_params"],
        )
        self.training_steps = int(state_dict.get("training_steps", 0))
        self.total_env_steps = int(state_dict.get("total_env_steps", 0))
        self.epsilon = float(state_dict.get("epsilon", self.epsilon_start))

    def load_actor_state_dict(self, state_dict: Dict[str, Any]):
        self.load_state_dict(state_dict)
