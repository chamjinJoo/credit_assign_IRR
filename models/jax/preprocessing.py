from __future__ import annotations

from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from .common import gather_other_agent_obs

QMIX_PACKED_KEYS = (
    "agent_obs",
    "joint_obs",
    "actions",
    "rewards",
    "agent_rewards",
    "terminated",
    "masks",
    "actor_hidden",
    "prev_actions",
    "agent_dones",
    "available_actions",
)


def qmix_transition_from_rollout(
    rollout_np: Dict[str, np.ndarray],
    actor_hidden: np.ndarray,
    prev_actions: np.ndarray,
    agent_dones: np.ndarray,
    env_idx: int,
    t: int,
) -> Dict[str, np.ndarray]:
    return {
        "obs": rollout_np["obs"][t, env_idx],
        "avail_actions": rollout_np["avail_actions"][t, env_idx],
        "actor_hidden": actor_hidden[t, env_idx],
        "prev_actions": prev_actions[t, env_idx],
        "actions": rollout_np["actions"][t, env_idx],
        "rewards": rollout_np["rewards"][t, env_idx],
        "done": bool(rollout_np["dones"][t, env_idx] > 0.5),
        "agent_dones": agent_dones[t, env_idx],
        "next_obs": rollout_np["next_obs"][t, env_idx],
        "next_avail_actions": rollout_np["next_avail_actions"][t, env_idx],
    }


def qmix_pack_episode_arrays(
    episode: Dict[str, np.ndarray],
    *,
    episode_limit: int | None,
    num_agents: int,
    policy_obs_dim: int,
    state_dim: int,
    hidden_dim: int,
    action_dim: int,
    global_reward: bool,
    reward_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    episode_len = int(episode["actions"].shape[0])
    max_t = episode_limit or episode_len
    max_t = max(max_t, episode_len)
    tp1 = max_t + 1

    agent_obs = np.zeros((tp1, num_agents, policy_obs_dim), dtype=np.float32)
    joint_obs = np.zeros((tp1, state_dim), dtype=np.float32)
    actions = np.zeros((tp1, num_agents), dtype=np.int32)
    rewards = np.zeros((max_t,), dtype=np.float32)
    agent_rewards = np.zeros((max_t, num_agents), dtype=np.float32)
    terminated = np.zeros((max_t,), dtype=np.float32)
    masks = np.zeros((max_t,), dtype=np.float32)
    actor_hidden = np.zeros((max_t, num_agents, hidden_dim), dtype=np.float32)
    prev_actions = np.zeros((max_t, num_agents, action_dim), dtype=np.float32)
    agent_dones = np.zeros((max_t, num_agents), dtype=np.float32)
    available_actions = np.ones((tp1, num_agents, action_dim), dtype=np.float32)

    obs = episode["obs"].astype(np.float32, copy=False)
    agent_obs[:episode_len] = obs
    joint_obs[:episode_len] = obs.reshape(episode_len, -1)
    actions[:episode_len] = episode["actions"].astype(np.int32, copy=False)
    available_actions[:episode_len] = episode["avail_actions"].astype(np.float32, copy=False)
    actor_hidden[:episode_len] = episode["actor_hidden"].astype(np.float32, copy=False)
    prev_actions[:episode_len] = episode["prev_actions"].astype(np.float32, copy=False)
    agent_dones[:episode_len] = episode["agent_dones"].astype(np.float32, copy=False)
    reward_seq = episode["rewards"].astype(np.float32, copy=False) * float(reward_scale)
    agent_rewards[:episode_len] = reward_seq
    rewards[:episode_len] = (
        np.mean(reward_seq, axis=-1) if global_reward else np.sum(reward_seq, axis=-1)
    )
    terminated[:episode_len] = episode["dones"].astype(np.float32, copy=False)
    masks[:episode_len] = 1.0
    if episode_len > 1:
        masks[1:episode_len] *= 1.0 - terminated[: episode_len - 1]

    next_obs = episode["next_obs"][-1].astype(np.float32, copy=False)
    agent_obs[episode_len] = next_obs
    joint_obs[episode_len] = next_obs.reshape(-1)
    available_actions[episode_len] = episode["next_avail_actions"][-1].astype(
        np.float32,
        copy=False,
    )

    return {
        "agent_obs": agent_obs,
        "joint_obs": joint_obs,
        "actions": actions,
        "rewards": rewards,
        "agent_rewards": agent_rewards,
        "terminated": terminated,
        "masks": masks,
        "actor_hidden": actor_hidden,
        "prev_actions": prev_actions,
        "agent_dones": agent_dones,
        "available_actions": available_actions,
        "length": np.asarray(episode_len, dtype=np.int32),
    }


def qmix_pack_rollout_slice(
    rollout_np: Dict[str, np.ndarray],
    actor_hidden: np.ndarray,
    prev_actions: np.ndarray,
    agent_dones: np.ndarray,
    env_idx: int,
    start: int,
    end: int,
    **pack_kwargs: Any,
) -> Dict[str, np.ndarray]:
    sl = slice(start, end)
    episode = {
        "obs": rollout_np["obs"][sl, env_idx],
        "avail_actions": rollout_np["avail_actions"][sl, env_idx],
        "actor_hidden": actor_hidden[sl, env_idx],
        "prev_actions": prev_actions[sl, env_idx],
        "actions": rollout_np["actions"][sl, env_idx],
        "rewards": rollout_np["rewards"][sl, env_idx],
        "dones": rollout_np["dones"][sl, env_idx],
        "agent_dones": agent_dones[sl, env_idx],
        "next_obs": rollout_np["next_obs"][sl, env_idx],
        "next_avail_actions": rollout_np["next_avail_actions"][sl, env_idx],
    }
    return qmix_pack_episode_arrays(episode, **pack_kwargs)


def qmix_pack_transition_episode(
    episode: List[Dict[str, np.ndarray]],
    *,
    episode_limit: int | None,
    num_agents: int,
    policy_obs_dim: int,
    state_dim: int,
    hidden_dim: int,
    action_dim: int,
    global_reward: bool,
    reward_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    episode_len = len(episode)
    max_t = episode_limit or episode_len
    max_t = max(max_t, episode_len)
    tp1 = max_t + 1

    arrays = {
        "obs": np.zeros((episode_len, num_agents, policy_obs_dim), dtype=np.float32),
        "avail_actions": np.ones((episode_len, num_agents, action_dim), dtype=np.float32),
        "actor_hidden": np.zeros((episode_len, num_agents, hidden_dim), dtype=np.float32),
        "prev_actions": np.zeros((episode_len, num_agents, action_dim), dtype=np.float32),
        "actions": np.zeros((episode_len, num_agents), dtype=np.int32),
        "rewards": np.zeros((episode_len, num_agents), dtype=np.float32),
        "dones": np.zeros((episode_len,), dtype=np.float32),
        "agent_dones": np.zeros((episode_len, num_agents), dtype=np.float32),
        "next_obs": np.zeros((episode_len, num_agents, policy_obs_dim), dtype=np.float32),
        "next_avail_actions": np.ones((episode_len, num_agents, action_dim), dtype=np.float32),
    }
    eye_actions = np.eye(action_dim, dtype=np.float32)

    for t, transition in enumerate(episode):
        arrays["obs"][t] = transition["obs"].astype(np.float32)
        arrays["avail_actions"][t] = transition["avail_actions"].astype(np.float32)
        arrays["actor_hidden"][t] = transition.get(
            "actor_hidden",
            np.zeros((num_agents, hidden_dim), dtype=np.float32),
        ).astype(np.float32)
        if "prev_actions" in transition:
            arrays["prev_actions"][t] = transition["prev_actions"].astype(np.float32)
        elif t > 0:
            arrays["prev_actions"][t] = eye_actions[arrays["actions"][t - 1]]
        arrays["actions"][t] = transition["actions"].astype(np.int32)
        arrays["rewards"][t] = transition["rewards"].astype(np.float32)
        arrays["dones"][t] = 1.0 if transition["done"] else 0.0
        arrays["agent_dones"][t] = transition.get(
            "agent_dones",
            np.zeros((num_agents,), dtype=np.float32),
        ).astype(np.float32)
        arrays["next_obs"][t] = transition["next_obs"].astype(np.float32)
        arrays["next_avail_actions"][t] = transition["next_avail_actions"].astype(np.float32)

    packed = qmix_pack_episode_arrays(
        arrays,
        episode_limit=max_t,
        num_agents=num_agents,
        policy_obs_dim=policy_obs_dim,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        global_reward=global_reward,
        reward_scale=reward_scale,
    )
    packed["length"] = np.asarray(episode_len, dtype=np.int32)
    return packed


def qmix_stack_episodes(
    episodes: List[Any],
    *,
    num_agents: int,
    policy_obs_dim: int,
    state_dim: int,
    hidden_dim: int,
    action_dim: int,
    global_reward: bool,
    reward_scale: float = 1.0,
) -> Dict[str, jnp.ndarray]:
    if episodes and isinstance(episodes[0], dict):
        return {
            key: jnp.asarray(np.stack([episode[key] for episode in episodes], axis=0))
            for key in QMIX_PACKED_KEYS
        }

    max_t = max(len(episode) for episode in episodes)
    packed = [
        qmix_pack_transition_episode(
            episode,
            episode_limit=max_t,
            num_agents=num_agents,
            policy_obs_dim=policy_obs_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            global_reward=global_reward,
            reward_scale=reward_scale,
        )
        for episode in episodes
    ]
    return {
        key: jnp.asarray(np.stack([episode[key] for episode in packed], axis=0))
        for key in QMIX_PACKED_KEYS
    }


def flatten_actor_batch(
    rollout: Dict[str, jnp.ndarray],
    advantages: jnp.ndarray,
    *,
    num_agents: int,
    policy_obs_dim: int,
    action_dim: int,
    actor_state_dim: int,
    normalize_advantages,
) -> Dict[str, jnp.ndarray]:
    total_steps = rollout["obs"].shape[0] * rollout["obs"].shape[1]
    masks = jnp.broadcast_to(
        rollout["masks"].reshape(rollout["masks"].shape + (1,)),
        rollout["actions"].shape,
    ).reshape(total_steps, num_agents)

    if advantages.ndim == 2:
        adv = jnp.repeat(advantages[..., None], num_agents, axis=-1)
    else:
        adv = advantages
    adv = normalize_advantages(adv.reshape(total_steps, num_agents))

    return {
        "obs": rollout["obs"].reshape(total_steps, num_agents, policy_obs_dim),
        "avail_actions": rollout["avail_actions"].reshape(total_steps, num_agents, action_dim),
        "actions": rollout["actions"].reshape(total_steps, num_agents),
        "old_log_probs": rollout["log_probs"].reshape(total_steps, num_agents),
        "prev_actions": rollout["prev_actions"].reshape(total_steps, num_agents, action_dim),
        "hidden": rollout["actor_hidden"].reshape(total_steps, num_agents, actor_state_dim),
        "masks": masks,
        "loss_masks": masks,
        "advantages": adv,
    }


def flatten_critic_batch(
    rollout: Dict[str, jnp.ndarray],
    returns: jnp.ndarray,
    *,
    num_agents: int,
    policy_obs_dim: int,
    action_dim: int,
    critic_state_dim: int,
    use_centralized_critic: bool,
) -> Dict[str, jnp.ndarray]:
    if use_centralized_critic:
        total_steps = rollout["obs"].shape[0] * rollout["obs"].shape[1]
        base_masks = rollout["masks"].reshape(-1)
        if returns.ndim == 2:
            returns_flat = returns.reshape(-1)
            loss_masks = base_masks
        else:
            returns_flat = returns.reshape(-1, num_agents)
            loss_masks = jnp.broadcast_to(base_masks[:, None], (total_steps, num_agents))
        return {
            "obs": rollout["obs"].reshape(rollout["obs"].shape[0], rollout["obs"].shape[1], -1).reshape(-1, policy_obs_dim * num_agents),
            "prev_actions": rollout["prev_actions"].reshape(rollout["prev_actions"].shape[0], rollout["prev_actions"].shape[1], -1).reshape(-1, action_dim * num_agents),
            "hidden": rollout["critic_hidden"].reshape(total_steps, critic_state_dim),
            "masks": base_masks,
            "loss_masks": loss_masks,
            "returns": returns_flat,
        }

    total_steps = rollout["obs"].shape[0] * rollout["obs"].shape[1] * rollout["obs"].shape[2]
    per_agent_masks = jnp.broadcast_to(
        rollout["masks"].reshape(rollout["masks"].shape + (1,)),
        rollout["actions"].shape,
    ).reshape(-1)
    return {
        "obs": rollout["obs"].reshape(-1, policy_obs_dim),
        "prev_actions": rollout["prev_actions"].reshape(-1, action_dim),
        "hidden": rollout["critic_hidden"].reshape(total_steps, critic_state_dim),
        "masks": per_agent_masks,
        "loss_masks": per_agent_masks,
        "returns": returns.reshape(-1),
    }


def flatten_delta_critic_batch(
    rollout: Dict[str, jnp.ndarray],
    delta_returns: jnp.ndarray,
    *,
    policy_obs_dim: int,
    action_dim: int,
    delta_critic_state_dim: int,
) -> Dict[str, jnp.ndarray]:
    total_steps = rollout["obs"].shape[0] * rollout["obs"].shape[1] * rollout["obs"].shape[2]
    per_agent_masks = jnp.broadcast_to(
        rollout["masks"].reshape(rollout["masks"].shape + (1,)),
        rollout["actions"].shape,
    ).reshape(-1)
    return {
        "obs": rollout["obs"].reshape(-1, policy_obs_dim),
        "prev_actions": rollout["prev_actions"].reshape(-1, action_dim),
        "hidden": rollout["delta_critic_hidden"].reshape(total_steps, delta_critic_state_dim),
        "masks": per_agent_masks,
        "loss_masks": per_agent_masks,
        "returns": delta_returns.reshape(-1),
    }


def preprocess_causal_rollout(
    rollout: Dict[str, jnp.ndarray],
    *,
    num_agents: int,
    policy_obs_dim: int,
    action_dim: int,
    actor_state_dim: int,
    bridge_hidden_dim: int,
    bridge_prev_action_dim: int,
    use_recurrent_policy: bool,
    global_reward: bool,
):
    obs = rollout["obs"]
    time_steps, n_envs = obs.shape[:2]
    flat_batch = time_steps * n_envs

    obs_4d = obs.reshape(flat_batch, num_agents, policy_obs_dim)
    local_obs = obs_4d.reshape(-1, policy_obs_dim)
    others_obs = gather_other_agent_obs(obs_4d).reshape(
        -1,
        (num_agents - 1) * policy_obs_dim,
    )
    prev_actions_4d = rollout["prev_actions"].reshape(flat_batch, num_agents, action_dim)
    prev_actions_others = gather_other_agent_obs(prev_actions_4d).reshape(
        -1,
        bridge_prev_action_dim,
    )

    if use_recurrent_policy:
        hidden_4d = rollout["actor_hidden"].reshape(flat_batch, num_agents, actor_state_dim)
        z = hidden_4d.reshape(-1, actor_state_dim)
        hidden_others = gather_other_agent_obs(hidden_4d).reshape(-1, bridge_hidden_dim)
    else:
        n_flat = flat_batch * num_agents
        z = jnp.zeros((n_flat, 0), dtype=jnp.float32)
        hidden_others = jnp.zeros((n_flat, 0), dtype=jnp.float32)

    bridge_context = jnp.concatenate([hidden_others, prev_actions_others], axis=-1)
    actions = rollout["actions"].reshape(-1)
    agent_ids = jnp.tile(jnp.arange(num_agents, dtype=jnp.int32), flat_batch)
    if global_reward:
        team_rewards = rollout["rewards"].sum(axis=-1) / num_agents
        reward_targets = jnp.repeat(team_rewards.reshape(-1), num_agents, axis=0)[:, None]
    else:
        reward_targets = jnp.repeat(
            rollout["rewards"][:, :, None, :],
            num_agents,
            axis=2,
        ).reshape(-1, num_agents)

    return local_obs, others_obs, z, bridge_context, actions, reward_targets, agent_ids


def causal_batch_from_rollout(
    rollout: Dict[str, jnp.ndarray],
    *,
    num_agents: int,
    **preprocess_kwargs: Any,
) -> Dict[str, jnp.ndarray]:
    local_obs, others_obs, z, bridge_context, actions, targets, agent_ids = preprocess_causal_rollout(
        rollout,
        num_agents=num_agents,
        **preprocess_kwargs,
    )
    loss_mask = jnp.broadcast_to(
        rollout["masks"].reshape(rollout["masks"].shape + (1,)),
        rollout["actions"].shape,
    ).reshape(-1).astype(jnp.float32)
    return {
        "z": z.astype(jnp.float32),
        "w": others_obs.astype(jnp.float32),
        "x_obs": local_obs.astype(jnp.float32),
        "x_hid": bridge_context.astype(jnp.float32),
        "actions": actions.astype(jnp.int32),
        "targets": targets.astype(jnp.float32),
        "agent_ids": agent_ids.astype(jnp.int32),
        "loss_mask": loss_mask,
    }


def bridge_inputs_from_rollout(
    rollout: Dict[str, jnp.ndarray],
    *,
    num_agents: int,
    action_dim: int,
    **preprocess_kwargs: Any,
):
    local_obs, others_obs, _, bridge_context, actions, reward_targets, _ = preprocess_causal_rollout(
        rollout,
        num_agents=num_agents,
        action_dim=action_dim,
        **preprocess_kwargs,
    )
    probs = rollout["probs"].reshape(-1, action_dim)
    avail_actions = rollout["avail_actions"].reshape(-1, action_dim)
    return local_obs, others_obs, bridge_context, actions, reward_targets, probs, avail_actions


def slice_batch(batch: Dict[str, jnp.ndarray], idx: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    return {key: value[idx] for key, value in batch.items()}


def minibatch_indices(size: int, *, batch_size: int, num_batches: int):
    if size <= 0:
        return
    if batch_size > 0:
        for start in range(0, size, batch_size):
            yield jnp.arange(start, min(start + batch_size, size))
        return

    n_batches = min(num_batches, size)
    for batch_idx in range(n_batches):
        start = (size * batch_idx) // n_batches
        end = (size * (batch_idx + 1)) // n_batches
        if end > start:
            yield jnp.arange(start, end)


def causal_minibatch_indices(te_total: int, *, num_agents: int, causal_mini_batch_size: int):
    te_idx = jnp.arange(te_total)
    for te_start in range(0, te_total, causal_mini_batch_size):
        te_end = te_start + causal_mini_batch_size
        if te_end <= te_total:
            te_chunk = te_idx[te_start:te_end]
        else:
            pad = causal_mini_batch_size - (te_total - te_start)
            te_chunk = jnp.concatenate([te_idx[te_start:], te_idx[:pad]])
        yield (te_chunk[:, None] * num_agents + jnp.arange(num_agents)[None, :]).reshape(-1)


def qmix_batch_to_bridge_rollout(
    batch: Dict[str, jnp.ndarray],
    *,
    action_dim: int,
    masked_q_values,
    mac_out: jnp.ndarray | None = None,
    actor_hidden: jnp.ndarray | None = None,
) -> Dict[str, jnp.ndarray]:
    obs = jnp.swapaxes(batch["agent_obs"][:, :-1], 0, 1)
    actions = jnp.swapaxes(batch["actions"][:, :-1], 0, 1)
    rewards = jnp.swapaxes(batch["agent_rewards"], 0, 1)
    masks = jnp.swapaxes(batch["masks"], 0, 1)
    dones = jnp.swapaxes(batch["terminated"], 0, 1)
    avail_actions = jnp.swapaxes(batch["available_actions"][:, :-1], 0, 1)
    if actor_hidden is None:
        actor_hidden = batch["actor_hidden"]
    actor_hidden = jnp.swapaxes(actor_hidden, 0, 1)
    prev_actions = jnp.swapaxes(batch["prev_actions"], 0, 1)
    agent_dones = jnp.swapaxes(batch["agent_dones"], 0, 1)

    if mac_out is None:
        probs = jnp.ones_like(avail_actions) / float(action_dim)
    else:
        q_values = jnp.swapaxes(mac_out[:, :-1], 0, 1)
        masked_q = masked_q_values(q_values, avail_actions)
        probs = jax.nn.softmax(masked_q, axis=-1)

    return {
        "obs": obs.astype(jnp.float32),
        "avail_actions": avail_actions.astype(jnp.float32),
        "actor_hidden": actor_hidden.astype(jnp.float32),
        "prev_actions": prev_actions.astype(jnp.float32),
        "actions": actions.astype(jnp.int32),
        "rewards": rewards.astype(jnp.float32),
        "dones": dones.astype(jnp.float32),
        "masks": masks.astype(jnp.float32),
        "agent_dones": agent_dones.astype(jnp.float32),
        "probs": probs.astype(jnp.float32),
    }


def masked_delta_metrics(delta_agent: jnp.ndarray, masks: jnp.ndarray) -> Dict[str, float]:
    agent_masks = jnp.broadcast_to(masks[..., None], delta_agent.shape)
    denom = jnp.maximum(jnp.sum(agent_masks), 1.0)
    mean = jnp.sum(delta_agent * agent_masks) / denom
    centered = (delta_agent - mean) * agent_masks
    std = jnp.sqrt(jnp.sum(jnp.square(centered)) / denom)
    abs_mean = jnp.sum(jnp.abs(delta_agent) * agent_masks) / denom
    team_delta = jnp.sum(delta_agent, axis=-1)
    team_mean = jnp.sum(team_delta * masks) / jnp.maximum(jnp.sum(masks), 1.0)
    return {
        "delta_mean": float(mean),
        "delta_std": float(std),
        "delta_abs_mean": float(abs_mean),
        "delta_team_mean": float(team_mean),
    }
