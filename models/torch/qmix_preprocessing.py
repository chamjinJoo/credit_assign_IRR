from typing import Dict, List

import torch

from models.torch.common import classify_staghunt_delta_scenario



def qmix_extract_policy_hidden(hidden_entry, hidden_dim: int) -> torch.Tensor:
    if isinstance(hidden_entry, (tuple, list)):
        hidden_entry = hidden_entry[0] if hidden_entry else None
    if hidden_entry is None:
        return torch.zeros(hidden_dim)
    hidden = hidden_entry.detach().to("cpu").float().reshape(-1)
    if hidden.numel() == 0:
        return torch.zeros(hidden_dim)
    if hidden.numel() < hidden_dim:
        padded = torch.zeros(hidden_dim)
        padded[-hidden.numel():] = hidden
        return padded
    return hidden[-hidden_dim:]


def _obs_tensor_cpu(obs: torch.Tensor) -> torch.Tensor:
    return obs.detach().to("cpu").float()


def qmix_pack_episode(
    episode: List[Dict],
    *,
    obs_dim: int,
    num_agents: int,
    action_dim: int,
    hidden_dim: int,
    reward_scale: float,
    env_required_agents=None,
) -> Dict[str, torch.Tensor]:
    t_len = len(episode)
    tp1 = t_len + 1
    state_dim = obs_dim * num_agents

    joint_obs = torch.zeros(tp1, state_dim, dtype=torch.float32)
    agent_obs = torch.zeros(tp1, num_agents, obs_dim, dtype=torch.float32)
    actions = torch.zeros(tp1, num_agents, dtype=torch.long)
    rewards = torch.zeros(t_len, dtype=torch.float32)
    raw_agent_rewards = torch.zeros(t_len, num_agents, dtype=torch.float32)
    terminated = torch.zeros(t_len, dtype=torch.float32)
    masks = torch.ones(t_len, dtype=torch.float32)
    available_actions = torch.ones(tp1, num_agents, action_dim, dtype=torch.float32)
    prev_actions = torch.zeros(t_len, num_agents, dtype=torch.long)
    hidden_states = torch.zeros(t_len, num_agents, hidden_dim, dtype=torch.float32)
    staghunt_delta_categories = torch.zeros(t_len, num_agents, 4, dtype=torch.bool)
    has_available_actions = False

    required_agents = env_required_agents
    if required_agents is None and episode:
        first_info = episode[0].get("info", {}) or {}
        huntable_objects = first_info.get("huntable_objects", {}) or {}
        huntable_stags = huntable_objects.get("huntable_stags", {}) or {}
        required_agents = {
            int(stag_id): list(stag_info.get("required_agents", []))
            for stag_id, stag_info in huntable_stags.items()
            if isinstance(stag_info, dict) and "required_agents" in stag_info
        } or None

    category_index = {
        "stag_reward": 0,
        "forage_reward": 1,
        "focal_forage_reward": 2,
        "no_reward": 3,
    }

    for t, trans in enumerate(episode):
        obs_t = [_obs_tensor_cpu(trans["bold_o_t"][i]) for i in range(num_agents)]
        obs_stacked = torch.stack(obs_t)
        agent_obs[t] = obs_stacked
        joint_obs[t] = obs_stacked.reshape(-1)

        action_t = [int(trans["bold_a_t"][i]) for i in range(num_agents)]
        actions[t] = torch.tensor(action_t, dtype=torch.long)

        reward_t = torch.tensor(
            [float(trans["bold_r_t"].get(i, 0.0)) for i in range(num_agents)],
            dtype=torch.float32,
        ) * float(reward_scale)
        raw_agent_rewards[t] = reward_t
        terminated[t] = 1.0 if any(trans["dones"].values()) else 0.0

        if trans.get("info", {}).get("stag_hunts") is not None:
            for agent_id in range(num_agents):
                scenario = classify_staghunt_delta_scenario(
                    trans,
                    agent_id,
                    num_agents,
                    required_agents,
                )
                staghunt_delta_categories[t, agent_id, category_index[scenario]] = True

        prev_action_dict = trans.get("bold_a_t_minus_1", {})
        prev_actions[t] = torch.tensor(
            [int(prev_action_dict.get(i, 0)) for i in range(num_agents)],
            dtype=torch.long,
        )

        hidden_dict = trans.get("hidden_states", {}) or {}
        for agent_id in range(num_agents):
            hidden_states[t, agent_id] = qmix_extract_policy_hidden(
                hidden_dict.get(agent_id),
                hidden_dim,
            )

        avail = trans.get("available_actions", None)
        if avail is not None:
            has_available_actions = True
            for i in range(num_agents):
                if i in avail:
                    available_actions[t, i] = torch.as_tensor(avail[i], dtype=torch.float32)

        if t == t_len - 1 and "next_observations" in trans:
            next_obs = [
                _obs_tensor_cpu(trans["next_observations"][i])
                for i in range(num_agents)
            ]
            next_obs_stacked = torch.stack(next_obs)
            agent_obs[t + 1] = next_obs_stacked
            joint_obs[t + 1] = next_obs_stacked.reshape(-1)
            if avail is not None:
                for i in range(num_agents):
                    if i in avail:
                        available_actions[t + 1, i] = torch.as_tensor(
                            avail[i],
                            dtype=torch.float32,
                        )

    if t_len > 1:
        masks[1:] *= 1.0 - terminated[:-1]

    rewards[:t_len] = raw_agent_rewards[:t_len].sum(dim=-1)

    return {
        "packed": True,
        "length": torch.tensor(t_len, dtype=torch.long),
        "joint_obs": joint_obs,
        "agent_obs": agent_obs,
        "actions": actions,
        "rewards": rewards,
        "raw_agent_rewards": raw_agent_rewards,
        "terminated": terminated,
        "masks": masks,
        "available_actions": available_actions,
        "has_available_actions": torch.tensor(1 if has_available_actions else 0, dtype=torch.bool),
        "prev_actions": prev_actions,
        "hidden_states": hidden_states,
        "staghunt_delta_categories": staghunt_delta_categories,
    }


def qmix_stack_packed_episodes(
    episodes: List[Dict[str, torch.Tensor]],
    *,
    obs_dim: int,
    num_agents: int,
    action_dim: int,
    hidden_dim: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    batch_size = len(episodes)
    max_t = max(int(ep["length"].item()) for ep in episodes)
    tp1 = max_t + 1
    state_dim = obs_dim * num_agents

    joint_obs = torch.zeros(batch_size, tp1, state_dim, device=device)
    agent_obs = torch.zeros(batch_size, tp1, num_agents, obs_dim, device=device)
    actions = torch.zeros(batch_size, tp1, num_agents, dtype=torch.long, device=device)
    rewards = torch.zeros(batch_size, max_t, device=device)
    raw_agent_rewards = torch.zeros(batch_size, max_t, num_agents, device=device)
    terminated = torch.zeros(batch_size, max_t, device=device)
    masks = torch.zeros(batch_size, max_t, device=device)
    available_actions = torch.ones(batch_size, tp1, num_agents, action_dim, device=device)
    prev_actions = torch.zeros(batch_size, max_t, num_agents, dtype=torch.long, device=device)
    hidden_states = torch.zeros(batch_size, max_t, num_agents, hidden_dim, device=device)
    staghunt_delta_categories = torch.zeros(batch_size, max_t, num_agents, 4, dtype=torch.bool, device=device)
    has_staghunt_delta_categories = False
    has_available_actions = False

    for b, episode in enumerate(episodes):
        t_len = int(episode["length"].item())
        joint_obs[b, : t_len + 1] = episode["joint_obs"][: t_len + 1].to(device)
        agent_obs[b, : t_len + 1] = episode["agent_obs"][: t_len + 1].to(device)
        actions[b, : t_len + 1] = episode["actions"][: t_len + 1].to(device)
        rewards[b, :t_len] = episode["rewards"][:t_len].to(device)
        raw_agent_rewards[b, :t_len] = episode["raw_agent_rewards"][:t_len].to(device)
        terminated[b, :t_len] = episode["terminated"][:t_len].to(device)
        masks[b, :t_len] = episode["masks"][:t_len].to(device)
        prev_actions[b, :t_len] = episode["prev_actions"][:t_len].to(device)
        hidden_states[b, :t_len] = episode["hidden_states"][:t_len].to(device)
        if "staghunt_delta_categories" in episode:
            staghunt_delta_categories[b, :t_len] = episode["staghunt_delta_categories"][:t_len].to(device)
            has_staghunt_delta_categories = True
        if bool(episode["has_available_actions"].item()):
            has_available_actions = True
            available_actions[b, : t_len + 1] = episode["available_actions"][: t_len + 1].to(device)

    result = {
        "joint_obs": joint_obs,
        "agent_obs": agent_obs,
        "actions": actions,
        "rewards": rewards,
        "raw_agent_rewards": raw_agent_rewards,
        "terminated": terminated,
        "masks": masks,
        "prev_actions": prev_actions,
        "hidden_states": hidden_states,
    }
    if has_staghunt_delta_categories:
        result["staghunt_delta_categories"] = staghunt_delta_categories
    if has_available_actions:
        result["available_actions"] = available_actions
    return result


def qmix_pad_and_stack_legacy(
    episodes: List[List[Dict]],
    *,
    obs_dim: int,
    num_agents: int,
    action_dim: int,
    reward_scale: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    batch_size = len(episodes)
    max_t = max(len(ep) for ep in episodes)
    tp1 = max_t + 1

    joint_obs = torch.zeros(batch_size, tp1, obs_dim * num_agents, device=device)
    agent_obs = torch.zeros(batch_size, tp1, num_agents, obs_dim, device=device)
    actions = torch.zeros(batch_size, tp1, num_agents, dtype=torch.long, device=device)
    available_actions = torch.ones(batch_size, tp1, num_agents, action_dim, device=device)
    has_available_actions = False

    rewards = torch.zeros(batch_size, max_t, device=device)
    terminated = torch.zeros(batch_size, max_t, device=device)
    masks = torch.zeros(batch_size, max_t, device=device)

    for b, episode in enumerate(episodes):
        t_len = len(episode)
        for t, trans in enumerate(episode):
            obs_t = torch.stack([trans["bold_o_t"][i] for i in range(num_agents)]).to(device)
            agent_obs[b, t] = obs_t
            joint_obs[b, t] = obs_t.reshape(-1)
            actions[b, t] = torch.tensor(
                [trans["bold_a_t"][i] for i in range(num_agents)],
                dtype=torch.long,
                device=device,
            )
            rewards[b, t] = sum(trans["bold_r_t"].values()) * float(reward_scale)
            terminated[b, t] = 1.0 if any(trans["dones"].values()) else 0.0

            avail = trans.get("available_actions", None)
            if avail is not None:
                has_available_actions = True
                for i in range(num_agents):
                    if i in avail:
                        available_actions[b, t, i] = torch.tensor(
                            avail[i],
                            dtype=torch.float32,
                            device=device,
                        )

            if t == t_len - 1 and "next_observations" in trans:
                next_obs = torch.stack(
                    [trans["next_observations"][i] for i in range(num_agents)]
                ).to(device)
                agent_obs[b, t + 1] = next_obs
                joint_obs[b, t + 1] = next_obs.reshape(-1)

        masks[b, :t_len] = 1.0

    if max_t > 1:
        masks[:, 1:] = masks[:, 1:] * (1 - terminated[:, :-1])

    result = {
        "joint_obs": joint_obs,
        "agent_obs": agent_obs,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "masks": masks,
    }
    if has_available_actions:
        result["available_actions"] = available_actions
    return result
