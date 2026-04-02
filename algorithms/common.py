from typing import Dict, List, Optional, Sequence, Tuple

import torch


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    next_value: float,
    dones: Sequence[bool],
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    advantages = []
    returns = []
    gae = 0

    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[step]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[step]
            next_val = values[step + 1]

        delta = rewards[step] + gamma * next_val * next_non_terminal - values[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[step])

    return advantages, returns


def chunk_sequence(tensor: torch.Tensor, chunk_len: int) -> torch.Tensor:
    t_total = tensor.size(0)
    num_chunks = t_total // chunk_len
    if num_chunks == 0:
        return tensor.unsqueeze(1)

    tensor = tensor[:num_chunks * chunk_len]
    shape = (num_chunks, chunk_len) + tensor.shape[1:]
    tensor = tensor.view(shape)

    permute_dims = (1, 0) + tuple(range(2, tensor.dim()))
    return tensor.permute(permute_dims).contiguous()


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.numel() > 1:
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return torch.zeros_like(advantages)


def concatenate_agent_observations(
    observations: Dict[int, torch.Tensor],
    num_agents: int,
    obs_dim: int,
    device: torch.device,
) -> torch.Tensor:
    sorted_obs = [observations[i] for i in range(num_agents) if i in observations]
    if not sorted_obs:
        return torch.zeros(num_agents * obs_dim, device=device)
    return torch.cat(sorted_obs, dim=-1)


def concatenate_agent_actions(
    actions: Dict[int, int],
    num_agents: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [actions[i] for i in range(num_agents) if i in actions],
        device=device,
    )


def build_available_actions_batch(
    batch_data: List[Dict],
    num_agents: int,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    available_actions_batch = {}
    if 'available_actions' in batch_data[0] and batch_data[0]['available_actions'] is not None:
        for agent_id in range(num_agents):
            if agent_id in batch_data[0]['available_actions']:
                avail_list = [t['available_actions'][agent_id] for t in batch_data]
                available_actions_batch[agent_id] = torch.tensor(
                    avail_list,
                    dtype=torch.float32,
                    device=device,
                )
    return available_actions_batch


def build_agent_masks_batch(
    batch_data: List[Dict],
    num_agents: int,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    masks_batch = {}
    for agent_id in range(num_agents):
        masks_batch[agent_id] = torch.tensor(
            [t['masks'][agent_id] for t in batch_data],
            device=device,
        ).unsqueeze(1)
    return masks_batch


def classify_staghunt_delta_scenario(
    transition: Dict,
    agent_id: int,
    num_agents: int,
    env_required_agents: Optional[Dict[int, List[int]]],
) -> str:
    info = transition.get('info', {})
    stag_hunts = info.get('stag_hunts', {})
    other_stag = False
    other_forage = False

    for other_id in range(num_agents):
        if other_id != agent_id:
            if transition['bold_r_t'][other_id] == 5.0:
                other_stag = True
            elif transition['bold_r_t'][other_id] == 3.0:
                other_forage = True

    if other_stag and env_required_agents is not None:
        for stag_id, participants in stag_hunts.items():
            required = env_required_agents.get(stag_id, [])
            if agent_id in required and agent_id in participants:
                return 'stag_reward'

    if other_forage:
        return 'forage_reward'

    return 'no_reward'


def split_action_and_log_probs(
    actions: Dict[int, Tuple[int, float]],
) -> Tuple[Dict[int, int], Dict[int, float]]:
    actions_only = {agent_id: action for agent_id, (action, _) in actions.items()}
    log_probs = {agent_id: log_prob for agent_id, (_, log_prob) in actions.items()}
    return actions_only, log_probs


def build_transition_kwargs(
    *,
    prev_observations,
    prev_rewards,
    prev_actions,
    observations,
    actions,
    rewards,
    next_observations,
    dones,
    masks,
    hidden_states,
    log_probs,
    info=None,
    available_actions=None,
    env_id: Optional[int] = None,
) -> Dict:
    kwargs = {
        'prev_observations': prev_observations,
        'prev_rewards': prev_rewards,
        'prev_actions': prev_actions,
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'next_observations': next_observations,
        'dones': dones,
        'masks': masks,
        'hidden_states': hidden_states,
        'available_actions': available_actions,
        'log_probs': log_probs,
        'info': info,
    }
    if env_id is not None:
        kwargs['env_id'] = env_id
    return kwargs


def prepare_env_info(info: Optional[Dict], env_id: int) -> Dict:
    if info is None:
        info = {}
    info['env_id'] = env_id
    return info


def move_modules_to_device(device: torch.device, *modules) -> None:
    for module in modules:
        if module is not None:
            module.to(device)
