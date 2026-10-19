import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class ProxyShuffleDiagnostics:
    requested: int = 0
    actual: int = 0
    skipped: int = 0
    fixed_points: int = 0
    donor_total: int = 0
    donor_groups: int = 0

    def update(self, other: "ProxyShuffleDiagnostics") -> None:
        self.requested += other.requested
        self.actual += other.actual
        self.skipped += other.skipped
        self.fixed_points += other.fixed_points
        self.donor_total += other.donor_total
        self.donor_groups += other.donor_groups

    def as_metrics(self) -> Dict[str, float]:
        denom = max(self.requested, 1)
        actual_denom = max(self.actual, 1)
        donor_denom = max(self.donor_groups, 1)
        return {
            "proxy_shuffle/actual_fraction": float(self.actual) / float(denom),
            "proxy_shuffle/skipped_fraction": float(self.skipped) / float(denom),
            "proxy_shuffle/num_valid_donors": float(self.donor_total) / float(donor_denom),
            "proxy_shuffle/fixed_point_fraction": float(self.fixed_points) / float(actual_denom),
        }


def assign_proxy_shuffle_metadata(trajectories: List[List[Dict]]) -> None:
    next_episode_id = 0
    for traj in trajectories:
        timestep = 0
        episode_id = next_episode_id
        for transition in traj:
            transition["_proxy_episode_id"] = episode_id
            transition["_proxy_timestep"] = timestep
            done = any(bool(v) for v in transition.get("dones", {}).values())
            if done:
                next_episode_id += 1
                episode_id = next_episode_id
                timestep = 0
            else:
                timestep += 1
        next_episode_id += 1


def _stable_seed(base_seed: int, training_step: int, timestep: int, proxy_type: str) -> int:
    payload = f"{int(base_seed)}:{int(training_step)}:{int(timestep)}:{proxy_type}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)


def _select_deranged_donors(
    row_pool: torch.Tensor,
    row_episode_ids: torch.Tensor,
    forbidden_donor_pool: Optional[torch.Tensor],
    generator: torch.Generator,
) -> Optional[torch.Tensor]:
    count = int(row_pool.numel())
    if count < 2:
        return None

    candidate_shifts = []
    for shift in range(1, count):
        donor_pool = torch.roll(row_pool, shifts=shift)
        no_fixed_episode = bool(torch.all(row_episode_ids[row_pool] != row_episode_ids[donor_pool]).item())
        if not no_fixed_episode:
            continue
        if forbidden_donor_pool is not None and torch.any(donor_pool == forbidden_donor_pool):
            continue
        candidate_shifts.append(shift)

    if not candidate_shifts:
        for shift in range(1, count):
            donor_pool = torch.roll(row_pool, shifts=shift)
            no_fixed_episode = bool(torch.all(row_episode_ids[row_pool] != row_episode_ids[donor_pool]).item())
            if no_fixed_episode:
                candidate_shifts.append(shift)

    if not candidate_shifts:
        return None

    shift_idx = int(torch.randint(0, len(candidate_shifts), (1,), generator=generator).item())
    return torch.roll(row_pool, shifts=candidate_shifts[shift_idx])


def apply_episode_axis_proxy_shuffle(
    proxy: torch.Tensor,
    transition_indices: torch.Tensor,
    episode_ids: torch.Tensor,
    timesteps: torch.Tensor,
    valid_mask: torch.Tensor,
    num_agents: int,
    mode: str,
    prob: float,
    base_seed: int,
    training_step: int,
    proxy_type: str,
    strategy: str = "episode_axis",
    forbidden_donors: Optional[Dict[int, torch.Tensor]] = None,
) -> tuple[torch.Tensor, ProxyShuffleDiagnostics, Dict[int, torch.Tensor]]:
    if strategy != "episode_axis":
        raise ValueError(f"Unsupported proxy shuffle strategy: {strategy}")
    if mode not in {"none", "w", "z", "wz"}:
        raise ValueError(f"Unsupported proxy shuffle mode: {mode}")
    if prob < 0.0 or prob > 1.0:
        raise ValueError(f"proxy_shuffle_prob must be in [0, 1], got {prob}")
    if proxy_type not in {"w", "z"}:
        raise ValueError(f"proxy_type must be 'w' or 'z', got {proxy_type}")

    diagnostics = ProxyShuffleDiagnostics()
    donor_by_timestep: Dict[int, torch.Tensor] = {}
    if mode == "none" or proxy_type not in mode or prob <= 0.0:
        return proxy, diagnostics, donor_by_timestep

    if proxy.dim() != 2:
        raise ValueError(f"proxy must be [B*N, D], got shape {tuple(proxy.shape)}")
    if proxy.size(0) % num_agents != 0:
        raise ValueError(f"proxy first dimension must be divisible by num_agents={num_agents}")

    rows = proxy.size(0) // num_agents
    proxy_by_agent = proxy.reshape(rows, num_agents, -1)
    valid_by_agent = valid_mask.reshape(rows, num_agents).bool()

    row_transition_idx = transition_indices.reshape(rows, num_agents)[:, 0].detach().cpu()
    row_episode_ids = episode_ids[row_transition_idx].detach().cpu()
    row_timesteps = timesteps[row_transition_idx].detach().cpu()
    row_valid = valid_by_agent.detach().cpu().any(dim=1)

    out = proxy_by_agent.clone()
    unique_timesteps = torch.unique(row_timesteps[row_valid])

    for timestep_tensor in unique_timesteps:
        timestep = int(timestep_tensor.item())
        row_pool = torch.nonzero((row_timesteps == timestep) & row_valid, as_tuple=False).flatten()
        if row_pool.numel() == 0:
            continue

        valid_agents_cpu = valid_by_agent[row_pool].detach().cpu()
        diagnostics.requested += int(valid_agents_cpu.sum().item())

        if row_pool.numel() < 2:
            diagnostics.skipped += int(valid_agents_cpu.sum().item())
            continue

        seed = _stable_seed(base_seed, training_step, timestep, proxy_type)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        forbidden_donor_pool = None
        if forbidden_donors is not None:
            forbidden_donor_pool = forbidden_donors.get(timestep)
        donor_pool = _select_deranged_donors(
            row_pool,
            row_episode_ids,
            forbidden_donor_pool,
            generator,
        )
        if donor_pool is None:
            diagnostics.skipped += int(valid_agents_cpu.sum().item())
            continue

        donor_by_timestep[timestep] = donor_pool
        diagnostics.donor_total += int(row_pool.numel())
        diagnostics.donor_groups += 1

        corrupt_cpu = (torch.rand(valid_agents_cpu.shape, generator=generator) < prob) & valid_agents_cpu
        corrupt_count = int(corrupt_cpu.sum().item())
        if corrupt_count == 0:
            continue

        recipient_device = row_pool.to(proxy.device)
        donor_device = donor_pool.to(proxy.device)
        corrupt_device = corrupt_cpu.to(proxy.device)
        expanded_donor = proxy_by_agent[donor_device]
        out[recipient_device] = torch.where(corrupt_device.unsqueeze(-1), expanded_donor, out[recipient_device])

        fixed_points = (row_episode_ids[row_pool] == row_episode_ids[donor_pool]).unsqueeze(1) & corrupt_cpu
        diagnostics.fixed_points += int(fixed_points.sum().item())
        diagnostics.actual += corrupt_count

    return out.reshape_as(proxy), diagnostics, donor_by_timestep
