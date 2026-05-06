"""
Shared utilities for Stag-Hunt bridge analysis.

Handles checkpoint loading, environment creation, rollout collection,
and scenario classification.
"""

import sys
import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJ_ROOT)

from algorithms.agent_common import RecurrentPolicyAgent
from algorithms.networks import OutcomeBridgeNetwork
from envs.multi_agent_env import StagHuntEnv

# ── Dimension constants (inferred from final_checkpoint.pt) ───────────────────
NUM_AGENTS      = 3
OBS_DIM         = 175                                          # env.obs_dim
ACTION_DIM      = 5
HIDDEN_DIM      = 64
RECURRENT_N     = 1
W_DIM           = OBS_DIM * (NUM_AGENTS - 1)                  # 350
BRIDGE_HID_DIM  = HIDDEN_DIM * RECURRENT_N * (NUM_AGENTS - 1) # 128
BRIDGE_ACT_DIM  = ACTION_DIM * (NUM_AGENTS - 1)               # 10
X_DIM           = OBS_DIM + BRIDGE_HID_DIM + BRIDGE_ACT_DIM   # 313

ACTION_NAMES    = {0: "Stay", 1: "Up", 2: "Down", 3: "Left", 4: "Right"}
SCENARIO_ORDER  = ["Stag", "Hare", "Null"]

CKPT_BASE = os.path.join(PROJ_ROOT, "checkpoints_for_eval")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def get_checkpoint_paths(variant: str, model_type: str) -> List[str]:
    """
    variant   : 'sh3' | 'sh3_noise'
    model_type: 'ipci' | 'mpci'
    """
    pattern = os.path.join(
        CKPT_BASE, variant, model_type,
        "*", "staghuntenv", "checkpoints", "final_checkpoint.pt",
    )
    paths = []
    for path in sorted(glob.glob(pattern)):
        if not os.path.isfile(path):
            continue
        if os.path.getsize(path) <= 0:
            print(f"[WARN] Skipping empty checkpoint: {path}")
            continue
        paths.append(path)
    return paths


def load_bridge_and_agents(
    ckpt_path: str, device: str = "cpu"
) -> Tuple[OutcomeBridgeNetwork, Dict[int, RecurrentPolicyAgent]]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if os.path.getsize(ckpt_path) <= 0:
        raise ValueError(f"Checkpoint file is empty: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    except EOFError as exc:
        raise ValueError(f"Checkpoint file is truncated or empty: {ckpt_path}") from exc
    algo = ckpt["algorithm_state"]

    bridge_net = OutcomeBridgeNetwork(
        action_dim=ACTION_DIM,
        num_agents=NUM_AGENTS,
        w_dim=W_DIM,
        outcome_dim=1,
        hidden_dim=HIDDEN_DIM,
        action_embedding_dim=8,
        x_dim=X_DIM,
    )
    bridge_net.load_state_dict(algo["outcome_bridge_net"])
    bridge_net.eval().to(device)

    agents: Dict[int, RecurrentPolicyAgent] = {}
    for i in range(NUM_AGENTS):
        agent = RecurrentPolicyAgent(
            agent_id=i,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            recurrent_N=RECURRENT_N,
        )
        agent.policy_net.load_state_dict(algo["agents"][i]["policy_net"])
        agent.policy_net.eval()
        agents[i] = agent

    return bridge_net, agents


def make_random_bridge() -> OutcomeBridgeNetwork:
    """Return a randomly-initialised bridge with the same architecture."""
    bridge = OutcomeBridgeNetwork(
        action_dim=ACTION_DIM,
        num_agents=NUM_AGENTS,
        w_dim=W_DIM,
        outcome_dim=1,
        hidden_dim=HIDDEN_DIM,
        action_embedding_dim=8,
        x_dim=X_DIM,
    )
    return bridge.eval()


# ── Environment factory ───────────────────────────────────────────────────────

def make_env(obs_noise_std: float = 0.0) -> StagHuntEnv:
    return StagHuntEnv(
        num_agents=NUM_AGENTS,
        max_steps=50,
        grid_observation=True,
        forage_reward=3.0,
        stag_reward=5.0,
        mauling_punishment=0.0,
        coop_bonus=0.0,
        obs_noise_std=obs_noise_std,
    )


# ── Scenario classification ───────────────────────────────────────────────────

def classify_scenario(
    agent_id: int,
    rewards: Dict[int, float],
    stag_hunts: Dict,
    required_agents: Dict,
) -> str:
    """Returns 'Stag', 'Hare', or 'Null' from the focal-agent perspective."""
    other_stag  = any(rewards.get(j, 0.0) == 5.0 for j in range(NUM_AGENTS) if j != agent_id)
    other_forage = any(rewards.get(j, 0.0) == 3.0 for j in range(NUM_AGENTS) if j != agent_id)

    if other_stag:
        for stag_id, participants in stag_hunts.items():
            req = required_agents.get(int(stag_id), [])
            if agent_id in req and agent_id in participants:
                return "Stag"

    if other_forage:
        return "Hare"

    return "Null"


# ── Rollout data collection ───────────────────────────────────────────────────


def collect_rollout_data(
    agents: Dict[int, RecurrentPolicyAgent],
    env: StagHuntEnv,
    n_episodes: int = 200,
) -> List[Dict]:
    """
    Run n_episodes with the given policy agents and return a list of records.

    Each record (one per agent per timestep) contains:
        episode (int), agent_id (int), action (int), global_reward (float),
        scenario (str),
        obs (Tensor[OBS_DIM]),   -- o^i_t
        w   (Tensor[W_DIM]),     -- W^i_t = concat of others' observations
        x   (Tensor[X_DIM]),     -- X^i_t = (o^i_t, h^{-i}_{t-1}, a^{-i}_{t-1})
        z   (Tensor[HIDDEN_DIM]) -- Z^i_t = h^i_{t-1}
    """
    records: List[Dict] = []

    for ep in range(n_episodes):
        obs_dict = env.reset()
        for agent in agents.values():
            agent.reset_rnn_state()

        done = False
        while not done:
            # Observations as tensors
            obs_t = {
                i: torch.tensor(obs_dict[i], dtype=torch.float32)
                for i in range(NUM_AGENTS)
            }

            # Snapshot hidden states and prev-actions BEFORE the step
            h_before   = {i: agents[i].rnn_state_policy.clone() for i in range(NUM_AGENTS)}
            a_prev_oh  = {i: agents[i].prev_action.clone()       for i in range(NUM_AGENTS)}

            # Get policy probs for all actions (before RNN state update) and sample action
            actions: Dict[int, int] = {}
            policy_probs: Dict[int, np.ndarray] = {}
            for i in range(NUM_AGENTS):
                net = agents[i].policy_net
                with torch.no_grad():
                    logits, _ = net(
                        obs_t[i].unsqueeze(0),
                        h_before[i].unsqueeze(0),
                        torch.ones(1, 1),
                        prev_actions=a_prev_oh[i].unsqueeze(0),
                    )
                    policy_probs[i] = F.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
                act, _, _, _ = agents[i].get_action(obs_t[i], masks=torch.ones(1))
                actions[i] = act

            next_obs, rewards, dones, info = env.step(actions)
            done = all(dones.values())

            global_r   = sum(rewards.values()) / NUM_AGENTS
            stag_hunts = info.get("stag_hunts", {})

            for i in range(NUM_AGENTS):
                others = [j for j in range(NUM_AGENTS) if j != i]

                # W^i_t : other agents' current observations (outcome-side proxy)
                w_i = torch.cat([obs_t[j] for j in others], dim=0)

                # h^{-i}_{t-1} : others' pre-action hidden states (part of X)
                h_others = torch.cat([h_before[j].view(-1) for j in others], dim=0)

                # a^{-i}_{t-1} : others' prev-action one-hots (part of X)
                a_others = torch.cat([a_prev_oh[j] for j in others], dim=0)

                # X^i_t = (o^i_t, h^{-i}_{t-1}, a^{-i}_{t-1})
                x_i = torch.cat([obs_t[i], h_others, a_others], dim=0)

                # Z^i_t = h^i_{t-1}
                z_i = h_before[i].view(-1)

                scenario = classify_scenario(i, rewards, stag_hunts, env.required_agents)

                records.append({
                    "episode":       ep,
                    "agent_id":      i,
                    "action":        actions[i],
                    "global_reward": global_r,
                    "scenario":      scenario,
                    "obs":           obs_t[i].cpu(),
                    "w":             w_i.cpu(),
                    "x":             x_i.cpu(),
                    "z":             z_i.cpu(),
                    "policy_probs":  policy_probs[i],  # (ACTION_DIM,) for Eq.(13) baseline
                })

            obs_dict = next_obs

    return records


# ── Bridge evaluation helpers ─────────────────────────────────────────────────

def bridge_outputs_all_actions(
    bridge_net: OutcomeBridgeNetwork,
    records: List[Dict],
    device: str = "cpu",
) -> np.ndarray:
    """
    Evaluate bridge for every action a' ∈ {0,...,ACTION_DIM-1} at each record's context.

    Returns array of shape (N_records, ACTION_DIM).
    """
    ws = torch.stack([r["w"] for r in records]).to(device)  # (N, W_DIM)
    xs = torch.stack([r["x"] for r in records]).to(device)  # (N, X_DIM)
    N  = len(records)

    outputs = np.zeros((N, ACTION_DIM), dtype=np.float32)
    with torch.no_grad():
        for a in range(ACTION_DIM):
            a_tensor = torch.full((N,), a, dtype=torch.long, device=device)
            out = bridge_net(a_tensor, ws, x=xs).squeeze(-1).cpu().numpy()
            outputs[:, a] = out

    return outputs


def bridge_residuals(
    bridge_net: OutcomeBridgeNetwork,
    records: List[Dict],
    device: str = "cpu",
) -> np.ndarray:
    """
    Residual R - h_φ(W, A_factual, X) for each record.
    Returns array of shape (N_records,).
    """
    actions = torch.tensor([r["action"] for r in records], dtype=torch.long)
    ws      = torch.stack([r["w"] for r in records]).to(device)
    xs      = torch.stack([r["x"] for r in records]).to(device)
    rewards = np.array([r["global_reward"] for r in records], dtype=np.float32)

    with torch.no_grad():
        r_hat = bridge_net(actions.to(device), ws, x=xs).squeeze(-1).cpu().numpy()

    return rewards - r_hat
