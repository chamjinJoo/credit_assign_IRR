import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from envs.multi_agent_env import StagHuntEnv
from models.torch.agent_common import RecurrentActorCriticAgent
from models.torch.networks import OutcomeBridgeNetwork


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_ROOT = ROOT / "runs/checkpoints_for_sh/sh/ipci"
DEFAULT_RESULT_DIR = ROOT / "runs/gradient_analysis/staghunt"
DEFAULT_ITERATIONS = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 3999]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure global and counterfactual policy-gradient components at StagHunt PCI-IPPO checkpoints."
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--iterations", type=int, nargs="+", default=DEFAULT_ITERATIONS)
    parser.add_argument("--num-episodes", type=int, default=32)
    parser.add_argument("--rollout-seed", type=int, default=20260731)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit-seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--baseline-mode",
        choices=("leave_one_out_renorm", "full_weighted"),
        default="leave_one_out_renorm",
        help="Counterfactual baseline used to reconstruct the auxiliary reward.",
    )
    return parser.parse_args()


def discover_runs(root, limit_seeds=None):
    runs = []
    allowed = set(limit_seeds) if limit_seeds else None
    for run_dir in sorted(root.glob("*_seed*")):
        match = re.search(r"_seed(\d+)$", run_dir.name)
        if match is None:
            continue
        seed = int(match.group(1))
        if allowed is not None and seed not in allowed:
            continue
        env_dir = run_dir / "staghuntenv"
        config_path = env_dir / "simulation_config.json"
        if config_path.is_file():
            runs.append((seed, env_dir, config_path))
    return sorted(runs)


def make_env(config):
    env_config = config["config"]["environment"]
    return StagHuntEnv(
        num_agents=int(env_config["num_agents"]),
        max_steps=int(env_config["max_steps"]),
        stag_follows=bool(env_config.get("stag_follows", False)),
        run_away_after_maul=bool(env_config.get("run_away_after_maul", False)),
        forage_reward=float(env_config["forage_reward"]),
        stag_reward=float(env_config["stag_reward"]),
        mauling_punishment=float(env_config["mauling_punishment"]),
        coop_bonus=float(env_config["coop_bonus"]),
        global_reward=bool(env_config.get("global_reward", False)),
        grid_observation=bool(env_config["grid_observation"]),
        obs_noise_std=float(env_config["obs_noise_std"]),
    )


def load_model(checkpoint_path, env, config, device):
    network_config = config["config"]["network"]
    algorithm_config = config["config"]["algorithm"]
    num_agents = int(config["simulation_info"]["num_agents"])
    obs_dim = int(config["simulation_info"]["obs_dim"])
    action_dim = int(config["simulation_info"]["action_dim"])
    hidden_dim = int(network_config["hidden_dim"])
    recurrent_n = int(network_config["recurrent_N"])

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["algorithm_state"]
    policy = SimpleNamespace(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        recurrent_n=recurrent_n,
        agents={},
    )
    for agent_id in range(num_agents):
        agent = RecurrentActorCriticAgent(
            agent_id=agent_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            recurrent_N=recurrent_n,
            device=device,
        )
        saved = state["agents"].get(agent_id, state["agents"].get(str(agent_id)))
        agent.policy_net.load_state_dict(saved["policy_net"])
        agent.policy_net.eval()
        policy.agents[agent_id] = agent

    w_dim = obs_dim * (num_agents - 1)
    saved_bridge_input_dim = int(
        state["outcome_bridge_net"]["mlp.mlp.fc1.linear.weight"].shape[1]
    )
    x_dim = saved_bridge_input_dim - action_dim - w_dim
    bridge = OutcomeBridgeNetwork(
        action_dim=action_dim,
        num_agents=num_agents,
        w_dim=w_dim,
        outcome_dim=1,
        hidden_dim=hidden_dim,
        action_embedding_dim=8,
        x_dim=x_dim,
    ).to(device)
    bridge.load_state_dict(state["outcome_bridge_net"])
    bridge.eval()

    one_hot_x_dim = (
        obs_dim
        + (num_agents - 1) * hidden_dim
        + (num_agents - 1) * action_dim
    )
    embedding = None
    if "prev_action_embedding" in state:
        embedding_weight = state["prev_action_embedding"]["weight"]
        embedding = torch.nn.Embedding(
            int(embedding_weight.shape[0]),
            int(embedding_weight.shape[1]),
        ).to(device)
        embedding.load_state_dict(state["prev_action_embedding"])
        embedding.eval()

    embedding_x_dim = (
        obs_dim + (num_agents - 1) * hidden_dim
        + (num_agents - 1) * int(embedding.embedding_dim)
        if embedding is not None else None
    )
    if x_dim == one_hot_x_dim:
        policy.bridge_previous_action_embedding = None
        previous_action_representation = "one_hot"
    elif embedding is not None and x_dim == embedding_x_dim:
        policy.bridge_previous_action_embedding = embedding
        previous_action_representation = f"embedding_dim_{embedding.embedding_dim}"
    else:
        raise ValueError(
            "Bridge input dimension does not match one-hot or saved-embedding "
            f"previous actions: x_dim={x_dim}, one_hot={one_hot_x_dim}, "
            f"embedding={embedding_x_dim}"
        )

    metadata = {
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "training_total_steps": int(checkpoint["total_steps"]),
        "causal_lambda": float(algorithm_config["causal_lambda"]),
        "gamma": float(algorithm_config["gamma"]),
        "gae_lambda": float(algorithm_config["gae_lambda"]),
        "delta_critic_saved": "delta_value_net" in state,
        "bridge_x_dim": x_dim,
        "bridge_previous_action_representation": previous_action_representation,
    }
    return policy, bridge, metadata


def reset_policy(policy, device):
    for agent in policy.agents.values():
        agent.reset_rnn_state(device=device)


def tensor_observations(observations, device):
    return {
        agent_id: torch.as_tensor(value, dtype=torch.float32, device=device)
        for agent_id, value in observations.items()
    }


def policy_probabilities(policy, observations, hidden_states, previous_actions):
    probabilities = {}
    for agent_id in range(policy.num_agents):
        logits, _ = policy.agents[agent_id].policy_net(
            observations[agent_id].unsqueeze(0),
            hidden_states[agent_id].unsqueeze(0),
            torch.ones(1, 1, device=observations[agent_id].device),
            prev_actions=previous_actions[agent_id].unsqueeze(0),
        )
        probabilities[agent_id] = torch.softmax(logits, dim=-1).squeeze(0)
    return probabilities


def counterfactual_baselines(
    policy,
    bridge,
    observations,
    actions,
    action_probabilities,
    hidden_states,
    previous_actions,
    baseline_mode,
):
    baselines = {}
    with torch.no_grad():
        action_range = torch.arange(policy.action_dim, device=observations[0].device)
        for focal_agent in range(policy.num_agents):
            other_agents = [agent_id for agent_id in range(policy.num_agents) if agent_id != focal_agent]
            w = torch.cat([observations[agent_id] for agent_id in other_agents])
            other_hidden = torch.cat([hidden_states[agent_id].reshape(-1) for agent_id in other_agents])
            if policy.bridge_previous_action_embedding is None:
                other_previous_actions = torch.cat(
                    [previous_actions[agent_id] for agent_id in other_agents]
                )
            else:
                previous_action_indices = torch.tensor(
                    [
                        int(torch.argmax(previous_actions[agent_id]).item())
                        for agent_id in other_agents
                    ],
                    device=observations[0].device,
                )
                other_previous_actions = policy.bridge_previous_action_embedding(
                    previous_action_indices
                ).reshape(-1)
            x = torch.cat([observations[focal_agent], other_hidden, other_previous_actions])

            predictions = bridge(
                action_range,
                w.unsqueeze(0).expand(policy.action_dim, -1),
                x=x.unsqueeze(0).expand(policy.action_dim, -1),
            ).reshape(policy.action_dim, -1).sum(dim=-1)

            probabilities = action_probabilities[focal_agent].clone()
            if baseline_mode == "leave_one_out_renorm":
                probabilities[int(actions[focal_agent])] = 0.0
                probabilities = probabilities / probabilities.sum().clamp_min(1e-10)
            elif baseline_mode != "full_weighted":
                raise ValueError(f"Unsupported baseline mode: {baseline_mode}")
            baselines[focal_agent] = float(torch.sum(probabilities * predictions).item())
    return baselines


def collect_episodes(policy, bridge, env, num_episodes, rollout_seed, device, baseline_mode):
    episodes = []
    for episode_id in range(num_episodes):
        seed = rollout_seed + episode_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)
        observations_raw = env.reset()
        reset_policy(policy, device)
        episode = []
        done = False

        while not done and len(episode) < env.max_steps:
            observations = tensor_observations(observations_raw, device)
            hidden_states = {
                agent_id: policy.agents[agent_id].rnn_state_policy.detach().clone()
                for agent_id in range(policy.num_agents)
            }
            previous_actions = {
                agent_id: policy.agents[agent_id].prev_action.detach().clone()
                for agent_id in range(policy.num_agents)
            }
            with torch.no_grad():
                probabilities = policy_probabilities(
                    policy, observations, hidden_states, previous_actions
                )

            actions = {}
            for agent_id in range(policy.num_agents):
                action, _ = policy.agents[agent_id].get_action(
                    observations[agent_id],
                    torch.ones(1, device=device),
                    return_rnn_state=False,
                )
                actions[agent_id] = int(action)

            baselines = counterfactual_baselines(
                policy,
                bridge,
                observations,
                actions,
                probabilities,
                hidden_states,
                previous_actions,
                baseline_mode,
            )
            next_observations, rewards, dones, info = env.step(actions)
            global_reward = float(sum(rewards.values()))
            deltas = {
                agent_id: global_reward - baselines[agent_id]
                for agent_id in range(policy.num_agents)
            }
            episode.append(
                {
                    "observations": {
                        agent_id: observations[agent_id].detach().cpu()
                        for agent_id in range(policy.num_agents)
                    },
                    "previous_actions": {
                        agent_id: previous_actions[agent_id].detach().cpu()
                        for agent_id in range(policy.num_agents)
                    },
                    "actions": dict(actions),
                    "global_reward": global_reward,
                    "baseline": dict(baselines),
                    "delta": deltas,
                }
            )
            observations_raw = next_observations
            done = all(bool(value) for value in dones.values())

        success = float(
            not any(info.get("stag_alive", {}).values())
            and not any(info.get("hare_alive", {}).values())
        )
        episodes.append({"steps": episode, "success": success})
    return episodes


def critic_free_gae(rewards, gamma, gae_lambda):
    advantages = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    discount = gamma * gae_lambda
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + discount * running
        advantages[index] = running
    return advantages


def normalize_population(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.zeros_like(values)
    std = max(float(values.std(ddof=0)), 1e-8)
    return (values - float(values.mean())) / std


def prepare_advantages(episodes, num_agents, gamma, gae_lambda, causal_lambda):
    by_agent = {
        agent_id: {"global": [], "delta": [], "combined": []}
        for agent_id in range(num_agents)
    }
    for episode in episodes:
        steps = episode["steps"]
        global_rewards = [step["global_reward"] for step in steps]
        global_gae = critic_free_gae(global_rewards, gamma, gae_lambda)
        global_normalized = normalize_population(global_gae)
        for agent_id in range(num_agents):
            delta_rewards = [step["delta"][agent_id] for step in steps]
            delta_gae = critic_free_gae(delta_rewards, gamma, gae_lambda)
            delta_normalized = normalize_population(delta_gae)
            combined = global_normalized + causal_lambda * delta_normalized
            by_agent[agent_id]["global"].append(global_normalized)
            by_agent[agent_id]["delta"].append(delta_normalized)
            by_agent[agent_id]["combined"].append(combined)

    for agent_id in range(num_agents):
        global_values = np.concatenate(by_agent[agent_id]["global"])
        delta_values = np.concatenate(by_agent[agent_id]["delta"])
        combined_values = np.concatenate(by_agent[agent_id]["combined"])
        combined_std = max(float(combined_values.std(ddof=1)), 1e-8)
        by_agent[agent_id] = {
            "global": (global_values - global_values.mean()) / combined_std,
            "delta": causal_lambda * (delta_values - delta_values.mean()) / combined_std,
            "combined": (combined_values - combined_values.mean()) / combined_std,
            "global_unscaled": global_values,
            "delta_unscaled": delta_values,
        }
    return by_agent


def flatten_gradients(gradients, parameters):
    pieces = []
    for gradient, parameter in zip(gradients, parameters):
        if gradient is None:
            pieces.append(torch.zeros_like(parameter).reshape(-1))
        else:
            pieces.append(gradient.detach().reshape(-1))
    return torch.cat(pieces)


def policy_gradient_components(policy, episodes, advantages, agent_id, device):
    log_probs = []
    for episode in episodes:
        steps = episode["steps"]
        observations = torch.stack(
            [step["observations"][agent_id] for step in steps]
        ).to(device)
        previous_actions = torch.stack(
            [step["previous_actions"][agent_id] for step in steps]
        ).to(device)
        actions = torch.tensor(
            [step["actions"][agent_id] for step in steps],
            dtype=torch.long,
            device=device,
        )
        masks = torch.ones(len(steps), 1, device=device)
        initial_hidden = torch.zeros(
            1, policy.recurrent_n, policy.hidden_dim, device=device
        )
        episode_log_probs, _ = policy.agents[agent_id].policy_net.evaluate_actions(
            observations,
            initial_hidden,
            actions,
            masks,
            prev_actions=previous_actions,
        )
        log_probs.append(episode_log_probs)

    log_probs = torch.cat(log_probs)
    global_weights = torch.as_tensor(
        advantages[agent_id]["global"], dtype=torch.float32, device=device
    )
    delta_weights = torch.as_tensor(
        advantages[agent_id]["delta"], dtype=torch.float32, device=device
    )
    combined_weights = torch.as_tensor(
        advantages[agent_id]["combined"], dtype=torch.float32, device=device
    )
    parameters = [parameter for parameter in policy.agents[agent_id].policy_net.parameters()]

    global_objective = torch.mean(log_probs * global_weights)
    delta_objective = torch.mean(log_probs * delta_weights)
    combined_objective = torch.mean(log_probs * combined_weights)
    global_grad = flatten_gradients(
        torch.autograd.grad(global_objective, parameters, retain_graph=True),
        parameters,
    )
    delta_grad = flatten_gradients(
        torch.autograd.grad(delta_objective, parameters, retain_graph=True),
        parameters,
    )
    combined_direct = flatten_gradients(
        torch.autograd.grad(combined_objective, parameters),
        parameters,
    )
    combined_sum = global_grad + delta_grad
    residual = float(
        torch.linalg.vector_norm(combined_direct - combined_sum).item()
        / max(torch.linalg.vector_norm(combined_direct).item(), 1e-12)
    )
    return global_grad.cpu(), delta_grad.cpu(), combined_direct.cpu(), residual


def vector_metrics(global_grad, delta_grad, combined_grad, causal_lambda):
    eps = 1e-12
    global_norm = float(torch.linalg.vector_norm(global_grad).item())
    delta_norm = float(torch.linalg.vector_norm(delta_grad).item())
    combined_norm = float(torch.linalg.vector_norm(combined_grad).item())
    bias_free_norm = (1.0 + float(causal_lambda)) * global_norm
    dot_global_delta = float(torch.dot(global_grad, delta_grad).item())
    cosine = dot_global_delta / max(global_norm * delta_norm, eps)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    combined_cosine = float(
        np.clip(
            torch.dot(global_grad, combined_grad).item()
            / max(global_norm * combined_norm, eps),
            -1.0,
            1.0,
        )
    )
    parallel_ratio = dot_global_delta / max(global_norm * global_norm, eps)
    parallel_norm = dot_global_delta / max(global_norm, eps)
    orthogonal_norm = math.sqrt(max(delta_norm * delta_norm - parallel_norm * parallel_norm, 0.0))
    return {
        "global_grad_norm": global_norm,
        "delta_grad_norm": delta_norm,
        "combined_grad_norm": combined_norm,
        "bias_free_grad_norm": bias_free_norm,
        "gradient_gain": combined_norm / max(bias_free_norm, eps),
        "delta_to_global_norm_ratio": delta_norm / max(global_norm, eps),
        "delta_to_combined_norm_ratio": delta_norm / max(combined_norm, eps),
        "global_delta_cosine": cosine,
        "combined_global_cosine": combined_cosine,
        "combined_turn_angle_deg": math.degrees(math.acos(combined_cosine)),
        "delta_parallel_to_global_ratio": parallel_ratio,
        "delta_orthogonal_to_global_ratio": orthogonal_norm / max(global_norm, eps),
        "combined_global_projection_ratio": 1.0 + parallel_ratio,
    }


def advantage_metrics(advantages, agent_id):
    global_values = advantages[agent_id]["global_unscaled"]
    delta_values = advantages[agent_id]["delta_unscaled"]
    combined_values = global_values + delta_values
    correlation = float(np.corrcoef(global_values, delta_values)[0, 1])
    if not np.isfinite(correlation):
        correlation = 0.0
    nonzero = (np.abs(global_values) > 1e-8) | (np.abs(combined_values) > 1e-8)
    sign_flip = float(
        np.mean(np.sign(global_values[nonzero]) != np.sign(combined_values[nonzero]))
    ) if np.any(nonzero) else 0.0
    return {
        "advantage_global_delta_correlation": correlation,
        "advantage_sign_flip_fraction": sign_flip,
    }


def summarize_checkpoint(
    seed,
    iteration,
    checkpoint_path,
    metadata,
    episodes,
    advantages,
    policy,
    device,
):
    rows = []
    joint_global = []
    joint_delta = []
    joint_combined = []
    residuals = []
    for agent_id in range(policy.num_agents):
        global_grad, delta_grad, combined_grad, residual = policy_gradient_components(
            policy, episodes, advantages, agent_id, device
        )
        row = {
            "seed": seed,
            "iteration": iteration,
            "agent_id": agent_id,
            "checkpoint_path": str(checkpoint_path.resolve()),
            **metadata,
            **vector_metrics(
                global_grad,
                delta_grad,
                combined_grad,
                metadata["causal_lambda"],
            ),
            **advantage_metrics(advantages, agent_id),
            "gradient_additivity_relative_error": residual,
        }
        rows.append(row)
        joint_global.append(global_grad)
        joint_delta.append(delta_grad)
        joint_combined.append(combined_grad)
        residuals.append(residual)

    global_grad = torch.cat(joint_global)
    delta_grad = torch.cat(joint_delta)
    combined_grad = torch.cat(joint_combined)
    total_steps = sum(len(episode["steps"]) for episode in episodes)
    all_rewards = [
        step["global_reward"] for episode in episodes for step in episode["steps"]
    ]
    all_baselines = [
        step["baseline"][agent_id]
        for episode in episodes
        for step in episode["steps"]
        for agent_id in range(policy.num_agents)
    ]
    all_deltas = [
        step["delta"][agent_id]
        for episode in episodes
        for step in episode["steps"]
        for agent_id in range(policy.num_agents)
    ]
    joint_adv_correlations = [
        advantage_metrics(advantages, agent_id)["advantage_global_delta_correlation"]
        for agent_id in range(policy.num_agents)
    ]
    joint_sign_flips = [
        advantage_metrics(advantages, agent_id)["advantage_sign_flip_fraction"]
        for agent_id in range(policy.num_agents)
    ]
    rows.append(
        {
            "seed": seed,
            "iteration": iteration,
            "agent_id": -1,
            "checkpoint_path": str(checkpoint_path.resolve()),
            **metadata,
            **vector_metrics(
                global_grad,
                delta_grad,
                combined_grad,
                metadata["causal_lambda"],
            ),
            "advantage_global_delta_correlation": float(np.mean(joint_adv_correlations)),
            "advantage_sign_flip_fraction": float(np.mean(joint_sign_flips)),
            "gradient_additivity_relative_error": float(max(residuals)),
            "num_episodes": len(episodes),
            "num_transitions": total_steps,
            "mean_episode_return": float(
                np.mean(
                    [
                        sum(step["global_reward"] for step in episode["steps"])
                        for episode in episodes
                    ]
                )
            ),
            "success_rate": float(np.mean([episode["success"] for episode in episodes])),
            "global_reward_mean": float(np.mean(all_rewards)),
            "counterfactual_baseline_mean": float(np.mean(all_baselines)),
            "counterfactual_baseline_std": float(np.std(all_baselines)),
            "delta_reward_mean": float(np.mean(all_deltas)),
            "delta_reward_std": float(np.std(all_deltas)),
        }
    )
    return rows


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    joint_rows = [row for row in rows if row["agent_id"] == -1]
    metrics = [
        "global_grad_norm",
        "delta_grad_norm",
        "combined_grad_norm",
        "bias_free_grad_norm",
        "gradient_gain",
        "delta_to_global_norm_ratio",
        "delta_to_combined_norm_ratio",
        "global_delta_cosine",
        "combined_global_cosine",
        "combined_turn_angle_deg",
        "delta_parallel_to_global_ratio",
        "delta_orthogonal_to_global_ratio",
        "combined_global_projection_ratio",
        "advantage_global_delta_correlation",
        "advantage_sign_flip_fraction",
        "mean_episode_return",
        "success_rate",
        "counterfactual_baseline_mean",
        "counterfactual_baseline_std",
        "delta_reward_mean",
        "delta_reward_std",
    ]
    summary = []
    for iteration in sorted({int(row["iteration"]) for row in joint_rows}):
        selected = [row for row in joint_rows if int(row["iteration"]) == iteration]
        result = {"iteration": iteration, "n_seeds": len(selected)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(result)
    return summary


def plot_summary(path, summary):
    iterations = np.asarray([row["iteration"] for row in summary])
    panels = [
        ("global_delta_cosine", "Gradient cosine", (-1.05, 1.05)),
        ("delta_to_global_norm_ratio", "Delta / global gradient norm", None),
        ("combined_turn_angle_deg", "Turn from global direction (degrees)", None),
        ("advantage_sign_flip_fraction", "Advantage sign-flip fraction", (0.0, 1.0)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for axis, (metric, ylabel, ylim) in zip(axes.flat, panels):
        mean = np.asarray([row[f"{metric}_mean"] for row in summary])
        std = np.asarray([row[f"{metric}_std"] for row in summary])
        axis.plot(iterations, mean, color="#202020", marker="o", linewidth=1.8)
        axis.fill_between(iterations, mean - std, mean + std, color="#4C78A8", alpha=0.22)
        axis.set_xlabel("Checkpoint iteration")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, linewidth=0.6)
        if ylim is not None:
            axis.set_ylim(*ylim)
    fig.suptitle("StagHunt PCI-IPPO: global and counterfactual policy-gradient decomposition")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fmt(mean, std, digits=3):
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def write_report(path, summary, args, runs, all_rows):
    joint_rows = [row for row in all_rows if row["agent_id"] == -1]
    max_additivity_error = max(float(row["gradient_additivity_relative_error"]) for row in joint_rows)
    delta_critic_saved = all(bool(row["delta_critic_saved"]) for row in joint_rows)
    lines = [
        "# StagHunt PCI-IPPO policy-gradient decomposition",
        "",
        "## Scope",
        "",
        f"- Checkpoints: `{args.checkpoint_root}`",
        f"- Training seeds: {', '.join(str(seed) for seed, _, _ in runs)}",
        f"- Checkpoint iterations: {', '.join(str(value) for value in args.iterations)}",
        f"- On-policy stochastic episodes per checkpoint: {args.num_episodes}",
        f"- Common rollout seed base: {args.rollout_seed}",
        "- Gradient target: first PPO epoch at the loaded policy, where behavior and current policies coincide.",
        "- Advantage construction: critic-free GAE with the training gamma and GAE lambda, followed by the same per-episode component normalization and final combined normalization used by the training path.",
        "",
        "## Reproducibility limitation",
        "",
        "Historical checkpoints do not contain `delta_value_net` state. Therefore the exact finite-batch delta GAE used during the original optimizer step cannot be reconstructed. "
        "This analysis applies the same critic-free estimator to global rewards and PCI delta rewards, isolating the policy-gradient direction induced by the saved policy and outcome bridge without introducing a newly fitted critic.",
        f"The checkpoint delta-critic state was available: `{delta_critic_saved}`.",
        "",
        "## Results",
        "",
        "| Iteration | Cos(global, delta) | Delta/global norm | Turn angle | Orthogonal/global | Sign flips | Return | Success |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["iteration"]),
                    fmt(row["global_delta_cosine_mean"], row["global_delta_cosine_std"]),
                    fmt(row["delta_to_global_norm_ratio_mean"], row["delta_to_global_norm_ratio_std"]),
                    fmt(row["combined_turn_angle_deg_mean"], row["combined_turn_angle_deg_std"], 1),
                    fmt(
                        row["delta_orthogonal_to_global_ratio_mean"],
                        row["delta_orthogonal_to_global_ratio_std"],
                    ),
                    fmt(
                        row["advantage_sign_flip_fraction_mean"],
                        row["advantage_sign_flip_fraction_std"],
                    ),
                    fmt(row["mean_episode_return_mean"], row["mean_episode_return_std"], 2),
                    fmt(row["success_rate_mean"], row["success_rate_std"]),
                ]
            )
            + " |"
        )

    mean_cosine = float(np.mean([row["global_delta_cosine_mean"] for row in summary]))
    mean_ratio = float(np.mean([row["delta_to_global_norm_ratio_mean"] for row in summary]))
    mean_turn = float(np.mean([row["combined_turn_angle_deg_mean"] for row in summary]))
    mean_orthogonal = float(
        np.mean([row["delta_orthogonal_to_global_ratio_mean"] for row in summary])
    )
    mean_parallel = float(
        np.mean([row["delta_parallel_to_global_ratio_mean"] for row in summary])
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Across checkpoint stages, the mean global-delta gradient cosine was {mean_cosine:.3f}. "
            "Positive values indicate reinforcement of the global direction; values near zero indicate a largely transverse correction; negative values indicate local conflict.",
            f"The delta component had {mean_ratio:.3f} times the global gradient norm on average. "
            f"Its mean parallel contribution was {mean_parallel:.3f} times the global norm, while its orthogonal contribution was {mean_orthogonal:.3f} times the global norm.",
            f"Adding the counterfactual component turned the joint policy gradient by {mean_turn:.1f} degrees on average. "
            "This angle directly quantifies how strongly PCI moves the update away from the global-only optimization stream.",
            "",
            "The scalar PPO policy loss is not used as the influence measure because normalized advantages make its value close to zero at probability ratio one. "
            "Gradient norm, cosine, parallel projection, orthogonal projection, and turn angle measure the actual parameter-space effect.",
            "",
            "## Validation",
            "",
            f"- Maximum relative error between the directly differentiated combined gradient and the sum of its differentiated global and delta terms: `{max_additivity_error:.3e}`.",
            "- All checkpoint evaluations used fresh stochastic on-policy rollouts and did not update model parameters.",
            "- Raw checkpoint-agent rows are in `checkpoint_gradient_components.csv`; seed summaries are in `checkpoint_gradient_summary.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def validate_configs(runs):
    signatures = []
    for seed, _, config_path in runs:
        config = json.loads(config_path.read_text())
        algorithm = config["config"]["algorithm"]
        environment = config["config"]["environment"]
        signatures.append(
            (
                seed,
                algorithm["type"],
                bool(algorithm["use_centralized_critic"]),
                float(algorithm["causal_lambda"]),
                float(algorithm["gamma"]),
                float(algorithm["gae_lambda"]),
                int(environment["num_agents"]),
                int(environment["max_steps"]),
                float(environment["obs_noise_std"]),
            )
        )
    reference = signatures[0][1:]
    mismatches = [signature for signature in signatures if signature[1:] != reference]
    if mismatches:
        raise ValueError(f"Run configurations differ: {signatures}")


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    args.result_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.checkpoint_root, args.limit_seeds)
    if not runs:
        raise FileNotFoundError(f"No runs found under {args.checkpoint_root}")
    validate_configs(runs)

    all_rows = []
    for seed, env_dir, config_path in runs:
        config = json.loads(config_path.read_text())
        for iteration in args.iterations:
            checkpoint_path = env_dir / "checkpoints" / f"checkpoint_{iteration}.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            env = make_env(config)
            policy, bridge, metadata = load_model(
                checkpoint_path, env, config, device
            )
            episodes = collect_episodes(
                policy,
                bridge,
                env,
                args.num_episodes,
                args.rollout_seed,
                device,
                args.baseline_mode,
            )
            advantages = prepare_advantages(
                episodes,
                policy.num_agents,
                metadata["gamma"],
                metadata["gae_lambda"],
                metadata["causal_lambda"],
            )
            rows = summarize_checkpoint(
                seed,
                iteration,
                checkpoint_path,
                metadata,
                episodes,
                advantages,
                policy,
                device,
            )
            all_rows.extend(rows)
            write_csv(args.result_dir / "checkpoint_gradient_components.csv", all_rows)
            joint = rows[-1]
            print(
                f"seed={seed:>2} iteration={iteration:>4} "
                f"cos={joint['global_delta_cosine']:+.3f} "
                f"ratio={joint['delta_to_global_norm_ratio']:.3f} "
                f"turn={joint['combined_turn_angle_deg']:.1f}deg "
                f"return={joint['mean_episode_return']:.2f}",
                flush=True,
            )

    summary = aggregate_rows(all_rows)
    write_csv(args.result_dir / "checkpoint_gradient_summary.csv", summary)
    plot_summary(args.result_dir / "gradient_decomposition.png", summary)
    metadata = {
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "result_dir": str(args.result_dir.resolve()),
        "iterations": args.iterations,
        "num_episodes": args.num_episodes,
        "rollout_seed": args.rollout_seed,
        "device": str(device),
        "baseline_mode": args.baseline_mode,
        "training_seeds": [seed for seed, _, _ in runs],
    }
    (args.result_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Results saved to {args.result_dir}", flush=True)


if __name__ == "__main__":
    main()
