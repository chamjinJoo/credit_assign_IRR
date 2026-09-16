import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.torch.networks import NaivePredictor
from result_analysis.run_staghunt_policy_gradient_decomposition import (
    DEFAULT_ITERATIONS,
    discover_runs,
    load_model,
    make_env,
    policy_gradient_components,
    policy_probabilities,
    prepare_advantages,
    reset_policy,
    tensor_observations,
    vector_metrics,
)


DEFAULT_ROOTS = {
    "ippo": {
        "baseline": ROOT / "runs/checkpoints_for_sh/sh/ipci",
        "w_shuffle": ROOT / "runs/checkpoints_for_sh/sh_etc/ipci_w_shuffle/p_1.0",
        "prefix": "",
    },
    "mappo": {
        "baseline": ROOT / "runs/checkpoints_for_sh/sh/mpci",
        "w_shuffle": ROOT / "runs/checkpoints_for_sh/sh_etc/mpci_w_shuffle/p_1.0",
        "prefix": "staghunt_pci_mappo_",
    },
}
DEFAULT_RESULT_DIR = ROOT / "result_analysis/gradient_analysis"
METRICS = (
    "global_delta_cosine",
    "delta_to_global_norm_ratio",
    "combined_turn_angle_deg",
    "delta_orthogonal_to_global_ratio",
)
T_95_DF4 = 2.7764451051977987


def parse_args():
    parser = argparse.ArgumentParser(
        description="StagHunt PCI-IPPO/PCI-MAPPO gradient-geometry null controls."
    )
    parser.add_argument("--model", choices=("ippo", "mappo"), default="ippo")
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument("--w-shuffle-root", type=Path, default=None)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[DEFAULT_ITERATIONS[-1]],
        help="Regular checkpoint iterations to evaluate (default: final iteration only).",
    )
    parser.add_argument("--num-episodes", type=int, default=32)
    parser.add_argument("--rollout-seed", type=int, default=20260731)
    parser.add_argument("--random-draws", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    defaults = DEFAULT_ROOTS[args.model]
    if args.baseline_root is None:
        args.baseline_root = defaults["baseline"]
    if args.w_shuffle_root is None:
        args.w_shuffle_root = defaults["w_shuffle"]
    args.output_prefix = defaults["prefix"]
    return args


def result_path(args, name):
    return args.result_dir / f"{args.output_prefix}{name}"


def stable_seed(*parts):
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**63 - 1)


def load_naive_predictor(checkpoint_path, policy, metadata, config, device):
    naive_path = checkpoint_path.parent.parent / "checkpoints_naivepredictor" / checkpoint_path.name
    if not naive_path.is_file():
        return None, None

    network_config = config["config"]["network"]
    predictor = NaivePredictor(
        action_dim=policy.action_dim,
        num_agents=policy.num_agents,
        x_dim=int(metadata["bridge_x_dim"]),
        w_dim=policy.obs_dim * (policy.num_agents - 1),
        output_dim=1,
        hidden_dim=int(network_config["hidden_dim"]),
    ).to(device)
    checkpoint = torch.load(naive_path, map_location=device, weights_only=False)
    predictor.load_state_dict(checkpoint["naive_predictor_state"])
    predictor.eval()
    return predictor, naive_path


def select_previous_action_embedding(policy, bridge_x_dim):
    fixed_dim = (
        policy.obs_dim
        + (policy.num_agents - 1) * policy.recurrent_n * policy.hidden_dim
    )
    action_block_dim = int(bridge_x_dim) - fixed_dim
    one_hot_dim = (policy.num_agents - 1) * policy.action_dim
    embedding = policy.bridge_previous_action_embedding
    embedding_dim = (
        (policy.num_agents - 1) * int(embedding.embedding_dim)
        if embedding is not None
        else -1
    )
    if action_block_dim == one_hot_dim:
        return None, "one_hot"
    if embedding is not None and action_block_dim == embedding_dim:
        return embedding, f"embedding_dim_{embedding.embedding_dim}"
    raise ValueError(
        "Cannot match bridge X dimension to a previous-action representation: "
        f"x_dim={bridge_x_dim}, fixed_dim={fixed_dim}, action_block_dim={action_block_dim}, "
        f"one_hot_dim={one_hot_dim}, embedding_dim={embedding_dim}"
    )


def predictor_outputs(
    policy,
    predictor,
    observations,
    hidden_states,
    previous_actions,
    previous_action_embedding,
    w_observations=None,
):
    predictions = {}
    action_range = torch.arange(policy.action_dim, device=observations[0].device)
    with torch.no_grad():
        for focal_agent in range(policy.num_agents):
            other_agents = [
                agent_id
                for agent_id in range(policy.num_agents)
                if agent_id != focal_agent
            ]
            w_source = observations if w_observations is None else w_observations
            w = torch.cat([w_source[agent_id] for agent_id in other_agents])
            other_hidden = torch.cat(
                [hidden_states[agent_id].reshape(-1) for agent_id in other_agents]
            )
            if previous_action_embedding is None:
                other_previous_actions = torch.cat(
                    [previous_actions[agent_id] for agent_id in other_agents]
                )
            else:
                indices = torch.tensor(
                    [
                        int(torch.argmax(previous_actions[agent_id]).item())
                        for agent_id in other_agents
                    ],
                    device=observations[0].device,
                )
                other_previous_actions = previous_action_embedding(indices).reshape(-1)
            x = torch.cat(
                [observations[focal_agent], other_hidden, other_previous_actions]
            )
            predictions[focal_agent] = predictor(
                action_range,
                w.unsqueeze(0).expand(policy.action_dim, -1),
                x=x.unsqueeze(0).expand(policy.action_dim, -1),
            ).reshape(policy.action_dim, -1).sum(dim=-1)
    return predictions


def baselines_from_predictions(policy, predictions, actions, action_probabilities):
    baselines = {}
    for focal_agent in range(policy.num_agents):
        probabilities = action_probabilities[focal_agent].clone()
        probabilities[int(actions[focal_agent])] = 0.0
        probabilities = probabilities / probabilities.sum().clamp_min(1e-10)
        baselines[focal_agent] = float(
            torch.sum(probabilities * predictions[focal_agent]).item()
        )
    return baselines


def collect_control_episodes(
    policy,
    predictors,
    env,
    num_episodes,
    rollout_seed,
    device,
):
    episodes = {name: [] for name in predictors}
    disagreement_numerator = []
    disagreement_denominator = []

    for episode_id in range(num_episodes):
        seed = rollout_seed + episode_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)
        observations_raw = env.reset()
        reset_policy(policy, device)
        episode_steps = {name: [] for name in predictors}
        done = False

        while not done and len(episode_steps["true_bridge"]) < env.max_steps:
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

            prediction_sets = {}
            for name, (predictor, previous_action_embedding) in predictors.items():
                prediction_sets[name] = predictor_outputs(
                    policy,
                    predictor,
                    observations,
                    hidden_states,
                    previous_actions,
                    previous_action_embedding,
                )

            if "naive_predictor" in prediction_sets:
                for agent_id in range(policy.num_agents):
                    true_values = prediction_sets["true_bridge"][agent_id]
                    naive_values = prediction_sets["naive_predictor"][agent_id]
                    disagreement_numerator.append(
                        torch.abs(true_values - naive_values).detach().cpu()
                    )
                    disagreement_denominator.append(torch.abs(naive_values).detach().cpu())

            next_observations, rewards, dones, info = env.step(actions)
            global_reward = float(sum(rewards.values()))
            for name, predictions in prediction_sets.items():
                baselines = baselines_from_predictions(
                    policy, predictions, actions, probabilities
                )
                deltas = {
                    agent_id: global_reward - baselines[agent_id]
                    for agent_id in range(policy.num_agents)
                }
                episode_steps[name].append(
                    {
                        "observations": {
                            agent_id: observations[agent_id].detach().cpu()
                            for agent_id in range(policy.num_agents)
                        },
                        "previous_actions": {
                            agent_id: previous_actions[agent_id].detach().cpu()
                            for agent_id in range(policy.num_agents)
                        },
                        "hidden_states": {
                            agent_id: hidden_states[agent_id].detach().cpu()
                            for agent_id in range(policy.num_agents)
                        },
                        "action_probabilities": {
                            agent_id: probabilities[agent_id].detach().cpu()
                            for agent_id in range(policy.num_agents)
                        },
                        "actions": dict(actions),
                        "global_reward": global_reward,
                        "baseline": baselines,
                        "delta": deltas,
                    }
                )
            observations_raw = next_observations
            done = all(bool(value) for value in dones.values())

        success = float(
            not any(info.get("stag_alive", {}).values())
            and not any(info.get("hare_alive", {}).values())
        )
        for name in predictors:
            episodes[name].append({"steps": episode_steps[name], "success": success})

    disagreement = None
    if disagreement_numerator:
        numerator = torch.cat(disagreement_numerator).mean().item()
        denominator = torch.cat(disagreement_denominator).mean().item()
        disagreement = float(numerator / max(denominator, 1e-12))
    return episodes, disagreement


def build_eval_w_shuffle_episodes(
    policy,
    bridge,
    previous_action_embedding,
    source_episodes,
    training_seed,
    iteration,
    shuffle_seed,
    device,
):
    output = [{"steps": [], "success": episode["success"]} for episode in source_episodes]
    total_recipients = 0
    shuffled_recipients = 0
    skipped_recipients = 0
    fixed_points = 0
    max_timestep = max(len(episode["steps"]) for episode in source_episodes)

    for timestep in range(max_timestep):
        pool = [
            episode_id
            for episode_id, episode in enumerate(source_episodes)
            if timestep < len(episode["steps"])
        ]
        total_recipients += len(pool) * policy.num_agents
        if len(pool) < 2:
            donor_ids = list(pool)
            skipped_recipients += len(pool) * policy.num_agents
        else:
            generator = np.random.default_rng(
                stable_seed(
                    shuffle_seed,
                    training_seed,
                    iteration,
                    timestep,
                    "eval_w",
                )
            )
            shift = int(generator.integers(1, len(pool)))
            donor_ids = list(np.roll(np.asarray(pool), shift))
            shuffled_recipients += len(pool) * policy.num_agents

        for recipient_id, donor_id in zip(pool, donor_ids):
            if recipient_id == donor_id and len(pool) >= 2:
                fixed_points += policy.num_agents
            recipient = source_episodes[recipient_id]["steps"][timestep]
            donor = source_episodes[donor_id]["steps"][timestep]
            observations = tensor_observations(recipient["observations"], device)
            donor_observations = tensor_observations(donor["observations"], device)
            hidden_states = {
                agent_id: value.to(device)
                for agent_id, value in recipient["hidden_states"].items()
            }
            previous_actions = {
                agent_id: value.to(device)
                for agent_id, value in recipient["previous_actions"].items()
            }
            action_probabilities = {
                agent_id: value.to(device)
                for agent_id, value in recipient["action_probabilities"].items()
            }
            predictions = predictor_outputs(
                policy,
                bridge,
                observations,
                hidden_states,
                previous_actions,
                previous_action_embedding,
                w_observations=donor_observations,
            )
            baselines = baselines_from_predictions(
                policy,
                predictions,
                recipient["actions"],
                action_probabilities,
            )
            global_reward = float(recipient["global_reward"])
            output[recipient_id]["steps"].append(
                {
                    "observations": recipient["observations"],
                    "previous_actions": recipient["previous_actions"],
                    "actions": recipient["actions"],
                    "global_reward": global_reward,
                    "baseline": baselines,
                    "delta": {
                        agent_id: global_reward - baselines[agent_id]
                        for agent_id in range(policy.num_agents)
                    },
                }
            )

    if fixed_points != 0:
        raise AssertionError(f"Eval-time W shuffle produced {fixed_points} fixed points")
    for original, shuffled in zip(source_episodes, output):
        if len(original["steps"]) != len(shuffled["steps"]):
            raise AssertionError("Eval-time W shuffle changed episode length")
        for original_step, shuffled_step in zip(original["steps"], shuffled["steps"]):
            if original_step["actions"] != shuffled_step["actions"]:
                raise AssertionError("Eval-time W shuffle changed recipient actions")
            if original_step["global_reward"] != shuffled_step["global_reward"]:
                raise AssertionError("Eval-time W shuffle changed recipient rewards")

    diagnostics = {
        "total_recipients": total_recipients,
        "shuffled_recipients": shuffled_recipients,
        "skipped_recipients": skipped_recipients,
        "actual_fraction": shuffled_recipients / max(total_recipients, 1),
        "skipped_fraction": skipped_recipients / max(total_recipients, 1),
        "fixed_point_fraction": fixed_points / max(shuffled_recipients, 1),
    }
    return output, diagnostics


def condition_gradients(policy, episodes, advantages, device):
    by_agent = {}
    residuals = []
    for agent_id in range(policy.num_agents):
        values = policy_gradient_components(
            policy, episodes, advantages, agent_id, device
        )
        global_grad, delta_grad, combined_grad, residual = values
        by_agent[agent_id] = (global_grad, delta_grad, combined_grad)
        residuals.append(residual)
    joint = tuple(
        torch.cat([by_agent[agent_id][index] for agent_id in range(policy.num_agents)])
        for index in range(3)
    )
    return by_agent, joint, max(residuals)


def row_from_vectors(condition, seed, iteration, agent_id, vectors, **extra):
    global_grad, delta_grad, combined_grad = vectors
    return {
        "condition": condition,
        "seed": seed,
        "iteration": iteration,
        "agent_id": agent_id,
        "draw": -1,
        **vector_metrics(global_grad, delta_grad, combined_grad),
        **extra,
    }


def matched_random(delta_grad, sizes, generator, per_tensor):
    if not per_tensor:
        draw = torch.randn(delta_grad.shape, generator=generator)
        return draw * (
            torch.linalg.vector_norm(delta_grad)
            / torch.linalg.vector_norm(draw).clamp_min(1e-12)
        )

    pieces = []
    offset = 0
    for size in sizes:
        target = delta_grad[offset : offset + size]
        draw = torch.randn((size,), generator=generator)
        target_norm = torch.linalg.vector_norm(target)
        if target_norm.item() == 0.0:
            draw.zero_()
        else:
            draw *= target_norm / torch.linalg.vector_norm(draw).clamp_min(1e-12)
        pieces.append(draw)
        offset += size
    if offset != delta_grad.numel():
        raise ValueError("Parameter sizes do not cover the flattened gradient")
    return torch.cat(pieces)


def random_control_rows(seed, iteration, policy, true_by_agent, draws):
    rows = []
    sizes_by_agent = {
        agent_id: [
            parameter.numel()
            for parameter in policy.agents[agent_id].policy_net.parameters()
        ]
        for agent_id in range(policy.num_agents)
    }
    for condition, per_tensor in (
        ("random_isotropic", False),
        ("random_parameter_tensor", True),
    ):
        for draw_index in range(draws):
            joint_global = []
            joint_random = []
            for agent_id in range(policy.num_agents):
                global_grad, delta_grad, _ = true_by_agent[agent_id]
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    stable_seed(20260731, seed, iteration, agent_id, draw_index, condition)
                )
                random_grad = matched_random(
                    delta_grad,
                    sizes_by_agent[agent_id],
                    generator,
                    per_tensor,
                )
                metrics = vector_metrics(
                    global_grad, random_grad, global_grad + random_grad
                )
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "iteration": iteration,
                        "agent_id": agent_id,
                        "draw": draw_index,
                        **metrics,
                    }
                )
                joint_global.append(global_grad)
                joint_random.append(random_grad)
            global_grad = torch.cat(joint_global)
            random_grad = torch.cat(joint_random)
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "iteration": iteration,
                    "agent_id": -1,
                    "draw": draw_index,
                    **vector_metrics(
                        global_grad, random_grad, global_grad + random_grad
                    ),
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


def read_control_rows(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("seed", "iteration", "agent_id", "draw"):
            row[key] = int(row[key])
        for key in METRICS:
            row[key] = float(row[key])
    return rows


def aggregate_random_checkpoints(rows):
    selected = [
        row
        for row in rows
        if row["condition"] in {"random_isotropic", "random_parameter_tensor"}
    ]
    groups = sorted(
        {
            (
                row["condition"],
                int(row["seed"]),
                int(row["iteration"]),
                int(row["agent_id"]),
            )
            for row in selected
        }
    )
    summary = []
    for condition, seed, iteration, agent_id in groups:
        group = [
            row
            for row in selected
            if row["condition"] == condition
            and int(row["seed"]) == seed
            and int(row["iteration"]) == iteration
            and int(row["agent_id"]) == agent_id
        ]
        result = {
            "condition": condition,
            "seed": seed,
            "iteration": iteration,
            "agent_id": agent_id,
            "n_draws": len(group),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summary.append(result)
    return summary


def aggregate(rows):
    joint_rows = [row for row in rows if row["agent_id"] == -1]
    seed_rows = []
    conditions = sorted({row["condition"] for row in joint_rows})
    for condition in conditions:
        selected = [row for row in joint_rows if row["condition"] == condition]
        for seed in sorted({int(row["seed"]) for row in selected}):
            seed_selected = [row for row in selected if int(row["seed"]) == seed]
            seed_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "n_rows": len(seed_selected),
                    **{
                        metric: float(
                            np.mean([float(row[metric]) for row in seed_selected])
                        )
                        for metric in METRICS
                    },
                }
            )

    summary = []
    for condition in conditions:
        selected = [row for row in seed_rows if row["condition"] == condition]
        result = {
            "condition": condition,
            "n_seeds": len(selected),
            "n_joint_rows": sum(int(row["n_rows"]) for row in selected),
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half_width = (
                T_95_DF4 * std / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            result[f"{metric}_mean"] = mean
            result[f"{metric}_95ci_low"] = mean - half_width
            result[f"{metric}_95ci_high"] = mean + half_width
        summary.append(result)
    return seed_rows, summary


def separation_statistics(seed_rows):
    lookup = {
        (row["condition"], int(row["seed"])): row
        for row in seed_rows
    }
    true_seeds = sorted(
        seed for condition, seed in lookup if condition == "true_bridge"
    )
    results = {}
    for control in (
        "random_isotropic",
        "random_parameter_tensor",
        "eval_w_shuffle",
        "w_shuffle_bridge",
        "naive_predictor",
    ):
        paired = [
            lookup[("true_bridge", seed)]["global_delta_cosine"]
            - lookup[(control, seed)]["global_delta_cosine"]
            for seed in true_seeds
            if (control, seed) in lookup
        ]
        if paired:
            values = np.asarray(paired, dtype=np.float64)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half_width = (
                T_95_DF4 * std / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            results[control] = {
                "rho_paired_difference_mean": mean,
                "rho_paired_difference_std": std,
                "rho_paired_difference_95ci_low": mean - half_width,
                "rho_paired_difference_95ci_high": mean + half_width,
            }

    return results


def load_existing_true_rows(args):
    path = result_path(args, "checkpoint_gradient_components.csv")
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (int(row["seed"]), int(row["iteration"])): row
        for row in rows
        if int(row["agent_id"]) == -1
    }


def main():
    args = parse_args()
    if args.random_draws < 20:
        raise ValueError("At least 20 random draws are required")
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        rows = read_control_rows(result_path(args, "gradient_control_components.csv"))
        seed_rows, summary = aggregate(rows)
        separation = separation_statistics(seed_rows)
        write_csv(result_path(args, "gradient_control_seed_summary.csv"), seed_rows)
        write_csv(result_path(args, "gradient_control_summary.csv"), summary)
        write_csv(
            result_path(args, "gradient_random_control_checkpoint_summary.csv"),
            aggregate_random_checkpoints(rows),
        )
        metadata_path = result_path(args, "gradient_control_metadata.json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            metadata["separation"] = separation
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Control summaries refreshed in {args.result_dir}", flush=True)
        return

    baseline_runs = discover_runs(args.baseline_root, args.limit_seeds)
    w_runs = {
        seed: (env_dir, config_path)
        for seed, env_dir, config_path in discover_runs(
            args.w_shuffle_root, args.limit_seeds
        )
    }
    existing_true = load_existing_true_rows(args)

    all_rows = []
    checkpoint_rows = []
    max_additivity_error = 0.0
    max_true_reproduction_error = 0.0
    missing_controls = []

    for seed, env_dir, config_path in baseline_runs:
        config = json.loads(config_path.read_text())
        for iteration in args.iterations:
            checkpoint_path = env_dir / "checkpoints" / f"checkpoint_{iteration}.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            env = make_env(config)
            policy, true_bridge, metadata = load_model(
                checkpoint_path, env, config, device
            )
            true_embedding, true_representation = select_previous_action_embedding(
                policy, metadata["bridge_x_dim"]
            )
            predictors = {
                "true_bridge": (
                    true_bridge,
                    true_embedding,
                )
            }
            paths = {"true_bridge": checkpoint_path}

            naive, naive_path = load_naive_predictor(
                checkpoint_path, policy, metadata, config, device
            )
            if naive is not None:
                predictors["naive_predictor"] = (
                    naive,
                    true_embedding,
                )
                paths["naive_predictor"] = naive_path
            else:
                missing_controls.append((seed, iteration, "naive_predictor"))

            w_run = w_runs.get(seed)
            if w_run is not None:
                w_path = w_run[0] / "checkpoints" / f"checkpoint_{iteration}.pt"
                if w_path.is_file():
                    w_policy, w_bridge, w_metadata = load_model(
                        w_path, env, config, device
                    )
                    w_embedding, w_representation = select_previous_action_embedding(
                        w_policy, w_metadata["bridge_x_dim"]
                    )
                    predictors["w_shuffle_bridge"] = (
                        w_bridge,
                        w_embedding,
                    )
                    paths["w_shuffle_bridge"] = w_path
                else:
                    missing_controls.append((seed, iteration, "w_shuffle_bridge"))
            else:
                missing_controls.append((seed, iteration, "w_shuffle_bridge"))

            episodes_by_condition, disagreement = collect_control_episodes(
                policy,
                predictors,
                env,
                args.num_episodes,
                args.rollout_seed,
                device,
            )
            eval_w_episodes, eval_w_diagnostics = build_eval_w_shuffle_episodes(
                policy,
                true_bridge,
                true_embedding,
                episodes_by_condition["true_bridge"],
                training_seed=seed,
                iteration=iteration,
                shuffle_seed=args.rollout_seed,
                device=device,
            )
            episodes_by_condition["eval_w_shuffle"] = eval_w_episodes
            paths["eval_w_shuffle"] = checkpoint_path

            true_by_agent = None
            for condition, episodes in episodes_by_condition.items():
                advantages = prepare_advantages(
                    episodes,
                    policy.num_agents,
                    metadata["gamma"],
                    metadata["gae_lambda"],
                    metadata["causal_lambda"],
                )
                by_agent, joint, residual = condition_gradients(
                    policy, episodes, advantages, device
                )
                max_additivity_error = max(max_additivity_error, residual)
                extra = {
                    "checkpoint_path": str(paths[condition].resolve()),
                    "gradient_additivity_relative_error": residual,
                    "naive_relative_disagreement": disagreement
                    if condition in {"true_bridge", "naive_predictor"}
                    else None,
                }
                for agent_id, vectors in by_agent.items():
                    all_rows.append(
                        row_from_vectors(
                            condition,
                            seed,
                            iteration,
                            agent_id,
                            vectors,
                            **extra,
                        )
                    )
                all_rows.append(
                    row_from_vectors(
                        condition,
                        seed,
                        iteration,
                        -1,
                        joint,
                        **extra,
                    )
                )
                if condition == "true_bridge":
                    true_by_agent = by_agent
                    expected = existing_true.get((seed, iteration))
                    if expected is not None:
                        actual = vector_metrics(*joint)
                        max_true_reproduction_error = max(
                            max_true_reproduction_error,
                            max(
                                abs(actual[metric] - float(expected[metric]))
                                for metric in METRICS
                            ),
                        )

            if true_by_agent is None:
                raise RuntimeError("True bridge gradients were not computed")
            all_rows.extend(
                random_control_rows(
                    seed,
                    iteration,
                    policy,
                    true_by_agent,
                    args.random_draws,
                )
            )
            checkpoint_rows.append(
                {
                    "seed": seed,
                    "iteration": iteration,
                    "controls": sorted(episodes_by_condition),
                    "naive_relative_disagreement": disagreement,
                    "eval_w_shuffle": eval_w_diagnostics,
                    "true_previous_action_representation": true_representation,
                    "w_previous_action_representation": w_representation
                    if "w_shuffle_bridge" in predictors
                    else None,
                }
            )
            write_csv(result_path(args, "gradient_control_components.csv"), all_rows)
            print(
                f"seed={seed:>2} iteration={iteration:>4} "
                f"controls={','.join(sorted(episodes_by_condition))} "
                f"disagreement={disagreement if disagreement is not None else float('nan'):.3f}",
                flush=True,
            )

    seed_rows, summary = aggregate(all_rows)
    separation = separation_statistics(seed_rows)
    write_csv(result_path(args, "gradient_control_seed_summary.csv"), seed_rows)
    write_csv(result_path(args, "gradient_control_summary.csv"), summary)
    write_csv(
        result_path(args, "gradient_random_control_checkpoint_summary.csv"),
        aggregate_random_checkpoints(all_rows),
    )
    metadata = {
        "model": args.model,
        "baseline_root": str(args.baseline_root.resolve()),
        "w_shuffle_root": str(args.w_shuffle_root.resolve()),
        "iterations": args.iterations,
        "excluded_checkpoint_kinds": ["best"],
        "num_episodes": args.num_episodes,
        "rollout_seed": args.rollout_seed,
        "random_draws_per_checkpoint_seed_agent": args.random_draws,
        "random_nulls": {
            "random_isotropic": "flattened actor-parameter isotropic Gaussian, globally norm matched",
            "random_parameter_tensor": "independent Gaussian in each actor parameter tensor, each tensor norm matched",
        },
        "max_gradient_additivity_relative_error": max_additivity_error,
        "max_true_metric_reproduction_absolute_error": max_true_reproduction_error,
        "missing_controls": missing_controls,
        "separation": separation,
        "checkpoint_rows": checkpoint_rows,
        "device": str(device),
    }
    result_path(args, "gradient_control_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps({"summary": summary, "separation": separation}, indent=2))
    print(f"Results saved to {args.result_dir}", flush=True)


if __name__ == "__main__":
    main()
