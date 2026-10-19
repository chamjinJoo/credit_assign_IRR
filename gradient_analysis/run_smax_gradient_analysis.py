from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.jax.common import compute_gae
from gradient_analysis.smax_checkpoint_utils import (
    build_trainer_for_run,
    discover_runs,
)


DEFAULT_RESULT_DIR = ROOT / "runs/gradient_analysis/smax"
METRICS = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure global and counterfactual policy-gradient components at SMAX PCI checkpoints."
    )
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--checkpoint-choices", nargs="+", choices=("best", "final"), default=("best", "final"))
    parser.add_argument("--rollout-seed", type=int, default=20260731)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--baseline-mode",
        choices=("leave_one_out_renorm", "full_weighted"),
        default="leave_one_out_renorm",
        help="Counterfactual baseline used to reconstruct the auxiliary reward.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validate_runs(runs, limit_seeds: Iterable[int] | None):
    if runs.empty:
        raise FileNotFoundError("No SMAX runs were discovered")
    if limit_seeds:
        runs = runs[runs["seed"].isin(set(limit_seeds))]
    runs = runs.sort_values("seed").reset_index(drop=True)
    if runs.empty:
        raise FileNotFoundError("No SMAX runs remain after applying --limit-seeds")

    signatures = {
        (
            str(row.algorithm_type),
            str(row.map_name),
            str(row.constraint_key),
            int(row.config.algorithm.causal_lambda > 0.0),
            bool(row.config.algorithm.use_centralized_critic),
            float(row.config.algorithm.gamma),
            float(row.config.algorithm.gae_lambda),
            int(row.config.training.max_steps_per_iteration),
            int(row.config.training.n_rollout_threads),
        )
        for row in runs.itertuples()
    }
    if len(signatures) != 1:
        raise ValueError(f"Run configurations differ: {sorted(signatures)}")
    algorithm_type = str(runs.iloc[0]["algorithm_type"])
    if algorithm_type not in {"jax_pci_ippo", "jax_pci_mappo"}:
        raise ValueError(f"Expected a JAX PCI-IPPO/MAPPO root, got {algorithm_type}")
    return runs


def reset_rollout_state(trainer: Any, rollout_seed: int) -> None:
    base_rng = jax.random.PRNGKey(int(rollout_seed))
    trainer._env_rng = jax.random.fold_in(base_rng, 0)
    trainer._action_rng = jax.random.fold_in(base_rng, 1)
    (
        trainer._env_rng,
        trainer._batch_env_state,
        trainer._current_obs,
        trainer._current_avail,
    ) = trainer.env.reset_jax_pure(trainer._env_rng)
    trainer._runner_state = trainer.algorithm.initialize_runner_state(trainer.rollout_envs)
    trainer._running_episode_rewards = np.zeros(
        (trainer.rollout_envs, trainer.env.num_agents), dtype=np.float32
    )
    trainer._running_episode_lengths = np.zeros((trainer.rollout_envs,), dtype=np.int32)


def collect_training_rollout(trainer: Any, rollout_seed: int) -> Dict[str, jnp.ndarray]:
    reset_rollout_state(trainer, rollout_seed)
    trainer.collect_rollouts()
    rollout = dict(trainer.algorithm.latest_rollout)
    jax.block_until_ready(rollout["actions"])
    return rollout


def actor_advantage_components(
    algorithm: Any,
    rollout: Dict[str, jnp.ndarray],
    baseline_mode: str,
) -> tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    baseline_type = {
        "leave_one_out_renorm": "weighted",
        "full_weighted": "full_weighted",
    }[baseline_mode]
    c_matrix, causal_info = algorithm.compute_causal_rewards(
        rollout, baseline_type=baseline_type
    )
    delta = c_matrix.sum(axis=-1)

    raw_team_rewards = rollout["rewards"].sum(axis=-1) / algorithm.num_agents
    if algorithm.global_reward:
        if algorithm.use_centralized_critic:
            env_rewards = raw_team_rewards
        else:
            env_rewards = jnp.repeat(raw_team_rewards[..., None], algorithm.num_agents, axis=-1)
    else:
        env_rewards = rollout["rewards"]

    env_advantages, _ = compute_gae(
        env_rewards,
        rollout["values"],
        rollout["last_value"],
        rollout["dones"],
        algorithm.gamma,
        algorithm.gae_lambda,
    )
    if algorithm.delta_critic_state is not None:
        pci_advantages, _ = compute_gae(
            delta,
            rollout["delta_values"],
            rollout["last_delta_value"],
            rollout["dones"],
            algorithm.gamma,
            algorithm.gae_lambda,
        )
    else:
        pci_advantages, _ = compute_gae(
            delta,
            jnp.zeros_like(delta),
            jnp.zeros_like(delta[0]),
            rollout["dones"],
            algorithm.gamma,
            algorithm.gae_lambda,
        )

    env_for_actor = algorithm._env_advantages_for_actor(env_advantages)
    env_flat = env_for_actor.reshape(-1, algorithm.num_agents)
    pci_flat = pci_advantages.reshape(-1, algorithm.num_agents)
    env_normalized = algorithm._normalize_actor_advantages(env_flat)
    pci_normalized = algorithm._normalize_actor_advantages(pci_flat)
    causal_lambda = algorithm.get_current_causal_lambda()
    combined_raw = env_normalized + causal_lambda * pci_normalized

    combined_mean = combined_raw.mean(axis=0, keepdims=True)
    centered = combined_raw - combined_mean
    denom = max(int(combined_raw.shape[0]) - 1, 1)
    combined_std = jnp.sqrt(
        jnp.clip(jnp.sum(jnp.square(centered), axis=0, keepdims=True) / denom, a_min=1.0e-8)
    )
    global_weights = (env_normalized - env_normalized.mean(axis=0, keepdims=True)) / combined_std
    delta_weights = causal_lambda * (
        pci_normalized - pci_normalized.mean(axis=0, keepdims=True)
    ) / combined_std
    combined_weights = algorithm._normalize_actor_advantages(combined_raw)

    components = {
        "global": global_weights,
        "delta": delta_weights,
        "combined": combined_weights,
        "global_unscaled": env_normalized,
        "delta_unscaled": pci_normalized,
    }
    diagnostics = {
        "delta": delta,
        "raw_team_rewards": raw_team_rewards,
        "causal_info": causal_info,
        "c_matrix": c_matrix,
    }
    return components, diagnostics


def flatten_agent_jacobian(jacobian: Any, num_agents: int) -> np.ndarray:
    leaves = jax.tree_util.tree_leaves(jacobian)
    return np.concatenate(
        [np.asarray(jax.device_get(leaf)).reshape(num_agents, -1) for leaf in leaves],
        axis=1,
    )


def policy_gradient_components(
    algorithm: Any,
    rollout: Dict[str, jnp.ndarray],
    advantages: Dict[str, jnp.ndarray],
) -> Dict[str, np.ndarray]:
    num_agents = algorithm.num_agents
    batch_size = rollout["actions"].shape[0] * rollout["actions"].shape[1]
    obs = rollout["obs"].reshape(batch_size * num_agents, algorithm.policy_obs_dim)
    prev_actions = rollout["prev_actions"].reshape(
        batch_size * num_agents, algorithm.action_dim
    )
    hidden = rollout["actor_hidden"].reshape(
        batch_size * num_agents, algorithm.actor_state_dim
    )
    masks = jnp.broadcast_to(
        rollout["masks"][..., None], rollout["actions"].shape
    ).reshape(batch_size, num_agents)
    flat_masks = masks.reshape(-1)
    avail_actions = rollout["avail_actions"].reshape(
        batch_size * num_agents, algorithm.action_dim
    )
    actions = rollout["actions"].reshape(batch_size * num_agents)
    denom = jnp.sum(masks, axis=0) + 1.0e-8

    def objectives(params, weights):
        _, logits = algorithm.actor_state.apply_fn(
            {"params": params}, obs, prev_actions, hidden, flat_masks
        )
        masked_logits = jnp.where(avail_actions > 0, logits, -1.0e9)
        log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
        selected = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze(-1)
        selected = selected.reshape(batch_size, num_agents)
        return jnp.sum(selected * weights * masks, axis=0) / denom

    jacobian_fn = jax.jacrev(objectives)
    result = {}
    for name in ("global", "delta", "combined"):
        jacobian = jacobian_fn(algorithm.actor_state.params, advantages[name])
        result[name] = flatten_agent_jacobian(jacobian, num_agents)
    return result


def vector_metrics(
    global_grad: np.ndarray,
    delta_grad: np.ndarray,
    combined_grad: np.ndarray,
    causal_lambda: float,
) -> Dict[str, float]:
    eps = 1.0e-12
    global_norm = float(np.linalg.norm(global_grad))
    delta_norm = float(np.linalg.norm(delta_grad))
    combined_norm = float(np.linalg.norm(combined_grad))
    bias_free_norm = (1.0 + float(causal_lambda)) * global_norm
    cosine = float(np.clip(np.dot(global_grad, delta_grad) / max(global_norm * delta_norm, eps), -1.0, 1.0))
    combined_cosine = float(
        np.clip(np.dot(global_grad, combined_grad) / max(global_norm * combined_norm, eps), -1.0, 1.0)
    )
    parallel_ratio = float(np.dot(delta_grad, global_grad) / max(global_norm * global_norm, eps))
    parallel = parallel_ratio * global_grad
    orthogonal_ratio = float(np.linalg.norm(delta_grad - parallel) / max(global_norm, eps))
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
        "delta_orthogonal_to_global_ratio": orthogonal_ratio,
        "combined_global_projection_ratio": 1.0 + parallel_ratio,
    }


def advantage_metrics(advantages: Dict[str, jnp.ndarray], agent_id: int | None) -> Dict[str, float]:
    if agent_id is None:
        global_values = np.asarray(jax.device_get(advantages["global"])).reshape(-1)
        delta_values = np.asarray(jax.device_get(advantages["delta"])).reshape(-1)
    else:
        global_values = np.asarray(jax.device_get(advantages["global"][:, agent_id]))
        delta_values = np.asarray(jax.device_get(advantages["delta"][:, agent_id]))
    combined_values = global_values + delta_values
    correlation = 0.0
    if global_values.std() > 1.0e-12 and delta_values.std() > 1.0e-12:
        correlation = float(np.corrcoef(global_values, delta_values)[0, 1])
    nonzero = (np.abs(global_values) > 1.0e-8) | (np.abs(combined_values) > 1.0e-8)
    sign_flip = float(
        np.mean(np.sign(global_values[nonzero]) != np.sign(combined_values[nonzero]))
    ) if np.any(nonzero) else 0.0
    return {
        "advantage_global_delta_correlation": correlation,
        "advantage_sign_flip_fraction": sign_flip,
    }


def episode_metrics(rollout: Dict[str, jnp.ndarray], num_agents: int) -> Dict[str, float]:
    rewards = np.asarray(jax.device_get(rollout["rewards"])).sum(axis=-1) / num_agents
    dones = np.asarray(jax.device_get(rollout["dones"])).astype(bool)
    wins = np.asarray(jax.device_get(rollout["battle_won"]))
    running = np.zeros(rewards.shape[1], dtype=np.float64)
    returns: list[float] = []
    successes: list[float] = []
    for timestep in range(rewards.shape[0]):
        running += rewards[timestep]
        for env_id in np.flatnonzero(dones[timestep]):
            returns.append(float(running[env_id]))
            successes.append(float(wins[timestep, env_id] > 0.5))
            running[env_id] = 0.0
    return {
        "num_episodes": len(returns),
        "num_transitions": int(rewards.size),
        "mean_episode_return": float(np.mean(returns)) if returns else float("nan"),
        "success_rate": float(np.mean(successes)) if successes else float("nan"),
    }


def summarize_checkpoint(
    run_row: Dict[str, Any],
    checkpoint_choice: str,
    checkpoint_path: Path,
    trainer: Any,
    rollout: Dict[str, jnp.ndarray],
    baseline_mode: str,
) -> list[Dict[str, Any]]:
    algorithm = trainer.algorithm
    advantages, diagnostics = actor_advantage_components(
        algorithm, rollout, baseline_mode
    )
    gradients = policy_gradient_components(algorithm, rollout, advantages)
    episodes = episode_metrics(rollout, algorithm.num_agents)
    delta_values = np.asarray(jax.device_get(diagnostics["delta"]))
    alive = ~np.asarray(jax.device_get(rollout["agent_dones"])).astype(bool)
    team_rewards = np.asarray(jax.device_get(diagnostics["raw_team_rewards"]))
    repeated_rewards = np.repeat(team_rewards[..., None], algorithm.num_agents, axis=-1)
    baseline_values = repeated_rewards[alive] - delta_values[alive]

    common = {
        "seed": int(run_row["seed"]),
        "checkpoint_label": checkpoint_choice,
        "iteration": int(trainer.iteration),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "training_total_steps": int(trainer.total_steps),
        "algorithm_type": str(run_row["algorithm_type"]),
        "map_name": str(run_row["map_name"]),
        "constraint_key": str(run_row["constraint_key"]),
        "causal_lambda": float(algorithm.get_current_causal_lambda()),
        "gamma": float(algorithm.gamma),
        "gae_lambda": float(algorithm.gae_lambda),
        "delta_critic_saved": bool(algorithm.delta_critic_state is not None),
        "shared_actor_parameters": True,
        **episodes,
        "global_reward_mean": float(team_rewards.mean()),
        "counterfactual_baseline_mean": float(baseline_values.mean()),
        "counterfactual_baseline_std": float(baseline_values.std()),
        "delta_reward_mean": float(delta_values[alive].mean()),
        "delta_reward_std": float(delta_values[alive].std()),
    }

    rows: list[Dict[str, Any]] = []
    residuals: list[float] = []
    for agent_id in range(algorithm.num_agents):
        global_grad = gradients["global"][agent_id]
        delta_grad = gradients["delta"][agent_id]
        combined_grad = gradients["combined"][agent_id]
        residual = float(
            np.linalg.norm(combined_grad - (global_grad + delta_grad))
            / max(np.linalg.norm(combined_grad), 1.0e-12)
        )
        residuals.append(residual)
        rows.append(
            {
                **common,
                "agent_id": agent_id,
                **vector_metrics(
                    global_grad,
                    delta_grad,
                    combined_grad,
                    common["causal_lambda"],
                ),
                **advantage_metrics(advantages, agent_id),
                "gradient_additivity_relative_error": residual,
            }
        )

    global_joint = gradients["global"].mean(axis=0)
    delta_joint = gradients["delta"].mean(axis=0)
    combined_joint = gradients["combined"].mean(axis=0)
    joint_residual = float(
        np.linalg.norm(combined_joint - (global_joint + delta_joint))
        / max(np.linalg.norm(combined_joint), 1.0e-12)
    )
    rows.append(
        {
            **common,
            "agent_id": -1,
            **vector_metrics(
                global_joint,
                delta_joint,
                combined_joint,
                common["causal_lambda"],
            ),
            **advantage_metrics(advantages, None),
            "gradient_additivity_relative_error": max(joint_residual, max(residuals)),
        }
    )
    return rows


def aggregate_rows(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    joint_rows = [row for row in rows if int(row["agent_id"]) == -1]
    summary = []
    for checkpoint_label in ("best", "final"):
        selected = [row for row in joint_rows if row["checkpoint_label"] == checkpoint_label]
        if not selected:
            continue
        result: Dict[str, Any] = {"checkpoint_label": checkpoint_label, "n_seeds": len(selected)}
        iterations = np.asarray([float(row["iteration"]) for row in selected])
        result["checkpoint_iteration_mean"] = float(iterations.mean())
        result["checkpoint_iteration_std"] = float(iterations.std(ddof=1)) if len(iterations) > 1 else 0.0
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(result)
    return summary


def plot_summary(path: Path, summary: list[Dict[str, Any]], title: str) -> None:
    labels = [row["checkpoint_label"] for row in summary]
    x = np.arange(len(labels))
    panels = (
        ("global_delta_cosine", "Gradient cosine", (-1.05, 1.05)),
        ("delta_to_global_norm_ratio", "Delta / global gradient norm", None),
        ("combined_turn_angle_deg", "Turn from global direction (degrees)", None),
        ("advantage_sign_flip_fraction", "Advantage sign-flip fraction", (0.0, 1.0)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for axis, (metric, ylabel, ylim) in zip(axes.flat, panels):
        mean = np.asarray([row[f"{metric}_mean"] for row in summary])
        std = np.asarray([row[f"{metric}_std"] for row in summary])
        axis.errorbar(x, mean, yerr=std, color="#202020", marker="o", linewidth=1.8, capsize=4)
        axis.set_xticks(x, labels)
        axis.set_xlabel("Checkpoint")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, linewidth=0.6)
        if ylim is not None:
            axis.set_ylim(*ylim)
    fig.suptitle(title)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    runs = validate_runs(discover_runs(args.checkpoint_root, project_filter=None), args.limit_seeds)
    if args.dry_run:
        for run in runs.to_dict(orient="records"):
            checkpoints = {
                choice: str(run[f"{choice}_checkpoint"])
                for choice in args.checkpoint_choices
            }
            missing = [path for path in checkpoints.values() if not Path(path).is_file()]
            if missing:
                seed = int(run["seed"])
                raise FileNotFoundError(f"Missing checkpoints for seed {seed}: {missing}")
            print(
                json.dumps(
                    {
                        "seed": int(run["seed"]),
                        "algorithm_type": run["algorithm_type"],
                        "map_name": run["map_name"],
                        "constraint_key": run["constraint_key"],
                        "checkpoints": checkpoints,
                    },
                    sort_keys=True,
                )
            )
        return
    all_rows: list[Dict[str, Any]] = []

    for run in runs.to_dict(orient="records"):
        for checkpoint_choice in args.checkpoint_choices:
            trainer, checkpoint_path, _ = build_trainer_for_run(
                run,
                checkpoint_choice=checkpoint_choice,
                device=args.device,
                eval_episodes=1,
                eval_seed=args.rollout_seed,
                eval_batch_envs=int(run["config"].training.n_rollout_threads),
            )
            trainer.load_checkpoint(str(checkpoint_path), actor_only=False)
            rollout = collect_training_rollout(trainer, args.rollout_seed)
            rows = summarize_checkpoint(
                run,
                checkpoint_choice,
                checkpoint_path,
                trainer,
                rollout,
                args.baseline_mode,
            )
            all_rows.extend(rows)
            components_path = args.result_dir / f"{args.experiment_name}_checkpoint_gradient_components.csv"
            write_csv(components_path, all_rows)
            joint = rows[-1]
            print(
                f"seed={int(run['seed']):>2} checkpoint={checkpoint_choice:<5} "
                f"iter={int(trainer.iteration):>5} cos={joint['global_delta_cosine']:+.3f} "
                f"ratio={joint['delta_to_global_norm_ratio']:.3f} "
                f"turn={joint['combined_turn_angle_deg']:.1f}deg "
                f"episodes={joint['num_episodes']}",
                flush=True,
            )

    summary = aggregate_rows(all_rows)
    summary_path = args.result_dir / f"{args.experiment_name}_checkpoint_gradient_summary.csv"
    figure_path = args.result_dir / f"{args.experiment_name}_gradient_decomposition.png"
    metadata_path = args.result_dir / f"{args.experiment_name}_analysis_metadata.json"
    write_csv(summary_path, summary)
    plot_summary(figure_path, summary, args.experiment_name.replace("_", " "))
    max_error = max(
        float(row["gradient_additivity_relative_error"])
        for row in all_rows
        if int(row["agent_id"]) == -1
    )
    metadata = {
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "experiment_name": args.experiment_name,
        "checkpoint_choices": list(args.checkpoint_choices),
        "rollout_seed": args.rollout_seed,
        "device": args.device,
        "training_seeds": [int(seed) for seed in runs["seed"].tolist()],
        "gradient_target": "first PPO epoch at behavior=current policy",
        "counterfactual_baseline": args.baseline_mode,
        "shared_actor_parameters": True,
        "max_gradient_additivity_relative_error": max_error,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Results saved with prefix {args.experiment_name} in {args.result_dir}", flush=True)


if __name__ == "__main__":
    main()
