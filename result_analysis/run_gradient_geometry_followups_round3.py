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

from result_analysis.run_gradient_geometry_followups import (
    DATA_DIR,
    EPS,
    FINAL_ITERATION,
    NUM_EPISODES,
    RESULT_DIR,
    ROLLOUT_SEED,
    SEEDS,
    mean_ci,
    read_csv,
    write_csv,
)
from result_analysis.run_gradient_geometry_followups_round2 import (
    MODEL_NAMES,
    correlation,
    distribution,
    logit_correction,
)
from result_analysis.run_staghunt_gradient_null_controls import (
    DEFAULT_ROOTS,
    baselines_from_predictions,
    build_eval_w_shuffle_episodes,
    condition_gradients,
    load_naive_predictor,
    predictor_outputs,
    select_previous_action_embedding,
)
from result_analysis.run_staghunt_policy_gradient_decomposition import (
    discover_runs,
    load_model,
    make_env,
    policy_probabilities,
    prepare_advantages,
    reset_policy,
    tensor_observations,
)


CONDITIONS = ("true_bridge", "naive_predictor")
COMPARISONS = (
    ("eval_w_shuffle", "eval_w_shuffle"),
    ("naive_predictor", "naive_predictor"),
)
TIMESTEP_BINS = (
    (0, 9, "00-09"),
    (10, 19, "10-19"),
    (20, 29, "20-29"),
    (30, 39, "30-39"),
    (40, 49, "40-49"),
)


def collect_round3_episodes(
    policy,
    predictors,
    env,
    num_episodes,
    rollout_seed,
    model,
    training_seed,
    device,
):
    episodes = {name: [] for name in predictors}
    sample_rows = []

    for episode_id in range(num_episodes):
        episode_seed = rollout_seed + episode_id
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        env.seed(episode_seed)
        observations_raw = env.reset()
        reset_policy(policy, device)
        episode_steps = {name: [] for name in predictors}
        episode_samples = []
        event_steps = []
        stag_event_steps = []
        done = False

        while not done and len(episode_steps["true_bridge"]) < env.max_steps:
            timestep = len(episode_steps["true_bridge"])
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

            prediction_sets = {
                name: predictor_outputs(
                    policy,
                    predictor,
                    observations,
                    hidden_states,
                    previous_actions,
                    embedding,
                )
                for name, (predictor, embedding) in predictors.items()
            }
            next_observations, rewards, dones, info = env.step(actions)
            global_reward = float(sum(rewards.values()))
            stag_event = int(bool(info.get("stag_hunts")))
            hare_event = int(bool(info.get("hare_hunts")))
            decisive_event = int(bool(stag_event or hare_event))
            if decisive_event:
                event_steps.append(timestep)
            if stag_event:
                stag_event_steps.append(timestep)
            huntable = info.get("huntable_objects", {})

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
                for agent_id in range(policy.num_agents):
                    probability = probabilities[agent_id].to(
                        device=device, dtype=torch.float64
                    )
                    probability = probability / probability.sum().clamp_min(EPS)
                    values = predictions[agent_id]
                    delta_h = float((values.max() - values.min()).item())
                    bz_l1 = float(
                        torch.linalg.vector_norm(
                            logit_correction(probability, values), ord=1
                        ).item()
                    )
                    episode_samples.append(
                        {
                            "model": model,
                            "condition": name,
                            "seed": training_seed,
                            "episode_id": episode_id,
                            "timestep": timestep,
                            "agent_id": agent_id,
                            "pi_max": float(probability.max().item()),
                            "delta_h": delta_h,
                            "bz_l1": bz_l1,
                            "global_reward": global_reward,
                            "decisive_event": decisive_event,
                            "stag_hunt_event": stag_event,
                            "hare_hunt_event": hare_event,
                            "stag_coordination_ready": int(
                                int(huntable.get("total_huntable_stags", 0)) > 0
                            ),
                            "hare_huntable": int(
                                int(huntable.get("total_huntable_hares", 0)) > 0
                            ),
                        }
                    )
            observations_raw = next_observations
            done = all(bool(value) for value in dones.values())

        success = int(
            not any(info.get("stag_alive", {}).values())
            and not any(info.get("hare_alive", {}).values())
        )
        episode_length = len(episode_steps["true_bridge"])
        for name in predictors:
            episodes[name].append(
                {"steps": episode_steps[name], "success": float(success)}
            )
        for row in episode_samples:
            timestep = row["timestep"]
            row["episode_success"] = success
            row["episode_length"] = episode_length
            row["relative_timestep"] = timestep / max(episode_length - 1, 1)
            row["steps_to_episode_end"] = episode_length - 1 - timestep
            row["episode_has_decisive_event"] = int(bool(event_steps))
            row["nearest_decisive_event_distance"] = (
                min(abs(timestep - event_step) for event_step in event_steps)
                if event_steps
                else -1
            )
            row["nearest_stag_event_distance"] = (
                min(abs(timestep - event_step) for event_step in stag_event_steps)
                if stag_event_steps
                else -1
            )
            sample_rows.append(row)
    return episodes, sample_rows


def projection_metrics(global_grad, reference_delta, comparison_delta):
    global_grad = global_grad.to(torch.float64)
    reference_delta = reference_delta.to(torch.float64)
    comparison_delta = comparison_delta.to(torch.float64)
    global_norm_squared = float(torch.dot(global_grad, global_grad).item())

    def split(delta):
        coefficient = float(torch.dot(global_grad, delta).item()) / max(
            global_norm_squared, EPS
        )
        parallel = coefficient * global_grad
        orthogonal = delta - parallel
        return coefficient, parallel, orthogonal

    reference_coefficient, reference_parallel, reference_orthogonal = split(
        reference_delta
    )
    comparison_coefficient, comparison_parallel, comparison_orthogonal = split(
        comparison_delta
    )
    total_change = reference_delta - comparison_delta
    parallel_change = reference_parallel - comparison_parallel
    orthogonal_change = reference_orthogonal - comparison_orthogonal
    total_change_squared = float(torch.dot(total_change, total_change).item())
    parallel_change_squared = float(torch.dot(parallel_change, parallel_change).item())
    orthogonal_change_squared = float(
        torch.dot(orthogonal_change, orthogonal_change).item()
    )

    reference_parallel_norm = float(torch.linalg.vector_norm(reference_parallel).item())
    reference_orthogonal_norm = float(
        torch.linalg.vector_norm(reference_orthogonal).item()
    )
    parallel_change_norm = math.sqrt(max(parallel_change_squared, 0.0))
    orthogonal_change_norm = math.sqrt(max(orthogonal_change_squared, 0.0))
    reference_norm = float(torch.linalg.vector_norm(reference_delta).item())
    comparison_norm = float(torch.linalg.vector_norm(comparison_delta).item())
    rho_reference = reference_coefficient * math.sqrt(global_norm_squared) / max(
        reference_norm, EPS
    )
    rho_comparison = comparison_coefficient * math.sqrt(global_norm_squared) / max(
        comparison_norm, EPS
    )
    reconstruction_error = max(
        float(
            torch.linalg.vector_norm(
                reference_parallel + reference_orthogonal - reference_delta
            ).item()
        )
        / max(reference_norm, EPS),
        float(
            torch.linalg.vector_norm(
                comparison_parallel + comparison_orthogonal - comparison_delta
            ).item()
        )
        / max(comparison_norm, EPS),
    )
    energy_error = abs(
        total_change_squared - parallel_change_squared - orthogonal_change_squared
    ) / max(total_change_squared, EPS)
    return {
        "reference_parallel_norm": reference_parallel_norm,
        "reference_orthogonal_norm": reference_orthogonal_norm,
        "parallel_change_norm": parallel_change_norm,
        "orthogonal_change_norm": orthogonal_change_norm,
        "parallel_relative_change": parallel_change_norm
        / max(reference_parallel_norm, EPS),
        "orthogonal_relative_change": orthogonal_change_norm
        / max(reference_orthogonal_norm, EPS),
        "parallel_change_energy_fraction": parallel_change_squared
        / max(total_change_squared, EPS),
        "orthogonal_change_energy_fraction": orthogonal_change_squared
        / max(total_change_squared, EPS),
        "total_change_norm": math.sqrt(max(total_change_squared, 0.0)),
        "total_change_to_reference_norm_ratio": math.sqrt(
            max(total_change_squared, 0.0)
        )
        / max(reference_norm, EPS),
        "kappa_rho_reference": reference_coefficient,
        "kappa_rho_comparison": comparison_coefficient,
        "kappa_rho_change": reference_coefficient - comparison_coefficient,
        "rho_reference": rho_reference,
        "rho_comparison": rho_comparison,
        "rho_change": rho_reference - rho_comparison,
        "projection_reconstruction_relative_error": reconstruction_error,
        "change_energy_identity_relative_error": energy_error,
    }


def global_direction_metrics(reference_global, comparison_global):
    reference_global = reference_global.to(torch.float64)
    comparison_global = comparison_global.to(torch.float64)
    reference_norm_squared = float(torch.dot(reference_global, reference_global).item())
    scale = float(torch.dot(reference_global, comparison_global).item()) / max(
        reference_norm_squared, EPS
    )
    aligned_residual = comparison_global - scale * reference_global
    denominator = max(float(torch.linalg.vector_norm(comparison_global).item()), EPS)
    cosine = float(torch.dot(reference_global, comparison_global).item()) / max(
        float(torch.linalg.vector_norm(reference_global).item()) * denominator, EPS
    )
    return {
        "global_bitwise_equal": int(torch.equal(reference_global, comparison_global)),
        "global_direction_cosine": cosine,
        "global_best_scale": scale,
        "global_scaled_direction_relative_error": float(
            torch.linalg.vector_norm(aligned_residual).item()
        )
        / denominator,
    }


def part_a_rows(
    policy,
    episodes,
    model,
    seed,
    gamma,
    gae_lambda,
    causal_lambda,
    device,
):
    advantages = {
        condition: prepare_advantages(
            condition_episodes,
            policy.num_agents,
            gamma,
            gae_lambda,
            causal_lambda,
        )
        for condition, condition_episodes in episodes.items()
    }
    gradients = {}
    for condition, condition_episodes in episodes.items():
        gradients[condition] = condition_gradients(
            policy, condition_episodes, advantages[condition], device
        )

    rows = []
    for comparison_name, condition in COMPARISONS:
        for agent_id in tuple(range(policy.num_agents)) + (-1,):
            if agent_id == -1:
                reference_vectors = gradients["true_bridge"][1]
                comparison_vectors = gradients[condition][1]
            else:
                reference_vectors = gradients["true_bridge"][0][agent_id]
                comparison_vectors = gradients[condition][0][agent_id]
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "comparison": comparison_name,
                    "agent_id": agent_id,
                    **projection_metrics(
                        reference_vectors[0],
                        reference_vectors[1],
                        comparison_vectors[1],
                    ),
                    **global_direction_metrics(
                        reference_vectors[0], comparison_vectors[0]
                    ),
                    "reference_gradient_additivity_relative_error": gradients[
                        "true_bridge"
                    ][2],
                    "comparison_gradient_additivity_relative_error": gradients[
                        condition
                    ][2],
                }
            )
    return rows


def aggregate_part_a(rows):
    metrics = [
        key
        for key in rows[0]
        if key not in {"model", "seed", "comparison", "agent_id"}
    ]
    seed_rows = []
    aggregate_rows = []
    for model in MODEL_NAMES.values():
        for comparison, _ in COMPARISONS:
            for seed in SEEDS:
                selected = [
                    row
                    for row in rows
                    if row["model"] == model
                    and row["comparison"] == comparison
                    and row["seed"] == seed
                    and row["agent_id"] == -1
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"Expected one joint row for {model}, {comparison}, {seed}"
                    )
                seed_rows.append(selected[0])
            selected_seeds = [
                row
                for row in seed_rows
                if row["model"] == model and row["comparison"] == comparison
            ]
            result = {
                "model": model,
                "comparison": comparison,
                "n_seeds": len(selected_seeds),
            }
            for metric in metrics:
                mean, ci = mean_ci([row[metric] for row in selected_seeds])
                result[f"{metric}_mean"] = mean
                result[f"{metric}_ci95_half_width"] = ci
            aggregate_rows.append(result)
    return seed_rows, aggregate_rows


def top_fraction_mask(values, fraction):
    values = np.asarray(values, dtype=np.float64)
    count = max(int(math.ceil(values.size * fraction)), 1)
    order = np.argsort(values, kind="mergesort")
    mask = np.zeros(values.size, dtype=bool)
    mask[order[-count:]] = True
    return mask


def composition_row(rows, set_name):
    timesteps = np.asarray([row["timestep"] for row in rows], dtype=np.float64)
    relative = np.asarray([row["relative_timestep"] for row in rows], dtype=np.float64)
    distances = np.asarray(
        [row["nearest_decisive_event_distance"] for row in rows], dtype=np.float64
    )
    has_event = distances >= 0
    result = {
        "model": rows[0]["model"],
        "condition": rows[0]["condition"],
        "seed": rows[0]["seed"],
        "set": set_name,
        "num_samples": len(rows),
    }
    for prefix, values in (("timestep", timesteps), ("relative_timestep", relative)):
        stats = distribution(values)
        for name in ("mean", "median", "q25", "q75"):
            result[f"{prefix}_{name}"] = stats[name]
    result.update(
        {
            "success_fraction": float(
                np.mean([row["episode_success"] for row in rows])
            ),
            "decisive_event_fraction": float(
                np.mean([row["decisive_event"] for row in rows])
            ),
            "stag_hunt_event_fraction": float(
                np.mean([row["stag_hunt_event"] for row in rows])
            ),
            "hare_hunt_event_fraction": float(
                np.mean([row["hare_hunt_event"] for row in rows])
            ),
            "stag_coordination_ready_fraction": float(
                np.mean([row["stag_coordination_ready"] for row in rows])
            ),
            "no_decisive_event_episode_fraction": float(np.mean(~has_event)),
            "within_1_step_of_event_fraction": float(
                np.mean(has_event & (distances <= 1))
            ),
            "within_3_steps_of_event_fraction": float(
                np.mean(has_event & (distances <= 3))
            ),
            "within_5_steps_of_event_fraction": float(
                np.mean(has_event & (distances <= 5))
            ),
            "nearest_event_distance_mean_when_available": float(
                distances[has_event].mean() if np.any(has_event) else np.nan
            ),
            "nearest_event_distance_median_when_available": float(
                np.median(distances[has_event]) if np.any(has_event) else np.nan
            ),
            "terminal_3_steps_fraction": float(
                np.mean([row["steps_to_episode_end"] <= 3 for row in rows])
            ),
        }
    )
    for lower, upper, label in TIMESTEP_BINS:
        result[f"timestep_{label}_fraction"] = float(
            np.mean((timesteps >= lower) & (timesteps <= upper))
        )
    return result


def part_b(sample_rows):
    enriched_rows = []
    composition_rows = []
    relationship_rows = []
    decile_rows = []
    for model in MODEL_NAMES.values():
        for condition in CONDITIONS:
            for seed in SEEDS:
                selected = [
                    dict(row)
                    for row in sample_rows
                    if row["model"] == model
                    and row["condition"] == condition
                    and row["seed"] == seed
                ]
                delta_h = np.asarray([row["delta_h"] for row in selected])
                pi_max = np.asarray([row["pi_max"] for row in selected])
                bz_l1 = np.asarray([row["bz_l1"] for row in selected])
                threshold = float(np.quantile(delta_h, 0.95))
                high_mask = delta_h > threshold
                top_five_mask = top_fraction_mask(delta_h, 0.05)
                top_one_mask = top_fraction_mask(delta_h, 0.01)
                for index, row in enumerate(selected):
                    row["high_delta_h_threshold"] = threshold
                    row["high_delta_h"] = int(high_mask[index])
                    row["top_5_percent_delta_h"] = int(top_five_mask[index])
                    row["top_1_percent_delta_h"] = int(top_one_mask[index])
                    enriched_rows.append(row)
                composition_rows.append(composition_row(selected, "all"))
                composition_rows.append(
                    composition_row(
                        [row for row, high in zip(selected, high_mask) if high],
                        "high_delta_h",
                    )
                )
                relationship_rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "seed": seed,
                        "num_samples": len(selected),
                        "high_delta_h_threshold": threshold,
                        "high_delta_h_fraction": float(np.mean(high_mask)),
                        "spearman_delta_h_vs_pi_max": correlation(
                            delta_h, pi_max, ranked=True
                        ),
                        "bz_l1_share_top_5_percent_delta_h": float(
                            bz_l1[top_five_mask].sum() / max(bz_l1.sum(), EPS)
                        ),
                        "bz_l1_share_top_1_percent_delta_h": float(
                            bz_l1[top_one_mask].sum() / max(bz_l1.sum(), EPS)
                        ),
                    }
                )
                order = np.argsort(pi_max, kind="mergesort")
                for decile, indices in enumerate(np.array_split(order, 10), start=1):
                    decile_rows.append(
                        {
                            "model": model,
                            "condition": condition,
                            "seed": seed,
                            "pi_max_decile": decile,
                            "num_samples": len(indices),
                            "pi_max_min": float(pi_max[indices].min()),
                            "pi_max_median": float(np.median(pi_max[indices])),
                            "pi_max_max": float(pi_max[indices].max()),
                            "delta_h_median": float(np.median(delta_h[indices])),
                            "bz_l1_median": float(np.median(bz_l1[indices])),
                        }
                    )
    return enriched_rows, composition_rows, relationship_rows, decile_rows


def aggregate_part_b(composition_rows, relationship_rows, decile_rows):
    composition_summary = []
    relationship_summary = []
    decile_summary = []
    composition_metrics = [
        key
        for key in composition_rows[0]
        if key not in {"model", "condition", "seed", "set", "num_samples"}
    ]
    for model in MODEL_NAMES.values():
        for condition in CONDITIONS:
            for set_name in ("all", "high_delta_h"):
                selected = [
                    row
                    for row in composition_rows
                    if row["model"] == model
                    and row["condition"] == condition
                    and row["set"] == set_name
                ]
                result = {
                    "model": model,
                    "condition": condition,
                    "set": set_name,
                    "n_seeds": len(selected),
                }
                for metric in composition_metrics:
                    mean, ci = mean_ci([row[metric] for row in selected])
                    result[f"{metric}_mean"] = mean
                    result[f"{metric}_ci95_half_width"] = ci
                composition_summary.append(result)

            selected_relationships = [
                row
                for row in relationship_rows
                if row["model"] == model and row["condition"] == condition
            ]
            result = {
                "model": model,
                "condition": condition,
                "n_seeds": len(selected_relationships),
            }
            for metric in (
                "high_delta_h_threshold",
                "high_delta_h_fraction",
                "spearman_delta_h_vs_pi_max",
                "bz_l1_share_top_5_percent_delta_h",
                "bz_l1_share_top_1_percent_delta_h",
            ):
                mean, ci = mean_ci([row[metric] for row in selected_relationships])
                result[f"{metric}_mean"] = mean
                result[f"{metric}_ci95_half_width"] = ci
            relationship_summary.append(result)

            for decile in range(1, 11):
                selected_deciles = [
                    row
                    for row in decile_rows
                    if row["model"] == model
                    and row["condition"] == condition
                    and row["pi_max_decile"] == decile
                ]
                result = {
                    "model": model,
                    "condition": condition,
                    "pi_max_decile": decile,
                    "n_seeds": len(selected_deciles),
                }
                for metric in (
                    "pi_max_min",
                    "pi_max_median",
                    "pi_max_max",
                    "delta_h_median",
                    "bz_l1_median",
                ):
                    mean, ci = mean_ci([row[metric] for row in selected_deciles])
                    result[f"{metric}_mean"] = mean
                    result[f"{metric}_ci95_half_width"] = ci
                decile_summary.append(result)
    return composition_summary, relationship_summary, decile_summary


def validate_round2_samples(round3_rows):
    round2_rows = read_csv(
        RESULT_DIR / "gradient_followup_round2_partA_samples.csv"
    )
    lookup = {
        (
            row["model"],
            int(row["seed"]),
            int(row["episode_id"]),
            int(row["timestep"]),
            int(row["agent_id"]),
        ): row
        for row in round2_rows
    }
    maximum = {"pi_max": 0.0, "delta_h": 0.0, "bz_l1": 0.0}
    true_rows = [row for row in round3_rows if row["condition"] == "true_bridge"]
    for row in true_rows:
        key = (
            row["model"],
            row["seed"],
            row["episode_id"],
            row["timestep"],
            row["agent_id"],
        )
        reference = lookup[key]
        for metric in maximum:
            maximum[metric] = max(
                maximum[metric], abs(row[metric] - float(reference[metric]))
            )
    if len(true_rows) != len(round2_rows):
        raise ValueError(
            f"Round 2/3 sample count mismatch: {len(round2_rows)} vs {len(true_rows)}"
        )
    return {"num_matched_samples": len(true_rows), **maximum}


def main():
    device = torch.device("cpu")
    part_a_observations = []
    part_b_samples = []
    naive_paths = []
    shuffle_rows = []

    for model_key in ("ippo", "mappo"):
        runs = discover_runs(DEFAULT_ROOTS[model_key]["baseline"], SEEDS)
        if tuple(seed for seed, _, _ in runs) != SEEDS:
            raise ValueError(f"Unexpected {model_key} seeds: {[seed for seed, _, _ in runs]}")
        for seed, env_dir, config_path in runs:
            config = json.loads(config_path.read_text())
            checkpoint_path = env_dir / "checkpoints" / f"checkpoint_{FINAL_ITERATION}.pt"
            env = make_env(config)
            policy, bridge, metadata = load_model(checkpoint_path, env, config, device)
            embedding, _ = select_previous_action_embedding(
                policy, metadata["bridge_x_dim"]
            )
            naive, naive_path = load_naive_predictor(
                checkpoint_path, policy, metadata, config, device
            )
            if naive is None:
                raise FileNotFoundError(
                    f"Missing final naive predictor for {model_key}, seed {seed}"
                )
            naive_paths.append(
                {"model": MODEL_NAMES[model_key], "seed": seed, "path": str(naive_path)}
            )
            episodes, samples = collect_round3_episodes(
                policy,
                {
                    "true_bridge": (bridge, embedding),
                    "naive_predictor": (naive, embedding),
                },
                env,
                NUM_EPISODES,
                ROLLOUT_SEED,
                MODEL_NAMES[model_key],
                seed,
                device,
            )
            shuffled, shuffle_diagnostics = build_eval_w_shuffle_episodes(
                policy,
                bridge,
                embedding,
                episodes["true_bridge"],
                seed,
                FINAL_ITERATION,
                ROLLOUT_SEED,
                device,
            )
            episodes["eval_w_shuffle"] = shuffled
            algorithm = config["config"]["algorithm"]
            part_a_observations.extend(
                part_a_rows(
                    policy,
                    episodes,
                    MODEL_NAMES[model_key],
                    seed,
                    float(algorithm["gamma"]),
                    float(algorithm["gae_lambda"]),
                    float(algorithm["causal_lambda"]),
                    device,
                )
            )
            part_b_samples.extend(samples)
            shuffle_rows.append(
                {"model": MODEL_NAMES[model_key], "seed": seed, **shuffle_diagnostics}
            )
            print(
                f"Round 3 {model_key} seed={seed}: samples={len(samples)}, "
                f"success={np.mean([row['episode_success'] for row in samples]):.3f}",
                flush=True,
            )

    part_a_seed_rows, part_a_summary = aggregate_part_a(part_a_observations)
    (
        enriched_samples,
        composition_rows,
        relationship_rows,
        decile_rows,
    ) = part_b(part_b_samples)
    (
        composition_summary,
        relationship_summary,
        decile_summary,
    ) = aggregate_part_b(composition_rows, relationship_rows, decile_rows)
    replay_validation = validate_round2_samples(enriched_samples)

    outputs = {
        "gradient_followup_round3_partA_observations.csv": part_a_observations,
        "gradient_followup_round3_partA_seed_summary.csv": part_a_seed_rows,
        "gradient_followup_round3_partA_summary.csv": part_a_summary,
        "gradient_followup_round3_partB_samples.csv": enriched_samples,
        "gradient_followup_round3_partB_composition_seed.csv": composition_rows,
        "gradient_followup_round3_partB_composition_summary.csv": composition_summary,
        "gradient_followup_round3_partB_relationship_seed.csv": relationship_rows,
        "gradient_followup_round3_partB_relationship_summary.csv": relationship_summary,
        "gradient_followup_round3_partB_pi_deciles_seed.csv": decile_rows,
        "gradient_followup_round3_partB_pi_deciles_summary.csv": decile_summary,
        "gradient_followup_round3_naive_checkpoint_paths.csv": naive_paths,
        "gradient_followup_round3_shuffle_diagnostics.csv": shuffle_rows,
    }
    for name, rows in outputs.items():
        write_csv(RESULT_DIR / name, rows)

    metadata = {
        "final_iteration": FINAL_ITERATION,
        "training_seeds": list(SEEDS),
        "num_episodes": NUM_EPISODES,
        "rollout_seed": ROLLOUT_SEED,
        "device": str(device),
        "training_updates": 0,
        "decisive_event_marker": (
            "Successful stag_hunts or hare_hunts reported by StagHuntEnv.step info"
        ),
        "replay_validation_against_round2": replay_validation,
        "part_a_max_projection_reconstruction_relative_error": max(
            row["projection_reconstruction_relative_error"]
            for row in part_a_observations
        ),
        "part_a_max_change_energy_identity_relative_error": max(
            row["change_energy_identity_relative_error"]
            for row in part_a_observations
        ),
        "part_a_max_global_scaled_direction_relative_error": max(
            row["global_scaled_direction_relative_error"]
            for row in part_a_observations
        ),
        "part_a_max_gradient_additivity_relative_error": max(
            max(
                row["reference_gradient_additivity_relative_error"],
                row["comparison_gradient_additivity_relative_error"],
            )
            for row in part_a_observations
        ),
    }
    (RESULT_DIR / "gradient_followup_round3_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps({name: len(rows) for name, rows in outputs.items()}, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
