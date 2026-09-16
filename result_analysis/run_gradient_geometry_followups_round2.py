import csv
import json
import math
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
    T_95_DF4,
    mean_ci,
    read_csv,
    write_csv,
)
from result_analysis.run_staghunt_gradient_null_controls import (
    DEFAULT_ROOTS,
    build_eval_w_shuffle_episodes,
    collect_control_episodes,
    condition_gradients,
    predictor_outputs,
    select_previous_action_embedding,
)
from result_analysis.run_staghunt_policy_gradient_decomposition import (
    critic_free_gae,
    discover_runs,
    load_model,
    make_env,
    normalize_population,
    prepare_advantages,
    tensor_observations,
)


MODEL_NAMES = {"ippo": "PCI-IPPO", "mappo": "PCI-MAPPO"}
BOUND_TOL = 1.0e-10


def logit_correction(probabilities, predictions):
    probabilities = probabilities.to(torch.float64)
    probabilities = probabilities / probabilities.sum().clamp_min(EPS)
    predictions = predictions.to(torch.float64)
    weighted_mean = torch.sum(probabilities * predictions)
    denominator = (1.0 - probabilities).clamp_min(EPS)
    identified_contrast = (predictions - weighted_mean) / denominator
    phi = torch.sum(probabilities.square() * identified_contrast)
    return probabilities * (probabilities * identified_contrast - phi)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(left, right, ranked=False):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if ranked:
        left = rankdata(left)
        right = rankdata(right)
    if left.std() <= EPS or right.std() <= EPS:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def relative_l2(reference, changed):
    reference = np.asarray(reference, dtype=np.float64)
    changed = np.asarray(changed, dtype=np.float64)
    return float(np.linalg.norm(reference - changed) / max(np.linalg.norm(reference), EPS))


def cosine(left, right):
    left = left.to(torch.float64)
    right = right.to(torch.float64)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right).item() / max(float(denominator.item()), EPS))


def part_a_samples(policy, bridge, embedding, episodes, model, seed, device):
    rows = []
    for episode_id, episode in enumerate(episodes):
        for timestep, step in enumerate(episode["steps"]):
            observations = tensor_observations(step["observations"], device)
            hidden_states = {
                agent_id: value.to(device)
                for agent_id, value in step["hidden_states"].items()
            }
            previous_actions = {
                agent_id: value.to(device)
                for agent_id, value in step["previous_actions"].items()
            }
            predictions = predictor_outputs(
                policy,
                bridge,
                observations,
                hidden_states,
                previous_actions,
                embedding,
            )
            for agent_id in range(policy.num_agents):
                probabilities = step["action_probabilities"][agent_id].to(
                    device=device, dtype=torch.float64
                )
                probabilities = probabilities / probabilities.sum().clamp_min(EPS)
                values = predictions[agent_id]
                pi_max = float(probabilities.max().item())
                delta_h = float((values.max() - values.min()).item())
                correction_l1 = float(
                    torch.linalg.vector_norm(
                        logit_correction(probabilities, values), ord=1
                    ).item()
                )
                deterministic_bound = 2.0 * delta_h * (1.0 - pi_max)
                universal_bound = 0.5 * delta_h
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "episode_id": episode_id,
                        "timestep": timestep,
                        "agent_id": agent_id,
                        "pi_max": pi_max,
                        "one_minus_pi_max": 1.0 - pi_max,
                        "delta_h": delta_h,
                        "bz_l1": correction_l1,
                        "d1_bound": deterministic_bound,
                        "universal_bound": universal_bound,
                        "d1_tightness": correction_l1 / max(deterministic_bound, EPS),
                        "universal_tightness": correction_l1 / max(universal_bound, EPS),
                        "d1_violation": int(
                            correction_l1 > deterministic_bound + BOUND_TOL
                        ),
                        "universal_violation": int(
                            correction_l1 > universal_bound + BOUND_TOL
                        ),
                        "d1_bound_active": int(
                            deterministic_bound <= universal_bound
                        ),
                    }
                )
    return rows


def summarize_part_a_seed(rows):
    result = {
        "model": rows[0]["model"],
        "seed": rows[0]["seed"],
        "num_samples": len(rows),
    }
    for prefix, field in (
        ("pi_max", "pi_max"),
        ("delta_h", "delta_h"),
        ("bz_l1", "bz_l1"),
        ("d1_tightness", "d1_tightness"),
        ("universal_tightness", "universal_tightness"),
    ):
        for name, value in distribution([row[field] for row in rows]).items():
            result[f"{prefix}_{name}"] = value
    pi_max = np.asarray([row["pi_max"] for row in rows])
    bz_l1 = np.asarray([row["bz_l1"] for row in rows])
    one_minus = np.asarray([row["one_minus_pi_max"] for row in rows])
    d1_bound = np.asarray([row["d1_bound"] for row in rows])
    result.update(
        {
            "pi_max_above_0_99_fraction": float(np.mean(pi_max > 0.99)),
            "pi_max_above_0_999_fraction": float(np.mean(pi_max > 0.999)),
            "pearson_bz_vs_one_minus_pi_max": correlation(bz_l1, one_minus),
            "spearman_bz_vs_one_minus_pi_max": correlation(
                bz_l1, one_minus, ranked=True
            ),
            "pearson_bz_vs_d1_bound": correlation(bz_l1, d1_bound),
            "spearman_bz_vs_d1_bound": correlation(bz_l1, d1_bound, ranked=True),
            "d1_violation_fraction": float(
                np.mean([row["d1_violation"] for row in rows])
            ),
            "universal_violation_fraction": float(
                np.mean([row["universal_violation"] for row in rows])
            ),
            "d1_bound_active_fraction": float(
                np.mean([row["d1_bound_active"] for row in rows])
            ),
            "delta_h_below_1e_6_fraction": float(
                np.mean([row["delta_h"] < 1.0e-6 for row in rows])
            ),
        }
    )
    return result


def aggregate_part_a(sample_rows, seed_rows):
    output = []
    metrics = [
        key
        for key in seed_rows[0]
        if key not in {"model", "seed", "num_samples"}
    ]
    for model in MODEL_NAMES.values():
        model_samples = [row for row in sample_rows if row["model"] == model]
        model_seeds = [row for row in seed_rows if row["model"] == model]
        result = {"model": model, "n_seeds": len(model_seeds)}
        pooled = summarize_part_a_seed(model_samples)
        for metric in metrics:
            mean, ci = mean_ci([row[metric] for row in model_seeds])
            result[f"{metric}_seed_mean"] = mean
            result[f"{metric}_ci95_half_width"] = ci
            result[f"{metric}_pooled"] = pooled[metric]
        output.append(result)
    return output


def part_b_stages(
    policy,
    real_episodes,
    shuffled_episodes,
    model,
    seed,
    gamma,
    gae_lambda,
    causal_lambda,
):
    real_advantages = prepare_advantages(
        real_episodes, policy.num_agents, gamma, gae_lambda, causal_lambda
    )
    shuffled_advantages = prepare_advantages(
        shuffled_episodes, policy.num_agents, gamma, gae_lambda, causal_lambda
    )
    offsets = {agent_id: 0 for agent_id in range(policy.num_agents)}
    rows = []
    for episode_id, (real_episode, shuffled_episode) in enumerate(
        zip(real_episodes, shuffled_episodes)
    ):
        length = len(real_episode["steps"])
        for agent_id in range(policy.num_agents):
            real_baseline = np.asarray(
                [step["baseline"][agent_id] for step in real_episode["steps"]]
            )
            shuffled_baseline = np.asarray(
                [step["baseline"][agent_id] for step in shuffled_episode["steps"]]
            )
            real_reward = np.asarray(
                [step["delta"][agent_id] for step in real_episode["steps"]]
            )
            shuffled_reward = np.asarray(
                [step["delta"][agent_id] for step in shuffled_episode["steps"]]
            )
            real_raw = critic_free_gae(real_reward, gamma, gae_lambda)
            shuffled_raw = critic_free_gae(shuffled_reward, gamma, gae_lambda)
            real_component = normalize_population(real_raw)
            shuffled_component = normalize_population(shuffled_raw)
            start = offsets[agent_id]
            end = start + length
            real_final = real_advantages[agent_id]["delta"][start:end]
            shuffled_final = shuffled_advantages[agent_id]["delta"][start:end]
            offsets[agent_id] = end

            difference = real_baseline - shuffled_baseline
            total_squared = float(np.sum(np.square(difference)))
            mean_squared = float(length * np.square(difference.mean()))
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "episode_id": episode_id,
                    "agent_id": agent_id,
                    "episode_length": length,
                    "baseline_relative_l2": relative_l2(
                        real_baseline, shuffled_baseline
                    ),
                    "shaped_reward_relative_l2": relative_l2(
                        real_reward, shuffled_reward
                    ),
                    "raw_gae_relative_l2": relative_l2(real_raw, shuffled_raw),
                    "component_normalized_relative_l2": relative_l2(
                        real_component, shuffled_component
                    ),
                    "final_normalized_relative_l2": relative_l2(
                        real_final, shuffled_final
                    ),
                    "baseline_change_episode_mean_squared_fraction": (
                        mean_squared / max(total_squared, EPS)
                    ),
                    "baseline_reference_l2": float(np.linalg.norm(real_baseline)),
                    "baseline_difference_l2": float(np.linalg.norm(difference)),
                    "shaped_reward_reference_l2": float(np.linalg.norm(real_reward)),
                    "shaped_reward_difference_l2": float(
                        np.linalg.norm(real_reward - shuffled_reward)
                    ),
                    "raw_gae_reference_l2": float(np.linalg.norm(real_raw)),
                    "raw_gae_difference_l2": float(
                        np.linalg.norm(real_raw - shuffled_raw)
                    ),
                    "component_normalized_reference_l2": float(
                        np.linalg.norm(real_component)
                    ),
                    "component_normalized_difference_l2": float(
                        np.linalg.norm(real_component - shuffled_component)
                    ),
                    "final_normalized_reference_l2": float(
                        np.linalg.norm(real_final)
                    ),
                    "final_normalized_difference_l2": float(
                        np.linalg.norm(real_final - shuffled_final)
                    ),
                }
            )
    return rows, real_advantages, shuffled_advantages


def summarize_part_b_seed(rows):
    result = {
        "model": rows[0]["model"],
        "seed": rows[0]["seed"],
        "num_episode_agent_pairs": len(rows),
    }
    stages = (
        "baseline",
        "shaped_reward",
        "raw_gae",
        "component_normalized",
        "final_normalized",
    )
    for stage in stages:
        values = [row[f"{stage}_relative_l2"] for row in rows]
        result[f"{stage}_relative_l2_mean"] = float(np.mean(values))
        result[f"{stage}_relative_l2_median"] = float(np.median(values))
        result[f"{stage}_relative_l2_ratio_of_norm_sums"] = float(
            np.sum([row[f"{stage}_difference_l2"] for row in rows])
            / max(np.sum([row[f"{stage}_reference_l2"] for row in rows]), EPS)
        )
    mean_fraction = [
        row["baseline_change_episode_mean_squared_fraction"] for row in rows
    ]
    result["baseline_change_episode_mean_squared_fraction_mean"] = float(
        np.mean(mean_fraction)
    )
    result["baseline_change_episode_mean_squared_fraction_median"] = float(
        np.median(mean_fraction)
    )
    return result


def part_b_gradient_rows(
    policy,
    real_episodes,
    shuffled_episodes,
    real_advantages,
    shuffled_advantages,
    model,
    seed,
    device,
):
    real_by_agent, real_joint, real_residual = condition_gradients(
        policy, real_episodes, real_advantages, device
    )
    shuffled_by_agent, shuffled_joint, shuffled_residual = condition_gradients(
        policy, shuffled_episodes, shuffled_advantages, device
    )
    def comparison(agent_id, real_vectors, shuffled_vectors):
        global_grad = real_vectors[0]
        real_delta = real_vectors[1]
        shuffled_delta = shuffled_vectors[1]
        difference = real_delta - shuffled_delta
        global_norm = float(torch.linalg.vector_norm(global_grad).item())
        difference_norm = float(torch.linalg.vector_norm(difference).item())
        return {
            "model": model,
            "seed": seed,
            "agent_id": agent_id,
            "auxiliary_gradient_cosine_real_vs_eval_w": cosine(
                real_delta, shuffled_delta
            ),
            "rho_real": cosine(global_grad, real_delta),
            "rho_eval_w": cosine(global_grad, shuffled_delta),
            "rho_real_minus_eval_w": cosine(global_grad, real_delta)
            - cosine(global_grad, shuffled_delta),
            "auxiliary_difference_vs_global_cosine": cosine(
                difference, global_grad
            ),
            "auxiliary_difference_to_real_norm_ratio": difference_norm
            / max(float(torch.linalg.vector_norm(real_delta).item()), EPS),
            "auxiliary_difference_parallel_to_global_ratio": float(
                torch.dot(difference, global_grad).item()
            )
            / max(global_norm * global_norm, EPS),
            "real_gradient_additivity_relative_error": real_residual,
            "shuffled_gradient_additivity_relative_error": shuffled_residual,
        }

    rows = []
    for agent_id in range(policy.num_agents):
        rows.append(comparison(agent_id, real_by_agent[agent_id], shuffled_by_agent[agent_id]))
    rows.append(comparison(-1, real_joint, shuffled_joint))
    return rows


def aggregate_part_b(seed_rows, gradient_rows):
    output = []
    stages = (
        "baseline",
        "shaped_reward",
        "raw_gae",
        "component_normalized",
        "final_normalized",
    )
    for model in MODEL_NAMES.values():
        selected = [row for row in seed_rows if row["model"] == model]
        result = {"model": model, "n_seeds": len(selected)}
        for stage in stages:
            for suffix in (
                "relative_l2_mean",
                "relative_l2_median",
                "relative_l2_ratio_of_norm_sums",
            ):
                metric = f"{stage}_{suffix}"
                mean, ci = mean_ci([row[metric] for row in selected])
                result[f"{metric}_mean"] = mean
                result[f"{metric}_ci95_half_width"] = ci
        for metric in (
            "baseline_change_episode_mean_squared_fraction_mean",
            "baseline_change_episode_mean_squared_fraction_median",
        ):
            mean, ci = mean_ci([row[metric] for row in selected])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_ci95_half_width"] = ci

        joint = [
            row
            for row in gradient_rows
            if row["model"] == model and row["agent_id"] == -1
        ]
        per_agent_seed_means = []
        for seed in SEEDS:
            values = [
                row["auxiliary_gradient_cosine_real_vs_eval_w"]
                for row in gradient_rows
                if row["model"] == model
                and row["seed"] == seed
                and row["agent_id"] >= 0
            ]
            per_agent_seed_means.append(float(np.mean(values)))
        mean, ci = mean_ci(
            [row["auxiliary_gradient_cosine_real_vs_eval_w"] for row in joint]
        )
        result["joint_auxiliary_gradient_cosine_mean"] = mean
        result["joint_auxiliary_gradient_cosine_ci95_half_width"] = ci
        mean, ci = mean_ci(per_agent_seed_means)
        result["per_agent_auxiliary_gradient_cosine_mean"] = mean
        result["per_agent_auxiliary_gradient_cosine_ci95_half_width"] = ci
        for metric in (
            "rho_real_minus_eval_w",
            "auxiliary_difference_vs_global_cosine",
            "auxiliary_difference_to_real_norm_ratio",
            "auxiliary_difference_parallel_to_global_ratio",
        ):
            mean, ci = mean_ci([row[metric] for row in joint])
            result[f"joint_{metric}_mean"] = mean
            result[f"joint_{metric}_ci95_half_width"] = ci
        output.append(result)
    return output


def final_rows(path, field, value, agent_filter):
    return [
        row
        for row in read_csv(path)
        if str(row[field]) == str(value) and agent_filter(int(row["agent_id"]))
    ]


def part_c():
    specs = (
        (
            "StagHunt PCI-IPPO",
            DATA_DIR / "checkpoint_gradient_components.csv",
            "iteration",
            FINAL_ITERATION,
            1.0,
        ),
        (
            "StagHunt PCI-MAPPO",
            DATA_DIR / "staghunt_vanilla_pci_mappo_checkpoint_gradient_components.csv",
            "iteration",
            FINAL_ITERATION,
            1.0,
        ),
        (
            "SMAX 3m:hard PCI-IPPO",
            DATA_DIR / "smax_3m_hard_pci_ippo_checkpoint_gradient_components.csv",
            "checkpoint_label",
            "final",
            1.0,
        ),
        (
            "SMAX 3m:hard PCI-MAPPO",
            DATA_DIR / "smax_3m_hard_pci_mappo_checkpoint_gradient_components.csv",
            "checkpoint_label",
            "final",
            1.0,
        ),
        (
            "SMAX smacv2_5_units:hard PCI-IPPO",
            DATA_DIR
            / "smax_smacv2_5_units_hard_pci_ippo_checkpoint_gradient_components.csv",
            "checkpoint_label",
            "final",
            0.1,
        ),
        (
            "SMAX smacv2_5_units:hard PCI-MAPPO",
            DATA_DIR
            / "smax_smacv2_5_units_hard_pci_mappo_checkpoint_gradient_components.csv",
            "checkpoint_label",
            "final",
            0.1,
        ),
    )
    aggregate_rows = []
    per_agent_rows = []
    summary_rows = []
    for domain_model, path, field, value, causal_lambda in specs:
        aggregate_source = final_rows(path, field, value, lambda agent_id: agent_id == -1)
        agent_source = final_rows(path, field, value, lambda agent_id: agent_id >= 0)
        if len(aggregate_source) != len(SEEDS):
            raise ValueError(f"Expected five aggregate rows for {domain_model}")
        for row in aggregate_source:
            rho = float(row["global_delta_cosine"])
            kappa_weighted = float(row["delta_to_global_norm_ratio"])
            # Stored delta gradients already include causal_lambda.  Dividing it out
            # recovers the conceptual unweighted kappa used in 1 + lambda*kappa*rho.
            kappa_unweighted = kappa_weighted / causal_lambda
            coefficient = 1.0 + causal_lambda * kappa_unweighted * rho
            direct_coefficient = float(row["combined_global_projection_ratio"])
            aggregate_rows.append(
                {
                    "domain_model": domain_model,
                    "seed": int(row["seed"]),
                    "causal_lambda": causal_lambda,
                    "rho": rho,
                    "kappa_unweighted": kappa_unweighted,
                    "kappa_weighted_stored": kappa_weighted,
                    "parallel_coefficient": coefficient,
                    "parallel_coefficient_saved_direct": direct_coefficient,
                    "absolute_error": abs(coefficient - direct_coefficient),
                    "reversed": int(coefficient < 0.0),
                }
            )
        for row in agent_source:
            rho = float(row["global_delta_cosine"])
            kappa_weighted = float(row["delta_to_global_norm_ratio"])
            coefficient = 1.0 + kappa_weighted * rho
            per_agent_rows.append(
                {
                    "domain_model": domain_model,
                    "seed": int(row["seed"]),
                    "agent_id": int(row["agent_id"]),
                    "causal_lambda": causal_lambda,
                    "rho": rho,
                    "kappa_unweighted": kappa_weighted / causal_lambda,
                    "kappa_weighted_stored": kappa_weighted,
                    "parallel_coefficient": coefficient,
                    "reversed": int(coefficient < 0.0),
                }
            )
        selected_aggregate = [
            row for row in aggregate_rows if row["domain_model"] == domain_model
        ]
        selected_agents = [
            row for row in per_agent_rows if row["domain_model"] == domain_model
        ]
        summary_rows.append(
            {
                "domain_model": domain_model,
                "causal_lambda": causal_lambda,
                "num_aggregate_observations": len(selected_aggregate),
                "num_negative_aggregate_coefficients": sum(
                    row["reversed"] for row in selected_aggregate
                ),
                "minimum_aggregate_coefficient": min(
                    row["parallel_coefficient"] for row in selected_aggregate
                ),
                "num_per_agent_observations": len(selected_agents),
                "num_negative_per_agent_coefficients": sum(
                    row["reversed"] for row in selected_agents
                ),
                "num_negative_per_agent_rho": sum(
                    row["rho"] < 0.0 for row in selected_agents
                ),
                "minimum_per_agent_coefficient": min(
                    row["parallel_coefficient"] for row in selected_agents
                ),
            }
        )
    return aggregate_rows, per_agent_rows, summary_rows


def main():
    device = torch.device("cpu")
    part_a_sample_rows = []
    part_a_seed_rows = []
    part_b_stage_rows = []
    part_b_seed_rows = []
    part_b_gradient_rows_output = []
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
            episodes_by_condition, _ = collect_control_episodes(
                policy,
                {"true_bridge": (bridge, embedding)},
                env,
                NUM_EPISODES,
                ROLLOUT_SEED,
                device,
            )
            real_episodes = episodes_by_condition["true_bridge"]
            shuffled_episodes, shuffle_diagnostics = build_eval_w_shuffle_episodes(
                policy,
                bridge,
                embedding,
                real_episodes,
                seed,
                FINAL_ITERATION,
                ROLLOUT_SEED,
                device,
            )
            model = MODEL_NAMES[model_key]
            samples = part_a_samples(
                policy, bridge, embedding, real_episodes, model, seed, device
            )
            part_a_sample_rows.extend(samples)
            part_a_seed_rows.append(summarize_part_a_seed(samples))

            algorithm = config["config"]["algorithm"]
            stage_rows, real_advantages, shuffled_advantages = part_b_stages(
                policy,
                real_episodes,
                shuffled_episodes,
                model,
                seed,
                float(algorithm["gamma"]),
                float(algorithm["gae_lambda"]),
                float(algorithm["causal_lambda"]),
            )
            part_b_stage_rows.extend(stage_rows)
            part_b_seed_rows.append(summarize_part_b_seed(stage_rows))
            part_b_gradient_rows_output.extend(
                part_b_gradient_rows(
                    policy,
                    real_episodes,
                    shuffled_episodes,
                    real_advantages,
                    shuffled_advantages,
                    model,
                    seed,
                    device,
                )
            )
            shuffle_rows.append({"model": model, "seed": seed, **shuffle_diagnostics})
            print(
                f"Round 2 {model_key} seed={seed}: "
                f"pi_max={part_a_seed_rows[-1]['pi_max_mean']:.6f}, "
                f"final_adv_change={part_b_seed_rows[-1]['final_normalized_relative_l2_mean']:.6f}",
                flush=True,
            )

    part_a_aggregate = aggregate_part_a(part_a_sample_rows, part_a_seed_rows)
    part_b_aggregate = aggregate_part_b(
        part_b_seed_rows, part_b_gradient_rows_output
    )
    part_c_aggregate, part_c_agents, part_c_summary = part_c()

    outputs = {
        "gradient_followup_round2_partA_samples.csv": part_a_sample_rows,
        "gradient_followup_round2_partA_seed_summary.csv": part_a_seed_rows,
        "gradient_followup_round2_partA_summary.csv": part_a_aggregate,
        "gradient_followup_round2_partB_episode_agent_stages.csv": part_b_stage_rows,
        "gradient_followup_round2_partB_seed_summary.csv": part_b_seed_rows,
        "gradient_followup_round2_partB_gradient_cosines.csv": part_b_gradient_rows_output,
        "gradient_followup_round2_partB_summary.csv": part_b_aggregate,
        "gradient_followup_round2_shuffle_diagnostics.csv": shuffle_rows,
        "gradient_followup_round2_partC_aggregate_coefficients.csv": part_c_aggregate,
        "gradient_followup_round2_partC_per_agent_coefficients.csv": part_c_agents,
        "gradient_followup_round2_partC_summary.csv": part_c_summary,
    }
    for name, rows in outputs.items():
        write_csv(RESULT_DIR / name, rows)
    metadata = {
        "final_iteration": FINAL_ITERATION,
        "training_seeds": list(SEEDS),
        "num_episodes": NUM_EPISODES,
        "rollout_seed": ROLLOUT_SEED,
        "device": str(device),
        "bound_tolerance": BOUND_TOL,
        "training_updates": 0,
        "note": (
            "Parts A and B deterministically recreate the same final-checkpoint "
            "rollouts and eval-time W donor permutations as Round 1. Part C reads "
            "the existing final-checkpoint gradient component CSVs."
        ),
    }
    (RESULT_DIR / "gradient_followup_round2_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps({name: len(rows) for name, rows in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
