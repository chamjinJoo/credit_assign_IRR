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

from result_analysis.run_staghunt_gradient_null_controls import (
    DEFAULT_ROOTS,
    collect_control_episodes,
    predictor_outputs,
    select_previous_action_embedding,
    stable_seed,
)
from result_analysis.run_staghunt_policy_gradient_decomposition import (
    discover_runs,
    load_model,
    make_env,
    tensor_observations,
)


RESULT_DIR = ROOT / "result_analysis/gradient_analysis"
DATA_DIR = RESULT_DIR / "log"
SEEDS = (1, 2, 31, 32, 42)
FINAL_ITERATION = 3999
ROLLOUT_SEED = 20260731
NUM_EPISODES = 32
T_95_DF4 = 2.7764451051977987
EPS = 1.0e-12


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size != len(SEEDS):
        raise ValueError(f"Expected {len(SEEDS)} seed values, got {values.size}")
    mean = float(values.mean())
    half_width = float(T_95_DF4 * values.std(ddof=1) / math.sqrt(values.size))
    return mean, half_width


def part1_correlations():
    paths = {
        "PCI-IPPO": DATA_DIR / "gradient_control_seed_summary.csv",
        "PCI-MAPPO": DATA_DIR / "staghunt_pci_mappo_gradient_control_seed_summary.csv",
    }
    pair_rows = []
    summaries = []
    for model, path in paths.items():
        rows = read_csv(path)
        lookup = {
            (row["condition"], int(row["seed"])): float(row["global_delta_cosine"])
            for row in rows
        }
        controls = (
            "naive_predictor",
            "eval_w_shuffle",
            "w_shuffle_bridge",
        )
        for control in controls:
            true_values = np.asarray(
                [lookup[("true_bridge", seed)] for seed in SEEDS],
                dtype=np.float64,
            )
            control_values = np.asarray(
                [lookup[(control, seed)] for seed in SEEDS],
                dtype=np.float64,
            )
            differences = true_values - control_values
            correlation = float(np.corrcoef(true_values, control_values)[0, 1])
            summaries.append(
                {
                    "model": model,
                    "control": control,
                    "pearson_r": correlation,
                    "difference_mean": float(differences.mean()),
                    "difference_ci95_half_width": float(
                        T_95_DF4
                        * differences.std(ddof=1)
                        / math.sqrt(differences.size)
                    ),
                }
            )
            for seed, true_value, control_value, difference in zip(
                SEEDS, true_values, control_values, differences
            ):
                pair_rows.append(
                    {
                        "model": model,
                        "control": control,
                        "seed": seed,
                        "rho_true": float(true_value),
                        "rho_control": float(control_value),
                        "rho_difference": float(difference),
                    }
                )
    return pair_rows, summaries


def final_agent_rows(path, checkpoint_field, checkpoint_value):
    rows = read_csv(path)
    return [
        row
        for row in rows
        if str(row[checkpoint_field]) == str(checkpoint_value)
        and int(row["agent_id"]) >= 0
    ]


def part2_parallel_coefficients():
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
    observations = []
    summaries = []
    for domain_model, path, field, value, causal_lambda in specs:
        rows = final_agent_rows(path, field, value)
        if not rows:
            raise ValueError(f"No final per-agent rows in {path}")
        for row in rows:
            rho = float(row["global_delta_cosine"])
            kappa = float(row["delta_to_global_norm_ratio"])
            coefficient = 1.0 + causal_lambda * kappa * rho
            observations.append(
                {
                    "domain_model": domain_model,
                    "seed": int(row["seed"]),
                    "agent_id": int(row["agent_id"]),
                    "causal_lambda": causal_lambda,
                    "rho": rho,
                    "kappa": kappa,
                    "parallel_coefficient": coefficient,
                    "reversed": int(coefficient < 0.0),
                }
            )
        selected = [
            row for row in observations if row["domain_model"] == domain_model
        ]
        reversed_rows = [row for row in selected if row["reversed"]]
        summaries.append(
            {
                "domain_model": domain_model,
                "causal_lambda": causal_lambda,
                "num_observations": len(selected),
                "num_reversed": len(reversed_rows),
                "reversed_fraction": len(reversed_rows) / len(selected),
                "minimum_parallel_coefficient": min(
                    row["parallel_coefficient"] for row in selected
                ),
                "minimum_rho": min(row["rho"] for row in selected),
                "num_negative_rho": sum(row["rho"] < 0.0 for row in selected),
                "reversed_rho_values": ";".join(
                    f"seed={row['seed']}:agent={row['agent_id']}:rho={row['rho']:.9f}"
                    for row in reversed_rows
                ),
            }
        )
    return observations, summaries


def gain_from_row(row):
    global_norm = float(row["global_grad_norm"])
    direct = float(row["combined_grad_norm"]) / max(global_norm, EPS)
    rho = float(row["global_delta_cosine"])
    kappa = float(row["delta_to_global_norm_ratio"])
    closed = math.sqrt(max(1.0 + 2.0 * kappa * rho + kappa * kappa, 0.0))
    return direct, closed


def part3_gains():
    paths = {
        "PCI-IPPO": DATA_DIR / "gradient_control_components.csv",
        "PCI-MAPPO": DATA_DIR / "staghunt_pci_mappo_gradient_control_components.csv",
    }
    observation_rows = []
    seed_rows = []
    aggregate_rows = []
    max_absolute_error = 0.0
    max_relative_error = 0.0
    for model, path in paths.items():
        rows = read_csv(path)
        for row in rows:
            if int(row["iteration"]) != FINAL_ITERATION or int(row["agent_id"]) < 0:
                continue
            if row["condition"] not in {"true_bridge", "random_isotropic"}:
                continue
            direct, closed = gain_from_row(row)
            absolute_error = abs(direct - closed)
            relative_error = absolute_error / max(abs(direct), EPS)
            max_absolute_error = max(max_absolute_error, absolute_error)
            max_relative_error = max(max_relative_error, relative_error)
            observation_rows.append(
                {
                    "model": model,
                    "condition": row["condition"],
                    "seed": int(row["seed"]),
                    "agent_id": int(row["agent_id"]),
                    "draw": int(row["draw"]),
                    "gain_direct": direct,
                    "gain_closed_form": closed,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                }
            )

        for seed in SEEDS:
            true_values = [
                row["gain_direct"]
                for row in observation_rows
                if row["model"] == model
                and row["condition"] == "true_bridge"
                and row["seed"] == seed
            ]
            null_values = [
                row["gain_direct"]
                for row in observation_rows
                if row["model"] == model
                and row["condition"] == "random_isotropic"
                and row["seed"] == seed
            ]
            if not true_values or not null_values:
                raise ValueError(f"Missing gain rows for {model}, seed {seed}")
            true_mean = float(np.mean(true_values))
            null_mean = float(np.mean(null_values))
            seed_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "true_gain_mean": true_mean,
                    "isotropic_null_gain_mean": null_mean,
                    "true_minus_null_gain": true_mean - null_mean,
                    "num_agents": len(true_values),
                    "num_null_agent_draws": len(null_values),
                }
            )

        model_rows = [row for row in seed_rows if row["model"] == model]
        true_mean, true_ci = mean_ci([row["true_gain_mean"] for row in model_rows])
        null_mean, null_ci = mean_ci(
            [row["isotropic_null_gain_mean"] for row in model_rows]
        )
        difference_mean, difference_ci = mean_ci(
            [row["true_minus_null_gain"] for row in model_rows]
        )
        aggregate_rows.append(
            {
                "model": model,
                "true_gain_mean": true_mean,
                "true_gain_ci95_half_width": true_ci,
                "isotropic_null_gain_mean": null_mean,
                "isotropic_null_gain_ci95_half_width": null_ci,
                "true_minus_null_gain_mean": difference_mean,
                "true_minus_null_gain_ci95_half_width": difference_ci,
            }
        )
    validation = {
        "max_absolute_direct_vs_closed_error": max_absolute_error,
        "max_relative_direct_vs_closed_error": max_relative_error,
    }
    return observation_rows, seed_rows, aggregate_rows, validation


def logit_correction(probabilities, predictions):
    probabilities = probabilities.to(torch.float64)
    predictions = predictions.to(torch.float64)
    weighted_mean = torch.sum(probabilities * predictions)
    denominator = (1.0 - probabilities).clamp_min(1.0e-10)
    identified_contrast = (predictions - weighted_mean) / denominator
    phi = torch.sum(probabilities.square() * identified_contrast)
    return probabilities * (probabilities * identified_contrast - phi)


def part4_seed_diagnostics(model, device):
    runs = discover_runs(DEFAULT_ROOTS[model]["baseline"], SEEDS)
    if tuple(seed for seed, _, _ in runs) != SEEDS:
        raise ValueError(f"Unexpected {model} seeds: {[seed for seed, _, _ in runs]}")
    rows = []
    for training_seed, env_dir, config_path in runs:
        config = json.loads(config_path.read_text())
        checkpoint_path = (
            env_dir / "checkpoints" / f"checkpoint_{FINAL_ITERATION}.pt"
        )
        env = make_env(config)
        policy, bridge, metadata = load_model(
            checkpoint_path, env, config, device
        )
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
        episodes = episodes_by_condition["true_bridge"]

        w_ratios = []
        level_difference_sum = 0.0
        level_reference_sum = 0.0
        action_contrast_ratios = []
        action_contrast_reference_norms = []
        action_contrast_difference_norms = []
        contrast_zero_denominators = 0
        contrast_below_1e8 = 0
        contrast_below_1e6 = 0
        contrast_below_1e4 = 0
        w_zero_denominators = 0
        total_recipients = 0
        shuffled_recipients = 0
        skipped_recipients = 0
        max_timestep = max(len(episode["steps"]) for episode in episodes)

        for timestep in range(max_timestep):
            pool = [
                episode_id
                for episode_id, episode in enumerate(episodes)
                if timestep < len(episode["steps"])
            ]
            if len(pool) < 2:
                donor_ids = list(pool)
                skipped_recipients += len(pool) * policy.num_agents
            else:
                generator = np.random.default_rng(
                    stable_seed(
                        ROLLOUT_SEED,
                        training_seed,
                        FINAL_ITERATION,
                        timestep,
                        "eval_w",
                    )
                )
                shift = int(generator.integers(1, len(pool)))
                donor_ids = list(np.roll(np.asarray(pool), shift))
                shuffled_recipients += len(pool) * policy.num_agents

            for recipient_id, donor_id in zip(pool, donor_ids):
                recipient = episodes[recipient_id]["steps"][timestep]
                donor = episodes[donor_id]["steps"][timestep]
                observations = tensor_observations(
                    recipient["observations"], device
                )
                donor_observations = tensor_observations(
                    donor["observations"], device
                )
                hidden_states = {
                    agent_id: value.to(device)
                    for agent_id, value in recipient["hidden_states"].items()
                }
                previous_actions = {
                    agent_id: value.to(device)
                    for agent_id, value in recipient["previous_actions"].items()
                }
                probabilities = {
                    agent_id: value.to(device)
                    for agent_id, value in recipient[
                        "action_probabilities"
                    ].items()
                }
                real_predictions = predictor_outputs(
                    policy,
                    bridge,
                    observations,
                    hidden_states,
                    previous_actions,
                    embedding,
                )
                shuffled_predictions = predictor_outputs(
                    policy,
                    bridge,
                    observations,
                    hidden_states,
                    previous_actions,
                    embedding,
                    w_observations=donor_observations,
                )

                for focal_agent in range(policy.num_agents):
                    other_agents = [
                        agent_id
                        for agent_id in range(policy.num_agents)
                        if agent_id != focal_agent
                    ]
                    w_real = torch.cat(
                        [observations[agent_id] for agent_id in other_agents]
                    )
                    w_donor = torch.cat(
                        [
                            donor_observations[agent_id]
                            for agent_id in other_agents
                        ]
                    )
                    w_norm = float(torch.linalg.vector_norm(w_real).item())
                    if w_norm <= EPS:
                        w_zero_denominators += 1
                    w_ratios.append(
                        float(
                            torch.linalg.vector_norm(w_real - w_donor).item()
                            / max(w_norm, EPS)
                        )
                    )

                    real = real_predictions[focal_agent]
                    shuffled = shuffled_predictions[focal_agent]
                    level_difference_sum += float(
                        torch.abs(real - shuffled).sum().item()
                    )
                    level_reference_sum += float(torch.abs(real).sum().item())

                    real_correction = logit_correction(
                        probabilities[focal_agent], real
                    )
                    shuffled_correction = logit_correction(
                        probabilities[focal_agent], shuffled
                    )
                    contrast_denominator = float(
                        torch.linalg.vector_norm(
                            real_correction, ord=1
                        ).item()
                    )
                    contrast_difference = float(
                        torch.linalg.vector_norm(
                            real_correction - shuffled_correction,
                            ord=1,
                        ).item()
                    )
                    action_contrast_reference_norms.append(contrast_denominator)
                    action_contrast_difference_norms.append(contrast_difference)
                    if contrast_denominator <= EPS:
                        contrast_zero_denominators += 1
                    if contrast_denominator < 1.0e-8:
                        contrast_below_1e8 += 1
                    if contrast_denominator < 1.0e-6:
                        contrast_below_1e6 += 1
                    if contrast_denominator < 1.0e-4:
                        contrast_below_1e4 += 1
                    action_contrast_ratios.append(
                        contrast_difference / max(contrast_denominator, EPS)
                    )
                    total_recipients += 1

        rows.append(
            {
                "model": "PCI-IPPO" if model == "ippo" else "PCI-MAPPO",
                "seed": training_seed,
                "w_intervention_strength": float(np.mean(w_ratios)),
                "bridge_level_sensitivity": (
                    level_difference_sum / max(level_reference_sum, EPS)
                ),
                "action_contrast_sensitivity": float(
                    np.mean(action_contrast_ratios)
                ),
                "action_contrast_sensitivity_median": float(
                    np.median(action_contrast_ratios)
                ),
                "action_contrast_sensitivity_p90": float(
                    np.quantile(action_contrast_ratios, 0.9)
                ),
                "action_contrast_sensitivity_ratio_of_means": (
                    float(np.mean(action_contrast_difference_norms))
                    / max(float(np.mean(action_contrast_reference_norms)), EPS)
                ),
                "action_contrast_reference_l1_mean": float(
                    np.mean(action_contrast_reference_norms)
                ),
                "action_contrast_difference_l1_mean": float(
                    np.mean(action_contrast_difference_norms)
                ),
                "num_recipient_agent_samples": total_recipients,
                "actual_shuffle_fraction": (
                    shuffled_recipients / max(total_recipients, 1)
                ),
                "skipped_shuffle_fraction": (
                    skipped_recipients / max(total_recipients, 1)
                ),
                "w_zero_denominator_fraction": (
                    w_zero_denominators / max(total_recipients, 1)
                ),
                "contrast_zero_denominator_fraction": (
                    contrast_zero_denominators / max(total_recipients, 1)
                ),
                "contrast_below_1e8_fraction": (
                    contrast_below_1e8 / max(total_recipients, 1)
                ),
                "contrast_below_1e6_fraction": (
                    contrast_below_1e6 / max(total_recipients, 1)
                ),
                "contrast_below_1e4_fraction": (
                    contrast_below_1e4 / max(total_recipients, 1)
                ),
            }
        )
        print(
            f"Part 4 {model} seed={training_seed}: "
            f"a={rows[-1]['w_intervention_strength']:.6f}, "
            f"b={rows[-1]['bridge_level_sensitivity']:.6f}, "
            f"c={rows[-1]['action_contrast_sensitivity']:.6f}",
            flush=True,
        )
    return rows


def part4_aggregate(seed_rows):
    output = []
    for model in ("PCI-IPPO", "PCI-MAPPO"):
        rows = [row for row in seed_rows if row["model"] == model]
        result = {"model": model, "n_seeds": len(rows)}
        for metric in (
            "w_intervention_strength",
            "bridge_level_sensitivity",
            "action_contrast_sensitivity",
            "action_contrast_sensitivity_median",
            "action_contrast_sensitivity_ratio_of_means",
            "action_contrast_reference_l1_mean",
            "action_contrast_difference_l1_mean",
            "contrast_below_1e8_fraction",
            "contrast_below_1e6_fraction",
            "contrast_below_1e4_fraction",
        ):
            mean, ci = mean_ci([row[metric] for row in rows])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_ci95_half_width"] = ci
        output.append(result)
    return output


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    part1_pairs, part1_summary = part1_correlations()
    part2_observations, part2_summary = part2_parallel_coefficients()
    (
        part3_observations,
        part3_seed_rows,
        part3_summary,
        part3_validation,
    ) = part3_gains()

    device = torch.device("cpu")
    part4_seed_rows = []
    for model in ("ippo", "mappo"):
        part4_seed_rows.extend(part4_seed_diagnostics(model, device))
    part4_summary = part4_aggregate(part4_seed_rows)

    outputs = {
        "gradient_followup_part1_pairs.csv": part1_pairs,
        "gradient_followup_part1_summary.csv": part1_summary,
        "gradient_followup_part2_observations.csv": part2_observations,
        "gradient_followup_part2_summary.csv": part2_summary,
        "gradient_followup_part3_observations.csv": part3_observations,
        "gradient_followup_part3_seed_summary.csv": part3_seed_rows,
        "gradient_followup_part3_summary.csv": part3_summary,
        "gradient_followup_part4_seed_metrics.csv": part4_seed_rows,
        "gradient_followup_part4_summary.csv": part4_summary,
    }
    for name, rows in outputs.items():
        write_csv(RESULT_DIR / name, rows)

    metadata = {
        "final_iteration": FINAL_ITERATION,
        "training_seeds": list(SEEDS),
        "num_episodes": NUM_EPISODES,
        "rollout_seed": ROLLOUT_SEED,
        "device": str(device),
        "part3_validation": part3_validation,
        "part4_rollout_note": (
            "The original paired W tensors were not serialized. The script "
            "deterministically recreates the same final-checkpoint rollouts and "
            "eval-W donor permutations from the original seeds."
        ),
    }
    (RESULT_DIR / "gradient_followup_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps({
        "part1": part1_summary,
        "part2": part2_summary,
        "part3": part3_summary,
        "part3_validation": part3_validation,
        "part4": part4_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
