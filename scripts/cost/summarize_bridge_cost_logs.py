#!/usr/bin/env python3
"""Build bridge-cost timing CSVs and a markdown report from Slurm logs."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

NAME_RE = re.compile(
    r"slurm-cost_(?P<env>[^_]+)_(?P<variant>vanilla|pci)_(?P<backbone>ippo|mappo)"
    r"(?:_s(?P<seed>\d+))?-(?P<jobid>\d+)\.out$"
)
METRIC_RE = re.compile(r"(?:^|\s{2,})(?P<key>[A-Za-z0-9_/]+):\s+(?P<value>[-+0-9.eE]+)(?:\s|$)")
META_RE = re.compile(r"^\[(?P<section>cost_meta|cost_cuda)\]\s+(?P<key>[^=]+)=(?P<value>.*)$")
GPU_RE = re.compile(r"^\[cost_gpu\]\s+(?P<value>.*)$")
TRAINER_RE = {
    "num_agents": re.compile(r"Number of agents:\s+(?P<value>\d+)"),
    "horizon": re.compile(r"Steps per iteration:\s+(?P<value>\d+)"),
    "max_iterations": re.compile(r"Max iterations:\s+(?P<value>\d+)"),
    "device_logged": re.compile(r"Device:\s+(?P<value>\S+)"),
}

DEFAULT_METRICS = [
    "iteration_time_sec",
    "rollout_time_sec",
    "algorithm_update_time_sec",
    "samples_per_algorithm_update_sec",
    "causal_update_time_sec",
    "causal_prepare_time_sec",
    "causal_gram_time_sec",
    "bridge_solve_time_sec",
    "bridge_net_time_sec",
    "outcome_bridge_train_time_sec",
    "naive_predictor_train_time_sec",
    "causal_num_minibatches",
    "causal_samples",
]
TIME_METRICS = [
    "algorithm_update_time_sec",
    "causal_update_time_sec",
    "causal_gram_time_sec",
    "bridge_solve_time_sec",
    "bridge_net_time_sec",
    "outcome_bridge_train_time_sec",
]
DERIVED_FIELDS = [
    "backbone_relative_overhead_pct",
    "causal_pipeline_share_pct",
    "gram_construction_share_pct",
    "solve_share_pct",
    "gram_related_share_pct",
    "bridge_net_share_pct",
    "preprocessing_share_pct",
    "bridge_optimizer_share_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["slurms/logs"])
    parser.add_argument("--drop-first", type=int, default=10)
    parser.add_argument("--output-dir", default="result_analysis/bridge_cost")
    parser.add_argument("--job-range", default=None)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--solve-benchmark-csv", default=None, help="Optional JAX shape-matched solve benchmark CSV")
    return parser.parse_args()


def iter_logs(paths: list[str]):
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(path.glob("slurm-cost_*.out"))
        elif path.is_file():
            yield path


def parse_config_value(path: str, key: str):
    if not path:
        return ""
    cfg = Path(path)
    if not cfg.exists():
        return ""
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*([^#\n]+)")
    for line in cfg.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return match.group(1).strip().strip('"\'')
    return ""


def finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def stats(values: list[float]) -> dict[str, float]:
    clean = finite(values)
    if not clean:
        return {"count": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def fmt_mean_std(mean: float, std: float, suffix: str = "") -> str:
    if not math.isfinite(mean):
        return "n/a"
    if not math.isfinite(std):
        std = 0.0
    return f"{mean:.3f} +/- {std:.3f}{suffix}"


def parse_log(path: Path, metrics: list[str]) -> dict:
    name = NAME_RE.search(path.name)
    if not name:
        raise ValueError(f"unrecognized log name: {path}")
    condition = name.groupdict()
    condition["seed"] = condition.get("seed") or ""
    meta = {}
    gpu_lines = []
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        meta_match = META_RE.search(line)
        if meta_match:
            meta[meta_match.group("key").strip()] = meta_match.group("value").strip()
            continue
        gpu_match = GPU_RE.search(line)
        if gpu_match:
            gpu_lines.append(gpu_match.group("value").strip())
            continue
        for key, pattern in TRAINER_RE.items():
            trainer_match = pattern.search(line)
            if trainer_match and key not in meta:
                meta[key] = trainer_match.group("value")
        metric_match = METRIC_RE.search(line)
        if metric_match:
            key = metric_match.group("key")
            if key in metrics:
                values_by_metric[key].append(float(metric_match.group("value")))

    cuda_available = meta.get("torch_cuda_available", "") == "True"
    training_completed = "Training Completed!" in text
    run_done = "[run] done" in text
    exit0 = training_completed and run_done
    gpu_model = meta.get("torch_device_name", "")
    if not gpu_model and gpu_lines:
        parts = [p.strip() for p in gpu_lines[0].split(",")]
        if len(parts) >= 2:
            gpu_model = parts[1]

    config = meta.get("config", "")
    ppo_epochs = parse_config_value(config, "ppo_epochs")
    num_batches = parse_config_value(config, "num_batches")
    bridge_epoch = parse_config_value(config, "bridge_epoch")
    causal_mini_batch_size = parse_config_value(config, "causal_mini_batch_size")

    return {
        **condition,
        "path": path,
        "meta": meta,
        "gpu_model": gpu_model,
        "cuda_available": cuda_available,
        "exit0": exit0,
        "ppo_epochs": ppo_epochs,
        "num_batches": num_batches,
        "bridge_epoch": bridge_epoch,
        "causal_mini_batch_size": causal_mini_batch_size,
        "values_by_metric": values_by_metric,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_rows(records: list[dict], drop_first: int):
    raw_rows = []
    seed_metric_rows = []
    seed_wide = {}

    for rec in records:
        base = {k: rec[k] for k in ("env", "variant", "backbone", "seed", "jobid")}
        for metric, values in rec["values_by_metric"].items():
            for i, value in enumerate(values):
                raw_rows.append({
                    **base,
                    "metric": metric,
                    "logged_index": i,
                    "dropped_warmup": int(i < drop_first),
                    "value": value,
                    "source": str(rec["path"]),
                })
            used = values[drop_first:] if len(values) > drop_first else values
            metric_stats = stats(used)
            seed_metric_rows.append({
                **base,
                "metric": metric,
                **metric_stats,
                "source": str(rec["path"]),
            })
            seed_wide.setdefault(tuple(base.items()), {**base})[metric] = metric_stats["mean"]
            seed_wide[tuple(base.items())][metric + "_count"] = metric_stats["count"]

        wide = seed_wide.setdefault(tuple(base.items()), {**base})
        wide.update({
            "gpu_model": rec["gpu_model"],
            "cuda_available": rec["cuda_available"],
            "exit0": rec["exit0"],
            "n_rollout_threads": rec["meta"].get("n_rollout_threads", ""),
            "horizon": rec["meta"].get("horizon") or rec["meta"].get("max_steps", ""),
            "max_iterations": rec["meta"].get("max_iterations", ""),
            "ppo_epochs": rec["ppo_epochs"],
            "num_batches": rec["num_batches"],
            "bridge_epoch": rec["bridge_epoch"],
            "causal_mini_batch_size": rec["causal_mini_batch_size"],
            "num_agents": rec["meta"].get("num_agents", ""),
            "source": str(rec["path"]),
        })

    seed_wide_rows = list(seed_wide.values())
    return raw_rows, seed_metric_rows, seed_wide_rows



def load_solve_benchmarks(path: str | None) -> dict[tuple[str, str, str], dict]:
    if not path:
        return {}
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    out: dict[tuple[str, str, str], dict] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("env", ""), row.get("backbone", ""), row.get("seed", ""))
            out[key] = row
            out[(key[0], key[1], "")] = row
    return out


def as_float(value, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def derive_pair_rows(seed_wide_rows: list[dict], solve_benchmarks: dict[tuple[str, str, str], dict] | None = None) -> list[dict]:
    solve_benchmarks = solve_benchmarks or {}
    by_pair = defaultdict(dict)
    for row in seed_wide_rows:
        by_pair[(row["env"], row["backbone"], row["seed"])][row["variant"]] = row

    pair_rows = []
    for (env, backbone, seed), variants in sorted(by_pair.items()):
        if "vanilla" not in variants or "pci" not in variants:
            continue
        vanilla = variants["vanilla"]
        pci = variants["pci"]
        vanilla_update = as_float(vanilla.get("algorithm_update_time_sec"))
        pci_update = as_float(pci.get("algorithm_update_time_sec"))
        causal = as_float(pci.get("causal_update_time_sec"), 0.0)
        prepare = as_float(pci.get("causal_prepare_time_sec"), 0.0)
        gram = as_float(pci.get("causal_gram_time_sec"), 0.0)
        bridge = as_float(pci.get("outcome_bridge_train_time_sec"), 0.0)
        solve = as_float(pci.get("bridge_solve_time_sec"))
        solve_source = "in_loop_timer" if math.isfinite(solve) and solve > 0.0 else ""
        bench = solve_benchmarks.get((env, backbone, seed)) or solve_benchmarks.get((env, backbone, ""))
        if (not math.isfinite(solve) or solve <= 0.0) and bench:
            solve = as_float(bench.get("bridge_solve_time_sec"))
            solve_source = bench.get("source", "jax_shape_microbenchmark")
        if not math.isfinite(solve):
            solve = 0.0
        bridge_net = as_float(pci.get("bridge_net_time_sec"))
        if not math.isfinite(bridge_net):
            bridge_net = max(0.0, bridge - solve)
        gram_related = gram + solve
        num_agents = as_float(pci.get("num_agents"), 0.0)
        causal_samples = as_float(pci.get("causal_samples"), 0.0)
        causal_minibatches = as_float(pci.get("causal_num_minibatches"), 0.0)
        gram_batch_size = math.nan
        if num_agents > 0 and causal_minibatches > 0:
            gram_batch_size = causal_samples / num_agents / causal_minibatches
        overhead = math.nan
        if math.isfinite(vanilla_update) and vanilla_update > 0:
            overhead = 100.0 * (pci_update - vanilla_update) / vanilla_update
        share = lambda value: 100.0 * value / pci_update if math.isfinite(pci_update) and pci_update > 0 else math.nan
        pair_rows.append({
            "env": env,
            "backbone": backbone,
            "seed": seed,
            "vanilla_jobid": vanilla.get("jobid", ""),
            "pci_jobid": pci.get("jobid", ""),
            "gpu_model": pci.get("gpu_model") or vanilla.get("gpu_model", ""),
            "cuda_available": pci.get("cuda_available") and vanilla.get("cuda_available"),
            "exit0": pci.get("exit0") and vanilla.get("exit0"),
            "n_rollout_threads": pci.get("n_rollout_threads", ""),
            "horizon": pci.get("horizon", ""),
            "num_agents": pci.get("num_agents", ""),
            "ppo_epochs": pci.get("ppo_epochs") or vanilla.get("ppo_epochs", ""),
            "num_batches": pci.get("num_batches") or vanilla.get("num_batches", ""),
            "bridge_epoch": pci.get("bridge_epoch", ""),
            "causal_mini_batch_size": pci.get("causal_mini_batch_size", ""),
            "causal_num_minibatches": causal_minibatches,
            "gram_batch_size": gram_batch_size,
            "logged_updates_after_drop": int(min(
                vanilla.get("algorithm_update_time_sec_count", 0) or 0,
                pci.get("algorithm_update_time_sec_count", 0) or 0,
            )),
            "vanilla_algorithm_update_time_sec": vanilla_update,
            "pci_algorithm_update_time_sec": pci_update,
            "causal_pipeline_time_sec": causal,
            "preprocessing_time_sec": prepare,
            "gram_time_sec": gram,
            "bridge_solve_time_sec": solve,
            "bridge_solve_source": solve_source,
            "gram_related_time_sec": gram_related,
            "bridge_net_time_sec": bridge_net,
            "bridge_optimizer_time_sec": bridge,
            "backbone_relative_overhead_pct": overhead,
            "causal_pipeline_share_pct": share(causal),
            "gram_construction_share_pct": share(gram),
            "solve_share_pct": share(solve),
            "gram_related_share_pct": share(gram_related),
            "bridge_net_share_pct": share(bridge_net),
            "preprocessing_share_pct": share(prepare),
            "bridge_optimizer_share_pct": share(bridge),
        })
    return pair_rows


def aggregate_pair_rows(pair_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in pair_rows:
        grouped[(row["env"], row["backbone"])].append(row)
    aggregate_rows = []
    for (env, backbone), rows in sorted(grouped.items()):
        out = {
            "env": env,
            "backbone": backbone,
            "seeds": " ".join(str(r["seed"]) for r in rows),
            "n_seeds": len(rows),
            "gpu_model": "; ".join(sorted({str(r.get("gpu_model", "")) for r in rows if r.get("gpu_model")})),
            "all_cuda_available": all(bool(r.get("cuda_available")) for r in rows),
            "all_exit0": all(bool(r.get("exit0")) for r in rows),
            "solve_sources": "; ".join(sorted({str(r.get("bridge_solve_source", "")) for r in rows if r.get("bridge_solve_source")})),
            "n_rollout_threads": "; ".join(sorted({str(r.get("n_rollout_threads", "")) for r in rows})),
            "horizon": "; ".join(sorted({str(r.get("horizon", "")) for r in rows})),
            "num_agents": "; ".join(sorted({str(r.get("num_agents", "")) for r in rows})),
            "ppo_epochs": "; ".join(sorted({str(r.get("ppo_epochs", "")) for r in rows})),
            "num_batches": "; ".join(sorted({str(r.get("num_batches", "")) for r in rows})),
            "bridge_epoch": "; ".join(sorted({str(r.get("bridge_epoch", "")) for r in rows})),
            "causal_mini_batch_size": "; ".join(sorted({str(r.get("causal_mini_batch_size", "")) for r in rows})),
            "logged_updates_after_drop_min": min(int(r.get("logged_updates_after_drop", 0)) for r in rows),
        }
        for field in [
            "gram_batch_size",
            "vanilla_algorithm_update_time_sec",
            "pci_algorithm_update_time_sec",
            "causal_pipeline_time_sec",
            "preprocessing_time_sec",
            "gram_time_sec",
            "bridge_solve_time_sec",
            "gram_related_time_sec",
            "bridge_net_time_sec",
            "bridge_optimizer_time_sec",
            *DERIVED_FIELDS,
        ]:
            s = stats([as_float(r.get(field)) for r in rows])
            out[field + "_mean"] = s["mean"]
            out[field + "_std"] = s["std"]
        aggregate_rows.append(out)
    return aggregate_rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def fmt_report_value(row: dict, field: str, suffix: str = "") -> str:
    mean = as_float(row.get(field + "_mean"))
    std = as_float(row.get(field + "_std"), 0.0)
    n = int(row.get("n_seeds", 1) or 1)
    if not math.isfinite(mean):
        return "n/a"
    if n <= 1:
        return f"{mean:.3f}{suffix}"
    return fmt_mean_std(mean, std, suffix)


def build_report(job_range: str, aggregate_rows: list[dict], pair_rows: list[dict], drop_first: int) -> str:
    settings_rows = []
    for row in aggregate_rows:
        settings_rows.append([
            row["env"],
            row["backbone"],
            row["seeds"],
            fmt_report_value(row, "gram_batch_size"),
            row["n_rollout_threads"],
            row["horizon"],
            row["num_agents"],
            row["ppo_epochs"],
            row["gpu_model"],
            str(row["logged_updates_after_drop_min"]),
        ])

    timing_rows = []
    for row in aggregate_rows:
        timing_rows.append([
            row["env"],
            row["backbone"],
            fmt_report_value(row, "vanilla_algorithm_update_time_sec"),
            fmt_report_value(row, "pci_algorithm_update_time_sec"),
            fmt_report_value(row, "causal_pipeline_time_sec"),
            fmt_report_value(row, "preprocessing_time_sec"),
            fmt_report_value(row, "gram_time_sec"),
            fmt_report_value(row, "bridge_solve_time_sec"),
            fmt_report_value(row, "gram_related_time_sec"),
            fmt_report_value(row, "bridge_net_time_sec"),
            fmt_report_value(row, "bridge_optimizer_time_sec"),
            row.get("solve_sources", ""),
        ])

    share_rows = []
    for row in aggregate_rows:
        share_rows.append([
            row["env"],
            row["backbone"],
            fmt_report_value(row, "backbone_relative_overhead_pct", "%"),
            fmt_report_value(row, "causal_pipeline_share_pct", "%"),
            fmt_report_value(row, "preprocessing_share_pct", "%"),
            fmt_report_value(row, "gram_construction_share_pct", "%"),
            fmt_report_value(row, "solve_share_pct", "%"),
            fmt_report_value(row, "gram_related_share_pct", "%"),
            fmt_report_value(row, "bridge_net_share_pct", "%"),
        ])

    gram_rows = []
    by_batch = defaultdict(list)
    for row in pair_rows:
        batch = as_float(row.get("gram_batch_size"))
        if math.isfinite(batch):
            by_batch[(row["env"], row["backbone"], round(batch, 3))].append(as_float(row["gram_related_time_sec"]))
    for (env, backbone, batch_size), vals in sorted(by_batch.items()):
        s = stats(vals)
        if len(vals) <= 1:
            value = f"{s['mean']:.3f}" if math.isfinite(s["mean"]) else "n/a"
        else:
            value = fmt_mean_std(s["mean"], s["std"])
        gram_rows.append([env, backbone, f"{batch_size:.0f}", value])

    negative = [
        f"{row['env']}/{row['backbone']} ({row['backbone_relative_overhead_pct_mean']:.2f}%)"
        for row in aggregate_rows
        if math.isfinite(row["backbone_relative_overhead_pct_mean"]) and row["backbone_relative_overhead_pct_mean"] < 0.0
    ]
    failed = [f"{row['env']}/{row['backbone']}" for row in aggregate_rows if not row["all_cuda_available"] or not row["all_exit0"]]
    gram_related_values = [as_float(row.get("gram_related_share_pct_mean")) for row in aggregate_rows]
    gram_related_values = [v for v in gram_related_values if math.isfinite(v)]
    bridge_net_values = [as_float(row.get("bridge_net_share_pct_mean")) for row in aggregate_rows]
    bridge_net_values = [v for v in bridge_net_values if math.isfinite(v)]
    if gram_related_values:
        gram_related_sentence = f"Gram construction plus kernel-ridge solve accounts for {min(gram_related_values):.1f}%--{max(gram_related_values):.1f}% of a PCI update across these single-seed conditions."
    else:
        gram_related_sentence = "Gram-related share could not be computed from the parsed logs."
    if bridge_net_values:
        net_sentence = f"The bridge network component accounts for {min(bridge_net_values):.1f}%--{max(bridge_net_values):.1f}% of a PCI update."
    else:
        net_sentence = "The bridge network component could not be computed."
    interpretation = (
        "These results are single-seed, indicative measurements with the first "
        f"{drop_first} logged updates dropped for warmup. "
        "The JAX solve cannot be isolated with a host timer inside the jitted gradient function without changing the training computation, so JAX solve entries are shape-matched microbenchmarks; Torch StagHunt solve entries are in-loop timers. "
        + gram_related_sentence + " " + net_sentence
    )
    if negative:
        interpretation += " Conditions with negative PCI-vs-vanilla overhead are timing noise or config-path interaction, not evidence of negative cost: " + ", ".join(negative) + "."
    if failed:
        interpretation += " WARNING: at least one aggregate includes a run that did not satisfy the parsed exit/CUDA checks: " + ", ".join(failed) + "."

    return "\n\n".join([
        f"# PCI bridge computational cost with solve split ({job_range})",
        f"Single-seed, indicative results. Post-warmup means drop the first {drop_first} logged updates from each seed/log. Existing in-loop timers use device synchronization before each recorded timer stops.",
        "## Settings",
        markdown_table(["env", "backbone", "seeds", "Gram B", "n_envs", "horizon", "agents", "PPO epochs", "GPU", "updates"], settings_rows),
        "## Per-condition timing seconds",
        markdown_table(["env", "backbone", "vanilla update", "PCI update", "causal pipeline", "preprocessing", "Gram construction", "kernel-ridge solve", "Gram-related", "bridge net", "bridge optimizer", "solve source"], timing_rows),
        "## PCI update shares",
        markdown_table(["env", "backbone", "backbone-relative overhead", "causal pipeline", "preprocessing", "Gram construction", "solve", "Gram-related", "bridge net"], share_rows),
        "## Gram-related time vs batch size",
        markdown_table(["env", "backbone", "Gram B", "gram_related_time_sec"], gram_rows) if gram_rows else "Not measured beyond the main conditions.",
        "## Interpretation",
        interpretation,
    ]) + "\n"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        for path in list(out_dir.glob("*.md")) + list(out_dir.glob("*.csv")):
            path.unlink()

    records = [parse_log(path, DEFAULT_METRICS) for path in iter_logs(args.paths) if NAME_RE.search(path.name)]
    if not records:
        print("no matching slurm-cost logs found", file=sys.stderr)
        return 2
    job_ids = sorted(int(rec["jobid"]) for rec in records)
    job_range = args.job_range or f"{job_ids[0]}_{job_ids[-1]}"

    raw_rows, seed_metric_rows, seed_wide_rows = build_rows(records, args.drop_first)
    solve_benchmarks = load_solve_benchmarks(args.solve_benchmark_csv)
    pair_rows = derive_pair_rows(seed_wide_rows, solve_benchmarks)
    aggregate_rows = aggregate_pair_rows(pair_rows)

    write_csv(out_dir / f"bridge_cost_raw_updates_{job_range}.csv", raw_rows,
              ["env", "variant", "backbone", "seed", "jobid", "metric", "logged_index", "dropped_warmup", "value", "source"])
    write_csv(out_dir / f"bridge_cost_seed_metrics_{job_range}.csv", seed_metric_rows,
              ["env", "variant", "backbone", "seed", "jobid", "metric", "count", "mean", "std", "min", "max", "source"])
    wide_fields = ["env", "variant", "backbone", "seed", "jobid", "gpu_model", "cuda_available", "exit0", "n_rollout_threads", "horizon", "max_iterations", "num_agents", "ppo_epochs", "num_batches", "bridge_epoch", "causal_mini_batch_size", "source"]
    for metric in DEFAULT_METRICS:
        wide_fields.extend([metric, metric + "_count"])
    write_csv(out_dir / f"bridge_cost_seed_summary_{job_range}.csv", seed_wide_rows, wide_fields)
    pair_fields = [
        "env", "backbone", "seed", "vanilla_jobid", "pci_jobid", "gpu_model", "cuda_available", "exit0",
        "n_rollout_threads", "horizon", "num_agents", "ppo_epochs", "num_batches", "bridge_epoch", "causal_mini_batch_size",
        "causal_num_minibatches", "gram_batch_size", "logged_updates_after_drop", "vanilla_algorithm_update_time_sec", "pci_algorithm_update_time_sec",
        "causal_pipeline_time_sec", "preprocessing_time_sec", "gram_time_sec", "bridge_solve_time_sec", "bridge_solve_source",
        "gram_related_time_sec", "bridge_net_time_sec", "bridge_optimizer_time_sec", *DERIVED_FIELDS,
    ]
    write_csv(out_dir / f"bridge_cost_seed_pairs_{job_range}.csv", pair_rows, pair_fields)
    aggregate_fields = list(aggregate_rows[0].keys()) if aggregate_rows else []
    write_csv(out_dir / f"bridge_cost_condition_summary_{job_range}.csv", aggregate_rows, aggregate_fields)
    report = build_report(job_range, aggregate_rows, pair_rows, args.drop_first)
    (out_dir / f"bridge_cost_report_{job_range}.md").write_text(report)
    print(out_dir / f"bridge_cost_report_{job_range}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
