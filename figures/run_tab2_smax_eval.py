#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd


def resolve_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "main.py").is_file() and (candidate / "configs").is_dir():
            return candidate
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = resolve_repo_root()
    parser = argparse.ArgumentParser(description="Evaluate SMAX final checkpoints for Table 2.")
    parser.add_argument("--checkpoints-dir", type=Path, default=root / "runs" / "checkpoints_smax")
    parser.add_argument("--figures-dir", type=Path, default=root / "figures")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-choice", choices=("best", "final"), default="final")
    parser.add_argument("--eval-seed-start", type=int, default=10001)
    parser.add_argument("--eval-seed-count", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-seed-batch-size", type=int, default=16)
    parser.add_argument("--duplicate-policy", choices=("error", "latest", "best_metric", "all"), default="latest")
    parser.add_argument("--maps", nargs="*", help="Optional subset of maps to evaluate.")
    parser.add_argument("--smoke-only", action="store_true", help="Run one train seed x two eval seeds only.")
    parser.add_argument(
        "--force-reevaluate",
        action="store_true",
        help="Ignore cached per-run evaluation results and overwrite them with fresh rollouts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    figures_dir = args.figures_dir.resolve()
    checkpoints_dir = args.checkpoints_dir.resolve()

    for path in (repo_root, figures_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    import tab2_smax_checkpoint_eval_utils as scu

    scu = importlib.reload(scu)

    eval_seeds = list(range(args.eval_seed_start, args.eval_seed_start + args.eval_seed_count))
    print(f"Repo root: {repo_root}")
    print(f"Checkpoint root: {checkpoints_dir}")
    print(f"Checkpoint choice: {args.checkpoint_choice}")
    print(f"Eval seeds: {eval_seeds[0]}..{eval_seeds[-1]} ({len(eval_seeds)})")
    print(f"Use cached run evaluations: {not args.force_reevaluate}")

    scu.bootstrap_project(args.device)
    runs_df = scu.discover_runs(checkpoints_dir, project_filter=None)
    if runs_df.empty:
        raise RuntimeError(f"No runs were discovered under {checkpoints_dir}")

    checkpoint_column = f"{args.checkpoint_choice}_checkpoint"
    missing_checkpoint_runs = runs_df[~runs_df[checkpoint_column].map(lambda path: Path(path).exists())].copy()
    if not missing_checkpoint_runs.empty:
        print(f"Skipping {len(missing_checkpoint_runs)} runs without {args.checkpoint_choice} checkpoints:")
        print(
            missing_checkpoint_runs[
                ["map_name", "constraint_key", "model_label", "seed", "run_dir", checkpoint_column]
            ].to_string(index=False)
        )
    runs_df = runs_df[runs_df[checkpoint_column].map(lambda path: Path(path).exists())].reset_index(drop=True)
    print(f"Discovered evaluable runs: {len(runs_df)}")
    print(runs_df.groupby(["map_name", "constraint_key", "model_label"]).size().to_string())

    target_maps = args.maps or scu.ordered_map_names(runs_df["map_name"].dropna().unique().tolist())
    target_models = scu.ordered_model_labels(runs_df["model_label"].dropna().unique().tolist())
    print(f"Target maps: {target_maps}")
    print(f"Target models: {target_models}")

    if args.smoke_only:
        smoke_eval_seeds = eval_seeds[:2]
        smoke_row = runs_df.sort_values(["map_name", "constraint_key", "model_label", "seed", "run_dir"]).iloc[0]
        smoke_runs = runs_df[
            (runs_df["map_name"] == smoke_row["map_name"])
            & (runs_df["constraint_key"] == smoke_row["constraint_key"])
            & (runs_df["model_label"] == smoke_row["model_label"])
            & (runs_df["seed"] == smoke_row["seed"])
        ].copy()
        result = scu.evaluate_map_constraint(
            smoke_runs,
            smoke_row["map_name"],
            smoke_row["constraint_key"],
            checkpoint_choice=args.checkpoint_choice,
            device=args.device,
            eval_episodes=args.eval_episodes,
            eval_seeds=smoke_eval_seeds,
            eval_seed_batch_size=min(args.eval_seed_batch_size, len(smoke_eval_seeds)),
            duplicate_policy=args.duplicate_policy,
            use_cache=not args.force_reevaluate,
        )
        print(result["run_results"].to_string(index=False))
        print(result["seed_summary"].to_string(index=False))
        print(result["model_summary"].to_string(index=False))
        return

    map_results = {}
    for map_name in target_maps:
        result = scu.evaluate_map(
            runs_df,
            map_name,
            checkpoint_choice=args.checkpoint_choice,
            device=args.device,
            eval_episodes=args.eval_episodes,
            eval_seeds=eval_seeds,
            eval_seed_batch_size=args.eval_seed_batch_size,
            model_order=target_models,
            duplicate_policy=args.duplicate_policy,
            use_cache=not args.force_reevaluate,
        )
        if result["model_summary"].empty:
            print(f"Skipping {map_name}: no discovered runs")
            continue
        map_results[map_name] = result
        print(f"\n=== {map_name} ===")
        print(result["model_summary"].to_string(index=False))

    if not map_results:
        raise RuntimeError("No map results were produced.")

    combined_model_summary = pd.concat(
        [map_results[map_name]["model_summary"] for map_name in target_maps if map_name in map_results],
        ignore_index=True,
    )
    summary_paths = scu.save_summary_tables(
        combined_model_summary,
        figures_dir,
        checkpoint_choice=args.checkpoint_choice,
        map_order=target_maps,
        model_order=target_models,
    )
    bundle_dir = figures_dir / f"smax_checkpoint_eval_{args.checkpoint_choice}_bundle"
    bundle_paths = scu.save_full_evaluation_bundle(
        map_results,
        bundle_dir,
        checkpoint_choice=args.checkpoint_choice,
        eval_episodes=args.eval_episodes,
        eval_seeds=eval_seeds,
        map_order=target_maps,
        model_order=target_models,
    )

    combined_seed_summary = pd.concat(
        [map_results[map_name]["seed_summary"] for map_name in target_maps if map_name in map_results],
        ignore_index=True,
    )
    seedwise_groups = {
        "part1": ["3m", "2s3z", "3s5z"],
        "part2": ["smacv2_5_units", "smacv2_10_units", "6h_vs_8z"],
    }
    for suffix, map_group in seedwise_groups.items():
        available_group = [map_name for map_name in map_group if map_name in target_maps]
        if not available_group:
            continue
        latex = scu.format_seedwise_cross_map_latex(
            combined_seed_summary,
            checkpoint_choice=args.checkpoint_choice,
            map_order=available_group,
            model_order=target_models,
            label=f"tab:smax_seedwise_{suffix}",
        )
        path = figures_dir / f"smax_checkpoint_eval_{args.checkpoint_choice}_seedwise_{suffix}.tex"
        path.write_text(latex)
        print(f"Saved seedwise TeX {suffix}: {path}")

    print(f"Saved summary CSV: {summary_paths['csv']}")
    print(f"Saved summary TeX: {summary_paths['tex']}")
    print(f"Saved bundle dir: {bundle_paths['dir']}")
    print(f"Saved run results: {bundle_paths['run_results']}")
    print(f"Saved seed summary: {bundle_paths['seed_summary']}")
    print(f"Saved model summary: {bundle_paths['model_summary']}")
    print(f"Saved bundle TeX: {bundle_paths['tex']}")


if __name__ == "__main__":
    main()
