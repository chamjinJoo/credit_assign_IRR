from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

mpl_config_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"))
if not mpl_config_dir.exists() or not os.access(mpl_config_dir, os.W_OK):
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt

try:
    from wandb_training_curves import (
        DEFAULT_ENTITY,
        build_plot_config,
        extract_final_performance,
        fetch_wandb_training_curves,
        plot,
        plot_final_distribution,
        plot_horizontal_legend,
    )
except ModuleNotFoundError:
    from figures.wandb_training_curves import (
        DEFAULT_ENTITY,
        build_plot_config,
        extract_final_performance,
        fetch_wandb_training_curves,
        plot,
        plot_final_distribution,
        plot_horizontal_legend,
    )


FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent
DEFAULT_CACHE_ROOT = FIGURES_DIR / "wandb_cache" / "LBF"
DEFAULT_OUTPUT_DIR = FIGURES_DIR / "lbf_success_rate_plots"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "runs" / "checkpoints"
DEFAULT_ENV_IDS = (
    "2s-8x8-2p-2f-coop",
    "2s-10x10-3p-3f",
    "2s-15x15-4p-3f",
    "2s-15x15-4p-5f",
)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")


def env_ids_from_checkpoints(source_root: Path = DEFAULT_SOURCE_ROOT) -> list[str]:
    env_ids: set[str] = set()
    for config_path in sorted(source_root.rglob("jumanjilbfbatchwrapper/simulation_config.json")):
        try:
            payload = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            continue
        cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
        run_name = str(((cfg.get("wandb") or {}).get("run_name")) or "")
        if run_name.startswith("JumanjiLBF-"):
            env_ids.add(run_name.removeprefix("JumanjiLBF-"))
    return sorted(env_ids)


def lbf_cache_dir(cache_root: Path, env_id: str) -> Path:
    return cache_root / safe_name(env_id)


def fetch_lbf_env_curves(
    env_id: str,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    entity: str = DEFAULT_ENTITY,
    project: str | None = None,
    state: str = "finished",
    max_runs: int | None = None,
) -> Path:
    output_dir = lbf_cache_dir(cache_root, env_id)
    fetch_wandb_training_curves(
        preset="lbf",
        entity=entity,
        project=project,
        output_dir=output_dir,
        state=state,
        name_contains=(f"JumanjiLBF-{env_id}",),
        max_runs=max_runs,
    )
    return output_dir


def plot_lbf_env(
    env_id: str,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    window_size: int = 20,
    last_n: int = 50,
    xlim: tuple[float, float] = (0.0, 2.1e6),
    ylim: tuple[float, float] = (0.0, 1.0),
) -> list[dict]:
    cache_dir = lbf_cache_dir(cache_root, env_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_config = build_plot_config("lbf", cache_dir)

    prefix = output_dir / safe_name(env_id)
    plot(
        plot_config,
        window_size=window_size,
        figsize=(5, 4.5),
        xlim=xlim,
        ylim=ylim,
        mean_only=True,
        show_std_shading=True,
        save_path=prefix.with_suffix(".success_rate.png"),
        show=False,
    )
    plot_final_distribution(
        plot_config,
        window_size=window_size,
        last_n=last_n,
        figsize=(6.4, 5),
        ylim=ylim,
        show_seed_points=False,
        save_path=prefix.with_suffix(".final_distribution.png"),
        show=False,
    )
    return extract_final_performance(plot_config, window_size=window_size, last_n=last_n)


def write_summary(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["env_id", "algorithm", "mean", "std", "n_seeds", "min", "max"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LBF success-rate curves by env_id.")
    parser.add_argument("--env-id", action="append", default=[], help="Env id to plot. Repeatable.")
    parser.add_argument("--discover-env-ids", action="store_true", help="Discover env ids from runs/checkpoints.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=None)
    parser.add_argument("--state", default="finished")
    parser.add_argument("--fetch", action="store_true", help="Fetch curves from W&B before plotting.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--last-n", type=int, default=50)
    parser.add_argument("--xlim", nargs=2, type=float, default=(0.0, 2.1e6))
    parser.add_argument("--ylim", nargs=2, type=float, default=(0.0, 1.0))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.env_id:
        env_ids = args.env_id
    elif args.discover_env_ids:
        env_ids = env_ids_from_checkpoints(args.source_root)
    else:
        env_ids = list(DEFAULT_ENV_IDS)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    legend_config = build_plot_config("lbf", args.cache_root / safe_name(env_ids[0]))
    plot_horizontal_legend(
        legend_config,
        figsize=(5, 0.8),
        save_path=args.output_dir / "legend.png",
        show=False,
    )

    summary_rows = []
    for env_id in env_ids:
        if args.fetch:
            fetch_lbf_env_curves(
                env_id,
                cache_root=args.cache_root,
                entity=args.entity,
                project=args.project,
                state=args.state,
                max_runs=args.max_runs,
            )
        results = plot_lbf_env(
            env_id,
            cache_root=args.cache_root,
            output_dir=args.output_dir,
            window_size=args.window_size,
            last_n=args.last_n,
            xlim=tuple(args.xlim),
            ylim=tuple(args.ylim),
        )
        for row in results:
            summary_rows.append({"env_id": env_id, **row})

    summary_path = args.output_dir / "lbf_success_rate_summary.csv"
    write_summary(summary_rows, summary_path)
    plt.close("all")
    print(f"[done] env_ids={len(env_ids)} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
