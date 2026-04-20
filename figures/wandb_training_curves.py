from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ENTITY = "chamjin-korea-advanced-institute-of-science-and-technology"
DEFAULT_METRIC = "rollout/mean_success_rate"
DEFAULT_STEP = "total_steps"
SH_NOISE_TOKEN = "noise0.1"
SH_NOISE_SUFFIX = "_noise0.1"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    keyword: str
    color: str
    aliases: tuple[str, ...]
    linestyle: str = "-"


SH_MODELS = (
    ModelSpec("PCI_rippo", "PCI+IPPO", "PCI_rippo", "blue", ("PCI_rippo", "PCI_rIPPO", "pci-rippo")),
    ModelSpec("PCI_rmappo", "PCI+MAPPO", "PCI_rmappo", "red", ("PCI_rmappo", "PCI_rMAPPO", "pci-rmappo")),
    ModelSpec("r_ippo", "IPPO", "rippo", "skyblue", ("r_ippo", "rIPPO", "r-ippo")),
    ModelSpec("r_mappo", "MAPPO", "rmappo", "orange", ("r_mappo", "rMAPPO", "r-mappo")),
)

SMAX_MODELS = (
    ModelSpec("pci_ippo", "PCI+IPPO", "pci_ippo", "blue", ("pci_ippo", "jax_pci_ippo", "jax-pci-ippo")),
    ModelSpec("pci_mappo", "PCI+MAPPO", "pci_mappo", "red", ("pci_mappo", "jax_pci_mappo", "jax-pci-mappo")),
    ModelSpec("ippo", "IPPO", "ippo", "skyblue", ("ippo", "jax_ippo", "jax-ippo")),
    ModelSpec("mappo", "MAPPO", "mappo", "orange", ("mappo", "jax_mappo", "jax-mappo")),
)


PRESETS = {
    "sh": {
        "project": "pci-marl-staghunt",
        "models": SH_MODELS,
        "name_contains": (),
        "exclude_name_contains": (SH_NOISE_TOKEN,),
        "name_suffixes": (),
        "exclude_name_suffixes": (SH_NOISE_SUFFIX,),
        "xlim": (0.0, 3e6),
    },
    "sh-noise": {
        "project": "pci-marl-staghunt",
        "models": SH_MODELS,
        "name_contains": (SH_NOISE_TOKEN,),
        "exclude_name_contains": (),
        "name_suffixes": (),
        "exclude_name_suffixes": (),
        "xlim": (0.0, 3e6),
    },
    "smax": {
        "project": "pci-marl-smax-jax-v2",
        "models": SMAX_MODELS,
        "name_contains": (),
        "exclude_name_contains": (),
        "name_suffixes": (),
        "exclude_name_suffixes": (),
        "xlim": (0.0, 1e7),
    },
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")


def _run_text(run) -> str:
    parts = [getattr(run, "name", ""), getattr(run, "display_name", ""), getattr(run, "group", "")]
    parts.extend(getattr(run, "tags", []) or [])
    return " ".join(str(p) for p in parts if p)


def _matches_filters(
    run,
    contains: Iterable[str],
    exclude_contains: Iterable[str],
    name_contains: Iterable[str] = (),
    exclude_name_contains: Iterable[str] = (),
    name_suffixes: Iterable[str] = (),
    exclude_name_suffixes: Iterable[str] = (),
) -> bool:
    run_name = getattr(run, "name", "")
    run_name_lower = run_name.lower()
    if any(term.lower() not in run_name_lower for term in name_contains):
        return False
    if any(term.lower() in run_name_lower for term in exclude_name_contains):
        return False

    suffixes = tuple(s for s in name_suffixes if s)
    if suffixes and not run_name_lower.endswith(tuple(s.lower() for s in suffixes)):
        return False

    excluded_suffixes = tuple(s for s in exclude_name_suffixes if s)
    if excluded_suffixes and run_name_lower.endswith(tuple(s.lower() for s in excluded_suffixes)):
        return False

    text = _run_text(run).lower()
    if any(term.lower() not in text for term in contains):
        return False
    return not any(term.lower() in text for term in exclude_contains)


def _match_model(run_name: str, specs: Iterable[ModelSpec]) -> str | None:
    normalized = _norm(run_name)
    for spec in specs:
        if any(_norm(alias) in normalized for alias in spec.aliases):
            return spec.key
    return None


def _config_get(config, *keys):
    cur = config or {}
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _match_model_from_run(run, preset: str, specs: Iterable[ModelSpec]) -> str | None:
    name_match = _match_model(run.name, specs)
    if name_match is not None:
        return name_match

    config = getattr(run, "config", {}) or {}
    algorithm_type = _norm(_config_get(config, "algorithm", "type") or "")
    centralized = _as_bool(_config_get(config, "algorithm", "use_centralized_critic"))

    if preset == "smax":
        if algorithm_type in {"jaxrpci", "rpci"}:
            return "pci_mappo" if centralized else "pci_ippo"
        if algorithm_type in {"jaxippo", "ippo"}:
            return "ippo"
        if algorithm_type in {"jaxmappo", "mappo"}:
            return "mappo"

    if preset.startswith("sh"):
        if algorithm_type in {"rpci", "pcirnn"}:
            return "PCI_rmappo" if centralized else "PCI_rippo"
        if algorithm_type in {"rippo"}:
            return "r_ippo"
        if algorithm_type in {"rmappo"}:
            return "r_mappo"

    return None


def _scan_metric_history(run, metric_key: str, step_key: str, page_size: int) -> pd.DataFrame:
    rows = list(run.scan_history(keys=[step_key, metric_key, "_step"], page_size=page_size))
    active_step_key = step_key
    if not rows and step_key != "_step":
        rows = list(run.scan_history(keys=["_step", metric_key], page_size=page_size))
        active_step_key = "_step"

    data = []
    for row in rows:
        metric_value = row.get(metric_key)
        step_value = row.get(active_step_key, row.get("_step"))
        raw_step_value = row.get("_step", step_value)
        if metric_value is None or step_value is None:
            continue
        data.append((step_value, metric_value, raw_step_value))

    if not data:
        return pd.DataFrame(columns=[DEFAULT_STEP, metric_key, "_step"])

    df = pd.DataFrame(data, columns=[DEFAULT_STEP, metric_key, "_step"])
    df = df.drop_duplicates(subset=[DEFAULT_STEP], keep="last")
    return df.sort_values(DEFAULT_STEP).reset_index(drop=True)


def fetch_wandb_training_curves(
    preset: str,
    entity: str = DEFAULT_ENTITY,
    project: str | None = None,
    output_dir: str | Path | None = None,
    metric_key: str = DEFAULT_METRIC,
    step_key: str = DEFAULT_STEP,
    state: str = "finished",
    contains: Iterable[str] = (),
    exclude_contains: Iterable[str] = (),
    name_contains: Iterable[str] = (),
    exclude_name_contains: Iterable[str] = (),
    name_suffixes: Iterable[str] = (),
    exclude_name_suffixes: Iterable[str] = (),
    page_size: int = 10000,
    max_runs: int | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb가 설치되어 있지 않습니다. `pip install wandb` 후 다시 실행하세요.") from exc

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose one of {sorted(PRESETS)}")

    preset_config = PRESETS[preset]
    specs = preset_config["models"]
    project = project or preset_config["project"]
    name_contains = tuple(preset_config.get("name_contains", ())) + tuple(name_contains)
    exclude_name_contains = tuple(preset_config.get("exclude_name_contains", ())) + tuple(exclude_name_contains)
    name_suffixes = tuple(preset_config.get("name_suffixes", ())) + tuple(name_suffixes)
    preset_exclude_suffixes = () if name_suffixes else tuple(preset_config.get("exclude_name_suffixes", ()))
    exclude_name_suffixes = preset_exclude_suffixes + tuple(exclude_name_suffixes)
    output_dir = Path(output_dir or Path("wandb_cache") / preset)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        if name_contains:
            print(f"[filter] include run name containing: {', '.join(name_contains)}")
        if exclude_name_contains:
            print(f"[filter] exclude run name containing: {', '.join(exclude_name_contains)}")
        if name_suffixes:
            print(f"[filter] include run name suffix: {', '.join(name_suffixes)}")
        if exclude_name_suffixes:
            print(f"[filter] exclude run name suffix: {', '.join(exclude_name_suffixes)}")

    filters = None if state == "all" else {"state": state}
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=filters)
    grouped: dict[str, list[tuple[object, pd.DataFrame]]] = {spec.key: [] for spec in specs}
    skipped = []

    for run in runs:
        if max_runs is not None and sum(len(v) for v in grouped.values()) >= max_runs:
            break
        if not _matches_filters(
            run,
            contains,
            exclude_contains,
            name_contains,
            exclude_name_contains,
            name_suffixes,
            exclude_name_suffixes,
        ):
            continue

        model_key = _match_model_from_run(run, preset, specs)
        if model_key is None:
            skipped.append(f"[model unmatched] {run.name}")
            continue

        history = _scan_metric_history(run, metric_key, step_key, page_size)
        if history.empty:
            skipped.append(f"[metric missing] {run.name}")
            continue

        grouped[model_key].append((run, history))
        if verbose:
            print(f"[collect] {model_key:<10} {run.name} ({len(history)} points)")

    csv_paths: dict[str, Path] = {}
    for spec in specs:
        items = grouped[spec.key]
        csv_path = output_dir / f"{spec.key}.csv"
        if not items:
            if csv_path.exists():
                csv_path.unlink()
                if verbose:
                    print(f"[remove stale] {csv_path}")
            if verbose:
                print(f"[WARN] no runs collected for {spec.key}")
            continue

        wide = None
        for run, history in items:
            run_name = _safe_filename(run.name)
            renamed = history.rename(
                columns={
                    metric_key: f"{run_name} - {metric_key}",
                    "_step": f"{run_name} - _step",
                }
            )
            wide = renamed if wide is None else pd.merge(wide, renamed, on=DEFAULT_STEP, how="outer")

        wide = wide.sort_values(DEFAULT_STEP).reset_index(drop=True)
        wide.to_csv(csv_path, index=False)
        csv_paths[spec.key] = csv_path
        if verbose:
            print(f"[write] {csv_path} ({len(items)} runs, {len(wide)} rows)")

    if skipped and verbose:
        print("\n[skipped]")
        print("\n".join(skipped[:50]))
        if len(skipped) > 50:
            print(f"... {len(skipped) - 50} more")

    return csv_paths


def build_plot_config(preset: str, csv_dir: str | Path) -> list[dict]:
    preset_config = PRESETS[preset]
    specs = preset_config["models"]
    exclude_terms = tuple(preset_config.get("exclude_name_contains", ()))
    csv_dir = Path(csv_dir)
    config = []
    for spec in specs:
        entry = {
            "path": str(csv_dir / f"{spec.key}.csv"),
            "label": spec.label,
            "keyword": spec.keyword,
            "color": spec.color,
            "linestyle": spec.linestyle,
        }
        if exclude_terms:
            entry["exclude"] = exclude_terms[0]
        config.append(entry)
    return config


def plot(plot_config, window_size=50, figsize=(8, 6), ylim=(0, 1), xlim=(0, 3e6), save_path=None, show=True):
    plt.close("all")
    plt.rc("font", size=14)
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=14)
    plt.rc("ytick", labelsize=14)
    plt.rc("legend", fontsize=14)

    fig, ax = plt.subplots(figsize=figsize)

    for entry in plot_config:
        file_path = entry["path"]
        label = entry["label"]
        keyword = entry.get("keyword", "success_rate")
        color = entry["color"]
        lsty = entry.get("linestyle", "-")
        lw = entry.get("linewidth", 3)
        alpha = entry.get("alpha", 0.08)
        exclude = entry.get("exclude", None)

        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"[WARN] File not found: {file_path}")
            continue

        clean_kw = keyword.lower().replace("_", "").replace("-", "")
        cols = []
        for col in df.columns:
            col_lower = col.lower()
            if "success_rate" not in col_lower:
                continue
            clean_col = col_lower.replace("_", "").replace("-", "")
            if clean_kw not in clean_col:
                continue
            if "MIN" in col or "MAX" in col or "_step" in col:
                continue
            if exclude and exclude.lower() in col_lower:
                continue
            cols.append(col)

        if not cols:
            print(f'[WARN] No matching columns for keyword "{keyword}" in {file_path}')
            continue

        x = df[DEFAULT_STEP]
        y_mean = df[cols].mean(axis=1).rolling(window=window_size, min_periods=1).mean()
        y_std = df[cols].std(axis=1).rolling(window=window_size, min_periods=1).mean()

        ax.plot(x, y_mean, label=label, color=color, linestyle=lsty, linewidth=lw)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=alpha)

    ax.set_xlabel("Total Steps")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[save] {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


def extract_final_performance(plot_config, window_size=50, last_n=100):
    results = []
    for entry in plot_config:
        file_path = entry["path"]
        label = entry["label"]
        keyword = entry.get("keyword", "success_rate")
        exclude = entry.get("exclude", None)

        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"[WARN] File not found: {file_path}")
            continue

        clean_kw = keyword.lower().replace("_", "").replace("-", "")
        cols = []
        for col in df.columns:
            col_lower = col.lower()
            if "success_rate" not in col_lower:
                continue
            clean_col = col_lower.replace("_", "").replace("-", "")
            if clean_kw not in clean_col:
                continue
            if "MIN" in col or "MAX" in col or "_step" in col:
                continue
            if exclude and exclude.lower() in col_lower:
                continue
            cols.append(col)

        if not cols:
            print(f'[WARN] No matching columns for keyword "{keyword}" in {file_path}')
            continue

        final_vals = []
        for col in cols:
            series = df[col].dropna()
            smoothed = series.rolling(window=window_size, min_periods=1).mean()
            val = smoothed.tail(last_n).mean() if last_n and last_n > 0 else smoothed.iloc[-1]
            final_vals.append(float(val))

        final_vals = np.array(final_vals)
        results.append(
            {
                "algorithm": label,
                "mean": float(np.mean(final_vals)),
                "std": float(np.std(final_vals)),
                "n_seeds": len(final_vals),
                "min": float(np.min(final_vals)),
                "max": float(np.max(final_vals)),
            }
        )

    return results


def print_performance_table(plot_config, window_size=50, last_n=200):
    results = extract_final_performance(plot_config, window_size=window_size, last_n=last_n)
    if not results:
        print("[ERROR] No performance data extracted.")
        return

    print("\n" + "=" * 80)
    print("Final Performance Summary (Success Rate)")
    print(f"Based on last {last_n} points with window_size={window_size}")
    print("=" * 80)
    print(f'{"Algorithm":<15} {"N":<5} {"Mean +/- Std":<15} {"Min":<8} {"Max":<8}')
    print("-" * 80)
    for result in results:
        mean_std = f"{result['mean']:.3f}+/-{result['std']:.3f}"
        print(f"{result['algorithm']:<15} {result['n_seeds']:<5} {mean_std:<15} {result['min']:<8.3f} {result['max']:<8.3f}")
    print("=" * 80)
    print("Note: Values are success rates (0-1 scale)")
    print("=" * 80)


def list_runs(
    preset: str,
    entity: str,
    project: str | None,
    state: str,
    contains,
    exclude_contains,
    name_contains=(),
    exclude_name_contains=(),
    name_suffixes=(),
    exclude_name_suffixes=(),
):
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb가 설치되어 있지 않습니다. `pip install wandb` 후 다시 실행하세요.") from exc

    preset_config = PRESETS[preset]
    specs = preset_config["models"]
    project = project or preset_config["project"]
    name_contains = tuple(preset_config.get("name_contains", ())) + tuple(name_contains)
    exclude_name_contains = tuple(preset_config.get("exclude_name_contains", ())) + tuple(exclude_name_contains)
    name_suffixes = tuple(preset_config.get("name_suffixes", ())) + tuple(name_suffixes)
    preset_exclude_suffixes = () if name_suffixes else tuple(preset_config.get("exclude_name_suffixes", ()))
    exclude_name_suffixes = preset_exclude_suffixes + tuple(exclude_name_suffixes)
    filters = None if state == "all" else {"state": state}
    api = wandb.Api()

    print(f"[runs] {entity}/{project} state={state}")
    for run in api.runs(f"{entity}/{project}", filters=filters):
        if not _matches_filters(
            run,
            contains,
            exclude_contains,
            name_contains,
            exclude_name_contains,
            name_suffixes,
            exclude_name_suffixes,
        ):
            continue
        model = _match_model_from_run(run, preset, specs) or "-"
        print(f"{model:<10} {run.state:<10} {run.name}")


def _parse_xlim(values):
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("--xlim expects exactly two numbers")
    return (float(values[0]), float(values[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fetch wandb success-rate histories and plot training curves.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="sh")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metric-key", default=DEFAULT_METRIC)
    parser.add_argument("--step-key", default=DEFAULT_STEP)
    parser.add_argument("--state", default="finished", help="wandb run state, or 'all'")
    parser.add_argument("--contains", nargs="*", default=[], help="run name/tag/group substrings that must all appear")
    parser.add_argument("--exclude-contains", nargs="*", default=[], help="run name/tag/group substrings to exclude")
    parser.add_argument("--name-contains", nargs="*", default=[], help="run name substrings that must all appear")
    parser.add_argument("--exclude-name-contains", nargs="*", default=[], help="run name substrings to exclude")
    parser.add_argument("--name-suffix", nargs="*", default=[], help="run name suffixes to include, e.g. _noise0.1")
    parser.add_argument("--exclude-name-suffix", nargs="*", default=[], help="run name suffixes to exclude")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--page-size", type=int, default=10000)
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-png", default=None)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--figsize", nargs=2, type=float, default=(8, 6))
    parser.add_argument("--xlim", nargs=2, default=None, help="example: --xlim 0 3000000. Omit for auto.")
    parser.add_argument("--no-table", action="store_true")
    args = parser.parse_args(argv)

    if args.list_runs:
        list_runs(
            args.preset,
            args.entity,
            args.project,
            args.state,
            args.contains,
            args.exclude_contains,
            args.name_contains,
            args.exclude_name_contains,
            args.name_suffix,
            args.exclude_name_suffix,
        )
        return 0

    default_output_dir = Path(__file__).resolve().parent / "wandb_cache" / args.preset
    output_dir = Path(args.output_dir or default_output_dir)
    fetch_wandb_training_curves(
        preset=args.preset,
        entity=args.entity,
        project=args.project,
        output_dir=output_dir,
        metric_key=args.metric_key,
        step_key=args.step_key,
        state=args.state,
        contains=args.contains,
        exclude_contains=args.exclude_contains,
        name_contains=args.name_contains,
        exclude_name_contains=args.exclude_name_contains,
        name_suffixes=args.name_suffix,
        exclude_name_suffixes=args.exclude_name_suffix,
        page_size=args.page_size,
        max_runs=args.max_runs,
    )

    plot_config = build_plot_config(args.preset, output_dir)
    if not args.no_table:
        print_performance_table(plot_config, window_size=args.window_size, last_n=200)
    if args.plot or args.save_png:
        xlim = _parse_xlim(args.xlim)
        if xlim is None:
            xlim = PRESETS[args.preset]["xlim"]
        plot(
            plot_config,
            window_size=args.window_size,
            figsize=tuple(args.figsize),
            xlim=xlim,
            save_path=args.save_png,
            show=args.plot,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
