#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "runs" / "checkpoints_gs1"
CHECKPOINT_NAMES = ("jax_checkpoint_best.pkl", "jax_checkpoint_final.pkl")
WRAPPER_DIR_NAMES = ("smaxbatchwrapper", "smaxwrapper")

MODEL_LABELS = {
    "jax_ippo": "ippo",
    "jax_mappo": "mappo",
    "jax_pci_ippo": "ipci",
    "jax_pci_mappo": "mpci",
    "ippo": "ippo",
    "mappo": "mappo",
    "pci_ippo": "ipci",
    "pci_mappo": "mpci",
}


def _get_nested(mapping: Any, *keys: str, default: Any = None) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def _slug(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace(os.sep, "_")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")


def _model_label(algorithm_type: Any) -> str:
    key = str(algorithm_type or "").strip().lower().replace("-", "_")
    return MODEL_LABELS.get(key, key.removeprefix("jax_") or "unknown_model")


def _constraint_label(env_cfg: dict[str, Any], wandb_cfg: dict[str, Any]) -> str:
    if not bool(env_cfg.get("use_allowed_targets", False)):
        return "vanilla"

    label_text = " ".join(
        str(wandb_cfg.get(key, "") or "") for key in ("run_name_postfix", "run_name")
    ).lower()
    if re.search(r"(^|[_\W])hard($|[_\W])", label_text):
        return "hard"
    if re.search(r"(^|[_\W])medium($|[_\W])", label_text):
        return "medium"

    window = env_cfg.get("allowed_target_window")
    return f"target_window_{window}" if window is not None else "allowed_targets"


def _run_seed(cfg: dict[str, Any], run_name: str) -> Any:
    seed = _get_nested(cfg, "training", "seed")
    if seed is not None:
        return seed
    seed = _get_nested(cfg, "environment", "seed")
    if seed is not None:
        return seed
    match = re.search(r"_seed(\d+)$", run_name)
    return int(match.group(1)) if match else None


def _safe_link_or_copy(src: Path, dst: Path, mode: str, dry_run: bool) -> str:
    if dst.exists():
        try:
            if src.samefile(dst):
                return "existing"
        except OSError:
            pass
        raise FileExistsError(f"refusing to overwrite existing file: {dst}")

    if dry_run:
        return "dry_run"

    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(src, dst)
        return "linked"
    shutil.copy2(src, dst)
    return "copied"


def _copy_config(src: Path, dst: Path, dry_run: bool) -> str:
    if dst.exists():
        if src.read_bytes() == dst.read_bytes():
            return "existing"
        raise FileExistsError(f"refusing to overwrite existing config: {dst}")

    if dry_run:
        return "dry_run"

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def _raw_config_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for wrapper_name in WRAPPER_DIR_NAMES:
        paths.extend(root.glob(f"*/{wrapper_name}/simulation_config.json"))
    return sorted(paths)


def _classified_config_paths(root: Path) -> list[Path]:
    paths = []
    for config_path in root.rglob("simulation_config.json"):
        if config_path.parent.name in WRAPPER_DIR_NAMES:
            try:
                rel_parts = config_path.relative_to(root).parts
            except ValueError:
                rel_parts = ()
            run_leaf = config_path.parents[1]
            leaf_is_already_classified = (
                (run_leaf / "simulation_config.json").exists()
                and (run_leaf / "checkpoints").is_dir()
            )
            if (
                len(rel_parts) >= 6
                and (config_path.parent / "checkpoints").is_dir()
                and not leaf_is_already_classified
            ):
                paths.append(config_path)
            continue
        if (config_path.parent / "checkpoints").is_dir():
            paths.append(config_path)
    return sorted(paths)


def _config_paths(root: Path, source_layout: str) -> list[Path]:
    if source_layout == "raw":
        return _raw_config_paths(root)
    if source_layout == "classified":
        return _classified_config_paths(root)
    if source_layout == "all":
        return sorted({*_raw_config_paths(root), *_classified_config_paths(root)})

    raw_paths = _raw_config_paths(root)
    return raw_paths if raw_paths else _classified_config_paths(root)


def _run_and_checkpoint_dir(config_path: Path) -> tuple[Path, Path, str]:
    if config_path.parent.name in WRAPPER_DIR_NAMES:
        return config_path.parents[1], config_path.parent / "checkpoints", "raw"
    return config_path.parent, config_path.parent / "checkpoints", "classified"


def classify(root: Path, mode: str, dry_run: bool, source_layout: str = "auto") -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    missing: list[dict[str, Any]] = []
    counts: collections.Counter[tuple[str, str, str]] = collections.Counter()
    checkpoint_action_counts: collections.Counter[str] = collections.Counter()
    config_action_counts: collections.Counter[str] = collections.Counter()

    config_paths = _config_paths(root, source_layout)
    for config_path in config_paths:
        run_dir, checkpoint_dir, detected_layout = _run_and_checkpoint_dir(config_path)
        try:
            payload = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            skipped.append({"path": str(config_path), "reason": f"invalid json: {exc}"})
            continue

        cfg = payload.get("config", {})
        env_cfg = cfg.get("environment", {})
        if env_cfg.get("env_name") != "SMAX":
            skipped.append({"path": str(config_path), "reason": "not SMAX"})
            continue

        map_name = _slug(env_cfg.get("map_name"))
        if not map_name:
            skipped.append({"path": str(config_path), "reason": "missing map_name"})
            continue

        algorithm_type = _get_nested(cfg, "algorithm", "type", default="")
        model = _slug(_model_label(algorithm_type))
        constraint = _slug(_constraint_label(env_cfg, cfg.get("wandb", {})))
        seed = _run_seed(cfg, run_dir.name)
        dest_run = root / map_name / constraint / model / run_dir.name

        config_action = _copy_config(
            config_path,
            dest_run / "simulation_config.json",
            dry_run=dry_run,
        )
        config_action_counts[config_action] += 1

        checkpoint_records: list[dict[str, str]] = []
        for checkpoint_name in CHECKPOINT_NAMES:
            src = checkpoint_dir / checkpoint_name
            dst = dest_run / "checkpoints" / checkpoint_name
            if not src.exists():
                missing.append(
                    {
                        "run": run_dir.name,
                        "map": map_name,
                        "constraint": constraint,
                        "model": model,
                        "seed": seed,
                        "missing_checkpoint": checkpoint_name,
                        "source": str(src),
                    }
                )
                continue

            action = _safe_link_or_copy(src, dst, mode=mode, dry_run=dry_run)
            checkpoint_action_counts[action] += 1
            checkpoint_records.append(
                {
                    "name": checkpoint_name,
                    "source": str(src),
                    "destination": str(dst),
                    "action": action,
                }
            )

        counts[(map_name, constraint, model)] += 1
        entries.append(
            {
                "run": run_dir.name,
                "map": map_name,
                "constraint": constraint,
                "model": model,
                "algorithm_type": algorithm_type,
                "allowed_target_window": env_cfg.get("allowed_target_window"),
                "use_allowed_targets": bool(env_cfg.get("use_allowed_targets", False)),
                "seed": seed,
                "config_source": str(config_path),
                "config_destination": str(dest_run / "simulation_config.json"),
                "source_layout": detected_layout,
                "checkpoints": checkpoint_records,
            }
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "dry_run": dry_run,
        "mode": mode,
        "source_layout": source_layout,
        "config_paths_scanned": len(config_paths),
        "smax_runs": len(entries),
        "skipped": skipped,
        "missing_checkpoints": missing,
        "config_actions": dict(sorted(config_action_counts.items())),
        "checkpoint_actions": dict(sorted(checkpoint_action_counts.items())),
        "counts_by_map_constraint_model": [
            {
                "map": map_name,
                "constraint": constraint,
                "model": model,
                "runs": count,
            }
            for (map_name, constraint, model), count in sorted(counts.items())
        ],
        "entries": entries,
    }

    manifest_path = root / "classification_manifest.json"
    if not dry_run:
        manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary["manifest"] = str(manifest_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify GS1 SMAX checkpoints by map, constraint, model, and run."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to place checkpoint pkl files in the classified tree.",
    )
    parser.add_argument(
        "--source-layout",
        choices=("auto", "raw", "classified", "all"),
        default="auto",
        help=(
            "Which simulation_config.json layout to scan. "
            "'raw' scans <run>/<wrapper>/simulation_config.json; "
            "'classified' scans already-classified <map>/<constraint>/<model>/<run>/simulation_config.json."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = classify(
        args.root.resolve(),
        mode=args.mode,
        dry_run=args.dry_run,
        source_layout=args.source_layout,
    )
    print(f"SMAX runs: {summary['smax_runs']}")
    print(f"Config paths scanned: {summary['config_paths_scanned']}")
    print(f"Source layout: {summary['source_layout']}")
    print(f"Manifest: {summary['manifest']}")
    print(f"Config actions: {summary['config_actions']}")
    print(f"Checkpoint actions: {summary['checkpoint_actions']}")
    print(f"Missing checkpoint files: {len(summary['missing_checkpoints'])}")
    for item in summary["counts_by_map_constraint_model"]:
        print(
            "{map}/{constraint}/{model}: {runs}".format(
                map=item["map"],
                constraint=item["constraint"],
                model=item["model"],
                runs=item["runs"],
            )
        )


if __name__ == "__main__":
    main()
