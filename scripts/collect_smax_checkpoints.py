#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "save_checkpoints"
DEFAULT_DEST_ROOT = REPO_ROOT / "checkpoints_for_eval"
DEFAULT_PROJECT_FILTER = "pci-marl-smax-jax-v3"
ALLOWED_ENVIRONMENTS = {"smaxwrapper", "smaxbatchwrapper"}
MODEL_PRETTY_NAMES = {
    "jax_pci_mappo": "pci_mappo",
    "jax_pci_ippo": "pci_ippo",
    "jax_mappo": "mappo",
    "jax_ippo": "ippo",
    "pci_mappo": "pci_mappo",
    "pci_ippo": "pci_ippo",
    "mappo": "mappo",
    "ippo": "ippo",
}
CONSTRAINT_PRETTY_NAMES = {
    "vanilla": "vanilla",
    "tar_w=2": "tar_w=2",
    "tar_w=3": "tar_w=3",
    "tar_w=4": "tar_w=4",
    "tar_w=5": "tar_w=5",
    "tar_w=6": "tar_w=6",
    "tar_w=8": "tar_w=8",
    "cb": "cb",
}


def _normalize_text(text: Any) -> str:
    return str(text or "").strip().lower()


def _get_nested(mapping: Any, *keys: str, default: Any = None) -> Any:
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def model_key_from_algorithm_type(algorithm_type: str) -> str:
    key = _normalize_text(algorithm_type).replace("-", "_")
    return key.removeprefix("jax_")


def model_label_from_algorithm_type(algorithm_type: str) -> str:
    key = model_key_from_algorithm_type(algorithm_type)
    return MODEL_PRETTY_NAMES.get(key, key)


def constraint_key_from_run_name(run_name: str, map_name: str | None = None) -> str:
    name = _normalize_text(run_name)
    if "smax_" in name:
        name = name[name.index("smax_") + 5 :]
    elif name.startswith("jax_"):
        name = name[4:]

    if map_name:
        map_prefix = _normalize_text(map_name)
        if name.startswith(map_prefix):
            name = name[len(map_prefix) :].lstrip("_")

    tar_match = re.search(r"tar_w=\d+", name)
    if tar_match is not None:
        return tar_match.group(0)
    if re.search(r"c[._]?b", name):
        return "cb"
    return "vanilla"


def checkpoint_filenames(algorithm_type: str) -> tuple[str, str]:
    key = _normalize_text(algorithm_type)
    if key.startswith("jax_"):
        return "jax_checkpoint_best.pkl", "jax_checkpoint_final.pkl"
    return "best_model.pt", "final_checkpoint.pt"


def iter_run_metadata(source_root: Path):
    for metadata_path in sorted(source_root.rglob("simulation_config.json")):
        try:
            payload = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            yield {
                "metadata_path": metadata_path,
                "skip_reason": "invalid simulation_config.json",
            }
            continue

        cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
        sim_info = payload.get("simulation_info", {}) if isinstance(payload, dict) else {}
        run_dir = metadata_path.parents[1]
        env_dir_name = metadata_path.parent.name
        wandb_project = _get_nested(cfg, "wandb", "project")
        wandb_name = (
            _get_nested(cfg, "wandb", "name")
            or _get_nested(cfg, "wandb", "run_name")
            or _get_nested(cfg, "wandb", "group")
            or run_dir.name
        )
        algorithm_type = _get_nested(cfg, "algorithm", "type")
        map_name = _get_nested(cfg, "environment", "map_name", default="unknown")
        env_name = _normalize_text(sim_info.get("environment"))
        constraint_key = constraint_key_from_run_name(str(wandb_name), map_name=str(map_name))
        model_label = model_label_from_algorithm_type(str(algorithm_type))
        best_name, final_name = checkpoint_filenames(str(algorithm_type))
        ckpt_dir = metadata_path.parent / "checkpoints"
        best_path = ckpt_dir / best_name
        final_path = ckpt_dir / final_name
        summary_path = metadata_path.parent / "results" / "training_summary.json"

        yield {
            "metadata_path": metadata_path,
            "run_dir": run_dir,
            "env_dir_name": env_dir_name,
            "config": cfg,
            "wandb_project": wandb_project,
            "wandb_name": str(wandb_name),
            "algorithm_type": str(algorithm_type),
            "map_name": str(map_name),
            "constraint_key": constraint_key,
            "model_label": model_label,
            "env_name": env_name,
            "best_path": best_path,
            "final_path": final_path,
            "summary_path": summary_path,
        }


def should_copy(entry: dict[str, Any], project_filter: str | None) -> tuple[bool, str]:
    if project_filter and entry["wandb_project"] != project_filter:
        return False, f"project mismatch: {entry['wandb_project']}"
    if entry["env_name"] not in ALLOWED_ENVIRONMENTS:
        return False, f"not a SMAX environment: {entry['env_name'] or 'unknown'}"
    if not entry["summary_path"].is_file():
        return False, "missing training_summary.json"
    try:
        summary = json.loads(entry["summary_path"].read_text())
    except json.JSONDecodeError:
        return False, "invalid training_summary.json"
    if "training_completed" not in summary:
        return False, "training_summary.json does not mark completion"
    if not entry["best_path"].is_file() or not entry["final_path"].is_file():
        return False, "missing best/final checkpoint"
    return True, "ok"


def destination_root(dest_root: Path, map_name: str, constraint_key: str, model_label: str, run_name: str, env_dir_name: str) -> Path:
    return dest_root / f"smax_{map_name}" / constraint_key / model_label / run_name / env_dir_name


def copy_file(src: Path, dst: Path, overwrite: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return "exists"
    shutil.copy2(src, dst)
    return "copied"


def collect_checkpoints(
    source_root: Path,
    dest_root: Path,
    *,
    project_filter: str | None,
    overwrite: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in iter_run_metadata(source_root):
        metadata_path = entry["metadata_path"]
        if "skip_reason" in entry:
            rows.append(
                {
                    "status": "skipped",
                    "reason": entry["skip_reason"],
                    "source": str(metadata_path),
                    "destination": "",
                    "map_name": "",
                    "constraint_key": "",
                    "model_label": "",
                    "run_name": "",
                }
            )
            continue

        keep, reason = should_copy(entry, project_filter)
        if not keep:
            rows.append(
                {
                    "status": "skipped",
                    "reason": reason,
                    "source": str(metadata_path),
                    "destination": "",
                    "map_name": entry["map_name"],
                    "constraint_key": entry["constraint_key"],
                    "model_label": entry["model_label"],
                    "run_name": entry["run_dir"].name,
                }
            )
            continue

        dst_dir = destination_root(
            dest_root,
            entry["map_name"],
            entry["constraint_key"],
            entry["model_label"],
            entry["run_dir"].name,
            entry["env_dir_name"],
        )
        best_dst = dst_dir / "checkpoints" / entry["best_path"].name
        final_dst = dst_dir / "checkpoints" / entry["final_path"].name
        summary_dst = dst_dir / "results" / "training_summary.json"
        readme_src = metadata_path.parent / "README.txt"
        readme_dst = dst_dir / "README.txt"
        config_dst = dst_dir / "simulation_config.json"

        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            copy_file(metadata_path, config_dst, overwrite=overwrite)
            if readme_src.is_file():
                copy_file(readme_src, readme_dst, overwrite=overwrite)
            if entry["summary_path"].is_file():
                copy_file(entry["summary_path"], summary_dst, overwrite=overwrite)
            copy_file(entry["best_path"], best_dst, overwrite=overwrite)
            copy_file(entry["final_path"], final_dst, overwrite=overwrite)

        rows.append(
            {
                "status": "copied" if not dry_run else "dry_run",
                "reason": reason,
                "source": str(metadata_path),
                "destination": str(dst_dir),
                "map_name": entry["map_name"],
                "constraint_key": entry["constraint_key"],
                "model_label": entry["model_label"],
                "run_name": entry["run_dir"].name,
            }
        )

    return rows


def write_manifest(rows: list[dict[str, Any]], dest_root: Path) -> Path:
    manifest_path = dest_root / "smax_checkpoint_copy_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["status", "reason", "source", "destination", "map_name", "constraint_key", "model_label", "run_name"]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect completed SMAX checkpoints from save_checkpoints into checkpoints_for_eval."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument("--project-filter", type=str, default=DEFAULT_PROJECT_FILTER)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files at the destination.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be copied without writing files.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    source_root = args.source_root.expanduser().resolve()
    dest_root = args.dest_root.expanduser().resolve()

    if not source_root.exists():
        print(f"[error] source root does not exist: {source_root}", file=sys.stderr)
        return 1

    rows = collect_checkpoints(
        source_root,
        dest_root,
        project_filter=args.project_filter,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    copied = sum(1 for row in rows if row["status"] in {"copied", "dry_run"})
    skipped = sum(1 for row in rows if row["status"] == "skipped")
    manifest_path = write_manifest(rows, dest_root)

    print(f"[done] matched {copied} runs, skipped {skipped} entries")
    print(f"[done] manifest written to {manifest_path}")
    if args.dry_run:
        print("[done] dry-run mode: no files were written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
