#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "runs" / "checkpoints"
DEFAULT_DEST_ROOT = REPO_ROOT / "not_for_submit" / "checkpoints_for_eval_LBF"
LBF_ENV_DIR_NAMES = {"jumanjilbfbatchwrapper"}

MODEL_LABELS = {
    "jax_pci_mappo": "mpci",
    "jax_pci_ippo": "ipci",
    "jax_mappo": "mappo",
    "jax_ippo": "ippo",
    "pci_mappo": "mpci",
    "pci_ippo": "ipci",
    "mappo": "mappo",
    "ippo": "ippo",
}

CHECKPOINT_NAMES = {
    "best": ("jax_checkpoint_best.pkl", "best_model.pt"),
    "final": ("jax_checkpoint_final.pkl", "final_checkpoint.pt"),
}


def _normalize(text: Any) -> str:
    return str(text or "").strip().lower()


def _get_nested(mapping: Any, *keys: str, default: Any = None) -> Any:
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _model_label(algorithm_type: Any) -> str:
    key = _normalize(algorithm_type).replace("-", "_")
    key = key.removeprefix("jax_") if key.startswith("jax_") and key not in MODEL_LABELS else key
    return MODEL_LABELS.get(_normalize(algorithm_type).replace("-", "_"), MODEL_LABELS.get(key, key))


def _env_id_from_config(cfg: dict[str, Any]) -> tuple[str, str]:
    run_name = str(_get_nested(cfg, "wandb", "run_name", default="") or "")
    match = re.search(r"(?:^|[_-])JumanjiLBF-(.+)$", run_name)
    if match:
        return match.group(1), "wandb.run_name"
    if run_name.startswith("JumanjiLBF-"):
        return run_name.removeprefix("JumanjiLBF-"), "wandb.run_name"

    env_id = str(_get_nested(cfg, "environment", "env_id", default="") or "")
    if env_id:
        return env_id.removeprefix("JumanjiLBF-"), "environment.env_id"
    return "unknown_env", "unknown"


def _find_checkpoint(ckpt_dir: Path, kind: str) -> Path | None:
    for filename in CHECKPOINT_NAMES[kind]:
        path = ckpt_dir / filename
        if path.is_file():
            return path
    return None


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "json root is not an object"
    return payload, None


def iter_lbf_runs(source_root: Path):
    for config_path in sorted(source_root.rglob("simulation_config.json")):
        env_dir = config_path.parent
        run_dir = env_dir.parent

        if env_dir.name.lower() not in LBF_ENV_DIR_NAMES:
            continue

        payload, error = _load_json(config_path)
        if error is not None or payload is None:
            yield {
                "status": "skipped",
                "reason": error or "invalid simulation_config.json",
                "source": str(config_path),
            }
            continue

        cfg = payload.get("config", {})
        sim_info = payload.get("simulation_info", {})
        summary_path = env_dir / "results" / "training_summary.json"
        summary, summary_error = _load_json(summary_path) if summary_path.is_file() else (None, None)
        algorithm_type = _get_nested(cfg, "algorithm", "type", default="")
        ckpt_dir = env_dir / "checkpoints"
        env_id, env_id_source = _env_id_from_config(cfg)

        yield {
            "run_dir": run_dir,
            "env_dir": env_dir,
            "config_path": config_path,
            "summary_path": summary_path,
            "readme_path": env_dir / "README.txt",
            "cfg": cfg,
            "sim_info": sim_info,
            "summary": summary,
            "summary_error": summary_error,
            "env_id": env_id,
            "env_id_source": env_id_source,
            "model_label": _model_label(algorithm_type),
            "algorithm_type": str(algorithm_type),
            "seed": str(_get_nested(cfg, "training", "seed", default="")),
            "wandb_project": str(_get_nested(cfg, "wandb", "project", default="") or ""),
            "wandb_run_name": str(_get_nested(cfg, "wandb", "run_name", default="") or ""),
            "simulation_environment": str(sim_info.get("environment", "")),
            "best_path": _find_checkpoint(ckpt_dir, "best"),
            "final_path": _find_checkpoint(ckpt_dir, "final"),
        }


def should_copy(
    entry: dict[str, Any],
    env_filter: set[str],
    model_filter: set[str],
    include_fallback_env_id: bool,
) -> tuple[bool, str]:
    if env_filter and entry["env_id"] not in env_filter:
        return False, f"env_id not selected: {entry['env_id']}"
    if model_filter and entry["model_label"] not in model_filter:
        return False, f"model not selected: {entry['model_label']}"
    if entry["env_id_source"] != "wandb.run_name" and not include_fallback_env_id:
        return False, f"env_id was not parsed from JumanjiLBF wandb.run_name: {entry['env_id_source']}"
    if entry["wandb_project"] and entry["wandb_project"] != "pci-marl-lbf":
        return False, f"project mismatch: {entry['wandb_project']}"
    if entry["summary_error"]:
        return False, entry["summary_error"]
    if not entry["summary_path"].is_file():
        return False, "missing training_summary.json"
    if not isinstance(entry["summary"], dict) or "training_completed" not in entry["summary"]:
        return False, "training did not complete"
    if entry["best_path"] is None or entry["final_path"] is None:
        return False, "missing best/final checkpoint"
    return True, "ok"


def copy_file(src: Path, dst: Path, overwrite: bool, execute: bool) -> str:
    if dst.exists() and not overwrite:
        return "exists"
    if execute:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return "copied" if execute else "would_copy"


def destination_dir(dest_root: Path, entry: dict[str, Any]) -> Path:
    return dest_root / entry["env_id"] / entry["model_label"] / entry["run_dir"].name / entry["env_dir"].name


def collect(
    *,
    source_root: Path,
    dest_root: Path,
    env_ids: set[str],
    models: set[str],
    overwrite: bool,
    execute: bool,
    include_fallback_env_id: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for entry in iter_lbf_runs(source_root):
        if "config_path" not in entry:
            rows.append({
                "status": "skipped",
                "reason": entry.get("reason", "unknown"),
                "source": entry.get("source", ""),
                "destination": "",
                "env_id": "",
                "model": "",
                "seed": "",
                "run_name": "",
                "best_checkpoint": "",
                "final_checkpoint": "",
            })
            continue

        keep, reason = should_copy(entry, env_ids, models, include_fallback_env_id)
        dst_root = destination_dir(dest_root, entry)
        row = {
            "status": "skipped",
            "reason": reason,
            "source": str(entry["env_dir"]),
            "destination": str(dst_root),
            "env_id": entry["env_id"],
            "model": entry["model_label"],
            "seed": entry["seed"],
            "run_name": entry["run_dir"].name,
            "wandb_run_name": entry["wandb_run_name"],
            "algorithm_type": entry["algorithm_type"],
            "env_id_source": entry["env_id_source"],
            "best_checkpoint": str(entry["best_path"] or ""),
            "final_checkpoint": str(entry["final_path"] or ""),
        }
        if not keep:
            rows.append(row)
            continue

        ckpt_dst = dst_root / "checkpoints"
        best_status = copy_file(
            entry["best_path"],
            ckpt_dst / entry["best_path"].name,
            overwrite,
            execute,
        )
        final_status = copy_file(
            entry["final_path"],
            ckpt_dst / entry["final_path"].name,
            overwrite,
            execute,
        )

        metadata_statuses = []
        for src, relative_dst in (
            (entry["config_path"], Path("simulation_config.json")),
            (entry["summary_path"], Path("results") / "training_summary.json"),
            (entry["readme_path"], Path("README.txt")),
        ):
            if src.is_file():
                metadata_statuses.append(copy_file(src, dst_root / relative_dst, overwrite, execute))

        row["status"] = "copied" if execute else "would_copy"
        if best_status == "exists" or final_status == "exists":
            row["status"] = "exists"
        row["reason"] = (
            f"best={best_status}; final={final_status}; "
            f"metadata={','.join(metadata_statuses) if metadata_statuses else 'none'}"
        )
        rows.append(row)
    return rows


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "reason",
        "source",
        "destination",
        "env_id",
        "model",
        "seed",
        "run_name",
        "wandb_run_name",
        "algorithm_type",
        "env_id_source",
        "best_checkpoint",
        "final_checkpoint",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect completed Jumanji LBF runs from runs/checkpoints into "
            "not_for_submit/checkpoints_for_eval_LBF/<env_id>/<model>/."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument(
        "--env-id",
        action="append",
        default=[],
        help="Only collect this env_id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(set(MODEL_LABELS.values())),
        default=[],
        help="Only collect this model label. Can be passed multiple times.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-fallback-env-id",
        action="store_true",
        help=(
            "Include older LBF runs whose env_id must be read from config "
            "because wandb.run_name is missing or not JumanjiLBF-*."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually copy files. Without this flag the script only writes a dry-run manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest CSV path. Defaults to <dest-root>/lbf_checkpoint_copy_manifest.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest or args.dest_root / "lbf_checkpoint_copy_manifest.csv"
    rows = collect(
        source_root=args.source_root,
        dest_root=args.dest_root,
        env_ids=set(args.env_id),
        models=set(args.model),
        overwrite=args.overwrite,
        execute=args.execute,
        include_fallback_env_id=args.include_fallback_env_id,
    )
    write_manifest(rows, manifest_path)

    copied = sum(1 for row in rows if row["status"] == "copied")
    would_copy = sum(1 for row in rows if row["status"] == "would_copy")
    exists = sum(1 for row in rows if row["status"] == "exists")
    skipped = sum(1 for row in rows if row["status"] == "skipped")
    mode = "execute" if args.execute else "dry-run"
    print(
        f"{mode}: copied={copied}, would_copy={would_copy}, "
        f"exists={exists}, skipped={skipped}, manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
