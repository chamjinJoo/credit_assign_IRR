#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import csv
import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TZ = ZoneInfo("Asia/Seoul")
DEFAULT_SOURCE_ROOT = REPO_ROOT / "runs" / "checkpoints"
DEFAULT_DEST_ROOT = REPO_ROOT / "checkpoints_for_eval"
DEFAULT_RUN_CSV_ROOT = REPO_ROOT / "runs" / "wandb_csv"
DEFAULT_HOSTNAME_FILTER = os.environ.get("CHECKPOINT_HOSTNAME_FILTER", "bmlgs4")
DEFAULT_WANDB_ENTITY = os.environ.get(
    "CHECKPOINT_WANDB_ENTITY",
    "chamjin-korea-advanced-institute-of-science-and-technology",
)
DEFAULT_WANDB_PROJECT = os.environ.get("CHECKPOINT_WANDB_PROJECT", "pci-marl-smax-jax-v3")
DEFAULT_CREATED_TIME_WINDOW_MINUTES = float(
    os.environ.get("CHECKPOINT_CREATED_TIME_WINDOW_MINUTES", "5")
)
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


def _parse_iso_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=LOCAL_TZ)
    return parsed


def _normalize_lr_value(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    try:
        return str(float(text))
    except ValueError:
        return text


def _path_mtime(path: Path) -> datetime | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _parse_run_dir_timestamp(run_dir_name: str) -> datetime | None:
    match = re.match(r"^(\d{8}_\d{6})", run_dir_name)
    if not match:
        return None
    try:
        parsed = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return parsed.replace(tzinfo=LOCAL_TZ)


def _candidate_import_paths() -> list[str]:
    """Return sys.path entries that do not point at this repository.

    The repository contains a top-level ``wandb/`` directory with saved run
    artifacts.  That directory can shadow the real W&B client package when the
    script is executed from the repo root.  We explicitly filter those path
    entries out before looking up the real client.
    """

    candidate_paths: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            candidate_paths.append(entry)
            continue
        if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
            continue
        candidate_paths.append(entry)
    return candidate_paths


def _import_wandb_client():
    """Import the real W&B client, bypassing the repo-local ``wandb`` folder."""

    try:
        import wandb as wandb_client  # type: ignore
    except Exception:
        wandb_client = None

    if wandb_client is not None and hasattr(wandb_client, "Api"):
        return wandb_client

    spec = importlib.machinery.PathFinder.find_spec("wandb", _candidate_import_paths())
    if spec is None or spec.loader is None:
        raise ImportError("wandb client not found outside the repository-local wandb/ directory")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def wandb_display_name_candidates_from_config(cfg: Any) -> tuple[str, ...]:
    """Return plausible W&B run names for a checkpoint config.

    Some runs were logged with `jax_pci_*` prefixes while others used the
    shorter `jax_*` form. We keep both candidate spellings so the collector can
    verify run existence without missing valid checkpoints.
    """

    algorithm_type = _normalize_text(_get_nested(cfg, "algorithm", "type"))
    wandb_algo_names = [algorithm_type]
    if algorithm_type in {"r_pci", "pci_rnn"}:
        if bool(_get_nested(cfg, "algorithm", "use_centralized_critic", default=False)):
            wandb_algo_names = ["PCI_rmappo"]
        else:
            wandb_algo_names = ["PCI_rippo"]
    elif algorithm_type in {"jax_qpci", "jax_q_pci"}:
        wandb_algo_names = ["JAX_qpci"]
    elif algorithm_type.startswith("jax_pci_"):
        wandb_algo_names = [algorithm_type, "jax_" + algorithm_type.removeprefix("jax_pci_")]

    if not bool(_get_nested(cfg, "network", "use_recurrent_policy", default=True)):
        wandb_algo_names = [f"{name}_FFAgent" for name in wandb_algo_names]

    run_name = str(_get_nested(cfg, "wandb", "run_name", default="") or "")
    run_name_postfix = str(_get_nested(cfg, "wandb", "run_name_postfix", default="") or "")
    run_name_variants: list[str] = []
    if run_name:
        run_name_variants.append(run_name)
    if run_name_postfix:
        normalized_run_name = _normalize_text(run_name)
        normalized_postfix = _normalize_text(run_name_postfix)
        if normalized_run_name and normalized_run_name.endswith(normalized_postfix):
            stripped = run_name[: max(0, len(run_name) - len(run_name_postfix))].rstrip("_")
            if stripped:
                run_name_variants.append(stripped)
        else:
            combined = f"{run_name}_{run_name_postfix}" if run_name else run_name_postfix
            run_name_variants.append(combined)

    seed = int(_get_nested(cfg, "training", "seed", default=-1) or -1)
    candidates = []
    for wandb_algo_name in wandb_algo_names:
        if run_name_variants:
            for run_name_variant in run_name_variants:
                if run_name_variant:
                    candidates.append(f"{wandb_algo_name}_{run_name_variant}_seed{seed}")
        else:
            candidates.append(wandb_algo_name)
    return tuple(dict.fromkeys(candidates))


def wandb_display_name_from_config(cfg: Any) -> str:
    candidates = wandb_display_name_candidates_from_config(cfg)
    return candidates[0] if candidates else ""


def run_signature_from_config(cfg: Any) -> tuple[str, str, str, str, str, str]:
    map_name = _normalize_text(_get_nested(cfg, "environment", "map_name"))
    wandb_run_name = str(_get_nested(cfg, "wandb", "run_name", default="") or "")
    return (
        map_name,
        _normalize_text(_get_nested(cfg, "algorithm", "type")),
        _normalize_text(constraint_key_from_run_name(wandb_run_name, map_name=map_name)),
        _normalize_text(_get_nested(cfg, "training", "seed")),
        _normalize_lr_value(_get_nested(cfg, "network", "lr_bridge")),
        _normalize_text(wandb_run_name),
    )


def checkpoint_filenames(algorithm_type: str) -> tuple[str, str]:
    key = _normalize_text(algorithm_type)
    if key.startswith("jax_"):
        return "jax_checkpoint_best.pkl", "jax_checkpoint_final.pkl"
    return "best_model.pt", "final_checkpoint.pt"


def checkpoint_time_check(
    entry: dict[str, Any],
    *,
    reference_timestamp: datetime | None,
    window_minutes: float = 5.0,
) -> tuple[bool, str]:
    """Compare the run directory timestamp against the exported CSV timestamp.

    The CSV now carries the exact W&B creation time, so we use that as the
    reference and accept a small symmetric window to tolerate clock skew and
    filesystem timestamp quirks.
    """

    if reference_timestamp is None:
        return False, "missing CSV Created timestamp"

    run_dir_ts = _parse_run_dir_timestamp(entry["run_dir"].name)
    if run_dir_ts is None:
        return False, "missing run directory timestamp"

    window = timedelta(minutes=float(window_minutes))
    earliest_allowed = reference_timestamp - window
    latest_allowed = reference_timestamp + window

    if run_dir_ts < earliest_allowed:
        return False, "run directory predates CSV Created timestamp"
    if run_dir_ts > latest_allowed:
        return False, "run directory too far from CSV Created timestamp"
    return True, "ok"


def live_wandb_run_exists(
    entity: str,
    project: str,
    run_name: str,
) -> tuple[bool, str]:
    try:
        wandb = _import_wandb_client()
    except ImportError as exc:
        return False, f"wandb client unavailable: {exc}"

    api = wandb.Api()
    try:
        runs = api.runs(f"{entity}/{project}")
    except Exception as exc:
        return False, f"wandb project query failed for live run check: {exc}"

    normalized_run_name = _normalize_text(run_name)
    for run in runs:
        run_state = _normalize_text(getattr(run, "state", "") or "")
        if run_state == "deleted":
            continue
        live_names = {
            _normalize_text(getattr(run, "display_name", "") or ""),
            _normalize_text(getattr(run, "name", "") or ""),
        }
        if normalized_run_name in live_names:
            return True, "ok"

    return False, f"wandb run not found in live project: {run_name}"


def _wandb_lookup_key(cfg: Any) -> tuple[str, str]:
    """Return the exact W&B run identity used for matching.

    We match on the reconstructed W&B display name plus the configured group so
    that archived and clean runs with similar settings never get merged into a
    single lookup bucket.
    """

    wandb_name = wandb_display_name_from_config(cfg)
    wandb_group = str(_get_nested(cfg, "wandb", "group", default="") or "")
    return _normalize_text(wandb_name), _normalize_text(wandb_group)

# 허용할 run state 화이트리스트 — 이 목록에 없으면 존재하지 않는 것으로 간주
VALID_RUN_STATES = {"finished", "running", "failed", "crashed", "killed"}


def resolve_wandb_run_lookup(
    entity: str, project: str, display_names: tuple[str, ...]
) -> dict[str, set[str]]:
    """display_name → groups 매핑 반환.

    수정 사항:
    - display_name 서버 필터 제거 (wandb 미지원 필드)
    - state 화이트리스트로 삭제/이동된 run 명시적 제거
    - display_name 클라이언트 측 매칭으로 변경
    """
    try:
        wandb = _import_wandb_client()
    except ImportError as exc:
        raise RuntimeError(
            "wandb is required. Activate the pcimarl conda env or install wandb."
        ) from exc

    if not display_names:
        return {}

    target_names = {_normalize_text(n) for n in display_names if n}
    api = wandb.Api()

    try:
        # display_name 서버 필터 제거 — 미지원 필드라 무시되거나 빈 결과 반환
        # state 화이트리스트로 삭제·이동된 run 명시적 제거
        runs = api.runs(
            f"{entity}/{project}",
            filters={"state": {"$in": sorted(VALID_RUN_STATES)}},
        )
    except Exception as exc:
        raise RuntimeError(
            f"api.runs() failed for {entity}/{project}: {exc}"
        ) from exc

    lookup: dict[str, set[str]] = {}
    for run in runs:
        run_display = _normalize_text(getattr(run, "display_name", "") or "")
        if run_display not in target_names:
            continue  # 클라이언트 측 display_name 매칭

        # VALID_RUN_STATES에 포함돼도 혹시 state가 다를 경우 이중 체크
        run_state = _normalize_text(getattr(run, "state", "") or "")
        if run_state and run_state not in VALID_RUN_STATES:
            continue

        group = _normalize_text(getattr(run, "group", "") or "")
        entry = lookup.setdefault(run_display, set())
        if group:
            entry.add(group)

    return lookup

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
        simulation_timestamp = sim_info.get("timestamp")
        wandb_project = _get_nested(cfg, "wandb", "project")
        wandb_name = (
            _get_nested(cfg, "wandb", "name")
            or _get_nested(cfg, "wandb", "run_name")
            or _get_nested(cfg, "wandb", "group")
            or run_dir.name
        )
        wandb_display_names = wandb_display_name_candidates_from_config(cfg)
        wandb_display_name = wandb_display_name_from_config(cfg)
        wandb_group = str(_get_nested(cfg, "wandb", "group", default="") or "")
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
            "simulation_timestamp": simulation_timestamp,
            "wandb_project": wandb_project,
            "wandb_name": str(wandb_name),
            "wandb_display_name": wandb_display_name,
            "wandb_display_names": wandb_display_names,
            "wandb_group": wandb_group,
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


def should_exclude_group(group: str) -> bool:
    group = _normalize_text(group)
    return any(token in group for token in EXCLUDED_GROUP_TOKENS)


def destination_root(dest_root: Path, map_name: str, constraint_key: str, model_label: str, run_name: str, env_dir_name: str) -> Path:
    return dest_root / f"smax_{map_name}" / constraint_key / model_label / run_name / env_dir_name


def copy_file(src: Path, dst: Path, overwrite: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return "exists"
    shutil.copy2(src, dst)
    return "copied"

def load_csv_allowlist(
    run_csv_root: Path, hostname_filter: str | None
) -> dict[str, dict[str, Any]]:
    allowlist: dict[str, dict[str, Any]] = {}
    csv_files = sorted(run_csv_root.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"no CSV files found under {run_csv_root}")

    normalized_hostname_filter = _normalize_text(hostname_filter) if hostname_filter else ""
    for csv_path in csv_files:
        csv_map = csv_path.parent.name
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                csv_name = str(row.get("Name", "") or "")
                csv_group = str(row.get("Group", row.get("wandb.group", "")) or "")
                csv_run_name = str(row.get("wandb.run_name", "") or "")
                normalized_csv_name = _normalize_text(csv_name)
                if not normalized_csv_name:
                    continue
                if not csv_map:
                    continue
                parsed_map = csv_group or csv_map
                parsed_constraint = constraint_key_from_run_name(
                    csv_name or csv_run_name,
                    map_name=parsed_map,
                )
                hostname = _normalize_text(row.get("Hostname", ""))
                if normalized_hostname_filter and hostname != normalized_hostname_filter:
                    continue
                if normalized_csv_name in allowlist:
                    continue
                allowlist[normalized_csv_name] = {
                    "csv_path": str(csv_path),
                    "csv_map": parsed_map,
                    "csv_name": csv_name,
                    "csv_name_normalized": normalized_csv_name,
                    "csv_created": _parse_iso_timestamp(row.get("Created")),
                    "csv_hostname": row.get("Hostname", ""),
                    "csv_group": csv_group,
                    "csv_run_name": csv_run_name,
                    "csv_run_name_postfix": row.get("wandb.run_name_postfix", ""),
                    "csv_constraint": parsed_constraint,
                }
    if normalized_hostname_filter and not allowlist:
        raise RuntimeError(
            f"no runs matched hostname filter {hostname_filter!r} under {run_csv_root}"
        )
    return allowlist


def collect_checkpoints(
    source_root: Path,
    dest_root: Path,
    *,
    run_csv_root: Path,
    hostname_filter: str | None,
    wandb_entity: str,
    wandb_project: str,
    overwrite: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    entries = list(iter_run_metadata(source_root))
    allowlist = load_csv_allowlist(run_csv_root, hostname_filter)

    rows: list[dict[str, Any]] = []
    for entry in entries:
        metadata_path = entry["metadata_path"]

        if "skip_reason" in entry:
            rows.append({
                "status": "skipped",
                "reason": entry["skip_reason"],
                "source": str(metadata_path),
                "destination": "",
                "map_name": "",
                "constraint_key": "",
                "model_label": "",
                "run_name": "",
                "wandb_display_name": "",
                "wandb_display_names": "",
                "simulation_timestamp": "",
                "best_mtime": "",
                "final_mtime": "",
                "csv_name": "",
                "csv_created": "",
                "csv_hostname": "",
                "csv_map": "",
                "csv_path": "",
            })
            continue

        keep, reason = should_copy(entry, None)
        if not keep:
            rows.append({
                "status": "skipped",
                "reason": reason,
                "source": str(metadata_path),
                "destination": "",
                "map_name": entry["map_name"],
                "constraint_key": entry["constraint_key"],
                "model_label": entry["model_label"],
                "run_name": entry["run_dir"].name,
                "wandb_display_name": entry["wandb_display_name"],
                "wandb_display_names": " | ".join(entry["wandb_display_names"]),
                "simulation_timestamp": str(entry.get("simulation_timestamp", "")),
                "best_mtime": "",
                "final_mtime": "",
                "csv_name": "",
                "csv_created": "",
                "csv_hostname": "",
                "csv_map": "",
                "csv_path": "",
            })
            continue

        candidate_names = [
            _normalize_text(name)
            for name in (
                entry.get("wandb_display_name"),
                *entry.get("wandb_display_names", ()),
                entry.get("wandb_name"),
            )
            if name
        ]
        matched_csv = None
        for candidate_name in candidate_names:
            matched_csv = allowlist.get(candidate_name)
            if matched_csv is not None:
                break

        if matched_csv is None:
            rows.append({
                "status": "skipped",
                "reason": (
                    "wandb run not found in CSV allowlist: "
                    f"{', '.join(candidate_names) if candidate_names else entry['run_dir'].name}"
                ),
                "source": str(metadata_path),
                "destination": "",
                "map_name": entry["map_name"],
                "constraint_key": entry["constraint_key"],
                "model_label": entry["model_label"],
                "run_name": entry["run_dir"].name,
                "wandb_display_name": entry["wandb_display_name"],
                "wandb_display_names": " | ".join(entry["wandb_display_names"]),
                "simulation_timestamp": str(entry.get("simulation_timestamp", "")),
                "best_mtime": "",
                "final_mtime": "",
                "csv_name": "",
                "csv_created": "",
                "csv_hostname": "",
                "csv_map": "",
                "csv_path": "",
            })
            continue

        time_ok, time_reason = checkpoint_time_check(
            entry,
            reference_timestamp=matched_csv["csv_created"],
            window_minutes=DEFAULT_CREATED_TIME_WINDOW_MINUTES,
        )
        if not time_ok:
            rows.append({
                "status": "skipped",
                "reason": time_reason,
                "source": str(metadata_path),
                "destination": "",
                "map_name": entry["map_name"],
                "constraint_key": entry["constraint_key"],
                "model_label": entry["model_label"],
                "run_name": entry["run_dir"].name,
                "wandb_display_name": entry["wandb_display_name"],
                "wandb_display_names": " | ".join(entry["wandb_display_names"]),
                "simulation_timestamp": str(entry.get("simulation_timestamp", "")),
                "best_mtime": "",
                "final_mtime": "",
                "csv_name": matched_csv["csv_name"],
                "csv_created": str(matched_csv["csv_created"] or ""),
                "csv_hostname": matched_csv["csv_hostname"],
                "csv_map": matched_csv["csv_map"],
                "csv_path": matched_csv["csv_path"],
            })
            continue

        run_ok, run_reason = live_wandb_run_exists(
            wandb_entity,
            wandb_project,
            matched_csv["csv_name"],
        )
        if not run_ok:
            rows.append({
                "status": "skipped",
                "reason": run_reason,
                "source": str(metadata_path),
                "destination": "",
                "map_name": entry["map_name"],
                "constraint_key": entry["constraint_key"],
                "model_label": entry["model_label"],
                "run_name": entry["run_dir"].name,
                "wandb_display_name": entry["wandb_display_name"],
                "wandb_display_names": " | ".join(entry["wandb_display_names"]),
                "simulation_timestamp": str(entry.get("simulation_timestamp", "")),
                "best_mtime": "",
                "final_mtime": "",
                "csv_name": matched_csv["csv_name"],
                "csv_created": str(matched_csv["csv_created"] or ""),
                "csv_hostname": matched_csv["csv_hostname"],
                "csv_map": matched_csv["csv_map"],
                "csv_path": matched_csv["csv_path"],
            })
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
            copy_file(entry["summary_path"], summary_dst, overwrite=overwrite)
            copy_file(entry["best_path"], best_dst, overwrite=overwrite)
            copy_file(entry["final_path"], final_dst, overwrite=overwrite)
            if readme_src.is_file():
                copy_file(readme_src, readme_dst, overwrite=overwrite)

        rows.append({
            "status": "copied" if not dry_run else "dry_run",
            "reason": reason,
            "source": str(metadata_path),
            "destination": str(dst_dir),
            "map_name": entry["map_name"],
            "constraint_key": entry["constraint_key"],
            "model_label": entry["model_label"],
            "run_name": entry["run_dir"].name,
            "wandb_display_name": entry["wandb_display_name"],
            "wandb_display_names": " | ".join(entry["wandb_display_names"]),
            "simulation_timestamp": str(entry.get("simulation_timestamp", "")),
            "best_mtime": str(_path_mtime(entry["best_path"])),
            "final_mtime": str(_path_mtime(entry["final_path"])),
            "csv_name": matched_csv["csv_name"],
            "csv_created": str(matched_csv["csv_created"] or ""),
            "csv_hostname": matched_csv["csv_hostname"],
            "csv_map": matched_csv["csv_map"],
            "csv_path": matched_csv["csv_path"],
        })

    return rows

def write_manifest(rows: list[dict[str, Any]], dest_root: Path) -> Path:
    manifest_path = dest_root / "smax_checkpoint_copy_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "reason",
        "source",
        "destination",
        "map_name",
        "constraint_key",
        "model_label",
        "run_name",
        "wandb_display_name",
        "wandb_display_names",
        "simulation_timestamp",
        "best_mtime",
        "final_mtime",
        "wandb_group",
        "csv_name",
        "csv_created",
        "csv_hostname",
        "csv_map",
        "csv_path",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect completed SMAX checkpoints from runs/checkpoints into checkpoints_for_eval."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument("--run-csv-root", type=Path, default=DEFAULT_RUN_CSV_ROOT)
    parser.add_argument("--hostname-filter", type=str, default=DEFAULT_HOSTNAME_FILTER)
    parser.add_argument("--wandb-entity", type=str, default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
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
        run_csv_root=args.run_csv_root.expanduser().resolve(),
        hostname_filter=args.hostname_filter,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
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
