from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from configs.config import Config

PROJECT_FILTER = "pci-marl-smax-jax-v3"

MAP_ORDER = ["3m", "2s3z", "3s5z", "smacv2_5_units", "smacv2_10_units", "6h_vs_8z"]
CONSTRAINT_ORDER = ["vanilla", "tar_w=2", "tar_w=3", "tar_w=4", "tar_w=5", "tar_w=6", "tar_w=8", "cb"]
MODEL_ORDER = ["pci-ippo", "ippo", "pci-mappo", "mappo"]

MODEL_PRETTY_NAMES = {
    "jax_pci_mappo": "pci-mappo",
    "jax_pci_ippo": "pci-ippo",
    "jax_mappo": "mappo",
    "jax_ippo": "ippo",
    "pci_mappo": "pci-mappo",
    "pci_ippo": "pci-ippo",
    "mappo": "mappo",
    "ippo": "ippo",
}

MAP_PRETTY_NAMES = {
    "3m": "3m",
    "2s3z": "2s3z",
    "3s5z": "3s5z",
    "smacv2_5_units": "SMACv2-5U",
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

_PROJECT_FACTORIES: Dict[str, Dict[str, Any]] = {}
_ENV_CACHE: Dict[str, Any] = {}
_RUN_EVAL_CACHE: Dict[tuple[Any, ...], Dict[str, Any]] = {}
_RUN_EVAL_DISK_CACHE: Dict[Path, Dict[str, Any]] = {}
_DUPLICATE_POLICIES = {"error", "latest", "best_metric", "all"}
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_TEMPLATE_PATHS = {
    "pci_mappo": _REPO_ROOT / "configs_jax/smax_pci_mappo.yaml",
    "pci_ippo": _REPO_ROOT / "configs_jax/smax_pci_ippo.yaml",
    "mappo": _REPO_ROOT / "configs_jax/smax_mappo.yaml",
    "ippo": _REPO_ROOT / "configs_jax/smax_ippo.yaml",
}
_MODEL_ALGORITHM_TYPES = {
    "pci_mappo": "jax_pci_mappo",
    "pci_ippo": "jax_pci_ippo",
    "mappo": "jax_mappo",
    "ippo": "jax_ippo",
}


def normalize_algorithm_type(algorithm_type: str) -> str:
    return str(algorithm_type).lower().replace("-", "_")


def model_key_from_algorithm_type(algorithm_type: str) -> str:
    key = normalize_algorithm_type(algorithm_type)
    return key.removeprefix("jax_")


def model_label_from_algorithm_type(algorithm_type: str) -> str:
    return MODEL_PRETTY_NAMES.get(model_key_from_algorithm_type(algorithm_type), model_key_from_algorithm_type(algorithm_type))


def map_label(map_name: str) -> str:
    return MAP_PRETTY_NAMES.get(map_name, map_name)


def constraint_label(constraint_key: str) -> str:
    return CONSTRAINT_PRETTY_NAMES.get(constraint_key, constraint_key)


def _normalize_name(text: str) -> str:
    return str(text).strip().lower()


def _constraint_sort_key(constraint_key: str) -> tuple[int, Any]:
    if constraint_key == "vanilla":
        return (0, 0)
    if constraint_key.startswith("tar_w="):
        match = re.fullmatch(r"tar_w=(\d+)", constraint_key)
        if match is not None:
            return (1, int(match.group(1)))
        return (1, constraint_key)
    if constraint_key == "cb":
        return (2, 0)
    return (3, constraint_key)


def _ordered_constraint_keys(keys: Iterable[str]) -> list[str]:
    keys = [str(key) for key in keys if str(key)]
    seen = set()
    ordered = []
    for key in CONSTRAINT_ORDER:
        if key in keys and key not in seen:
            ordered.append(key)
            seen.add(key)
    for key in sorted(set(keys) - seen, key=_constraint_sort_key):
        ordered.append(key)
    return ordered


def ordered_map_names(map_names: Iterable[str]) -> list[str]:
    map_names = [str(name) for name in map_names if str(name)]
    seen = set()
    ordered = []
    for name in MAP_ORDER:
        if name in map_names and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in map_names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def ordered_model_labels(model_labels: Iterable[str]) -> list[str]:
    model_labels = [str(name) for name in model_labels if str(name)]
    seen = set()
    ordered = []
    for name in MODEL_ORDER:
        if name in model_labels and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in model_labels:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _parse_map_name_from_checkpoint_tree(root_relative: Path) -> str:
    map_dir = root_relative.parts[0]
    if not map_dir.startswith("smax_"):
        return map_dir
    return map_dir[len("smax_") :]


def _parse_seed_from_run_dir(run_dir_name: str) -> int:
    match = re.search(r"_seed(\d+)$", str(run_dir_name))
    if match is None:
        raise ValueError(f"Could not infer training seed from run directory name: {run_dir_name}")
    return int(match.group(1))


def _build_checkpoint_only_config(
    *,
    model_key: str,
    map_name: str,
    constraint_key: str,
    seed: int,
) -> Config:
    template_path = _MODEL_TEMPLATE_PATHS.get(model_key)
    algorithm_type = _MODEL_ALGORITHM_TYPES.get(model_key)
    if template_path is None or algorithm_type is None:
        raise ValueError(f"No fallback template is registered for checkpoint-only model {model_key!r}")

    cfg = Config.from_yaml(str(template_path))
    cfg.algorithm.type = algorithm_type
    cfg.training.seed = int(seed)
    cfg.environment.map_name = str(map_name)
    cfg.environment.seed = int(seed)
    cfg.environment.use_allowed_targets = False
    cfg.environment.allowed_target_window = None
    cfg.environment.use_cooperative_kill_bonus = False

    if constraint_key.startswith("tar_w="):
        cfg.environment.use_allowed_targets = True
        cfg.environment.allowed_target_window = int(constraint_key.split("=", 1)[1])
    elif constraint_key == "cb":
        cfg.environment.use_cooperative_kill_bonus = True

    return cfg


def _discover_checkpoint_only_runs(root: Path, project_filter: Optional[str]) -> list[Dict[str, Any]]:
    root = Path(root)
    rows = []

    for final_checkpoint in sorted(root.rglob("jax_checkpoint_final.pkl")):
        checkpoint_dir = final_checkpoint.parent
        env_dir = checkpoint_dir.parent
        run_dir = env_dir.parent
        try:
            rel = run_dir.relative_to(root)
        except ValueError:
            continue

        if len(rel.parts) < 4:
            continue

        metadata_path = env_dir / "simulation_config.json"
        if metadata_path.exists():
            continue

        map_name = _parse_map_name_from_checkpoint_tree(rel)
        constraint_key = str(rel.parts[1])
        model_key = normalize_algorithm_type(rel.parts[2])
        seed = _parse_seed_from_run_dir(run_dir.name)
        try:
            cfg = _build_checkpoint_only_config(
                model_key=model_key,
                map_name=map_name,
                constraint_key=constraint_key,
                seed=seed,
            )
        except ValueError as exc:
            warnings.warn(f"Skipping checkpoint-only run at {run_dir}: {exc}", stacklevel=2)
            continue

        project = project_filter or PROJECT_FILTER
        wandb_run_name = f"smax_{map_name}_{constraint_key}_{model_key}_seed{seed}"
        algorithm_type = normalize_algorithm_type(cfg.algorithm.type)
        model_label = model_label_from_algorithm_type(cfg.algorithm.type)
        ckpt = _checkpoint_paths(run_dir, env_dir.name, cfg.algorithm.type)
        env_signature = json.dumps(
            {
                "algorithm_type": algorithm_type,
                "env": cfg.environment.get_relevant_config(),
                "seed": seed,
            },
            sort_keys=True,
            default=str,
        )

        rows.append(
            {
                "run_dir": run_dir,
                "metadata_path": metadata_path,
                "project": project,
                "map_name": map_name,
                "map_label": map_label(map_name),
                "wandb_run_name": wandb_run_name,
                "constraint_key": constraint_key,
                "constraint_label": constraint_label(constraint_key),
                "algorithm_type": algorithm_type,
                "model_key": model_key,
                "model_label": model_label,
                "seed": seed,
                "config": cfg,
                "env_dir_name": env_dir.name,
                "best_checkpoint": ckpt["best"],
                "final_checkpoint": ckpt["final"],
                "env_signature": env_signature,
            }
        )

    return rows


def constraint_key_from_run_name(run_name: str, map_name: str | None = None) -> str:
    """Infer the training constraint from a wandb run name.

    The repo's wandb naming convention encodes the map and then appends the
    active constraint token, e.g. ``smax_2s3z_tar_w=3``.  Anything after that
    token (such as ``_v2`` or ``_noSch``) is treated as a memo/variant rather
    than a separate constraint.
    """
    name = _normalize_name(run_name)
    if "smax_" in name:
        name = name[name.index("smax_") + 5 :]
    elif name.startswith("jax_"):
        name = name[4:]

    if map_name:
        map_prefix = _normalize_name(map_name)
        if name.startswith(map_prefix):
            name = name[len(map_prefix):].lstrip("_")

    if not name:
        return "vanilla"

    tar_match = re.search(r"tar_w=\d+", name)
    if tar_match is not None:
        return tar_match.group(0)

    if "cb" in name:
        return "cb"
    return "vanilla"


def _cfg_extra(env_cfg: Any, name: str, default: Any = None) -> Any:
    if hasattr(env_cfg, name):
        value = getattr(env_cfg, name)
        if value is not None:
            return value

    env_args = getattr(env_cfg, "env_args", None) or {}
    return env_args.get(name, default)


def bootstrap_project(device: str) -> Dict[str, Any]:
    """Import the project runtime after pinning the JAX backend."""
    cache_key = str(device).lower()
    if cache_key in _PROJECT_FACTORIES:
        return _PROJECT_FACTORIES[cache_key]

    if cache_key != "cpu":
        command_buffer_flag = "--xla_gpu_enable_command_buffer="
        existing_xla_flags = os.environ.get("XLA_FLAGS", "")
        if command_buffer_flag not in existing_xla_flags:
            os.environ["XLA_FLAGS"] = f"{existing_xla_flags} {command_buffer_flag}".strip()
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import main as project_main

    project_main.configure_jax_runtime(device)
    project_main.import_runtime_modules()

    from envs import SMAXBatchWrapper, SMAXWrapper, create_environment
    from runner import Trainer, JAXTrainer

    factories = {
        "main": project_main,
        "create_environment": create_environment,
        "SMAXWrapper": SMAXWrapper,
        "SMAXBatchWrapper": SMAXBatchWrapper,
        "Trainer": Trainer,
        "JAXTrainer": JAXTrainer,
    }
    _PROJECT_FACTORIES[cache_key] = factories
    return factories


def load_run_config(run_dir: Path) -> Config:
    run_dir = Path(run_dir)
    candidates = sorted(run_dir.rglob("simulation_config.json"))
    if not candidates:
        raise FileNotFoundError(f"simulation_config.json not found under {run_dir}")
    metadata_path = candidates[0]
    payload = json.loads(metadata_path.read_text())
    return Config.from_dict(payload["config"])


def _checkpoint_paths(run_dir: Path, env_dir_name: str, algorithm_type: str) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    env_dir_name = str(env_dir_name)
    key = normalize_algorithm_type(algorithm_type)
    if key.startswith("jax_"):
        ckpt_dir = run_dir / env_dir_name / "checkpoints"
        return {
            "best": ckpt_dir / "jax_checkpoint_best.pkl",
            "final": ckpt_dir / "jax_checkpoint_final.pkl",
        }

    ckpt_dir = run_dir / env_dir_name / "checkpoints"
    return {
        "best": ckpt_dir / "best_model.pt",
        "final": ckpt_dir / "final_checkpoint.pt",
    }


def discover_runs(root: Path, project_filter: Optional[str] = PROJECT_FILTER) -> pd.DataFrame:
    root = Path(root)
    rows = []

    for metadata_path in sorted(root.rglob("simulation_config.json")):
        run_dir = metadata_path.parents[1]
        env_dir_name = metadata_path.parent.name
        payload = json.loads(metadata_path.read_text())
        cfg = Config.from_dict(payload["config"])

        project = getattr(cfg.wandb, "project", None)
        if project_filter and project != project_filter:
            continue

        wandb_run_name = str(
            getattr(cfg.wandb, "name", None)
            or getattr(cfg.wandb, "run_name", None)
            or getattr(cfg.wandb, "group", None)
            or run_dir.name
        )
        algorithm_type = normalize_algorithm_type(cfg.algorithm.type)
        model_key = model_key_from_algorithm_type(cfg.algorithm.type)
        model = model_label_from_algorithm_type(cfg.algorithm.type)
        map_name = getattr(cfg.environment, "map_name", "unknown")
        constraint_key = constraint_key_from_run_name(wandb_run_name, map_name=map_name)
        seed = int(getattr(cfg.training, "seed", 0))

        ckpt = _checkpoint_paths(run_dir, env_dir_name, cfg.algorithm.type)
        env_signature = json.dumps(
            {
                "algorithm_type": algorithm_type,
                "env": cfg.environment.get_relevant_config(),
                "seed": seed,
            },
            sort_keys=True,
            default=str,
        )

        rows.append(
            {
                "run_dir": run_dir,
                "metadata_path": metadata_path,
                "project": project,
                "map_name": map_name,
                "map_label": map_label(map_name),
                "wandb_run_name": wandb_run_name,
                "constraint_key": constraint_key,
                "constraint_label": constraint_label(constraint_key),
                "algorithm_type": algorithm_type,
                "model_key": model_key,
                "model_label": model,
                "seed": seed,
                "config": cfg,
                "env_dir_name": env_dir_name,
                "best_checkpoint": ckpt["best"],
                "final_checkpoint": ckpt["final"],
                "env_signature": env_signature,
            }
        )

    rows.extend(_discover_checkpoint_only_runs(root, project_filter))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["run_dir"], keep="first")
    return df.sort_values(["map_name", "constraint_key", "model_key", "seed", "run_dir"]).reset_index(drop=True)


def _build_smax_env(cfg: Config, *, n_envs: int, batch: bool) -> Any:
    factories = bootstrap_project(cfg.training.device)
    env_cfg = cfg.environment
    env_signature = json.dumps(
        {
            "device": str(cfg.training.device).lower(),
            "batch": batch,
            "n_envs": int(n_envs),
            "warmup_episodes": 0 if batch else None,
            "algorithm_type": normalize_algorithm_type(cfg.algorithm.type),
            "env": env_cfg.get_relevant_config(),
            "seed": int(_cfg_extra(env_cfg, "seed", 0)),
        },
        sort_keys=True,
        default=str,
    )
    if env_signature in _ENV_CACHE:
        return _ENV_CACHE[env_signature]

    common_kwargs = {
        "map_name": getattr(env_cfg, "map_name", None),
        "num_allies": int(_cfg_extra(env_cfg, "num_allies", 5)),
        "num_enemies": int(_cfg_extra(env_cfg, "num_enemies", 5)),
        "seed": int(_cfg_extra(env_cfg, "seed", 0)),
        "max_steps": int(_cfg_extra(env_cfg, "max_steps", 100)),
        "use_individual_rewards": bool(_cfg_extra(env_cfg, "use_individual_rewards", False)),
        "enemy_shoots": bool(_cfg_extra(env_cfg, "enemy_shoots", True)),
        "attack_mode": _cfg_extra(env_cfg, "attack_mode", "closest"),
        "map_width": int(_cfg_extra(env_cfg, "map_width", 32)),
        "map_height": int(_cfg_extra(env_cfg, "map_height", 32)),
        "world_steps_per_env_step": int(_cfg_extra(env_cfg, "world_steps_per_env_step", 8)),
        "time_per_step": float(_cfg_extra(env_cfg, "time_per_step", 1.0 / 16)),
        "see_enemy_actions": bool(_cfg_extra(env_cfg, "see_enemy_actions", True)),
        "won_battle_bonus": float(_cfg_extra(env_cfg, "won_battle_bonus", 1.0)),
        "walls_cause_death": bool(_cfg_extra(env_cfg, "walls_cause_death", True)),
        "observation_type": _cfg_extra(env_cfg, "observation_type", "unit_list"),
        "smacv2_position_generation": bool(_cfg_extra(env_cfg, "smacv2_position_generation", False)),
        "smacv2_unit_type_generation": bool(_cfg_extra(env_cfg, "smacv2_unit_type_generation", False)),
        "use_allowed_targets": bool(_cfg_extra(env_cfg, "use_allowed_targets", False)),
        "allowed_target_window": _cfg_extra(env_cfg, "allowed_target_window", None),
        "allowed_targets": _cfg_extra(env_cfg, "allowed_targets", None),
        "use_partial_damage_restore": bool(_cfg_extra(env_cfg, "use_partial_damage_restore", False)),
        "use_cooperative_kill_bonus": bool(_cfg_extra(env_cfg, "use_cooperative_kill_bonus", False)),
        "cooperative_bonus_mode": _cfg_extra(env_cfg, "cooperative_bonus_mode", "kill"),
        "cooperative_size": int(_cfg_extra(env_cfg, "cooperative_size", _cfg_extra(env_cfg, "coop_size", 2))),
        "cooperative_kill_window": int(_cfg_extra(env_cfg, "cooperative_kill_window", 2)),
        "cooperative_kill_bonus": float(_cfg_extra(env_cfg, "cooperative_kill_bonus", 2.0)),
        "cooperative_kill_owner_ratio": float(_cfg_extra(env_cfg, "cooperative_kill_owner_ratio", 0.7)),
    }

    if batch:
        env = factories["SMAXBatchWrapper"](n_envs=n_envs, warmup_episodes=0, **common_kwargs)
    else:
        env = factories["SMAXWrapper"](**common_kwargs)

    _ENV_CACHE[env_signature] = env
    return env


def _evaluation_cache_key(
    run_row: Dict[str, Any],
    *,
    checkpoint_choice: str,
    device: str,
    eval_episodes: int,
    deterministic: bool,
    eval_seed: Optional[int],
) -> tuple[Any, ...]:
    checkpoint_path = Path(run_row[f"{checkpoint_choice}_checkpoint"])
    checkpoint_mtime = checkpoint_path.stat().st_mtime_ns if checkpoint_path.exists() else None
    return (
        str(run_row["run_dir"]),
        str(run_row["metadata_path"]),
        checkpoint_choice,
        str(device).lower(),
        int(eval_episodes),
        bool(deterministic),
        None if eval_seed is None else int(eval_seed),
        str(checkpoint_path),
        checkpoint_mtime,
    )


def _evaluation_cache_key_str(cache_key: tuple[Any, ...]) -> str:
    return json.dumps(list(cache_key), ensure_ascii=True, default=str)


def _normalize_eval_seeds(eval_seeds: Optional[Iterable[int]]) -> list[Optional[int]]:
    if eval_seeds is None:
        return [None]
    normalized = [int(seed) for seed in eval_seeds]
    if not normalized:
        raise ValueError("eval_seeds must contain at least one seed")
    return normalized


def _normalize_duplicate_policy(duplicate_policy: str) -> str:
    policy = str(duplicate_policy).strip().lower()
    if policy not in _DUPLICATE_POLICIES:
        raise ValueError(
            f"duplicate_policy must be one of {sorted(_DUPLICATE_POLICIES)}, got {duplicate_policy!r}"
        )
    return policy


def _checkpoint_best_metric(run_row: pd.Series | Dict[str, Any], checkpoint_choice: str) -> Optional[float]:
    if isinstance(run_row, pd.Series):
        run_row = run_row.to_dict()

    checkpoint_path = Path(run_row[f"{checkpoint_choice}_checkpoint"])
    if not checkpoint_path.exists():
        return None

    algorithm_type = normalize_algorithm_type(run_row["algorithm_type"])
    try:
        if algorithm_type.startswith("jax_"):
            with checkpoint_path.open("rb") as f:
                checkpoint = pickle.load(f)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    metric = checkpoint.get("best_eval_reward")
    if metric is None:
        return None
    return float(metric)


def _run_latest_timestamp_ns(run_row: pd.Series | Dict[str, Any], checkpoint_choice: str) -> int:
    if isinstance(run_row, pd.Series):
        run_row = run_row.to_dict()

    timestamps = []
    metadata_path = Path(run_row["metadata_path"])
    checkpoint_path = Path(run_row[f"{checkpoint_choice}_checkpoint"])
    for path in (metadata_path, checkpoint_path):
        try:
            timestamps.append(path.stat().st_mtime_ns)
        except OSError:
            continue
    return max(timestamps) if timestamps else -1


def _format_duplicate_report(duplicate_groups: list[pd.DataFrame], checkpoint_choice: str) -> str:
    lines = []
    for group in duplicate_groups:
        head = group.iloc[0]
        lines.append(
            "Duplicate group: "
            f"map={head['map_name']}, constraint_key={head['constraint_key']}, "
            f"model_label={head['model_label']}, seed={int(head['seed'])}"
        )
        for _, row in group.iterrows():
            metric = _checkpoint_best_metric(row, checkpoint_choice)
            latest_ts = _run_latest_timestamp_ns(row, checkpoint_choice)
            metric_text = "NA" if metric is None else f"{metric:.6f}"
            lines.append(
                "  - "
                f"run_dir={row['run_dir']}, "
                f"wandb_run_name={row.get('wandb_run_name', '')}, "
                f"checkpoint_path={row[f'{checkpoint_choice}_checkpoint']}, "
                f"best_eval_reward={metric_text}, "
                f"latest_mtime_ns={latest_ts}"
            )
    return "\n".join(lines)


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _metadata_fingerprint(run_row: pd.Series | Dict[str, Any]) -> Optional[str]:
    if isinstance(run_row, pd.Series):
        run_row = run_row.to_dict()

    metadata_path = Path(run_row["metadata_path"])
    if not metadata_path.exists():
        return None
    try:
        return _file_sha256(metadata_path)
    except OSError:
        return None


def _checkpoint_fingerprint(run_row: pd.Series | Dict[str, Any], checkpoint_choice: str) -> Optional[str]:
    if isinstance(run_row, pd.Series):
        run_row = run_row.to_dict()

    checkpoint_path = Path(run_row[f"{checkpoint_choice}_checkpoint"])
    if not checkpoint_path.exists():
        return None
    try:
        return _file_sha256(checkpoint_path)
    except OSError:
        return None


def _collapse_mirrored_duplicate_runs(
    runs_df: pd.DataFrame,
    *,
    checkpoint_choice: str,
) -> pd.DataFrame:
    duplicate_keys = ["map_name", "constraint_key", "model_label", "seed"]
    collapsed_groups = []
    dropped_notes = []

    for _, group in runs_df.groupby(duplicate_keys, sort=False, dropna=False):
        if len(group) == 1:
            collapsed_groups.append(group)
            continue

        work = group.copy()
        work["_run_dir_name"] = work["run_dir"].map(lambda value: Path(value).name)
        work["_run_dir_depth"] = work["run_dir"].map(lambda value: len(Path(value).parts))
        work["_metadata_fingerprint"] = work.apply(_metadata_fingerprint, axis=1)
        work["_checkpoint_fingerprint"] = work.apply(
            lambda row: _checkpoint_fingerprint(row, checkpoint_choice),
            axis=1,
        )

        kept_rows = []
        for _, basename_group in work.groupby("_run_dir_name", sort=False, dropna=False):
            fingerprint_keys = [
                "wandb_run_name",
                "env_dir_name",
                "_metadata_fingerprint",
                "_checkpoint_fingerprint",
            ]
            if (
                len(basename_group) > 1
                and basename_group["_metadata_fingerprint"].notna().all()
                and basename_group["_checkpoint_fingerprint"].notna().all()
                and basename_group[fingerprint_keys].nunique(dropna=False).max() == 1
            ):
                basename_group = basename_group.sort_values(
                    ["_run_dir_depth", "run_dir"],
                    ascending=[True, True],
                    kind="stable",
                )
                kept_rows.append(basename_group.head(1))
                kept = basename_group.iloc[0]["run_dir"]
                for dropped in basename_group.iloc[1:]["run_dir"].tolist():
                    dropped_notes.append(f"Collapsed mirrored duplicate run {dropped} into {kept}.")
                continue
            kept_rows.append(basename_group)

        collapsed = pd.concat(kept_rows, ignore_index=True)
        collapsed_groups.append(
            collapsed.drop(
                columns=[
                    "_run_dir_name",
                    "_run_dir_depth",
                    "_metadata_fingerprint",
                    "_checkpoint_fingerprint",
                ],
                errors="ignore",
            )
        )

    if dropped_notes:
        warnings.warn("\n".join(dropped_notes), stacklevel=2)
    return pd.concat(collapsed_groups, ignore_index=True)


def _resolve_duplicate_runs(
    runs_df: pd.DataFrame,
    *,
    checkpoint_choice: str,
    duplicate_policy: str,
) -> pd.DataFrame:
    policy = _normalize_duplicate_policy(duplicate_policy)
    duplicate_keys = ["map_name", "constraint_key", "model_label", "seed"]
    runs_df = _collapse_mirrored_duplicate_runs(
        runs_df,
        checkpoint_choice=checkpoint_choice,
    )
    duplicate_mask = runs_df.duplicated(subset=duplicate_keys, keep=False)
    if not duplicate_mask.any():
        return runs_df.reset_index(drop=True)

    duplicate_groups = [
        group.copy()
        for _, group in runs_df.loc[duplicate_mask].groupby(duplicate_keys, sort=False, dropna=False)
    ]
    report = _format_duplicate_report(duplicate_groups, checkpoint_choice)
    if policy == "error":
        raise ValueError(
            "Duplicate SMAX training runs detected for the same "
            f"{duplicate_keys}. Resolve the duplicates or choose a non-default duplicate_policy.\n"
            f"{report}"
        )

    if policy == "all":
        warnings.warn(
            "duplicate_policy='all' keeps duplicate runs and allows them to contribute to debug summaries. "
            "Do not use this mode for paper tables.\n"
            f"{report}",
            stacklevel=2,
        )
        return runs_df.reset_index(drop=True)

    selected_groups = []
    resolution_notes = []
    for _, group in runs_df.groupby(duplicate_keys, sort=False, dropna=False):
        if len(group) == 1:
            selected_groups.append(group)
            continue

        group = group.copy()
        if policy == "latest":
            group["_latest_mtime_ns"] = group.apply(
                lambda row: _run_latest_timestamp_ns(row, checkpoint_choice),
                axis=1,
            )
            group = group.sort_values("_latest_mtime_ns", ascending=False, kind="stable")
        else:
            group["_best_metric"] = group.apply(
                lambda row: _checkpoint_best_metric(row, checkpoint_choice),
                axis=1,
            )
            if group["_best_metric"].isna().any():
                raise ValueError(
                    "duplicate_policy='best_metric' requires every duplicate run to expose "
                    f"'best_eval_reward' in its checkpoint.\n{report}"
                )
            group["_latest_mtime_ns"] = group.apply(
                lambda row: _run_latest_timestamp_ns(row, checkpoint_choice),
                axis=1,
            )
            group = group.sort_values(
                ["_best_metric", "_latest_mtime_ns"],
                ascending=[False, False],
                kind="stable",
            )

        selected_row = group.head(1).drop(
            columns=[col for col in ["_latest_mtime_ns", "_best_metric"] if col in group.columns]
        )
        selected_groups.append(selected_row)
        chosen = selected_row.iloc[0]
        resolution_notes.append(
            f"Selected {chosen['run_dir']} for map={chosen['map_name']}, "
            f"constraint_key={chosen['constraint_key']}, model_label={chosen['model_label']}, seed={int(chosen['seed'])}."
        )

    warnings.warn(
        f"Resolved duplicate SMAX runs with duplicate_policy='{policy}'.\n"
        + "\n".join(resolution_notes)
        + "\n"
        + report,
        stacklevel=2,
    )
    return pd.concat(selected_groups, ignore_index=True)


def _validate_run_and_summary_protocol(
    run_df: pd.DataFrame,
    seed_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> None:
    if run_df.empty:
        return

    if (run_df["train_seed"] == run_df["eval_seed"]).any():
        collisions = run_df.loc[run_df["train_seed"] == run_df["eval_seed"], ["map_name", "constraint_key", "model_label", "train_seed", "eval_seed"]]
        raise AssertionError(
            "train_seed and eval_seed must differ for checkpoint evaluation.\n"
            f"{collisions.to_string(index=False)}"
        )

    eval_seed_sets = (
        run_df.groupby(["map_name", "constraint_key", "model_label", "seed"], dropna=False)["eval_seed"]
        .agg(lambda values: tuple(sorted({int(v) for v in values})))
        .reset_index(name="eval_seed_set")
    )
    seed_set_counts = (
        eval_seed_sets.groupby(["map_name", "constraint_key", "model_label"], dropna=False)["eval_seed_set"]
        .nunique()
    )
    if (seed_set_counts > 1).any():
        inconsistent = seed_set_counts[seed_set_counts > 1]
        raise AssertionError(
            "Each method must use the same eval_seed set across training seeds.\n"
            f"{inconsistent.to_string()}"
        )

    expected_seed_rows = (
        run_df.groupby(["map_name", "constraint_key", "constraint_label", "model_label"], dropna=False)["seed"]
        .nunique()
        .reset_index(name="expected_rows")
    )
    actual_seed_rows = (
        seed_summary.groupby(["map_name", "constraint_key", "constraint_label", "model_label"], dropna=False)
        .size()
        .reset_index(name="actual_rows")
    )
    merged_counts = expected_seed_rows.merge(
        actual_seed_rows,
        on=["map_name", "constraint_key", "constraint_label", "model_label"],
        how="left",
    )
    if not (merged_counts["expected_rows"] == merged_counts["actual_rows"]).all():
        raise AssertionError(
            "seed_summary row count must equal the number of unique training seeds.\n"
            f"{merged_counts.to_string(index=False)}"
        )

    expected_std = (
        seed_summary.groupby(["map_name", "constraint_key", "constraint_label", "model_label"], dropna=False)["eval_success_rate"]
        .std()
        .reset_index(name="expected_std_success_rate")
    )
    std_check = model_summary.merge(
        expected_std,
        on=["map_name", "constraint_key", "constraint_label", "model_label"],
        how="left",
    )
    expected_values = std_check["expected_std_success_rate"].fillna(0.0).to_numpy(dtype=float)
    actual_values = std_check["std_success_rate"].fillna(0.0).to_numpy(dtype=float)
    if not np.allclose(actual_values, expected_values, atol=1.0e-12, rtol=0.0):
        raise AssertionError(
            "model_summary.std_success_rate must be computed from training-seed variance.\n"
            f"{std_check[['map_name', 'constraint_key', 'model_label', 'std_success_rate', 'expected_std_success_rate']].to_string(index=False)}"
        )


def _evaluation_cache_path(run_row: Dict[str, Any]) -> Path:
    metadata_path = Path(run_row["metadata_path"])
    return metadata_path.parent / "checkpoint_eval_cache.json"


def _load_evaluation_disk_cache(cache_path: Path) -> Dict[str, Any]:
    cache_path = Path(cache_path)
    cached = _RUN_EVAL_DISK_CACHE.get(cache_path)
    if cached is not None:
        return cached

    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            payload = {}
    else:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}
    if not isinstance(payload.get("entries"), dict):
        payload["entries"] = {}
    payload.setdefault("version", 1)
    _RUN_EVAL_DISK_CACHE[cache_path] = payload
    return payload


def _save_evaluation_disk_cache(cache_path: Path, payload: Dict[str, Any]) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(cache_path)
    _RUN_EVAL_DISK_CACHE[cache_path] = payload


def _jax_actor_eval_checkpoint_path(checkpoint_path: Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.eval_actor{checkpoint_path.suffix}")


def _prepare_jax_actor_eval_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    actor_checkpoint_path = _jax_actor_eval_checkpoint_path(checkpoint_path)
    source_mtime = checkpoint_path.stat().st_mtime_ns

    if actor_checkpoint_path.exists():
        try:
            with actor_checkpoint_path.open("rb") as f:
                cached = pickle.load(f)
        except (OSError, pickle.PickleError, EOFError):
            cached = None

        if (
            isinstance(cached, dict)
            and cached.get("format") == "jax_eval_actor_only_v1"
            and cached.get("source_checkpoint_path") == str(checkpoint_path)
            and cached.get("source_checkpoint_mtime_ns") == source_mtime
            and isinstance(cached.get("algorithm_state"), dict)
            and "actor_state" in cached["algorithm_state"]
        ):
            return actor_checkpoint_path

    with checkpoint_path.open("rb") as f:
        checkpoint = pickle.load(f)

    algorithm_state = checkpoint.get("algorithm_state")
    if not isinstance(algorithm_state, dict) or "actor_state" not in algorithm_state:
        return checkpoint_path

    payload = {
        "format": "jax_eval_actor_only_v1",
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_mtime_ns": source_mtime,
        "iteration": int(checkpoint.get("iteration", 0)),
        "total_steps": int(checkpoint.get("total_steps", 0)),
        "best_eval_reward": float(checkpoint.get("best_eval_reward", float("-inf"))),
        "algorithm_state": {
            "actor_state": algorithm_state["actor_state"],
            "training_steps": int(algorithm_state.get("training_steps", 0)),
        },
    }
    tmp_path = actor_checkpoint_path.with_suffix(f"{actor_checkpoint_path.suffix}.tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f)
    tmp_path.replace(actor_checkpoint_path)
    return actor_checkpoint_path


def _is_jax_eval_init_solver_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    keywords = [
        "resource_exhausted",
        "unable to allocate workspace for geqrf",
        "gpusolverdncreate",
        "cusolver internal error",
        "solver_handle_pool",
    ]
    return any(keyword in text for keyword in keywords)


def _is_jax_eval_cuda_runtime_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    keywords = [
        "resource_exhausted",
        "cuda_error_out_of_memory",
        "out of memory",
        "gpusolverdncreate",
        "cusolver internal error",
        "unable to allocate workspace for geqrf",
        "solver_handle_pool",
    ]
    return any(keyword in text for keyword in keywords)


@contextmanager
def _cheap_jax_eval_init_context():
    import algorithms_jax.common as common

    original_relu = common._ORTH_RELU
    original_unit = common._ORTH_UNIT
    original_policy = common._ORTH_POLICY
    original_orthogonal = common.nn.initializers.orthogonal

    def _cheap_orthogonal(scale: float = 1.0, column_axis: int = -1, dtype=None):
        del column_axis
        return common.nn.initializers.normal(stddev=float(scale), dtype=dtype)

    common._ORTH_RELU = _cheap_orthogonal(float(np.sqrt(2.0)))
    common._ORTH_UNIT = _cheap_orthogonal(1.0)
    common._ORTH_POLICY = _cheap_orthogonal(1.0e-2)
    common.nn.initializers.orthogonal = _cheap_orthogonal
    try:
        yield
    finally:
        common._ORTH_RELU = original_relu
        common._ORTH_UNIT = original_unit
        common._ORTH_POLICY = original_policy
        common.nn.initializers.orthogonal = original_orthogonal


def _create_jax_algorithm_for_eval(main_mod: Any, env: Any, cfg: Config) -> Any:
    with _cheap_jax_eval_init_context():
        if str(getattr(cfg.training, "device", "")).lower() != "cuda":
            return main_mod.create_algorithm(env, cfg)

        import jax

        cpu_devices = [device for device in jax.devices() if device.platform == "cpu"]
        if not cpu_devices:
            return main_mod.create_algorithm(env, cfg)

        warnings.warn(
            "Initializing JAX eval algorithm parameters on CPU before loading the checkpoint actor state. "
            "This affects checkpoint evaluation only and avoids GPU QR/cuSolver initialization failures.",
            stacklevel=2,
        )
        with jax.default_device(cpu_devices[0]):
            return main_mod.create_algorithm(env, cfg)


def _move_jax_eval_state_to_device(trainer: Any, device: str) -> None:
    if trainer.__class__.__name__ != "JAXTrainer":
        return

    device_name = str(device).lower()
    if device_name == "cpu":
        return

    import jax

    target_devices = [dev for dev in jax.devices() if dev.platform in {"gpu", "cuda"}]
    if not target_devices:
        return
    target_device = target_devices[0]

    algorithm = trainer.algorithm
    for attr_name in ["actor_state", "critic_state", "bridge_state", "delta_critic_state"]:
        state = getattr(algorithm, attr_name, None)
        if state is not None:
            setattr(algorithm, attr_name, jax.device_put(state, device=target_device))


def _build_jax_eval_trainer(
    factories: Dict[str, Any],
    main_mod: Any,
    run_row: Dict[str, Any],
    cfg: Config,
    *,
    checkpoint_choice: str,
    eval_episodes: int,
    eval_batch_envs: int,
) -> tuple[Any, Path, Config]:
    env = _build_smax_env(cfg, n_envs=int(max(1, eval_batch_envs)), batch=True)
    algorithm = _create_jax_algorithm_for_eval(main_mod, env, cfg)
    trainer = factories["JAXTrainer"](
        config=cfg,
        env=env,
        algorithm=algorithm,
        max_iterations=1,
        max_steps_per_iteration=cfg.training.max_steps_per_iteration,
        episodes_per_iteration=getattr(cfg.training, "episodes_per_iteration", None),
        log_interval=1,
        save_interval=1,
        eval_interval=1,
        eval_episodes=eval_episodes,
        save_dir=None,
        use_wandb=False,
        wandb_project=cfg.wandb.project,
        wandb_run_name=None,
        wandb_group=None,
        device=cfg.training.device,
        n_rollout_threads=int(max(1, eval_batch_envs)),
        env_fn=None,
    )
    checkpoint_path = _load_trainer_checkpoint(
        trainer,
        run_row,
        checkpoint_choice=checkpoint_choice,
    )
    _move_jax_eval_state_to_device(trainer, str(cfg.training.device))
    return trainer, checkpoint_path, cfg


def _load_trainer_checkpoint(
    trainer: Any,
    run_row: pd.Series | Dict[str, Any],
    *,
    checkpoint_choice: str,
) -> Path:
    if isinstance(run_row, pd.Series):
        run_row = run_row.to_dict()

    checkpoint_path = Path(run_row[f"{checkpoint_choice}_checkpoint"])
    algorithm_type = normalize_algorithm_type(run_row["algorithm_type"])
    if algorithm_type.startswith("jax_"):
        eval_checkpoint_path = _prepare_jax_actor_eval_checkpoint(checkpoint_path)
        trainer.load_checkpoint(str(eval_checkpoint_path), actor_only=True)
    else:
        trainer.load_checkpoint(str(checkpoint_path))
    return checkpoint_path


def build_trainer_for_run(
    run_row: pd.Series | Dict[str, Any],
    *,
    checkpoint_choice: str,
    device: str,
    eval_episodes: int,
    eval_seed: Optional[int] = None,
    eval_batch_envs: int = 1,
) -> tuple[Any, Path, Config]:
    if isinstance(run_row, pd.Series):
        run_row = run_row.to_dict()

    cfg = copy.deepcopy(run_row["config"])
    cfg.training.device = device
    cfg.training.deterministic = True
    cfg.training.n_rollout_threads = int(max(1, eval_batch_envs))
    cfg.training.eval_episodes = int(eval_episodes)
    if eval_seed is not None:
        cfg.training.seed = int(eval_seed)
        if hasattr(cfg, "environment") and cfg.environment is not None:
            cfg.environment.seed = int(eval_seed)
            env_args = getattr(cfg.environment, "env_args", None)
            if isinstance(env_args, dict):
                env_args["seed"] = int(eval_seed)
    cfg.wandb.use_wandb = False

    factories = bootstrap_project(device)
    main_mod = factories["main"]
    algorithm_type = normalize_algorithm_type(cfg.algorithm.type)
    checkpoint_choice = str(checkpoint_choice).lower()
    if checkpoint_choice not in {"best", "final"}:
        raise ValueError("checkpoint_choice must be 'best' or 'final'")

    if algorithm_type.startswith("jax_"):
        try:
            trainer, checkpoint_path, cfg = _build_jax_eval_trainer(
                factories,
                main_mod,
                run_row,
                cfg,
                checkpoint_choice=checkpoint_choice,
                eval_episodes=eval_episodes,
                eval_batch_envs=eval_batch_envs,
            )
            return trainer, checkpoint_path, cfg
        except Exception as exc:
            if str(device).lower() != "cuda":
                raise

            import jax

            cpu_devices = [cpu_device for cpu_device in jax.devices() if cpu_device.platform == "cpu"]
            if not cpu_devices:
                raise

            warnings.warn(
                "CUDA JAX checkpoint evaluation setup failed; falling back to CPU for this eval-only trainer.\n"
                f"{type(exc).__name__}: {exc}",
                stacklevel=2,
            )
            cpu_cfg = copy.deepcopy(cfg)
            cpu_cfg.training.device = "cpu"
            with jax.default_device(cpu_devices[0]):
                trainer, checkpoint_path, cpu_cfg = _build_jax_eval_trainer(
                    factories,
                    main_mod,
                    run_row,
                    cpu_cfg,
                    checkpoint_choice=checkpoint_choice,
                    eval_episodes=eval_episodes,
                    eval_batch_envs=eval_batch_envs,
                )
            return trainer, checkpoint_path, cpu_cfg
    else:
        env = _build_smax_env(cfg, n_envs=1, batch=False)
        algorithm = main_mod.create_algorithm(env, cfg)
        trainer = factories["Trainer"](
            config=cfg,
            env=env,
            algorithm=algorithm,
            max_iterations=1,
            max_steps_per_iteration=cfg.training.max_steps_per_iteration,
            log_interval=1,
            save_interval=1,
            eval_interval=1,
            eval_episodes=eval_episodes,
            save_dir=None,
            use_wandb=False,
            wandb_project=cfg.wandb.project,
            wandb_run_name=None,
            wandb_group=None,
            device=device,
            n_rollout_threads=1,
            env_fn=None,
        )

    checkpoint_path = _load_trainer_checkpoint(
        trainer,
        run_row,
        checkpoint_choice=checkpoint_choice,
    )
    return trainer, checkpoint_path, cfg


def _greedy_action_for_agent(agent: Any, obs: torch.Tensor, available_actions: Optional[list], device: torch.device | str) -> int:
    obs_in = obs.unsqueeze(0)
    avail_tensor = None
    if available_actions is not None:
        avail_tensor = torch.as_tensor(available_actions, dtype=torch.float32, device=device).unsqueeze(0)

    if hasattr(agent, "rnn_state_policy"):
        masks = torch.ones(1, 1, device=device)
        rnn_state = agent.rnn_state_policy.unsqueeze(0)
        prev_action = getattr(agent, "prev_action", None)
        if prev_action is not None:
            prev_action = prev_action.unsqueeze(0)

        try:
            logits, next_state = agent.policy_net.forward(
                obs_in,
                rnn_state,
                masks,
                available_actions=avail_tensor,
                prev_actions=prev_action,
            )
        except TypeError:
            logits, next_state = agent.policy_net.forward(
                obs_in,
                rnn_state,
                masks,
                available_actions=avail_tensor,
            )

        action = torch.argmax(logits, dim=-1)
        agent.rnn_state_policy = next_state.squeeze(0)
        if hasattr(agent, "prev_action"):
            agent.prev_action = F.one_hot(action, agent.action_dim).float().squeeze(0)
        return int(action.item())

    try:
        logits = agent.policy_net.forward(obs_in, available_actions=avail_tensor)
    except TypeError:
        logits = agent.policy_net.forward(obs_in)

    if isinstance(logits, tuple):
        logits = logits[0]
    action = torch.argmax(logits, dim=-1)
    return int(action.item())


def deterministic_nonjax_evaluation(trainer: Any, num_episodes: int) -> Dict[str, float]:
    from algorithms.common import resolve_step_episode_done

    env = trainer.eval_env
    device = trainer.algorithm.device
    episode_limit = trainer._get_eval_episode_limit()
    for agent in trainer.algorithm.agents.values():
        policy_net = getattr(agent, "policy_net", None)
        if policy_net is not None and hasattr(policy_net, "eval"):
            policy_net.eval()

    eval_rewards: Dict[int, list] = {}
    eval_lengths = []
    eval_successes = []

    with torch.inference_mode():
        for _ in range(num_episodes):
            observations = env.reset()
            trainer.algorithm.reset_agents()
            observations = trainer._add_agent_id_to_obs(torch_like_obs_to_tensor(observations, device))

            episode_reward = {agent_id: 0.0 for agent_id in observations.keys()}
            episode_length = 0
            episode_success = 0.0
            done = False

            while not done and episode_length < episode_limit:
                available_actions = None
                if hasattr(env, "get_avail_actions"):
                    avail = env.get_avail_actions()
                    if avail is not None:
                        available_actions = {i: avail[i] for i in range(len(avail))}

                actions = {}
                for agent_id, obs in observations.items():
                    agent_avail = None
                    if available_actions is not None and agent_id in available_actions:
                        agent_avail = available_actions[agent_id]
                    actions[agent_id] = _greedy_action_for_agent(
                        trainer.algorithm.agents[agent_id],
                        obs,
                        agent_avail,
                        device,
                    )

                next_observations, rewards, dones, info = env.step(actions)
                next_observations = trainer._add_agent_id_to_obs(torch_like_obs_to_tensor(next_observations, device))

                if info.get("battle_won", False):
                    episode_success = 1.0

                for agent_id, reward in rewards.items():
                    episode_reward[agent_id] += float(reward)

                observations = next_observations
                episode_length += 1
                done = resolve_step_episode_done(dones, info)

            eval_lengths.append(episode_length)
            eval_successes.append(episode_success)
            for agent_id, reward in episode_reward.items():
                eval_rewards.setdefault(agent_id, []).append(reward)

    mean_per_agent = [float(np.mean(rewards)) for rewards in eval_rewards.values()] if eval_rewards else []
    return {
        "eval_success_rate": float(np.mean(eval_successes)) if eval_successes else 0.0,
        "eval_mean_reward": float(np.mean(mean_per_agent)) if mean_per_agent else 0.0,
        "eval_std_reward": float(np.std(mean_per_agent)) if mean_per_agent else 0.0,
        "eval_mean_length": float(np.mean(eval_lengths)) if eval_lengths else 0.0,
        "eval_min_reward": float(np.min(mean_per_agent)) if mean_per_agent else 0.0,
        "eval_max_reward": float(np.max(mean_per_agent)) if mean_per_agent else 0.0,
    }


def torch_like_obs_to_tensor(observations: Dict[int, Any], device: torch.device | str) -> Dict[int, torch.Tensor]:
    tensor_obs = {}
    for agent_id, obs in observations.items():
        tensor_obs[agent_id] = torch.as_tensor(obs, dtype=torch.float32, device=device)
    return tensor_obs


def _build_evaluation_result(
    run_row: Dict[str, Any],
    cfg: Config,
    checkpoint_path: Path,
    *,
    checkpoint_choice: str,
    eval_episodes: int,
    eval_seed: Optional[int],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    result = {
        "eval_success_rate": float(metrics["eval_success_rate"]),
        "eval_mean_reward": float(metrics["eval_mean_reward"]),
        "eval_std_reward": float(metrics["eval_std_reward"]),
        "eval_mean_length": float(metrics["eval_mean_length"]),
        "eval_min_reward": float(metrics["eval_min_reward"]),
        "eval_max_reward": float(metrics["eval_max_reward"]),
    }
    result.update(
        {
            "run_dir": str(run_row["run_dir"]),
            "metadata_path": str(run_row["metadata_path"]),
            "project": run_row["project"],
            "map_name": run_row["map_name"],
            "map_label": run_row["map_label"],
            "wandb_run_name": run_row.get("wandb_run_name", ""),
            "constraint_key": run_row.get("constraint_key", "vanilla"),
            "constraint_label": run_row.get("constraint_label", "vanilla"),
            "algorithm_type": run_row["algorithm_type"],
            "model_key": run_row["model_key"],
            "model_label": run_row["model_label"],
            "seed": int(run_row["seed"]),
            "train_seed": int(run_row["seed"]),
            "eval_seed": int(eval_seed) if eval_seed is not None else int(getattr(cfg.training, "seed", 0)),
            "checkpoint_choice": checkpoint_choice,
            "checkpoint_path": str(checkpoint_path),
            "num_episodes": int(eval_episodes),
        }
    )
    return result


def _serialize_run_row_for_subprocess(run_row: Dict[str, Any]) -> Dict[str, Any]:
    serialized = {}
    for key, value in run_row.items():
        if key == "config":
            serialized[key] = value.to_dict()
        elif isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def _evaluate_run_cpu_subprocess(
    run_row: Dict[str, Any],
    *,
    checkpoint_choice: str,
    eval_episodes: int,
    eval_seed: int,
    deterministic: bool,
) -> Dict[str, Any]:
    worker_path = _REPO_ROOT / "figures/smax_eval_subprocess_worker.py"
    with tempfile.TemporaryDirectory(prefix="smax_eval_cpu_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        payload_path = tmpdir_path / "payload.json"
        output_path = tmpdir_path / "result.json"
        payload = {
            "run_row": _serialize_run_row_for_subprocess(run_row),
            "checkpoint_choice": checkpoint_choice,
            "eval_episodes": int(eval_episodes),
            "eval_seed": int(eval_seed),
            "deterministic": bool(deterministic),
            "output_path": str(output_path),
        }
        payload_path.write_text(json.dumps(payload))

        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{_REPO_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(_REPO_ROOT)
        )
        completed = subprocess.run(
            [sys.executable, str(worker_path), str(payload_path)],
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "CPU subprocess checkpoint evaluation failed.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return json.loads(output_path.read_text())


def _load_cached_evaluation_result(
    run_row: Dict[str, Any],
    *,
    checkpoint_choice: str,
    device: str,
    eval_episodes: int,
    deterministic: bool,
    eval_seed: Optional[int],
    disk_cache: Dict[str, Any],
) -> tuple[Optional[Dict[str, Any]], tuple[Any, ...], str]:
    cache_key = _evaluation_cache_key(
        run_row,
        checkpoint_choice=checkpoint_choice,
        device=device,
        eval_episodes=eval_episodes,
        deterministic=deterministic,
        eval_seed=eval_seed,
    )
    cache_key_str = _evaluation_cache_key_str(cache_key)
    cached = _RUN_EVAL_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached), cache_key, cache_key_str

    disk_cached = disk_cache["entries"].get(cache_key_str)
    if isinstance(disk_cached, dict):
        _RUN_EVAL_CACHE[cache_key] = dict(disk_cached)
        return dict(disk_cached), cache_key, cache_key_str
    return None, cache_key, cache_key_str


def _store_cached_evaluation_result(
    result: Dict[str, Any],
    *,
    cache_key: tuple[Any, ...],
    cache_key_str: str,
    cache_path: Path,
    disk_cache: Dict[str, Any],
    save: bool = True,
) -> None:
    _RUN_EVAL_CACHE[cache_key] = dict(result)
    disk_cache["entries"][cache_key_str] = dict(result)
    if save:
        _save_evaluation_disk_cache(cache_path, disk_cache)


def _chunk_eval_seeds(eval_seeds: list[int], batch_size: int) -> list[list[int]]:
    batch_size = max(1, int(batch_size))
    return [eval_seeds[i : i + batch_size] for i in range(0, len(eval_seeds), batch_size)]


def evaluate_run_seeds(
    run_row: pd.Series | Dict[str, Any],
    *,
    checkpoint_choice: str,
    device: str,
    eval_episodes: int,
    eval_seeds: Iterable[int],
    deterministic: bool = True,
    eval_seed_batch_size: int = 16,
    trainer_pool: Optional[Dict[tuple[Any, ...], tuple[Any, Path, Config]]] = None,
) -> list[Dict[str, Any]]:
    if isinstance(run_row, pd.Series):
        run_row = run_row.to_dict()

    normalized_eval_seeds = [int(seed) for seed in eval_seeds]
    if not normalized_eval_seeds:
        raise ValueError("eval_seeds must contain at least one seed")

    cache_path = _evaluation_cache_path(run_row)
    disk_cache = _load_evaluation_disk_cache(cache_path)
    ordered_results: Dict[int, Dict[str, Any]] = {}
    pending_cache_keys: Dict[int, tuple[tuple[Any, ...], str]] = {}
    missing_seeds: list[int] = []

    for eval_seed in normalized_eval_seeds:
        cached, cache_key, cache_key_str = _load_cached_evaluation_result(
            run_row,
            checkpoint_choice=checkpoint_choice,
            device=device,
            eval_episodes=eval_episodes,
            deterministic=deterministic,
            eval_seed=eval_seed,
            disk_cache=disk_cache,
        )
        if cached is not None:
            ordered_results[eval_seed] = cached
            continue
        pending_cache_keys[eval_seed] = (cache_key, cache_key_str)
        missing_seeds.append(eval_seed)

    if not missing_seeds:
        return [dict(ordered_results[seed]) for seed in normalized_eval_seeds]

    algorithm_type = normalize_algorithm_type(run_row["algorithm_type"])
    if not algorithm_type.startswith("jax_"):
        for eval_seed in missing_seeds:
            result = evaluate_run(
                run_row,
                checkpoint_choice=checkpoint_choice,
                device=device,
                eval_episodes=eval_episodes,
                eval_seed=eval_seed,
                deterministic=deterministic,
            )
            ordered_results[eval_seed] = dict(result)
        return [dict(ordered_results[seed]) for seed in normalized_eval_seeds]

    trainer_cache = trainer_pool if trainer_pool is not None else {}
    pending_batches = _chunk_eval_seeds(missing_seeds, eval_seed_batch_size)
    while pending_batches:
        seed_batch = pending_batches.pop(0)
        batch_envs = len(seed_batch)
        trainer_key = (
            str(run_row["map_name"]),
            str(run_row.get("constraint_key", "vanilla")),
            str(run_row["algorithm_type"]),
            str(run_row["model_label"]),
            str(device).lower(),
            int(eval_episodes),
            int(batch_envs),
        )
        try:
            trainer_bundle = trainer_cache.get(trainer_key)
            if trainer_bundle is None:
                trainer_bundle = build_trainer_for_run(
                    run_row,
                    checkpoint_choice=checkpoint_choice,
                    device=device,
                    eval_episodes=eval_episodes,
                    eval_batch_envs=batch_envs,
                )
                trainer_cache[trainer_key] = trainer_bundle

            trainer, checkpoint_path, cfg = trainer_bundle
            checkpoint_path = _load_trainer_checkpoint(
                trainer,
                run_row,
                checkpoint_choice=checkpoint_choice,
            )
            batch_metrics = trainer.evaluate_policy_by_seeds(
                seed_batch,
                num_episodes=eval_episodes,
                deterministic=deterministic,
            )
        except Exception as exc:
            if str(device).lower() == "cuda" and len(seed_batch) > 1 and _is_jax_eval_cuda_runtime_error(exc):
                split = max(1, len(seed_batch) // 2)
                warnings.warn(
                    "CUDA batched checkpoint evaluation failed; retrying with a smaller eval-seed batch.\n"
                    f"batch_size={len(seed_batch)}, split={split}, error={type(exc).__name__}: {exc}",
                    stacklevel=2,
                )
                pending_batches.insert(0, seed_batch[split:])
                pending_batches.insert(0, seed_batch[:split])
                trainer_cache.pop(trainer_key, None)
                continue
            if str(device).lower() == "cuda" and len(seed_batch) == 1 and _is_jax_eval_cuda_runtime_error(exc):
                eval_seed = int(seed_batch[0])
                warnings.warn(
                    "CUDA single-seed checkpoint evaluation failed; retrying in a fresh CPU-only subprocess.\n"
                    f"eval_seed={eval_seed}, error={type(exc).__name__}: {exc}",
                    stacklevel=2,
                )
                result = _evaluate_run_cpu_subprocess(
                    run_row,
                    checkpoint_choice=checkpoint_choice,
                    eval_episodes=eval_episodes,
                    eval_seed=eval_seed,
                    deterministic=deterministic,
                )
                cache_key, cache_key_str = pending_cache_keys[eval_seed]
                _store_cached_evaluation_result(
                    result,
                    cache_key=cache_key,
                    cache_key_str=cache_key_str,
                    cache_path=cache_path,
                    disk_cache=disk_cache,
                    save=False,
                )
                ordered_results[eval_seed] = dict(result)
                _save_evaluation_disk_cache(cache_path, disk_cache)
                trainer_cache.pop(trainer_key, None)
                continue
            raise

        for metrics in batch_metrics:
            eval_seed = int(metrics["eval_seed"])
            cache_key, cache_key_str = pending_cache_keys[eval_seed]
            result = _build_evaluation_result(
                run_row,
                cfg,
                checkpoint_path,
                checkpoint_choice=checkpoint_choice,
                eval_episodes=eval_episodes,
                eval_seed=eval_seed,
                metrics=metrics,
            )
            _store_cached_evaluation_result(
                result,
                cache_key=cache_key,
                cache_key_str=cache_key_str,
                cache_path=cache_path,
                disk_cache=disk_cache,
                save=False,
            )
            ordered_results[eval_seed] = dict(result)
        _save_evaluation_disk_cache(cache_path, disk_cache)

    return [dict(ordered_results[seed]) for seed in normalized_eval_seeds]


def evaluate_run(
    run_row: pd.Series | Dict[str, Any],
    *,
    checkpoint_choice: str,
    device: str,
    eval_episodes: int,
    eval_seed: Optional[int] = None,
    deterministic: bool = True,
) -> Dict[str, Any]:
    if isinstance(run_row, pd.Series):
        run_row = run_row.to_dict()

    cache_key = _evaluation_cache_key(
        run_row,
        checkpoint_choice=checkpoint_choice,
        device=device,
        eval_episodes=eval_episodes,
        deterministic=deterministic,
        eval_seed=eval_seed,
    )
    cache_key_str = _evaluation_cache_key_str(cache_key)
    cached = _RUN_EVAL_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    cache_path = _evaluation_cache_path(run_row)
    disk_cache = _load_evaluation_disk_cache(cache_path)
    disk_cached = disk_cache["entries"].get(cache_key_str)
    if isinstance(disk_cached, dict):
        _RUN_EVAL_CACHE[cache_key] = dict(disk_cached)
        return dict(disk_cached)

    trainer, checkpoint_path, cfg = build_trainer_for_run(
        run_row,
        checkpoint_choice=checkpoint_choice,
        device=device,
        eval_episodes=eval_episodes,
        eval_seed=eval_seed,
    )

    if trainer.__class__.__name__ == "JAXTrainer":
        metrics = trainer.evaluate_policy(num_episodes=eval_episodes, deterministic=deterministic)
    else:
        metrics = deterministic_nonjax_evaluation(trainer, num_episodes=eval_episodes)

    result = _build_evaluation_result(
        run_row,
        cfg,
        checkpoint_path,
        checkpoint_choice=checkpoint_choice,
        eval_episodes=eval_episodes,
        eval_seed=eval_seed,
        metrics=metrics,
    )
    _RUN_EVAL_CACHE[cache_key] = dict(result)
    disk_cache["entries"][cache_key_str] = dict(result)
    _save_evaluation_disk_cache(cache_path, disk_cache)
    return result


def evaluate_map(
    runs_df: pd.DataFrame,
    map_name: str,
    *,
    checkpoint_choice: str,
    device: str,
    eval_episodes: int,
    eval_seeds: Optional[Iterable[int]] = None,
    eval_seed_batch_size: int = 16,
    model_order: Optional[list[str]] = None,
    duplicate_policy: str = "error",
) -> Dict[str, Any]:
    if model_order is None:
        model_order = ordered_model_labels(runs_df["model_label"].dropna().unique().tolist())

    map_rows = runs_df[runs_df["map_name"] == map_name].copy()
    if map_rows.empty:
        return {
            "map_name": map_name,
            "constraint_results": {},
            "run_results": pd.DataFrame(),
            "seed_summary": pd.DataFrame(),
            "model_summary": pd.DataFrame(),
            "latex": "",
        }

    constraint_results: Dict[str, Any] = {}
    run_results = []
    seed_results = []
    model_results = []

    constraint_keys = _ordered_constraint_keys(map_rows["constraint_key"].dropna().unique().tolist())
    if not constraint_keys:
        constraint_keys = ["vanilla"]

    for constraint_key in constraint_keys:
        result = evaluate_map_constraint(
            map_rows,
            map_name,
            constraint_key,
            checkpoint_choice=checkpoint_choice,
            device=device,
            eval_episodes=eval_episodes,
            eval_seeds=eval_seeds,
            eval_seed_batch_size=eval_seed_batch_size,
            model_order=model_order,
            duplicate_policy=duplicate_policy,
        )
        if result["run_results"].empty:
            continue
        constraint_results[constraint_key] = result
        run_results.append(result["run_results"])
        seed_results.append(result["seed_summary"])
        model_results.append(result["model_summary"])

    if not model_results:
        return {
            "map_name": map_name,
            "constraint_results": {},
            "run_results": pd.DataFrame(),
            "seed_summary": pd.DataFrame(),
            "model_summary": pd.DataFrame(),
            "latex": "",
        }

    run_df = pd.concat(run_results, ignore_index=True)
    seed_summary = pd.concat(seed_results, ignore_index=True)
    model_summary = pd.concat(model_results, ignore_index=True)

    constraint_rank = {
        name: idx
        for idx, name in enumerate(_ordered_constraint_keys(seed_summary["constraint_key"].unique().tolist()))
    }
    model_rank = {name: idx for idx, name in enumerate(model_order)}

    seed_summary["_constraint_rank"] = seed_summary["constraint_key"].map(constraint_rank)
    seed_summary["_model_rank"] = seed_summary["model_label"].map(model_rank)
    seed_summary = (
        seed_summary.sort_values(["map_name", "_constraint_rank", "_model_rank", "seed"])
        .drop(columns=["_constraint_rank", "_model_rank"])
        .reset_index(drop=True)
    )

    model_summary["_constraint_rank"] = model_summary["constraint_key"].map(constraint_rank)
    model_summary["_model_rank"] = model_summary["model_label"].map(model_rank)
    model_summary = (
        model_summary.sort_values(["map_name", "_constraint_rank", "_model_rank"])
        .drop(columns=["_constraint_rank", "_model_rank"])
        .reset_index(drop=True)
    )

    latex = format_map_constraint_latex(
        model_summary,
        map_name=map_name,
        checkpoint_choice=checkpoint_choice,
        model_order=model_order,
    )

    return {
        "map_name": map_name,
        "constraint_results": constraint_results,
        "run_results": run_df,
        "seed_summary": seed_summary,
        "model_summary": model_summary,
        "latex": latex,
    }


def evaluate_map_constraint(
    runs_df: pd.DataFrame,
    map_name: str,
    constraint_key: str,
    *,
    checkpoint_choice: str,
    device: str,
    eval_episodes: int,
    eval_seeds: Optional[Iterable[int]] = None,
    eval_seed_batch_size: int = 16,
    model_order: Optional[list[str]] = None,
    duplicate_policy: str = "error",
) -> Dict[str, Any]:
    if model_order is None:
        model_order = ordered_model_labels(runs_df["model_label"].dropna().unique().tolist())
    normalized_eval_seeds = _normalize_eval_seeds(eval_seeds)
    duplicate_policy = _normalize_duplicate_policy(duplicate_policy)

    constraint_rows = runs_df[
        (runs_df["map_name"] == map_name) & (runs_df["constraint_key"] == constraint_key)
    ].copy()
    if constraint_rows.empty:
        return {
            "map_name": map_name,
            "constraint_key": constraint_key,
            "run_results": pd.DataFrame(),
            "seed_summary": pd.DataFrame(),
            "model_summary": pd.DataFrame(),
            "latex": "",
        }
    constraint_rows = _resolve_duplicate_runs(
        constraint_rows,
        checkpoint_choice=checkpoint_choice,
        duplicate_policy=duplicate_policy,
    )

    run_results = []
    trainer_pool: Dict[tuple[Any, ...], tuple[Any, Path, Config]] = {}
    for _, row in constraint_rows.iterrows():
        run_results.extend(
            evaluate_run_seeds(
                row,
                checkpoint_choice=checkpoint_choice,
                device=device,
                eval_episodes=eval_episodes,
                eval_seeds=normalized_eval_seeds,
                deterministic=True,
                eval_seed_batch_size=eval_seed_batch_size,
                trainer_pool=trainer_pool,
            )
        )

    run_df = pd.DataFrame(run_results)
    seed_summary = (
        run_df.groupby(["map_name", "constraint_key", "constraint_label", "model_label", "seed"], as_index=False)
        .agg(
            eval_success_rate=("eval_success_rate", "mean"),
            eval_mean_reward=("eval_mean_reward", "mean"),
            eval_mean_length=("eval_mean_length", "mean"),
            n_runs=("run_dir", "nunique"),
            n_eval_seeds=("eval_seed", "nunique"),
        )
        .sort_values(["map_name", "constraint_key", "constraint_label", "model_label", "seed"])
        .reset_index(drop=True)
    )

    model_summary = (
        seed_summary.groupby(["map_name", "constraint_key", "constraint_label", "model_label"], as_index=False)
        .agg(
            mean_success_rate=("eval_success_rate", "mean"),
            std_success_rate=("eval_success_rate", "std"),
            mean_reward=("eval_mean_reward", "mean"),
            std_reward=("eval_mean_reward", "std"),
            mean_length=("eval_mean_length", "mean"),
            std_length=("eval_mean_length", "std"),
            n_seeds=("seed", "nunique"),
            n_runs=("n_runs", "sum"),
            n_eval_seeds=("n_eval_seeds", "max"),
        )
    )
    model_summary["_model_rank"] = model_summary["model_label"].map(
        {name: idx for idx, name in enumerate(model_order)}
    )
    model_summary = (
        model_summary.sort_values(["map_name", "constraint_key", "constraint_label", "_model_rank"])
        .drop(columns=["_model_rank"])
        .reset_index(drop=True)
    )
    single_seed_mask = model_summary["n_seeds"] == 1
    if single_seed_mask.any():
        warnings.warn(
            "std_success_rate is defined over training seeds. Filled 0.0 for rows with n_seeds=1.\n"
            f"{model_summary.loc[single_seed_mask, ['map_name', 'constraint_key', 'model_label', 'n_seeds']].to_string(index=False)}",
            stacklevel=2,
        )
        model_summary.loc[single_seed_mask, "std_success_rate"] = 0.0

    _validate_run_and_summary_protocol(run_df, seed_summary, model_summary)

    latex = format_single_map_latex(
        model_summary,
        map_name=map_name,
        constraint_key=constraint_key,
        checkpoint_choice=checkpoint_choice,
        model_order=model_order,
    )

    return {
        "map_name": map_name,
        "constraint_key": constraint_key,
        "run_results": run_df,
        "seed_summary": seed_summary,
        "model_summary": model_summary,
        "latex": latex,
    }


def format_single_map_latex(
    model_summary: pd.DataFrame,
    *,
    map_name: str,
    constraint_key: str = "vanilla",
    checkpoint_choice: str,
    model_order: Optional[list[str]] = None,
) -> str:
    if model_order is None:
        model_order = ordered_model_labels(model_summary["model_label"].dropna().unique().tolist())
    map_title = map_label(map_name)
    constraint_title = constraint_label(constraint_key)
    rows = []
    for model in model_order:
        sub = model_summary[
            (model_summary["constraint_label"] == constraint_title) & (model_summary["model_label"] == model)
        ]
        if sub.empty:
            cell = "--"
        else:
            mean = float(sub["mean_success_rate"].iloc[0]) * 100.0
            std = float(sub["std_success_rate"].iloc[0]) * 100.0
            cell = f"{mean:.1f} \scriptsize $\\pm$ {std:.1f}"
        rows.append(f"{model} & {cell} \\\\")

    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{Deterministic SMAX win rate on {map_title} with {constraint_title} using {checkpoint_choice} checkpoints. "
        "Mean $\\pm$ std are computed over seeds, not episodes.}}\n"
        f"\\label{{tab:smax_{map_name}_{constraint_key}_{checkpoint_choice}}}\n"
        "\\begin{tabular}{lc}\n"
        "\\toprule\n"
        "Model & Win rate (\\%) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


def format_map_constraint_latex(
    model_summary: pd.DataFrame,
    *,
    map_name: str,
    checkpoint_choice: str,
    constraint_order: Optional[list[str]] = None,
    model_order: Optional[list[str]] = None,
) -> str:
    if constraint_order is None:
        constraint_order = _ordered_constraint_keys(model_summary["constraint_label"].dropna().unique().tolist())
    if model_order is None:
        model_order = ordered_model_labels(model_summary["model_label"].dropna().unique().tolist())
    summary_lookup = {
        (row["constraint_key"], row["model_label"]): row
        for _, row in model_summary.iterrows()
    }

    def fmt(constraint: str, model: str) -> str:
        row = summary_lookup.get((constraint, model))
        if row is None:
            return "--"
        mean = float(row["mean_success_rate"]) * 100.0
        std = float(row["std_success_rate"]) * 100.0
        return f"{mean:.1f} \scriptsize $\\pm$ {std:.1f}"

    rows = []
    for constraint in constraint_order:
        row = " & ".join(fmt(constraint, model) for model in model_order)
        rows.append(f"{constraint_label(constraint)} & {row} \\\\")

    body = "\n".join(rows)
    map_title = map_label(map_name)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{Deterministic SMAX win rate on {map_title} using {checkpoint_choice} checkpoints. "
        "Rows are grouped by training constraint. Mean $\\pm$ std are computed over seeds, not episodes.}}\n"
        f"\\label{{tab:smax_{map_name}_{checkpoint_choice}}}\n"
        "\\begin{tabular}{l" + "c" * len(model_order) + "}\n"
        "\\toprule\n"
        "Constraint & " + " & ".join(model_order) + " \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


def format_cross_map_latex(
    model_summary: pd.DataFrame,
    *,
    checkpoint_choice: str,
    map_order: Optional[list[str]] = None,
    model_order: Optional[list[str]] = None,
) -> str:
    if map_order is None:
        map_order = ordered_map_names(model_summary["map_name"].dropna().unique().tolist())
    if model_order is None:
        model_order = ordered_model_labels(model_summary["model_label"].dropna().unique().tolist())
    summary_lookup = {
        (row["map_name"], row["constraint_key"], row["model_label"]): row
        for _, row in model_summary.iterrows()
    }
    constraints_by_map = {}
    for map_name in map_order:
        map_constraints = model_summary.loc[
            model_summary["map_name"] == map_name, "constraint_label"
        ].dropna().unique().tolist()
        constraints_by_map[map_name] = _ordered_constraint_keys(map_constraints)

    def fmt(map_name: str, constraint: str, model: str) -> str:
        row = summary_lookup.get((map_name, constraint, model))
        if row is None:
            return "--"
        mean = float(row["mean_success_rate"]) * 100.0
        std = float(row["std_success_rate"]) * 100.0
        return f"{mean:.1f} \scriptsize$\\pm$ {std:.1f}"

    header_cols = " & ".join(["Map", "Constraint"] + list(model_order))
    rows = []
    for map_name in map_order:
        for constraint in constraints_by_map.get(map_name, []):
            row = " & ".join(fmt(map_name, constraint, model) for model in model_order)
            rows.append(f"{map_label(map_name)} & {constraint_label(constraint)} & {row} \\\\")

    body = "\n".join(rows)
    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        f"\\caption{{Deterministic SMAX win rate using {checkpoint_choice} checkpoints. "
        "Rows are grouped by map and training constraint. Mean $\\pm$ std are computed over seeds, not episodes.}}\n"
        f"\\label{{tab:smax_cross_map_{checkpoint_choice}}}\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        "\\begin{tabular}{ll" + "c" * len(model_order) + "}\n"
        "\\toprule\n"
        f"{header_cols} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table*}"
    )


def save_summary_tables(
    model_summary: pd.DataFrame,
    output_dir: Path,
    *,
    checkpoint_choice: str,
    prefix: str = "smax_checkpoint_eval",
    map_order: Optional[list[str]] = None,
    model_order: Optional[list[str]] = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{prefix}_{checkpoint_choice}_summary.csv"
    tex_path = output_dir / f"{prefix}_{checkpoint_choice}_table.tex"
    model_summary.to_csv(csv_path, index=False)
    tex_path.write_text(
        format_cross_map_latex(
            model_summary,
            checkpoint_choice=checkpoint_choice,
            map_order=map_order,
            model_order=model_order,
        )
    )
    return {"csv": csv_path, "tex": tex_path}
