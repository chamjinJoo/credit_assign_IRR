from __future__ import annotations

import copy
import json
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from configs.config import Config


_PROJECT_FACTORIES: Dict[str, Dict[str, Any]] = {}
_ENV_CACHE: Dict[str, Any] = {}
_WRAPPER_DIR_NAMES = {"smaxbatchwrapper", "smaxwrapper"}


def _normalize_algorithm_type(algorithm_type: str) -> str:
    return str(algorithm_type).lower().replace("-", "_")


def _checkpoint_column(checkpoint_choice: str) -> str:
    choice = str(checkpoint_choice).strip().lower()
    if choice not in {"best", "final"}:
        raise ValueError("checkpoint_choice must be 'best' or 'final'")
    return f"{choice}_checkpoint"


def _filter_dataclass_section(section: Any, dataclass_obj: Any) -> Any:
    if not isinstance(section, dict):
        return section
    valid_keys = set(getattr(dataclass_obj, "__dataclass_fields__", {}).keys())
    return {key: value for key, value in section.items() if key in valid_keys}


def _config_from_checkpoint_dict(config_dict: Dict[str, Any]) -> Config:
    """Load historical metadata while ignoring retired non-environment fields."""
    normalized = copy.deepcopy(config_dict)
    defaults = Config()
    for section_name in ("network", "algorithm", "training", "wandb"):
        if section_name in normalized:
            normalized[section_name] = _filter_dataclass_section(
                normalized[section_name], getattr(defaults, section_name)
            )
    return Config.from_dict(normalized)


def _cfg_extra(env_cfg: Any, name: str, default: Any = None) -> Any:
    if hasattr(env_cfg, name):
        value = getattr(env_cfg, name)
        if value is not None:
            return value
    return (getattr(env_cfg, "env_args", None) or {}).get(name, default)


def _constraint_key(cfg: Config) -> str:
    env_cfg = cfg.environment
    if bool(_cfg_extra(env_cfg, "use_cooperative_kill_bonus", False)):
        return "cb"
    if bool(_cfg_extra(env_cfg, "use_allowed_targets", False)):
        window = _cfg_extra(env_cfg, "allowed_target_window", None)
        if window is not None:
            return f"tar_w={int(window)}"
    return "vanilla"


def _run_and_env_dir(metadata_path: Path) -> tuple[Path, str]:
    run_leaf = metadata_path.parent
    if run_leaf.name in _WRAPPER_DIR_NAMES:
        return metadata_path.parents[1], run_leaf.name
    if (run_leaf / "checkpoints").is_dir():
        return run_leaf, ""
    return metadata_path.parents[1], run_leaf.name


def _checkpoint_paths(run_dir: Path, env_dir_name: str) -> Dict[str, Path]:
    base_dir = run_dir / env_dir_name if env_dir_name else run_dir
    checkpoint_dir = base_dir / "checkpoints"
    return {
        "best": checkpoint_dir / "jax_checkpoint_best.pkl",
        "final": checkpoint_dir / "jax_checkpoint_final.pkl",
    }


def discover_runs(root: Path, project_filter: Optional[str] = None) -> pd.DataFrame:
    """Discover metadata-backed JAX SMAX runs for gradient decomposition."""
    rows: list[Dict[str, Any]] = []
    for metadata_path in sorted(Path(root).rglob("simulation_config.json")):
        payload = json.loads(metadata_path.read_text())
        if not isinstance(payload.get("config"), dict):
            raise ValueError(f"Missing config object in {metadata_path}")

        cfg = _config_from_checkpoint_dict(payload["config"])
        project = getattr(cfg.wandb, "project", None)
        if project_filter and project != project_filter:
            continue

        run_dir, env_dir_name = _run_and_env_dir(metadata_path)
        checkpoints = _checkpoint_paths(run_dir, env_dir_name)
        rows.append(
            {
                "run_dir": run_dir,
                "metadata_path": metadata_path,
                "project": project,
                "map_name": str(_cfg_extra(cfg.environment, "map_name", "unknown")),
                "constraint_key": _constraint_key(cfg),
                "algorithm_type": _normalize_algorithm_type(cfg.algorithm.type),
                "seed": int(cfg.training.seed),
                "config": cfg,
                "env_dir_name": env_dir_name,
                "best_checkpoint": checkpoints["best"],
                "final_checkpoint": checkpoints["final"],
                "_sort_path": str(run_dir),
            }
        )

    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs
    runs = runs.drop_duplicates(subset=["run_dir"], keep="first")
    runs = runs.sort_values(
        ["map_name", "constraint_key", "algorithm_type", "seed", "_sort_path"]
    )
    return runs.drop(columns="_sort_path").reset_index(drop=True)


def _bootstrap_project(device: str) -> Dict[str, Any]:
    cache_key = str(device).lower()
    if cache_key in _PROJECT_FACTORIES:
        return _PROJECT_FACTORIES[cache_key]

    if cache_key != "cpu":
        command_buffer_flag = "--xla_gpu_enable_command_buffer="
        existing_flags = os.environ.get("XLA_FLAGS", "")
        if command_buffer_flag not in existing_flags:
            os.environ["XLA_FLAGS"] = f"{existing_flags} {command_buffer_flag}".strip()
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import main as project_main

    project_main.configure_jax_runtime(device)
    project_main.import_runtime_modules()

    from envs import SMAXBatchWrapper
    from runner import JAXTrainer

    factories = {
        "main": project_main,
        "SMAXBatchWrapper": SMAXBatchWrapper,
        "JAXTrainer": JAXTrainer,
    }
    _PROJECT_FACTORIES[cache_key] = factories
    return factories


def _build_smax_env(cfg: Config, n_envs: int) -> Any:
    factories = _bootstrap_project(cfg.training.device)
    env_cfg = cfg.environment
    common_kwargs = {
        "map_name": _cfg_extra(env_cfg, "map_name", None),
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
        "smacv2_position_generation": bool(
            _cfg_extra(env_cfg, "smacv2_position_generation", False)
        ),
        "smacv2_unit_type_generation": bool(
            _cfg_extra(env_cfg, "smacv2_unit_type_generation", False)
        ),
        "use_allowed_targets": bool(_cfg_extra(env_cfg, "use_allowed_targets", False)),
        "allowed_target_window": _cfg_extra(env_cfg, "allowed_target_window", None),
        "allowed_targets": _cfg_extra(env_cfg, "allowed_targets", None),
        "use_partial_damage_restore": bool(
            _cfg_extra(env_cfg, "use_partial_damage_restore", False)
        ),
        "use_cooperative_kill_bonus": bool(
            _cfg_extra(env_cfg, "use_cooperative_kill_bonus", False)
        ),
        "cooperative_bonus_mode": _cfg_extra(env_cfg, "cooperative_bonus_mode", "kill"),
        "cooperative_size": int(
            _cfg_extra(env_cfg, "cooperative_size", _cfg_extra(env_cfg, "coop_size", 2))
        ),
        "cooperative_kill_window": int(_cfg_extra(env_cfg, "cooperative_kill_window", 2)),
        "cooperative_kill_bonus": float(_cfg_extra(env_cfg, "cooperative_kill_bonus", 2.0)),
        "cooperative_kill_owner_ratio": float(
            _cfg_extra(env_cfg, "cooperative_kill_owner_ratio", 0.7)
        ),
    }
    signature = json.dumps(
        {
            "device": str(cfg.training.device).lower(),
            "n_envs": int(n_envs),
            "algorithm_type": _normalize_algorithm_type(cfg.algorithm.type),
            "env_kwargs": common_kwargs,
        },
        sort_keys=True,
        default=str,
    )
    if signature not in _ENV_CACHE:
        _ENV_CACHE[signature] = factories["SMAXBatchWrapper"](
            n_envs=n_envs, warmup_episodes=0, **common_kwargs
        )
    return _ENV_CACHE[signature]


@contextmanager
def _cheap_jax_init_context():
    import models.jax.common as common

    original_relu = common._ORTH_RELU
    original_unit = common._ORTH_UNIT
    original_policy = common._ORTH_POLICY
    original_orthogonal = common.nn.initializers.orthogonal

    def cheap_orthogonal(scale: float = 1.0, column_axis: int = -1, dtype=None):
        del column_axis
        return common.nn.initializers.normal(stddev=float(scale), dtype=dtype)

    common._ORTH_RELU = cheap_orthogonal(float(np.sqrt(2.0)))
    common._ORTH_UNIT = cheap_orthogonal(1.0)
    common._ORTH_POLICY = cheap_orthogonal(1.0e-2)
    common.nn.initializers.orthogonal = cheap_orthogonal
    try:
        yield
    finally:
        common._ORTH_RELU = original_relu
        common._ORTH_UNIT = original_unit
        common._ORTH_POLICY = original_policy
        common.nn.initializers.orthogonal = original_orthogonal


def _create_algorithm(main_module: Any, env: Any, cfg: Config) -> Any:
    with _cheap_jax_init_context():
        if str(cfg.training.device).lower() != "cuda":
            return main_module.create_algorithm(env, cfg)

        import jax

        cpu_devices = [device for device in jax.devices() if device.platform == "cpu"]
        if not cpu_devices:
            return main_module.create_algorithm(env, cfg)
        with jax.default_device(cpu_devices[0]):
            return main_module.create_algorithm(env, cfg)


def _move_state_to_device(trainer: Any, device: str) -> None:
    if str(device).lower() == "cpu":
        return

    import jax

    target_devices = [item for item in jax.devices() if item.platform in {"gpu", "cuda"}]
    if not target_devices:
        return
    for name in ("actor_state", "critic_state", "bridge_state", "delta_critic_state"):
        state = getattr(trainer.algorithm, name, None)
        if state is not None:
            setattr(trainer.algorithm, name, jax.device_put(state, device=target_devices[0]))


def _build_trainer(
    run_row: Dict[str, Any],
    cfg: Config,
    checkpoint_choice: str,
    eval_episodes: int,
    eval_batch_envs: int,
) -> tuple[Any, Path, Config]:
    factories = _bootstrap_project(cfg.training.device)
    env = _build_smax_env(cfg, int(max(1, eval_batch_envs)))
    algorithm = _create_algorithm(factories["main"], env, cfg)
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
        device=cfg.training.device,
        n_rollout_threads=int(max(1, eval_batch_envs)),
        env_fn=None,
    )
    checkpoint_path = Path(run_row[_checkpoint_column(checkpoint_choice)])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    trainer.load_checkpoint(str(checkpoint_path), actor_only=True)
    _move_state_to_device(trainer, str(cfg.training.device))
    return trainer, checkpoint_path, cfg


def build_trainer_for_run(
    run_row: pd.Series | Dict[str, Any],
    *,
    checkpoint_choice: str,
    device: str,
    eval_episodes: int,
    eval_seed: Optional[int] = None,
    eval_batch_envs: int = 1,
) -> tuple[Any, Path, Config]:
    """Construct a checkpoint-compatible JAX SMAX trainer for gradient analysis."""
    if isinstance(run_row, pd.Series):
        run_row = run_row.to_dict()

    cfg = copy.deepcopy(run_row["config"])
    algorithm_type = _normalize_algorithm_type(cfg.algorithm.type)
    if algorithm_type not in {"jax_pci_ippo", "jax_pci_mappo"}:
        raise ValueError(f"Expected JAX PCI-IPPO/MAPPO, got {algorithm_type}")

    cfg.training.device = str(device)
    cfg.training.deterministic = True
    cfg.training.n_rollout_threads = int(max(1, eval_batch_envs))
    cfg.training.eval_episodes = int(eval_episodes)
    if eval_seed is not None:
        cfg.training.seed = int(eval_seed)
        cfg.environment.seed = int(eval_seed)
        if isinstance(cfg.environment.env_args, dict):
            cfg.environment.env_args["seed"] = int(eval_seed)
    cfg.wandb.use_wandb = False

    try:
        return _build_trainer(
            run_row, cfg, checkpoint_choice, eval_episodes, eval_batch_envs
        )
    except Exception as exc:
        if str(device).lower() != "cuda":
            raise

        import jax

        cpu_devices = [item for item in jax.devices() if item.platform == "cpu"]
        if not cpu_devices:
            raise
        warnings.warn(
            "CUDA checkpoint setup failed; using CPU for this gradient-analysis trainer.\n"
            f"{type(exc).__name__}: {exc}",
            stacklevel=2,
        )
        cpu_cfg = copy.deepcopy(cfg)
        cpu_cfg.training.device = "cpu"
        with jax.default_device(cpu_devices[0]):
            return _build_trainer(
                run_row, cpu_cfg, checkpoint_choice, eval_episodes, eval_batch_envs
            )
