from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from algorithms.common import resolve_step_episode_done
from configs.config import Config

PROJECT_FILTER = "pci-marl-smax-jax-v3"

MAP_ORDER = ["3m", "2s3z", "3s5z", "smacv2_5_units"]
CONSTRAINT_ORDER = ["vanilla", "tar_w=2", "tar_w=3", "tar_w=4", "tar_w=5", "tar_w=6", "tar_w=8", "cb"]
MODEL_ORDER = ["pci-mappo", "pci-ippo", "mappo", "ippo"]

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


def discover_runs(root: Path, project_filter: str = PROJECT_FILTER) -> pd.DataFrame:
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

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["map_name", "constraint_key", "model_key", "seed", "run_dir"]).reset_index(drop=True)


def _build_smax_env(cfg: Config, *, n_envs: int, batch: bool) -> Any:
    factories = bootstrap_project(cfg.training.device)
    env_cfg = cfg.environment
    env_signature = json.dumps(
        {
            "batch": batch,
            "n_envs": int(n_envs),
            "algorithm_type": normalize_algorithm_type(cfg.algorithm.type),
            "env": env_cfg.get_relevant_config(),
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
        env = factories["SMAXBatchWrapper"](n_envs=n_envs, **common_kwargs)
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
) -> tuple[Any, ...]:
    checkpoint_path = Path(run_row[f"{checkpoint_choice}_checkpoint"])
    checkpoint_mtime = checkpoint_path.stat().st_mtime_ns if checkpoint_path.exists() else None
    return (
        str(run_row["run_dir"]),
        str(run_row["metadata_path"]),
        checkpoint_choice,
        str(device).lower(),
        int(eval_episodes),
        str(checkpoint_path),
        checkpoint_mtime,
    )


def build_trainer_for_run(
    run_row: pd.Series | Dict[str, Any],
    *,
    checkpoint_choice: str,
    device: str,
    eval_episodes: int,
) -> tuple[Any, Path, Config]:
    if isinstance(run_row, pd.Series):
        run_row = run_row.to_dict()

    cfg = copy.deepcopy(run_row["config"])
    cfg.training.device = device
    cfg.training.deterministic = True
    cfg.training.n_rollout_threads = 1
    cfg.training.eval_episodes = int(eval_episodes)
    cfg.wandb.use_wandb = False

    factories = bootstrap_project(device)
    main_mod = factories["main"]
    algorithm_type = normalize_algorithm_type(cfg.algorithm.type)
    checkpoint_choice = str(checkpoint_choice).lower()
    if checkpoint_choice not in {"best", "final"}:
        raise ValueError("checkpoint_choice must be 'best' or 'final'")

    if algorithm_type.startswith("jax_"):
        env = _build_smax_env(cfg, n_envs=1, batch=True)
        algorithm = main_mod.create_algorithm(env, cfg)
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
            device=device,
            n_rollout_threads=1,
            env_fn=None,
        )
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

    checkpoint_path = Path(run_row[f"{checkpoint_choice}_checkpoint"])
    trainer.load_checkpoint(str(checkpoint_path))
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


def evaluate_run(
    run_row: pd.Series | Dict[str, Any],
    *,
    checkpoint_choice: str,
    device: str,
    eval_episodes: int,
) -> Dict[str, Any]:
    if isinstance(run_row, pd.Series):
        run_row = run_row.to_dict()

    cache_key = _evaluation_cache_key(
        run_row,
        checkpoint_choice=checkpoint_choice,
        device=device,
        eval_episodes=eval_episodes,
    )
    cached = _RUN_EVAL_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    trainer, checkpoint_path, cfg = build_trainer_for_run(
        run_row,
        checkpoint_choice=checkpoint_choice,
        device=device,
        eval_episodes=eval_episodes,
    )

    if trainer.__class__.__name__ == "JAXTrainer":
        metrics = trainer.evaluate_policy(num_episodes=eval_episodes)
        result = {
            "eval_success_rate": float(metrics["eval_success_rate"]),
            "eval_mean_reward": float(metrics["eval_mean_reward"]),
            "eval_std_reward": float(metrics["eval_std_reward"]),
            "eval_mean_length": float(metrics["eval_mean_length"]),
            "eval_min_reward": float(metrics["eval_min_reward"]),
            "eval_max_reward": float(metrics["eval_max_reward"]),
        }
    else:
        result = deterministic_nonjax_evaluation(trainer, num_episodes=eval_episodes)

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
            "checkpoint_choice": checkpoint_choice,
            "checkpoint_path": str(checkpoint_path),
            "num_episodes": int(eval_episodes),
        }
    )
    _RUN_EVAL_CACHE[cache_key] = dict(result)
    return result


def evaluate_map(
    runs_df: pd.DataFrame,
    map_name: str,
    *,
    checkpoint_choice: str,
    device: str,
    eval_episodes: int,
    model_order: Optional[list[str]] = None,
) -> Dict[str, Any]:
    if model_order is None:
        model_order = MODEL_ORDER

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
            model_order=model_order,
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

    constraint_rank = {name: idx for idx, name in enumerate(_ordered_constraint_keys(seed_summary["constraint_label"].unique().tolist()))}
    model_rank = {name: idx for idx, name in enumerate(model_order)}

    seed_summary["_constraint_rank"] = seed_summary["constraint_label"].map(constraint_rank)
    seed_summary["_model_rank"] = seed_summary["model_label"].map(model_rank)
    seed_summary = (
        seed_summary.sort_values(["map_name", "_constraint_rank", "_model_rank", "seed"])
        .drop(columns=["_constraint_rank", "_model_rank"])
        .reset_index(drop=True)
    )

    model_summary["_constraint_rank"] = model_summary["constraint_label"].map(constraint_rank)
    model_summary["_model_rank"] = model_summary["model_label"].map(model_rank)
    model_summary = (
        model_summary.sort_values(["map_name", "_constraint_rank", "_model_rank"])
        .drop(columns=["_constraint_rank", "_model_rank"])
        .reset_index(drop=True)
    )
    model_summary["std_success_rate"] = model_summary["std_success_rate"].fillna(0.0)

    latex = format_map_constraint_latex(
        model_summary,
        map_name=map_name,
        checkpoint_choice=checkpoint_choice,
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
    model_order: Optional[list[str]] = None,
) -> Dict[str, Any]:
    if model_order is None:
        model_order = MODEL_ORDER

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

    run_results = []
    for _, row in constraint_rows.iterrows():
        run_results.append(
            evaluate_run(
                row,
                checkpoint_choice=checkpoint_choice,
                device=device,
                eval_episodes=eval_episodes,
            )
        )

    run_df = pd.DataFrame(run_results)
    seed_summary = (
        run_df.groupby(["map_name", "constraint_label", "model_label", "seed"], as_index=False)
        .agg(
            eval_success_rate=("eval_success_rate", "mean"),
            eval_mean_reward=("eval_mean_reward", "mean"),
            eval_mean_length=("eval_mean_length", "mean"),
            n_runs=("run_dir", "count"),
        )
        .sort_values(["map_name", "constraint_label", "model_label", "seed"])
        .reset_index(drop=True)
    )

    model_summary = (
        seed_summary.groupby(["map_name", "constraint_label", "model_label"], as_index=False)
        .agg(
            mean_success_rate=("eval_success_rate", "mean"),
            std_success_rate=("eval_success_rate", "std"),
            mean_reward=("eval_mean_reward", "mean"),
            std_reward=("eval_mean_reward", "std"),
            mean_length=("eval_mean_length", "mean"),
            std_length=("eval_mean_length", "std"),
            n_seeds=("seed", "nunique"),
            n_runs=("n_runs", "sum"),
        )
    )
    model_summary["_model_rank"] = model_summary["model_label"].map(
        {name: idx for idx, name in enumerate(model_order)}
    )
    model_summary = (
        model_summary.sort_values(["map_name", "constraint_label", "_model_rank"])
        .drop(columns=["_model_rank"])
        .reset_index(drop=True)
    )
    model_summary["std_success_rate"] = model_summary["std_success_rate"].fillna(0.0)

    latex = format_single_map_latex(
        model_summary,
        map_name=map_name,
        constraint_key=constraint_key,
        checkpoint_choice=checkpoint_choice,
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
) -> str:
    map_title = map_label(map_name)
    constraint_title = constraint_label(constraint_key)
    rows = []
    for model in MODEL_ORDER:
        sub = model_summary[
            (model_summary["constraint_label"] == constraint_title) & (model_summary["model_label"] == model)
        ]
        if sub.empty:
            cell = "--"
        else:
            mean = float(sub["mean_success_rate"].iloc[0]) * 100.0
            std = float(sub["std_success_rate"].iloc[0]) * 100.0
            cell = f"{mean:.1f} $\\pm$ {std:.1f}"
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
        model_order = MODEL_ORDER
    summary_lookup = {
        (row["constraint_label"], row["model_label"]): row
        for _, row in model_summary.iterrows()
    }

    def fmt(constraint: str, model: str) -> str:
        row = summary_lookup.get((constraint, model))
        if row is None:
            return "--"
        mean = float(row["mean_success_rate"]) * 100.0
        std = float(row["std_success_rate"]) * 100.0
        return f"{mean:.1f} $\\pm$ {std:.1f}"

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
        map_order = MAP_ORDER
    if model_order is None:
        model_order = MODEL_ORDER
    summary_lookup = {
        (row["map_name"], row["constraint_label"], row["model_label"]): row
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
        return f"{mean:.1f} $\\pm$ {std:.1f}"

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
        )
    )
    return {"csv": csv_path, "tex": tex_path}
