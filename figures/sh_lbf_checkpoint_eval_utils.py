from __future__ import annotations

import copy
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import torch

from configs.config import Config

import tab2_smax_checkpoint_eval_utils as base


if not hasattr(base, "_sh_lbf_original_load_trainer_checkpoint"):
    base._sh_lbf_original_load_trainer_checkpoint = base._load_trainer_checkpoint
if not hasattr(base, "_sh_lbf_original_evaluation_cache_key"):
    base._sh_lbf_original_evaluation_cache_key = base._evaluation_cache_key

_BASE_LOAD_TRAINER_CHECKPOINT = base._sh_lbf_original_load_trainer_checkpoint
_BASE_EVALUATION_CACHE_KEY = base._sh_lbf_original_evaluation_cache_key


SH_MODEL_ORDER = ["ippo", "ipci", "mappo", "mpci"]
LBF_MODEL_ORDER = ["ippo", "ipci", "mappo", "mpci"]

SH_MODEL_NAMES = {
    "r_ippo": "ippo",
    "rippo": "ippo",
    "r_mappo": "mappo",
    "rmappo": "mappo",
    "r_pci": "mpci",
    "pci_rnn": "mpci",
    "pci_rippo": "ipci",
    "pci_rmappo": "mpci",
}

LBF_MODEL_NAMES = {
    "jax_ippo": "ippo",
    "ippo": "ippo",
    "jax_mappo": "mappo",
    "mappo": "mappo",
    "jax_pci_ippo": "ipci",
    "pci_ippo": "ipci",
    "jax_pci_mappo": "mpci",
    "pci_mappo": "mpci",
}


def bootstrap_project(device: str):
    return base.bootstrap_project(device)


def clear_runtime_caches() -> None:
    base.clear_runtime_caches()


def ordered_model_labels(model_labels: Iterable[str], model_order: Iterable[str]) -> list[str]:
    labels = [str(label) for label in model_labels if str(label)]
    seen = set()
    ordered = []
    for label in model_order:
        if label in labels and label not in seen:
            ordered.append(label)
            seen.add(label)
    for label in labels:
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    return ordered


def _normalize_algorithm_type(algorithm_type: Any) -> str:
    return str(algorithm_type or "").strip().lower().replace("-", "_")


def _parse_seed(run_dir_name: str) -> int:
    match = re.search(r"_seed(\d+)$", str(run_dir_name))
    if match is None:
        raise ValueError(f"Could not infer seed from run directory: {run_dir_name}")
    return int(match.group(1))


def _checkpoint_paths(run_dir: Path, env_dir_name: str, algorithm_type: str) -> dict[str, Path]:
    paths = base._checkpoint_paths(run_dir, env_dir_name, algorithm_type)
    if paths["final"].is_file():
        return paths

    key = _normalize_algorithm_type(algorithm_type)
    if key.startswith("jax_"):
        return paths

    checkpoint_dir = Path(run_dir) / str(env_dir_name) / "checkpoints"
    numbered = []
    for path in checkpoint_dir.glob("checkpoint_*.pt"):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
        if match is not None:
            numbered.append((int(match.group(1)), path))
    if numbered:
        paths["final"] = sorted(numbered, key=lambda item: item[0])[-1][1]
    return paths


def _selected_model_label(cfg: Config, *, model_dir: str, kind: str) -> str:
    algorithm_type = _normalize_algorithm_type(cfg.algorithm.type)
    if kind == "sh":
        if algorithm_type in {"r_pci", "pci_rnn"}:
            return "mpci" if bool(getattr(cfg.algorithm, "use_centralized_critic", False)) else "ipci"
        return SH_MODEL_NAMES.get(algorithm_type, model_dir.split("_")[-1])
    if kind == "lbf":
        return LBF_MODEL_NAMES.get(algorithm_type, model_dir)
    raise ValueError(f"unknown kind: {kind}")


def _env_id_from_lbf_config(cfg: Config) -> str:
    run_name = str(getattr(cfg.wandb, "run_name", "") or "")
    if run_name.startswith("JumanjiLBF-"):
        return run_name.removeprefix("JumanjiLBF-")
    env_id = str(getattr(cfg.environment, "env_id", "") or "")
    return env_id.removeprefix("JumanjiLBF-") if env_id else "unknown_env"


def _row(
    *,
    run_dir: Path,
    metadata_path: Path,
    cfg: Config,
    scenario: str,
    scenario_label: str,
    model_dir: str,
    model_label: str,
    env_dir_name: str,
    kind: str,
) -> dict[str, Any]:
    algorithm_type = _normalize_algorithm_type(cfg.algorithm.type)
    seed = int(getattr(cfg.training, "seed", _parse_seed(run_dir.name)))
    ckpt = _checkpoint_paths(run_dir, env_dir_name, cfg.algorithm.type)
    project = str(getattr(cfg.wandb, "project", "") or "")
    env_signature = json.dumps(
        {
            "kind": kind,
            "algorithm_type": algorithm_type,
            "env": cfg.environment.get_relevant_config(),
            "seed": seed,
        },
        sort_keys=True,
        default=str,
    )
    return {
        "run_dir": run_dir,
        "metadata_path": metadata_path,
        "project": project,
        "map_name": scenario,
        "map_label": scenario_label,
        "scenario": scenario,
        "scenario_label": scenario_label,
        "wandb_run_name": str(getattr(cfg.wandb, "run_name", "") or run_dir.name),
        "constraint_key": "default",
        "constraint_label": "default",
        "algorithm_type": algorithm_type,
        "model_key": model_label,
        "model_label": model_label,
        "model_dir": model_dir,
        "seed": seed,
        "config": cfg,
        "env_dir_name": env_dir_name,
        "best_checkpoint": ckpt["best"],
        "final_checkpoint": ckpt["final"],
        "env_signature": env_signature,
        "kind": kind,
    }


def discover_staghunt_runs(
    root: Path,
    *,
    scenario_filter: Optional[Iterable[str]] = ("sh_noise",),
) -> pd.DataFrame:
    root = Path(root)
    allowed = {str(item) for item in scenario_filter or []}
    rows = []
    for metadata_path in sorted(root.rglob("staghuntenv/simulation_config.json")):
        env_dir = metadata_path.parent
        run_dir = env_dir.parent
        rel = run_dir.relative_to(root)
        if len(rel.parts) < 3:
            continue
        scenario = rel.parts[0]
        if allowed and scenario not in allowed:
            continue
        model_dir = rel.parts[1]
        payload = json.loads(metadata_path.read_text())
        cfg = Config.from_dict(payload["config"])
        noise = float(getattr(cfg.environment, "obs_noise_std", 0.0) or 0.0)
        if scenario == "sh_noise" and abs(noise - 0.1) > 1e-9:
            continue
        if scenario == "sh" and noise != 0.0:
            continue
        model_label = _selected_model_label(cfg, model_dir=model_dir, kind="sh")
        rows.append(
            _row(
                run_dir=run_dir,
                metadata_path=metadata_path,
                cfg=cfg,
                scenario=scenario,
                scenario_label=f"{scenario} (obs_noise_std={noise:g})",
                model_dir=model_dir,
                model_label=model_label,
                env_dir_name=env_dir.name,
                kind="sh",
            )
        )
    return _finalize_runs(rows, ["map_name", "model_label", "seed", "run_dir"])


def discover_lbf_runs(
    root: Path,
    *,
    env_id_filter: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    root = Path(root)
    allowed = {str(item) for item in env_id_filter or []}
    rows = []
    for metadata_path in sorted(root.rglob("jumanjilbfbatchwrapper/simulation_config.json")):
        env_dir = metadata_path.parent
        run_dir = env_dir.parent
        rel = run_dir.relative_to(root)
        if len(rel.parts) < 3:
            continue
        env_id = rel.parts[0]
        if allowed and env_id not in allowed:
            continue
        model_dir = rel.parts[1]
        payload = json.loads(metadata_path.read_text())
        cfg = Config.from_dict(payload["config"])
        parsed_env_id = _env_id_from_lbf_config(cfg)
        if parsed_env_id != "unknown_env" and parsed_env_id != env_id:
            # The directory layout is the evaluation contract; avoid silently
            # evaluating a checkpoint under the wrong LBF scenario.
            continue
        model_label = _selected_model_label(cfg, model_dir=model_dir, kind="lbf")
        rows.append(
            _row(
                run_dir=run_dir,
                metadata_path=metadata_path,
                cfg=cfg,
                scenario=env_id,
                scenario_label=env_id,
                model_dir=model_dir,
                model_label=model_label,
                env_dir_name=env_dir.name,
                kind="lbf",
            )
        )
    return _finalize_runs(rows, ["map_name", "model_label", "seed", "run_dir"])


def _finalize_runs(rows: list[dict[str, Any]], sort_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for choice in ("best", "final"):
        path_col = f"{choice}_checkpoint"
        df = df[df[path_col].map(lambda path: Path(path).is_file())]
    return df.sort_values(sort_cols).reset_index(drop=True)


def _build_generic_env(cfg: Config, *, n_envs: int, batch: bool) -> Any:
    factories = base.bootstrap_project(cfg.training.device)
    env_cfg = copy.deepcopy(cfg.environment)
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = int(getattr(cfg.training, "seed", getattr(env_cfg, "seed", 0)) or 0)
    env_name = str(env_cfg.get_canonical_env_name()).lower().replace("-", "_")
    if env_name in {"jaxlbf", "jumanjilbf", "jax_lbf", "jumanji_lbf"}:
        from envs import JumanjiLBFBatchWrapper

        return JumanjiLBFBatchWrapper(
            n_envs=int(max(1, n_envs if batch else 1)),
            field_size=int(getattr(env_cfg, "field_size", 10)),
            num_agents=int(getattr(env_cfg, "num_agents", 3)),
            max_food=int(getattr(env_cfg, "max_food", 3)),
            player_level=tuple(getattr(env_cfg, "player_level", (1, 2))),
            sight=getattr(env_cfg, "sight", 2),
            max_episode_steps=int(getattr(env_cfg, "max_episode_steps", 100)),
            force_coop=bool(getattr(env_cfg, "force_coop", False)),
            seed=int(getattr(env_cfg, "seed", 0) or 0),
            grid_observation=bool(getattr(env_cfg, "grid_observation", True)),
            normalize_reward=bool(getattr(env_cfg, "normalize_reward", True)),
            penalty=float(getattr(env_cfg, "penalty", 0.0)),
        )
    if hasattr(env_cfg, "n_envs"):
        env_cfg.n_envs = int(max(1, n_envs if batch else 1))
    return factories["create_environment"](env_cfg)


def _load_policy_only_checkpoint(
    trainer: Any,
    run_row: pd.Series | dict[str, Any],
    *,
    checkpoint_choice: str,
) -> Path:
    if isinstance(run_row, pd.Series):
        run_row = run_row.to_dict()

    algorithm_type = _normalize_algorithm_type(run_row["algorithm_type"])
    if algorithm_type.startswith("jax_"):
        return _BASE_LOAD_TRAINER_CHECKPOINT(
            trainer,
            run_row,
            checkpoint_choice=checkpoint_choice,
        )

    checkpoint_path = Path(run_row[f"{checkpoint_choice}_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    trainer.iteration = checkpoint.get("iteration", trainer.iteration)
    trainer.total_steps = checkpoint.get("total_steps", trainer.total_steps)
    trainer.best_eval_reward = checkpoint.get("best_eval_reward", trainer.best_eval_reward)

    algorithm_state = checkpoint.get("algorithm_state", {})
    agent_states = algorithm_state.get("agents", {})
    if not hasattr(trainer.algorithm, "agents"):
        raise ValueError("Policy-only checkpoint loading requires an algorithm with agents.")

    for agent_id, agent in trainer.algorithm.agents.items():
        saved_agent_state = agent_states.get(agent_id, agent_states.get(str(agent_id)))
        if not saved_agent_state or "policy_net" not in saved_agent_state:
            continue
        agent.policy_net.load_state_dict(saved_agent_state["policy_net"])
        if hasattr(agent.policy_net, "eval"):
            agent.policy_net.eval()
    return checkpoint_path


def _save_evaluation_disk_cache_atomic(cache_path: Path, payload: dict[str, Any]) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(cache_path)
    base._RUN_EVAL_DISK_CACHE[cache_path] = payload


def _evaluation_cache_key_with_metric_version(
    run_row: dict[str, Any],
    *,
    checkpoint_choice: str,
    device: str,
    eval_episodes: int,
    deterministic: bool,
    eval_seed: Optional[int],
) -> tuple[Any, ...]:
    key = _BASE_EVALUATION_CACHE_KEY(
        run_row,
        checkpoint_choice=checkpoint_choice,
        device=device,
        eval_episodes=eval_episodes,
        deterministic=deterministic,
        eval_seed=eval_seed,
    )
    return (*key, "sh_lbf_eval_metric_v2")


def _deterministic_nonjax_evaluation(trainer: Any, num_episodes: int) -> dict[str, float]:
    from algorithms.common import resolve_step_episode_done

    env = trainer.eval_env
    device = trainer.algorithm.device
    episode_limit = trainer._get_eval_episode_limit()
    for agent in trainer.algorithm.agents.values():
        policy_net = getattr(agent, "policy_net", None)
        if policy_net is not None and hasattr(policy_net, "eval"):
            policy_net.eval()

    eval_rewards: dict[int, list[float]] = {}
    eval_lengths = []
    eval_successes = []
    is_staghunt = env.__class__.__name__ == "StagHuntEnv"

    with torch.inference_mode():
        for _ in range(num_episodes):
            observations = env.reset()
            trainer.algorithm.reset_agents()
            observations = base.torch_like_obs_to_tensor(observations, device)
            observations = trainer._add_agent_id_to_obs(observations)

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
                    actions[agent_id] = base._greedy_action_for_agent(
                        trainer.algorithm.agents[agent_id],
                        obs,
                        agent_avail,
                        device,
                    )

                next_observations, rewards, dones, info = env.step(actions)
                next_observations = base.torch_like_obs_to_tensor(next_observations, device)
                next_observations = trainer._add_agent_id_to_obs(next_observations)

                if is_staghunt:
                    stag_alive = info.get("stag_alive", {})
                    hare_alive = info.get("hare_alive", {})
                    if not any(stag_alive.values()) and not any(hare_alive.values()):
                        episode_success = 1.0
                elif info.get("battle_won", False):
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

    mean_per_agent = [float(base.np.mean(rewards)) for rewards in eval_rewards.values()] if eval_rewards else []
    return {
        "eval_success_rate": float(base.np.mean(eval_successes)) if eval_successes else 0.0,
        "eval_mean_reward": float(base.np.mean(mean_per_agent)) if mean_per_agent else 0.0,
        "eval_std_reward": float(base.np.std(mean_per_agent)) if mean_per_agent else 0.0,
        "eval_mean_length": float(base.np.mean(eval_lengths)) if eval_lengths else 0.0,
        "eval_min_reward": float(base.np.min(mean_per_agent)) if mean_per_agent else 0.0,
        "eval_max_reward": float(base.np.max(mean_per_agent)) if mean_per_agent else 0.0,
    }


base._build_smax_env = _build_generic_env
base._load_trainer_checkpoint = _load_policy_only_checkpoint
base._save_evaluation_disk_cache = _save_evaluation_disk_cache_atomic
base._evaluation_cache_key = _evaluation_cache_key_with_metric_version
base.deterministic_nonjax_evaluation = _deterministic_nonjax_evaluation

evaluate_map = base.evaluate_map
evaluate_map_constraint = base.evaluate_map_constraint
evaluate_run = base.evaluate_run
evaluate_run_seeds = base.evaluate_run_seeds
format_cross_map_latex = base.format_cross_map_latex
save_full_evaluation_bundle = base.save_full_evaluation_bundle
save_summary_tables = base.save_summary_tables
