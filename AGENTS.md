# Project Memory

This repository is a Python multi-agent reinforcement learning research codebase
for PCI-style algorithms and baselines. Treat this file as the first stop for
project orientation before making changes.

## Active Scope

- Main entrypoint: `main.py`.
- Current submit-ready paths are centered on:
  - Torch Stag Hunt experiments via `scripts/staghunt/`.
  - JAX SMAX experiments via `scripts/smax/`.
  - JAX/Jumanji LBF scripts under `scripts/lbf/`.
- Archived or older experiment code lives under `scripts/archive/`,
  `configs/unused/`, `legacy/`, and should not be treated as the active path
  unless the user explicitly asks.
- Runtime outputs and large/generated artifacts live under `runs/`,
  `save_checkpoints/`, `slurms/`, `figures/`, `not_for_submit/`, and similar
  directories. Avoid touching them unless the task is explicitly about results,
  checkpoints, SLURM jobs, or figures.

## Execution Flow

`main.py` follows this high-level sequence:

1. Parse CLI args using `configs.config.parse_args`.
2. Load YAML config into dataclass config objects with
   `Config.from_yaml` / `create_config_from_args`.
3. Apply CLI overrides on top of YAML.
4. Configure JAX CPU runtime early when `training.device == "cpu"`.
5. Seed Python, NumPy, Torch, and JAX-related runtime state through
   `utils.seed_utils`.
6. Create the environment.
7. Create the algorithm from `config.algorithm.type`.
8. Pick `JAXTrainer` when the algorithm exposes `requires_jax_trainer`,
   otherwise use the Torch `Trainer`.
9. Run train, eval, or test mode.

Config precedence is: dataclass defaults -> YAML file -> CLI overrides.

## Main Packages

- `configs/`: Dataclass-based configuration and YAML experiment presets.
  `configs/config.py` defines network, algorithm, training, environment, and
  wandb config schemas.
- `envs/`: Multi-agent environment interface and wrappers.
  - `base.py`: abstract `MultiAgentEnv` API: `reset`, `step`, `get_obs_dim`,
    `get_action_dim`.
  - `env_utils.py`: `create_environment` factory and tensor helpers.
  - `stag_hunt.py`: Torch-facing Stag Hunt environment.
  - `smax.py`: SMAX single and batched wrappers, including pure JAX reset/step.
  - `lbf.py`: Gym-style LBForaging wrapper and Jumanji batched LBF wrapper.
  - `parallel_env.py`: multiprocessing wrapper for non-native vectorized envs.
- `models/torch/`: Torch algorithms and networks used mainly for Stag Hunt.
  Active exports are `rPCIAlgorithm`, `rMAPPOAlgorithm`, `rIPPOAlgorithm`,
  and `rQMIXAlgorithm`.
- `models/jax/`: JAX/Flax implementations for SMAX/LBF-style batched training.
  Active top-level exports are `JAXPPOAlgorithm`, `JAXRPCIAlgorithm`,
  `JAXQMIXAlgorithm`, and `JAXPCIQMIXAlgorithm`.
- `runner/torch/trainer.py`: Default trainer for rollout collection, updates,
  eval, checkpointing, metadata, and wandb logging.
- `runner/jax/trainer.py`: Subclasses the Torch trainer but uses native pure
  JAX batched env APIs and JAX algorithm rollout/update paths.
- `utils/`: Buffers, logging, metrics, seed/reproducibility helpers, and small
  common utilities.
- `scripts/`: Shell launchers. Prefer active `staghunt`, `smax`, and `lbf`
  directories over archived scripts.
- `third_party/`: Vendored dependencies and assets.

## Algorithm Mapping

Torch path:

- `r-pci` / `r_pci` -> `models.torch.r_pci.rPCIAlgorithm`.
- `r-mappo` / `r_mappo` -> `models.torch.r_mappo.rMAPPOAlgorithm`.
- `r-ippo` / `r_ippo` -> `models.torch.r_ippo.rIPPOAlgorithm`.
- `qmix` / `r_qmix` -> `models.torch.qmix.rQMIXAlgorithm`.

JAX path:

- `jax-ippo`, `jax-mappo`, `jax-ppo` -> `JAXPPOAlgorithm`.
- `jax-rpci`, `jax-pci-ippo`, `jax-pci-mappo`, `jax-rippo`,
  `jax-rmappo` -> `JAXRPCIAlgorithm` with flags controlling PCI and
  centralized critic behavior.
- `jax-qmix` -> `JAXQMIXAlgorithm`.
- `jax-pci-qmix` -> `JAXPCIQMIXAlgorithm`.

Be careful with stale branches in `main.py`: older names such as `pci_mappo`,
`mappo`, `rippo`, and `coma` still appear in conditionals, but the current
top-level `models/__init__.py` does not export all of those symbols. Check the
active exports before assuming an old branch still runs.

## Environment Notes

- `envs.env_utils.create_environment` currently recognizes active names for
  `StagHunt`, `SMAX`, `LBForaging`/`LBF`, and JAX/Jumanji LBF aliases.
- `main.py` special-cases SMAX and Jumanji LBF when using JAX algorithms or
  multiple rollout threads, creating native batched wrappers directly.
- SMAX wrappers expose `reset_jax_pure` and `step_jax_pure`; JAXTrainer requires
  those methods.
- Available environment reporting in `get_available_environments` is minimal
  and not a full list of every alias accepted by the factory.

## Common Commands

- Setup: `conda create -n pcimarl python=3.10.18 -y`, then `bash install.sh`.
- Example Stag Hunt: `bash scripts/staghunt/train_ippo.sh`.
- Example SMAX: `bash scripts/smax/train_ippo.sh`.
- Most launchers expose key knobs near the top: `config`, `model`, seed(s),
  device, rollout threads, env options, and wandb settings.

## Development Guidance

- Prefer `rg` / `rg --files` for code search.
- Preserve user outputs and checkpoints. Do not clean `runs/`,
  `save_checkpoints/`, or result directories without an explicit request.
- Keep changes aligned with the active Torch/JAX split rather than refactoring
  archived paths.
- When adding a new algorithm or environment, update the relevant package
  exports, `main.py` factory logic, configs, and scripts together.

## Additional Notes
- Fail quickly without fallback code to prevent broken features from being hidden.
- Do not attach unnecessary annotations. If an annotation is needed, it should be simple and clear.
- Speculative edits are prohibited. Do not make edits if the cause is unknown.
- Variable names should be simple and clear while compatible to codebase.