## Setup

Create a conda environment and install dependencies:

```bash
conda create -n pcimarl python=3.10.18 -y
conda activate pcimarl
bash install.sh
```

The launcher scripts assume the conda environment name is `pcimarl`.

## Training Scripts

- `scripts/staghunt/`: Torch Stag Hunt experiments.
- `scripts/smax/`: JAX SMAX experiments.

Each script exposes its main settings near the top of the file, such as
`config`, `model`, `seed` or `seeds`, `device`, and environment-specific
options.

Example commands:

```bash
bash scripts/staghunt/train_ippo.sh
bash scripts/smax/train_ippo.sh
```

Some scripts enable Weights & Biases logging by default. If external logging is
not needed, remove or comment out the corresponding `--use-wandb` arguments in
the script before running it.

## Code Layout

- `models/torch/`: Torch algorithm implementations used for Stag Hunt.
- `models/jax/`: JAX algorithm implementations used for SMAX.
- `configs/`: YAML and dataclass-based experiment configuration.
- `envs/`: multi-agent environments and wrappers.
- `runner/torch/`, `runner/jax/`: training loops for Torch and JAX.
- `utils/`: shared utility code.
- `figures/`: analysis notebooks/scripts used to produce paper figures and tables.
- `legacy/`: retired source snapshots kept for reference only, not imported by anything.
- `third_party/`: vendored dependencies (`StarCraftII/` game assets, `gym-cooking/` env).
- `slurms/templates/`: SLURM `.sbatch` job templates; submit from the repo root, e.g. `sbatch slurms/templates/slurm_train_smax.sbatch`.
- `runs/`: gitignored training outputs (checkpoints, W&B logs/exports) consolidated under one directory.
- `not_for_submit/`: gitignored eval checkpoints excluded from paper submission.