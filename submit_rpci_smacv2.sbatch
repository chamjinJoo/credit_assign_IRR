#!/bin/bash
#SBATCH -p rtx2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -J rpci_smacv2
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@600
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Optional: keep this if your cluster requires explicit conda initialization in batch shells.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

bash scripts/smacv2/train_rpci.sh
