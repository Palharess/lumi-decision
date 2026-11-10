#!/usr/bin/env bash
#SBATCH --job-name=lumi-e0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Uso: sbatch -p "$LUMI_PARTITION" slurm/e0_sanidade.sbatch
# E0: checkpoint oficial BF16 nos 40 casos do piloto, com remote code e com a implementação nativa.
set -euo pipefail
source slurm/config.sh
lumi_activate
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

for impl in remote native; do
  flag=""; [ "$impl" = native ] && flag="--native"
  lumi-e0 $flag --source referencias/piloto40/casos.json --ref-run referencias/piloto40 \
    --out "runs/e0_${impl}_${SLURM_JOB_ID}" || echo "E0 ($impl) falhou"
done
