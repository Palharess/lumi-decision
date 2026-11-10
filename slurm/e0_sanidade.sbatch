#!/usr/bin/env bash
#SBATCH --job-name=lumi-e0
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# E0: checkpoint oficial BF16 nos 40 casos do piloto, com remote code e com a implementação nativa.
# Uso (da raiz do repo):  sbatch slurm/e0_sanidade.sbatch
set -euo pipefail
source slurm/config.sh
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv

for impl in remote native; do
  flag=""; [ "$impl" = native ] && flag="--native"
  lumi_exec lumi-e0 $flag --source referencias/piloto40/casos.json --ref-run referencias/piloto40 \
    --out "runs/e0_${impl}_${SLURM_JOB_ID}" || echo ">> E0 ($impl) falhou"
done
