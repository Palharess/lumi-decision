#!/usr/bin/env bash
#SBATCH --job-name=lumi-e4-san
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# E4: sanidade da pointer head (posições, gradientes, kernel fundido, sobreajuste, coerência, recarga
# conjunta, RNG). O job FALHA (código != 0) se alguma verificação obrigatória reprovar. Libera o treino do E4.
# Uso:  sbatch slurm/e4_sanidade.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
lumi_gpu_smoke
lumi_exec python -m lumi_decision.e4_ponteiro --out "runs/e4_sanidade_${SLURM_JOB_ID}"
