#!/usr/bin/env bash
#SBATCH --job-name=lumi-e2
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# E2: sanidade do LoRA no Lumi nativo (alvos, gradientes, sobreajuste, recarga) + memória por micro-lote.
# O job FALHA (código != 0) se alguma verificação obrigatória reprovar. Libera e dimensiona o E3.
# Uso:  sbatch slurm/e2_lora.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"   # funciona submetendo de qualquer pasta
source slurm/config.sh
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
lumi_gpu_smoke
lumi_exec python -m lumi_decision.e2_lora --out "runs/e2_${SLURM_JOB_ID}"
