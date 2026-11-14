#!/usr/bin/env bash
#SBATCH --job-name=lumi-orcamento
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Conta atualizações, exemplos vistos e tokens de treinos já feitos (sem GPU).
#   sbatch slurm/contar_orcamento.sbatch '{"seed": 1, "data_seed": 0}' '{"cap_per_family": 1300, ...}'
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
lumi_exec python scripts/contar_orcamento.py "$@"
