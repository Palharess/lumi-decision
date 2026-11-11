#!/usr/bin/env bash
#SBATCH --job-name=lumi-e1
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# E1 (H1): readout congelado + geração curta no dev das três fontes, duas ordens de opções.
# Rodar depois que o E0 passar.
# Uso:  sbatch slurm/e1_h1_dev.sbatch [MODELO] [REVISAO]
#       (Nemotron base como controle: sbatch slurm/e1_h1_dev.sbatch nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 main)
# Reentrante: as predições são gravadas linha a linha e casos já feitos são pulados;
# se o job cair (manutenção, limite de tempo), basta submeter de novo.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"   # funciona submetendo de qualquer pasta
source slurm/config.sh
export HF_HUB_OFFLINE=1

MODEL="${1:-ceia-br/lumi-4b}"
REV="${2:-27b30f40b930fd05cee5e1c1f8790f70fa5b816d}"
TAG="$(basename "$MODEL")"
IMPL_FLAG="${LUMI_IMPL_FLAG:-}"   # export LUMI_IMPL_FLAG=--native se o E0 indicar

for src in felhen telepatia traces; do
  lumi_exec lumi-eval run --backend hf --model "$MODEL" --revision "$REV" --dtype bfloat16 $IMPL_FLAG \
    --source "$src" --split dev --orders canonical,reversed \
    --out "runs/e1_${TAG}_${src}_dev"
done
