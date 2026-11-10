#!/usr/bin/env bash
#SBATCH --job-name=lumi-e1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Uso: sbatch -p "$LUMI_PARTITION" slurm/e1_h1_dev.sbatch [MODELO] [REVISAO]
# E1 (H1): readout congelado + geração curta no dev das três fontes, duas ordens de opções.
# Rodar depois que o E0 passar. MODELO padrão: Lumi; passe o Nemotron base para o controle.
set -euo pipefail
source slurm/config.sh
lumi_activate
export HF_HUB_OFFLINE=1

MODEL="${1:-ceia-br/lumi-4b}"
REV="${2:-27b30f40b930fd05cee5e1c1f8790f70fa5b816d}"
TAG="$(basename "$MODEL")"
IMPL_FLAG="${LUMI_IMPL_FLAG:-}"   # export LUMI_IMPL_FLAG=--native se o E0 indicar

for src in felhen telepatia traces; do
  lumi-eval run --backend hf --model "$MODEL" --revision "$REV" --dtype bfloat16 $IMPL_FLAG \
    --source "$src" --split dev --orders canonical,reversed \
    --out "runs/e1_${TAG}_${src}_dev"
done
