#!/usr/bin/env bash
#SBATCH --job-name=lumi-res2-dif
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Reservadas v2 (analises/reservadas_v2.md): filtro de dificuldade. SÓ o Lumi congelado, só a fatia
# `dificuldade` (50 grupos por família), em todas as rotações das opções. Saída: runs/res2dif_<família>.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
lumi_gpu_smoke || exit 1
BASE="--backend hf --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d --dtype bfloat16 --native --no-generate --source res2 --split dificuldade"
for spec in restricoes_oferta:canonical,rot1,rot2,rot3 regra_beneficio:canonical,rot1,rot2,rot3 \
            custo_plano:canonical,rot1,rot2 ordem_eventos:canonical,rot1,rot2,rot3; do
  fam=${spec%%:*}; ord=${spec#*:}
  lumi_exec lumi-eval run $BASE --families "$fam" --orders "$ord" --out "runs/res2dif_$fam" || { echo ">> $fam falhou"; exit 1; }
done
echo ">> filtro de dificuldade concluído"
