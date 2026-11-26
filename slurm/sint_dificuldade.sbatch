#!/usr/bin/env bash
#SBATCH --job-name=lumi-sint-dif
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# E5, controles do piloto sintético (analises/e5_dados.md, revisão de 25/11): o Lumi congelado em 3 condições
# (completo, sem_texto, so_versao_2), nas 4 rotações das opções. Saída: runs/sintdif_<condição><SUFIXO>.
#   sbatch slurm/sint_dificuldade.sbatch SUFIXO ARQUIVO_DE_CASOS...
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
SUF=${1:?SUFIXO}; shift
D=data/sint/controles$SUF
lumi_exec lumi-dados sint-controles --casos "$@" --saida-dir "$D" || exit 1
lumi_gpu_smoke || exit 1
BASE="--backend hf --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d --dtype bfloat16 --native --no-generate --split train --orders canonical,rot1,rot2,rot3"
for cond in completo sem_texto so_versao_2; do
  lumi_exec lumi-eval run $BASE --source "$D/$cond.jsonl" --out "runs/sintdif_${cond}$SUF" || { echo ">> $cond falhou"; exit 1; }
done
lumi_exec python3 scripts/analise_sint.py --sufixo "$SUF" || exit 1
echo ">> controles concluídos ($SUF)"
