#!/usr/bin/env bash
#SBATCH --job-name=lumi-e3br-cmp
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# E3b-r (analises/plano_fase2.md): confere o pareamento (mesmo nº real de atualizações) e roda o critério
# original do E3b (3 reservadas, 2 ordens). O critério principal (reservadas v2) é calculado com
# scripts/analise_m2.py sobre runs/res2dev_* (slurm/m2_res2.sbatch).
#   sbatch --dependency=afterok:<e3b>:<ctrl> slurm/e3br_comparar.sbatch <SEMENTE>
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
S="${1:?semente}"; E="runs/e3b_eikos_s$S"; C="runs/e3b_ctrl_s$S"
pe=$(grep -oE '"passos_total": *[0-9]+' "$E/treino_meta.json" | grep -oE '[0-9]+$')
pc=$(grep -oE '"passos_total": *[0-9]+' "$C/treino_meta.json" | grep -oE '[0-9]+$')
echo ">> atualizações: E3b s$S = $pe, controle s$S = $pc"
[ -n "$pe" ] && [ "$pe" = "$pc" ] || { echo ">> número de atualizações diferente: comparação não pareada"; exit 1; }
dirs() { echo "$1_felhen_dev,$1_telepatia_dev,$1_traces_dev"; }
lumi_exec lumi-eval h2 --base "$(dirs "$C")" --novo "$(dirs "$E")" --out "$E/e3b_menos_controle.json" || exit 1
grep -h '"veredito_h2"' "$E/e3b_menos_controle.json"
