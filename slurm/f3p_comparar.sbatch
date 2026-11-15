#!/usr/bin/env bash
#SBATCH --job-name=lumi-f3p-cmp
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Piloto F3 (analises/plano_fase2.md): braço diverso x controle (braço do F2.1) da mesma semente.
#   sbatch --dependency=afterok:<job diverso> slurm/f3p_comparar.sbatch <SEMENTE>
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
S="${1:?semente}"; N="runs/f3p_diverso_s$S"; A="runs/f21_balanceado_s$S"
pa=$(grep -oE '"passos_total": *[0-9]+' "$A/treino_meta.json" | grep -oE '[0-9]+$')
pn=$(grep -oE '"passos_total": *[0-9]+' "$N/treino_meta.json" | grep -oE '[0-9]+$')
echo ">> atualizações: controle s$S = $pa, diverso s$S = $pn"
[ -n "$pa" ] && [ "$pa" = "$pn" ] || { echo ">> número de atualizações diferente: comparação não pareada"; exit 1; }
dirs() { echo "$1_felhen_dev,$1_telepatia_dev,$1_traces_dev"; }
lumi_exec lumi-eval h3 --base "$(dirs "$A")" --novo "$(dirs "$N")" --readout-novo letras --out "$N/diverso_menos_controle.json" || exit 1
grep -A6 '"veredito"' "$N/diverso_menos_controle.json"
