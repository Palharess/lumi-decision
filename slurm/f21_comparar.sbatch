#!/usr/bin/env bash
#SBATCH --job-name=lumi-f21-cmp
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# F2.1 (analises/plano_fase2.md): mistura balanceada x E3 da mesma semente (mesmo nº de atualizações).
#   sbatch --dependency=afterok:<job balanceado> slurm/f21_comparar.sbatch <SEMENTE>
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
S="${1:?semente}"; N="runs/f21_balanceado_s$S"
if [ "$S" = 0 ]; then A=runs/e3_lora_r16; else A="runs/e3_lora_r16_s$S"; fi
dirs() { echo "$1_felhen_dev,$1_telepatia_dev,$1_traces_dev"; }
lumi_exec lumi-eval h3 --base "$(dirs "$A")" --novo "$(dirs "$N")" --readout-novo letras --out "$N/balanceado_menos_e3.json" || exit 1
grep -A6 '"veredito"' "$N/balanceado_menos_e3.json"
