#!/usr/bin/env bash
#SBATCH --job-name=lumi-f22-cmp
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# F2.2 (analises/plano_fase2.md): E3 sem treinar o LoRA do in_proj x E3 da mesma semente.
#   sbatch --dependency=afterok:<job f22> slurm/f22_comparar.sbatch <SEMENTE>
# Recusa comparar se o número real de atualizações não for o mesmo do E3 da mesma semente.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
S="${1:?semente}"; N="runs/f22_sem_inproj_s$S"
if [ "$S" = 0 ]; then A=runs/e3_lora_r16; else A="runs/e3_lora_r16_s$S"; fi
pa=$(grep -oE '"passos_total": *[0-9]+' "$A/treino_meta.json" | grep -oE '[0-9]+$')
pn=$(grep -oE '"passos_total": *[0-9]+' "$N/treino_meta.json" | grep -oE '[0-9]+$')
echo ">> atualizações: E3 s$S = $pa, sem in_proj s$S = $pn"
[ -n "$pa" ] && [ "$pa" = "$pn" ] || { echo ">> número de atualizações diferente: comparação não pareada"; exit 1; }
dirs() { echo "$1_felhen_dev,$1_telepatia_dev,$1_traces_dev"; }
lumi_exec lumi-eval h3 --base "$(dirs "$A")" --novo "$(dirs "$N")" --readout-novo letras --out "$N/sem_inproj_menos_e3.json" || exit 1
grep -A6 '"veredito"' "$N/sem_inproj_menos_e3.json"
