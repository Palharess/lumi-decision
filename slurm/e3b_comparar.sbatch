#!/usr/bin/env bash
#SBATCH --job-name=lumi-e3b-cmp
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# E3b: comparações registradas antes de rodar (README, "Protocolo do E3b"). Só CPU; submeter com
#   sbatch --dependency=afterok:<job E3b>:<job controle> slurm/e3b_comparar.sbatch [E3B] [CTRL]
# Principal: E3b − controle (mesmo nº de atualizações e cronograma). Secundária: E3b − Lumi congelado (E1).
# Diagnóstico: E3b − E3 original (508 atualizações).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
E3B="runs/${1:-e3b_eikos_s0}"; CTRL="runs/${2:-e3b_ctrl_s0}"; E3=runs/e3_lora_r16; E1=runs/e1_lumi_27b30f4_nativa
dirs() { echo "$1_felhen_dev,$1_telepatia_dev,$1_traces_dev"; }
e1dirs="${E1}_felhen_dev_completo,${E1}_telepatia_dev_completo,${E1}_traces_dev_completo"
lumi_exec lumi-eval h2 --base "$(dirs "$CTRL")" --novo "$(dirs "$E3B")" --out "$E3B/e3b_menos_controle.json" || exit 1
lumi_exec lumi-eval h2 --base "$e1dirs" --novo "$(dirs "$E3B")" --out "$E3B/e3b_menos_congelado.json" || exit 1
lumi_exec lumi-eval h2 --base "$(dirs "$E3")" --novo "$(dirs "$E3B")" --out "$E3B/e3b_menos_e3.json" || exit 1
lumi_exec lumi-eval h2 --base "$e1dirs" --novo "$(dirs "$CTRL")" --out "$CTRL/controle_menos_congelado.json" || exit 1
grep -h '"veredito_h2"' "$E3B"/e3b_menos_*.json "$CTRL/controle_menos_congelado.json"
