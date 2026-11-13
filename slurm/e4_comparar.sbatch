#!/usr/bin/env bash
#SBATCH --job-name=lumi-e4-cmp
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# E4 (H3): critério registrado no README ("Protocolo do E4"). Só CPU; submeter com
#   sbatch --dependency=afterok:<job do E4>[:<job do E3>] slurm/e4_comparar.sbatch [NOME_E4] [NOME_E3]
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
B="runs/${1:-e4_ponteiro_s0}"; A="runs/${2:-e3_lora_r16}"   # 2º argumento: o A1 pareado (réplicas do F1)
dirs() { echo "$1_felhen_dev,$1_telepatia_dev,$1_traces_dev"; }
lumi_exec lumi-eval h3 --base "$(dirs "$A")" --novo "$(dirs "$B")" --out "$B/h3.json" || exit 1
grep -A6 '"veredito"' "$B/h3.json"
