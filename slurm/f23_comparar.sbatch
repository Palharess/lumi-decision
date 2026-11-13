#!/usr/bin/env bash
#SBATCH --job-name=lumi-f23-cmp
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# F2.3a (analises/plano_fase2.md): pointer residual x controle (mesmo treino adicional só pelas letras), por semente.
#   sbatch --dependency=afterok:<residual>:<controle> slurm/f23_comparar.sbatch <SEMENTE>
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
S="${1:?semente}"; R="runs/f23_residual_s$S"; C="runs/f23_controle_s$S"
dirs() { echo "$1_felhen_dev,$1_telepatia_dev,$1_traces_dev"; }
lumi_exec lumi-eval h3 --base "$(dirs "$C")" --novo "$(dirs "$R")" --readout-novo residual --out "$R/residual_menos_controle.json" || exit 1
grep -A6 '"veredito"' "$R/residual_menos_controle.json"
