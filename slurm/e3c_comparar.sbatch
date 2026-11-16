#!/usr/bin/env bash
#SBATCH --job-name=lumi-e3c-cmp
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
# E3c (analises/plano_fase2.md): confere o pareamento dos três braços de uma semente (mesmo nº real de
# atualizações). Os critérios são calculados com scripts/analise_m2.py sobre runs/res2dev_* e runs/factck6_*.
#   sbatch --dependency=afterok:<proximo>:<resto>:<ctrl> slurm/e3c_comparar.sbatch <SEMENTE>
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
S="${1:?semente}"
passos() { grep -oE '"passos_total": *[0-9]+' "runs/$1/treino_meta.json" | grep -oE '[0-9]+$'; }
p=$(passos "e3c_proximo_s$S"); r=$(passos "e3c_resto_s$S"); c=$(passos "e3c_ctrl_s$S")
echo ">> atualizações s$S: próximo = $p, resto = $r, controle = $c"
[ -n "$p" ] && [ "$p" = "$r" ] && [ "$p" = "$c" ] || { echo ">> nº de atualizações diferente: não pareado"; exit 1; }
echo ">> pareado"
