#!/usr/bin/env bash
#SBATCH --job-name=lumi-prof-montar
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# Junta as partes da geração definitiva (analises/reservadas_v2.md, emenda 2) e monta os 650 grupos por família.
#   sbatch slurm/prof_montar.sbatch SUFIXO...   (ex.: _d0 _d1 _d2 _d3; saída data/prof/prof_v1.jsonl)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
D=data/prof
CAND=(); CONF=()
for s in "$@"; do CAND+=("$D/candidatos$s.jsonl"); CONF+=("$D/respostas_conferente$s.jsonl"); done
ls -la "${CAND[@]}" "${CONF[@]}" || exit 1
lumi_exec lumi-dados prof-montar --candidatos "${CAND[@]}" --conferencia "${CONF[@]}" --saida "$D/prof_v1.jsonl" || exit 1
echo ">> montagem concluída ($*)"
