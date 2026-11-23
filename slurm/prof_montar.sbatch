#!/usr/bin/env bash
#SBATCH --job-name=lumi-prof-montar
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# Junta as partes da geração definitiva (analises/reservadas_v2.md, emenda 2) e monta os 650 grupos por família.
#   sbatch slurm/prof_montar.sbatch PARTE...   (saída data/prof/prof_v1.jsonl)
# PARTE é um sufixo (_d2) ou um sufixo com famílias descartadas, separadas por "-"
# (_d0-resposta_indireta-inferencia_pt): as emendas 3 e 4 descartam, sem ler, as famílias geradas com pedidos antigos.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
D=data/prof
CAND=(); CONF=()
for p in "$@"; do
  s=${p%%-*}; f="$D/candidatos$s.jsonl"
  if [ "$s" != "$p" ]; then
    fams=${p#*-}; padrao="\"family\": \"(${fams//-/|})\""
    grep -v -E "$padrao" "$f" > "$D/candidatos$s-sem.jsonl" || exit 1
    echo ">> $s: $(grep -c -E "$padrao" "$f") casos de ${fams//-/, } descartados"
    f="$D/candidatos$s-sem.jsonl"
  fi
  CAND+=("$f"); CONF+=("$D/respostas_conferente$s.jsonl")
done
ls -la "${CAND[@]}" "${CONF[@]}" || exit 1
lumi_exec lumi-dados prof-montar --candidatos "${CAND[@]}" --conferencia "${CONF[@]}" --saida "$D/prof_v1.jsonl" || exit 1
echo ">> montagem concluída ($*)"
