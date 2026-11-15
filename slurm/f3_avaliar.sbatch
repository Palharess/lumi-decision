#!/usr/bin/env bash
#SBATCH --job-name=lumi-f3-aval
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Piloto F3 (analises/plano_fase2.md): avalia modelos no dev da fonte f3 (pss_mais_simples reservada,
# massive_intencao e goemo_emocao vistas), duas ordens, sem geração.
#   sbatch slurm/f3_avaliar.sbatch congelado f21_balanceado_s0 f3p_diverso_s1 ...
#   "congelado" = Lumi sem adaptador; qualquer outro nome = runs/<nome>/adapter. Saída: runs/f3dev_<nome>.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
lumi_gpu_smoke || exit 1
BASE="--backend hf --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d --dtype bfloat16 --native --no-generate --orders canonical,reversed --source f3 --split dev"
for nome in "$@"; do
  AD=""; [ "$nome" = congelado ] || AD="--adapter runs/$nome/adapter"
  lumi_exec lumi-eval run $BASE $AD --out "runs/f3dev_$nome" || { echo ">> $nome falhou"; exit 1; }
done
echo ">> avaliação f3 concluída: $*"
