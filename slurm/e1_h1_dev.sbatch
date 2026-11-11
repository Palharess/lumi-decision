#!/usr/bin/env bash
#SBATCH --job-name=lumi-e1
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# E1 (H1): readout congelado + geração curta no dev das três fontes, duas ordens de opções.
# Rodar depois que o E0 passar.
# Uso:  sbatch slurm/e1_h1_dev.sbatch [lumi|nemotron] [N_POR_FAMILIA]
#       N_POR_FAMILIA > 0 faz um E1 curto (amostra determinística); 0 = dev completo.
# Reentrante: predições gravadas linha a linha; ao ressubmeter com a MESMA configuração, os casos
# já feitos são pulados. Configuração diferente na mesma pasta é recusada pelo avaliador.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"   # funciona submetendo de qualquer pasta
source slurm/config.sh
export HF_HUB_OFFLINE=1

case "${1:-lumi}" in
  lumi)     MODEL=ceia-br/lumi-4b;                        REV=27b30f40b930fd05cee5e1c1f8790f70fa5b816d ;;
  nemotron) MODEL=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16;  REV=dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f ;;
  *) echo "modelo desconhecido: $1"; exit 1 ;;
esac
PER_FAMILY="${2:-0}"
IMPL_FLAG="${LUMI_IMPL_FLAG:-}"   # export LUMI_IMPL_FLAG=--native se o E0 indicar
IMPL="$([ -n "$IMPL_FLAG" ] && echo nativa || echo remote)"
SCOPE="$([ "$PER_FAMILY" -gt 0 ] && echo "curto${PER_FAMILY}" || echo completo)"

for src in felhen telepatia traces; do
  lumi_exec lumi-eval run --backend hf --model "$MODEL" --revision "$REV" --dtype bfloat16 $IMPL_FLAG \
    --source "$src" --split dev --orders canonical,reversed --per-family "$PER_FAMILY" \
    --out "runs/e1_${1:-lumi}_${REV:0:7}_${IMPL}_${src}_dev_${SCOPE}"
done
