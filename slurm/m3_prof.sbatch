#!/usr/bin/env bash
#SBATCH --job-name=lumi-m3-prof
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# M3 (analises/reservadas_v2.md, protocolo de 16/11/2026): checkpoints no dev das reservadas com professor, todas as
# rotações, com a leitura registrada de cada um. Só depois do registro (sha256 no splits.toml [prof]).
#   sbatch slurm/m3_prof.sbatch congelado e3_lora_r16 ...   ("congelado" = sem adaptador)
# Saída: runs/profdev_<nome>_r3 (3 opções: 3 rotações) e runs/profdev_<nome>_r2 (correferência: 2 rotações).
# Com SPLIT=dificuldade (só o congelado): o filtro de dificuldade, em runs/profdif_<nome>_r3 e _r2.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
SPLIT=${SPLIT:-dev}
PRE=$([ "$SPLIT" = dev ] && echo profdev || echo profdif)
lumi_gpu_smoke || exit 1
BASE="--backend hf --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d --dtype bfloat16 --native --no-generate --source prof --split $SPLIT"
for nome in "$@"; do
  AD=""
  if [ "$nome" != congelado ]; then
    [ -d "runs/$nome/adapter" ] || { echo ">> sem adaptador: runs/$nome/adapter"; exit 1; }
    AD="--adapter runs/$nome/adapter"
  fi
  lumi_exec lumi-eval run $BASE $AD --families postura_alvo,inferencia_pt,resposta_indireta \
    --orders canonical,rot1,rot2 --out "runs/${PRE}_${nome}_r3" || { echo ">> $nome r3 falhou"; exit 1; }
  lumi_exec lumi-eval run $BASE $AD --families correferencia \
    --orders canonical,rot1 --out "runs/${PRE}_${nome}_r2" || { echo ">> $nome r2 falhou"; exit 1; }
done
echo ">> M3 ($SPLIT) concluída: $*"
