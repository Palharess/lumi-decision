#!/usr/bin/env bash
#SBATCH --job-name=lumi-m1-factck6
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Medida M1 (analises/plano_fase2.md): factck_veracidade (dev, reservada) nas 6 permutações das opções, com a
# leitura registrada de cada checkpoint (letras, pointer head ou residual: o backend escolhe pelo adaptador).
#   sbatch slurm/m1_factck6.sbatch congelado e3_lora_r16 e4_ponteiro_s0 ...
#   "congelado" = Lumi sem adaptador; qualquer outro nome = runs/<nome>/adapter. Saída: runs/factck6_<nome>.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
lumi_gpu_smoke || exit 1
ORD=perm012,perm021,perm102,perm120,perm201,perm210
BASE="--backend hf --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d --dtype bfloat16 --native --no-generate --orders $ORD --source felhen --families factck_veracidade --held-out only --split dev"
for nome in "$@"; do
  AD=""
  if [ "$nome" != congelado ]; then
    [ -d "runs/$nome/adapter" ] || { echo ">> sem adaptador: runs/$nome/adapter"; exit 1; }
    AD="--adapter runs/$nome/adapter"
  fi
  lumi_exec lumi-eval run $BASE $AD --out "runs/factck6_$nome" || { echo ">> $nome falhou"; exit 1; }
done
echo ">> M1 concluída: $*"
