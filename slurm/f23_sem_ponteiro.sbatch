#!/usr/bin/env bash
#SBATCH --job-name=lumi-f23-sp
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Diagnóstico do F2.3a (analises/plano_fase2.md, protocolo F2.2): o adaptador residual lido só pelas letras
# (--sem-ponteiro), no dev do Felhen, nas mesmas duas ordens. Separa o efeito da cabeça do efeito do LoRA que
# treinou junto com ela.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
lumi_gpu_smoke || exit 1
BASE="--backend hf --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d --dtype bfloat16 --native --no-generate --orders canonical,reversed --source felhen --split dev"
for s in 0 1; do
  lumi_exec lumi-eval run $BASE --adapter "runs/f23_residual_s$s/adapter" --sem-ponteiro \
    --out "runs/f23_residual_s${s}_semponteiro_felhen_dev" || exit 1
done
echo ">> diagnóstico sem ponteiro concluído"
