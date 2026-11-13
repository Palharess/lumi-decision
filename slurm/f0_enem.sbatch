#!/usr/bin/env bash
#SBATCH --job-name=lumi-f0-enem
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# F0b (analises/plano_fase2.md): piloto do ENEM (VestBench-Jev v1.3), metade "explorar" (a "confirmar" fica
# intocada). Lumi congelado (A0), E3 (A1), E4 pela cabeça (B) e E4 pelas letras; duas ordens; sem geração.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
lumi_gpu_smoke || exit 1
E3=runs/e3_lora_r16/adapter; E4=runs/e4_ponteiro_s0/adapter
BASE="--backend hf --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d --dtype bfloat16 --native --no-generate --orders canonical,reversed --source vestbench_enem --split explorar"
lumi_exec lumi-eval run $BASE --out runs/f0_enem_A0 || exit 1
lumi_exec lumi-eval run $BASE --adapter "$E3" --out runs/f0_enem_A1 || exit 1
lumi_exec lumi-eval run $BASE --adapter "$E4" --out runs/f0_enem_B || exit 1
lumi_exec lumi-eval run $BASE --adapter "$E4" --sem-ponteiro --out runs/f0_enem_E4letras || exit 1
echo ">> F0b concluído"
