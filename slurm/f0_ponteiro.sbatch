#!/usr/bin/env bash
#SBATCH --job-name=lumi-f0-ptr
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# F0a (analises/plano_fase2.md): por que a pointer head do E4 errou. Só avaliação.
#  1) factck_veracidade nas 6 permutações: A0, E3, E4 pelas letras e E4 pela cabeça, com cruzamento q x k;
#  2) juristcu_area e invoice_processing em 5 ordens (canônica, invertida, rotações 1-3), mesmas 4 leituras.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
lumi_gpu_smoke || exit 1
OUT="runs/f0_ponteiro_${SLURM_JOB_ID}"
E3=runs/e3_lora_r16/adapter; E4=runs/e4_ponteiro_s0/adapter
lumi_exec python -m lumi_decision.diag_ponteiro --e3 "$E3" --e4 "$E4" --out "$OUT" || exit 1
ORD=canonical,reversed,rot1,rot2,rot3
BASE="--backend hf --model ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d --dtype bfloat16 --native --no-generate --orders $ORD --split dev --held-out only"
for spec in "felhen:juristcu_area" "telepatia:invoice_processing"; do
  src=${spec%%:*}; fam=${spec#*:}
  lumi_exec lumi-eval run $BASE --source "$src" --families "$fam" --out "${OUT}/A0_${fam}" || exit 1
  lumi_exec lumi-eval run $BASE --source "$src" --families "$fam" --adapter "$E3" --out "${OUT}/A1_${fam}" || exit 1
  lumi_exec lumi-eval run $BASE --source "$src" --families "$fam" --adapter "$E4" --sem-ponteiro --out "${OUT}/E4letras_${fam}" || exit 1
  lumi_exec lumi-eval run $BASE --source "$src" --families "$fam" --adapter "$E4" --out "${OUT}/B_${fam}" || exit 1
done
echo ">> F0a concluído em $OUT"
