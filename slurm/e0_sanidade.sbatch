#!/usr/bin/env bash
#SBATCH --job-name=lumi-e0
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# E0: checkpoint oficial BF16 nos 40 casos do piloto, com remote code e com a implementação nativa.
# O job FALHA (código != 0) se nenhuma das duas implementações passar nas verificações obrigatórias.
# Uso:  sbatch slurm/e0_sanidade.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"   # funciona submetendo de qualquer pasta
source slurm/config.sh
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
lumi_gpu_smoke   # falha cedo (set -e) se o Triton/mamba-ssm não rodar na GPU

aprovadas=()
for impl in remote native; do
  flag=""; [ "$impl" = native ] && flag="--native"
  out="runs/e0_${impl}_${SLURM_JOB_ID}"
  if lumi_exec lumi-e0 run $flag --source referencias/piloto40/casos.json --ref-run referencias/piloto40 --out "$out"; then
    aprovadas+=("$impl")
  else
    echo ">> E0 ($impl) REPROVADO ou com erro"
  fi
done

# Comparação direta entre as duas implementações BF16 (diagnóstico).
if [ -f "runs/e0_remote_${SLURM_JOB_ID}/predicoes.jsonl" ] && [ -f "runs/e0_native_${SLURM_JOB_ID}/predicoes.jsonl" ]; then
  echo ">> remote x native"
  lumi_exec lumi-e0 comparar "runs/e0_remote_${SLURM_JOB_ID}" "runs/e0_native_${SLURM_JOB_ID}" || true
fi

echo ">> implementações aprovadas: ${aprovadas[*]:-nenhuma}"
[ ${#aprovadas[@]} -gt 0 ]
