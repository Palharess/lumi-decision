#!/usr/bin/env bash
#SBATCH --job-name=lumi-db-inventario
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# E5 (analises/e5_dados.md): baixa o DecisionBank (625k, revisão fixa, ~1,6 GB de JSONL) para o cache do /raid e
# grava só contagens em analises/decisionbank_inventario.json. Sem GPU.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_DOWNLOAD_TIMEOUT=60
for t in 1 2 3; do
  lumi_exec python scripts/decisionbank_inventario.py analises/decisionbank_inventario.json && exit 0
  echo ">> tentativa $t falhou; de novo"
done
exit 1
