#!/usr/bin/env bash
#SBATCH --job-name=lumi-e5-prep
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# E5, braço DecisionBank (analises/e5_dados.md): baixa a revisão fixada, audita o filtro da `j11_generator` e a
# sobreposição com todas as avaliações, e conta o orçamento de cada braço ANTES do registro e do treino (sem GPU).
#   sbatch slurm/e5_preparar.sbatch
# Saída: analises/decisionbank_auditoria.json e analises/e5_orcamento.jsonl.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
lumi_exec lumi-dados baixar decisionbank || exit 1
export HF_HUB_OFFLINE=1
lumi_exec lumi-dados auditar-decisionbank > analises/decisionbank_auditoria.json || exit 1
cat analises/decisionbank_auditoria.json
NOVO=sources=felhen+telepatia+traces+eikos+decisionbank,epoch_offset=1
CTRL=sources=felhen+telepatia+traces+eikos,epoch_offset=1,epochs=2
lumi_exec python scripts/contar_orcamento.py "seed=0,$NOVO" "seed=1,data_seed=0,$NOVO" \
  "seed=0,$CTRL" "seed=1,data_seed=0,$CTRL" | grep '^{' | tee analises/e5_orcamento.jsonl || exit 1
echo ">> preparação do E5 concluída"
