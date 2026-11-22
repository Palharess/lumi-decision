#!/usr/bin/env bash
#SBATCH --job-name=lumi-prof-prep
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Reservadas com professor (analises/reservadas_v2.md): baixa a imagem do vLLM e os dois modelos, com a revisão
# fixada em slurm/prof_config.sh. Sem GPU. Tudo em /raid/$USER.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
source slurm/prof_config.sh
export HF_HUB_DOWNLOAD_TIMEOUT=60
if [ ! -f "$PROF_VLLM_IMAGE" ]; then
  apptainer pull "$PROF_VLLM_IMAGE" "docker://$PROF_VLLM_DOCKER" || { echo ">> pull do vLLM falhou"; exit 1; }
fi
for spec in "$PROF_ESCRITOR@$PROF_ESCRITOR_REV" "$PROF_CONFERENTE@$PROF_CONFERENTE_REV"; do
  # Job 37780: um arquivo do Gemma parou em ~2100 s/it e o job estourou o tempo. Cada tentativa tem limite de
  # 2 h; a seguinte retoma os arquivos parciais do cache.
  ok=0
  for t in 1 2 3 4; do
    timeout 2h bash -c "source slurm/config.sh; lumi_exec python scripts/prof_baixar.py '${spec%@*}' '${spec#*@}'" && { ok=1; break; }
    echo ">> tentativa $t de ${spec%@*} não terminou; retomando"
  done
  [ $ok = 1 ] || { echo ">> download de ${spec%@*} falhou"; exit 1; }
done
prof_vllm_exec python3 scripts/prof_versoes.py
echo ">> preparação concluída"
