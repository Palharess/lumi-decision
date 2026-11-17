#!/usr/bin/env bash
#SBATCH --job-name=lumi-prof-prep
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Reservadas com professor (analises/reservadas_v2.md): baixa a imagem do vLLM e os dois modelos, com a revisão
# fixada em slurm/prof_config.sh. Sem GPU. Tudo em /raid/$USER.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
source slurm/prof_config.sh
if [ ! -f "$PROF_VLLM_IMAGE" ]; then
  apptainer pull "$PROF_VLLM_IMAGE" "docker://$PROF_VLLM_DOCKER" || { echo ">> pull do vLLM falhou"; exit 1; }
fi
for spec in "$PROF_ESCRITOR@$PROF_ESCRITOR_REV" "$PROF_CONFERENTE@$PROF_CONFERENTE_REV"; do
  lumi_exec python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('${spec%@*}', revision='${spec#*@}', allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model', 'tokenizer*'])
print('>> baixado:', p)" || { echo ">> download de ${spec%@*} falhou"; exit 1; }
done
apptainer exec "$PROF_VLLM_IMAGE" python3 -c "import vllm, torch; print('>> vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
echo ">> preparação concluída"
