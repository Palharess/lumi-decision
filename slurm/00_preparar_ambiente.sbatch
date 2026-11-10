#!/usr/bin/env bash
#SBATCH --job-name=lumi-setup
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Prepara tudo em /raid/$USER, via Slurm (regra do manual):
#   1. baixa a imagem Apptainer (se ainda não existir);
#   2. cria o venv sobre o Python do container (--system-site-packages: reaproveita o torch do NGC);
#   3. instala o pacote, transformers/peft e compila mamba-ssm + causal-conv1d se faltarem;
#   4. baixa datasets, artefatos e checkpoints fixados (Lumi + Nemotron base);
#   5. checagens: CUDA, kernels, testes.
# Uso (da raiz do repo em /raid/$USER/lumi-decision):  mkdir -p logs && sbatch slurm/00_preparar_ambiente.sbatch
# Precisa de internet no nó. Se não houver, rode as partes 1 e 4 no nó de login e as demais aqui.
set -euo pipefail
source slurm/config.sh

if [ ! -f "$LUMI_IMAGE" ]; then
  echo ">> baixando imagem $LUMI_NGC_IMAGE"
  apptainer pull "$LUMI_IMAGE" "docker://$LUMI_NGC_IMAGE"
fi

apptainer exec --nv -B "$LUMI_RAID:$LUMI_RAID" "$LUMI_IMAGE" bash -euo pipefail -c "
  cd '$LUMI_REPO'
  [ -d '$LUMI_VENV' ] || python -m venv --system-site-packages '$LUMI_VENV'
  source '$LUMI_VENV/bin/activate'
  python -m pip install -U pip
  # Não reinstalar o torch do container.
  python -m pip install -e '.[baselines,dev]' 'transformers>=4.56' 'accelerate>=0.33' 'peft>=0.13' 'huggingface_hub[cli]'
  if ! python -c 'import mamba_ssm, causal_conv1d' 2>/dev/null; then
    echo '>> compilando causal-conv1d e mamba-ssm (pode levar dezenas de minutos)'
    export MAX_JOBS=\${SLURM_CPUS_PER_TASK:-8}
    python -m pip install --no-build-isolation causal-conv1d mamba-ssm
  fi
  lumi-dados baixar
  hf download ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d
  hf download nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
  python - <<'PY'
import torch, transformers, mamba_ssm, causal_conv1d
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0))
print('transformers', transformers.__version__, 'mamba_ssm', mamba_ssm.__version__)
PY
  python -m pytest -q
"
echo ">> ambiente pronto: $LUMI_VENV  imagem: $LUMI_IMAGE"
