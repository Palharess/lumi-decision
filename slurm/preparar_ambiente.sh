#!/usr/bin/env bash
# Prepara o ambiente no cluster. NÃO TESTADO AINDA: ajuste versões ao CUDA/driver do cluster.
#
# Parte 1 (nó de login, precisa de internet):  bash slurm/preparar_ambiente.sh baixar
# Parte 2 (nó com GPU, compila kernels):        srun -p $LUMI_PARTITION --gres=gpu:1 --time=01:00:00 \
#                                                   bash slurm/preparar_ambiente.sh kernels
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$LUMI_REPO"

case "${1:-}" in
  baixar)
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv --python 3.12 "$LUMI_VENV"
    source "$LUMI_VENV/bin/activate"
    # torch com CUDA: troque cu128 pela versão compatível com o driver do cluster.
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install -e ".[hf,baselines,dev]"
    lumi-dados baixar
    # Checkpoint oficial fixado + Nemotron base (controle).
    hf download ceia-br/lumi-4b --revision 27b30f40b930fd05cee5e1c1f8790f70fa5b816d
    hf download nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
    ;;
  kernels)
    lumi_modules
    source "$LUMI_VENV/bin/activate"
    uv pip install packaging ninja
    # Compilados contra o torch instalado; podem levar dezenas de minutos.
    uv pip install --no-build-isolation causal-conv1d mamba-ssm
    python -c "import mamba_ssm, causal_conv1d, torch; print('ok', torch.__version__, torch.version.cuda)"
    ;;
  *)
    echo "uso: $0 {baixar|kernels}"; exit 1 ;;
esac
