# Configuração do cluster DGX (manual "Orientações cluster DGX / Slurm").
# Regras do manual que este arquivo aplica:
#   - dados, modelos, caches, venvs, imagens e checkpoints SOMENTE em /raid/$USER (nunca na home);
#   - tudo que usa GPU roda via Slurm (sbatch/srun), nunca direto no nó;
#   - partição da máquina onde os dados estão: temos acesso ao H100-02 -> partição h100n2.
# Todos os .sbatch fazem `source slurm/config.sh` a partir da raiz do repo.

export LUMI_PARTITION="${LUMI_PARTITION:-h100n2}"
export LUMI_RAID="${LUMI_RAID:-/raid/$USER}"
export LUMI_REPO="${LUMI_REPO:-$LUMI_RAID/lumi-decision}"
export LUMI_VENV="${LUMI_VENV:-$LUMI_RAID/venvs/lumi-decision}"
export LUMI_CKPT="${LUMI_CKPT:-$LUMI_RAID/checkpoints/lumi-decision}"

# Imagem Apptainer (.sif) em /raid/$USER/containers. Padrão: NGC PyTorch (torch + CUDA + nvcc para
# compilar mamba-ssm). Tag conferida no NGC em 10/11/2026 (mais recente: 26.10).
export LUMI_NGC_IMAGE="${LUMI_NGC_IMAGE:-nvcr.io/nvidia/pytorch:26.09-py3}"
export LUMI_IMAGE="${LUMI_IMAGE:-$LUMI_RAID/containers/$(echo "${LUMI_NGC_IMAGE#nvcr.io/}" | tr '/:' '__').sif}"

# Caches fora da home.
export HF_HOME="$LUMI_RAID/cache/huggingface"
export PIP_CACHE_DIR="$LUMI_RAID/cache/pip"
export TORCH_HOME="$LUMI_RAID/cache/torch"
export TRITON_CACHE_DIR="$LUMI_RAID/cache/triton"
export XDG_CACHE_HOME="$LUMI_RAID/cache/xdg"
export APPTAINER_CACHEDIR="$LUMI_RAID/cache/apptainer"
export APPTAINER_TMPDIR="$LUMI_RAID/tmp"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" \
         "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$LUMI_CKPT" "$(dirname "$LUMI_VENV")" \
         "$LUMI_RAID/containers" 2>/dev/null || true

# Executa um comando dentro do container, com o venv ativado e na raiz do repo.
# O ambiente do host (HF_HOME etc.) é herdado pelo Apptainer.
lumi_exec() {
  apptainer exec --nv -B "$LUMI_RAID:$LUMI_RAID" "$LUMI_IMAGE" \
    bash -c "source '$LUMI_VENV/bin/activate' && cd '$LUMI_REPO' && $*"
}
