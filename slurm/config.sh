# Configuração do cluster DGX (manual "Orientações cluster DGX / Slurm").
# Regras do manual que este arquivo aplica:
#   - dados, modelos, caches, venvs, imagens e checkpoints SOMENTE em /raid/$USER (nunca na home);
#   - tudo que usa GPU roda via Slurm (sbatch/srun), nunca direto no nó;
#   - partição da máquina onde os dados estão: temos acesso ao H100-02 -> partição h100n2.
# Todos os .sbatch fazem `source slurm/config.sh` a partir da raiz do repo.

export LUMI_PARTITION="${LUMI_PARTITION:-h100n2}"
export LUMI_RAID="${LUMI_RAID:-/raid/$USER}"
export LUMI_REPO="${LUMI_REPO:-$LUMI_RAID/lumi-decision}"
export LUMI_CKPT="${LUMI_CKPT:-$LUMI_RAID/checkpoints/lumi-decision}"

# Imagem Apptainer (.sif) em /raid/$USER/containers: NGC PyTorch 26.04 (CUDA 13; driver do nó: 580/CUDA 13.0).
# Por que 26.04: mamba-ssm 2.3.2.post1 e causal-conv1d 1.6.2.post1 publicam wheels PRONTAS para o torch
# dessa imagem (`torch26.04`). Com a 26.09 (torch 2.14, headers C++20) a compilação do mamba-ssm falhou
# (job 36783). Trocar a imagem exige trocar as versões abaixo e cria outro venv.
export LUMI_NGC_IMAGE="${LUMI_NGC_IMAGE:-nvcr.io/nvidia/pytorch:26.04-py3}"
export LUMI_MAMBA_PKGS="${LUMI_MAMBA_PKGS:-causal-conv1d==1.6.2.post1 mamba-ssm==2.3.2.post1}"
export LUMI_IMAGE="${LUMI_IMAGE:-$LUMI_RAID/containers/$(echo "${LUMI_NGC_IMAGE#nvcr.io/}" | tr '/:' '__').sif}"
# Um venv por imagem (o venv usa o Python e o torch do container).
export LUMI_VENV="${LUMI_VENV:-$LUMI_RAID/venvs/lumi-decision-$(echo "${LUMI_NGC_IMAGE##*:}")}"

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

# Executa UM comando dentro do container, com o venv ativado e na raiz do repo. O comando substitui o
# bash (exec), então sinais repassados ao apptainer (ex.: SIGUSR1 do Slurm) chegam ao Python.
# O ambiente do host (HF_HOME etc.) é herdado pelo Apptainer.
# TRITON_LIBCUDA_PATH: o Triton procura a libcuda pelo cache do ldconfig da imagem, que aponta para
# /usr/local/cuda/compat/lib (sem a libcuda fora do entrypoint do NGC); com --nv a libcuda do driver fica em
# /.singularity.d/libs. Sem isso os kernels Triton do mamba-ssm falham com "libcuda.so cannot found" (job 36786).
_lumi_inner() {
  echo "for d in /.singularity.d/libs /usr/lib/x86_64-linux-gnu /usr/lib64; do
          if [ -e \"\$d/libcuda.so.1\" ]; then export TRITON_LIBCUDA_PATH=\"\$d\"; break; fi
        done
        source '$LUMI_VENV/bin/activate' && cd '$LUMI_REPO' && exec $*"
}
lumi_exec() {
  apptainer exec --nv -B "$LUMI_RAID:$LUMI_RAID" "$LUMI_IMAGE" bash -c "$(_lumi_inner "$@")"
}
# Igual, em segundo plano: o processo de fundo é o PRÓPRIO apptainer (PID em LUMI_BG_PID), que repassa
# sinais ao comando no container. Usado pelo treino para receber o SIGUSR1 do Slurm.
lumi_exec_bg() {
  apptainer exec --nv -B "$LUMI_RAID:$LUMI_RAID" "$LUMI_IMAGE" bash -c "$(_lumi_inner "$@")" &
  LUMI_BG_PID=$!
}

# Teste rápido na GPU: roda um kernel Triton do mamba-ssm de verdade (o setup só importava os módulos).
lumi_gpu_smoke() {
  lumi_exec python slurm/gpu_smoke.py
}
