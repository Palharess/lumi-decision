# Configuração do cluster — PREENCHER antes do primeiro job (ainda não verificado).
# Todos os .sbatch fazem `source slurm/config.sh`.

export LUMI_PARTITION="${LUMI_PARTITION:-TODO_particao_h100}"
export LUMI_ACCOUNT="${LUMI_ACCOUNT:-}"            # deixe vazio se o cluster não usa --account
export LUMI_REPO="${LUMI_REPO:-$HOME/lumi-decision}"
export LUMI_VENV="${LUMI_VENV:-$LUMI_REPO/.venv}"
# Cache do Hugging Face em armazenamento compartilhado (os nós de computação podem não ter internet:
# baixe tudo no nó de login com slurm/preparar_ambiente.sh).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Módulos do cluster, se houver (ex.: CUDA para compilar mamba-ssm). Ajuste ou deixe vazio.
lumi_modules() {
  : # module load cuda/12.x
}

lumi_activate() {
  lumi_modules
  # shellcheck disable=SC1091
  source "$LUMI_VENV/bin/activate"
  cd "$LUMI_REPO"
}
