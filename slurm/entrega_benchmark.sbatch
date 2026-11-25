#!/usr/bin/env bash
#SBATCH --job-name=lumi-entrega-bench
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Monta o pacote do E3b semente 1 (Lumi-4B + adaptador LoRA) para o time de benchmark e confere que ele reproduz a
# avaliação do dev (entrega/benchmark/verificar.py).
#   sbatch slurm/entrega_benchmark.sbatch
# Saída: $LUMI_RAID/entrega_benchmark/lumi-decision-e3b-s1/{base,adapter,exemplo.py,README.md,treino.json,...}
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
NOME=e3b_eikos_s1
REV=27b30f40b930fd05cee5e1c1f8790f70fa5b816d
SNAP="$HF_HOME/hub/models--ceia-br--lumi-4b/snapshots/$REV"
DEST="$LUMI_RAID/entrega_benchmark/lumi-decision-e3b-s1"

rm -rf "$DEST"; mkdir -p "$DEST/base" "$DEST/adapter"
cp -L "$SNAP"/* "$DEST/base/"
cp "runs/$NOME/adapter/adapter_config.json" "runs/$NOME/adapter/adapter_model.safetensors" "$DEST/adapter/"
cp "runs/$NOME/treino_meta.json" "$DEST/treino.json"
cp entrega/benchmark/exemplo.py entrega/benchmark/README.md "$DEST/"
lumi_exec python entrega/benchmark/versoes.py > "$DEST/ambiente.txt"
cat "$DEST/ambiente.txt"

lumi_gpu_smoke
lumi_exec python entrega/benchmark/verificar.py "$DEST" data/prof/prof_v1.jsonl "runs/profdev_${NOME}_r3/predicoes.jsonl" 10
lumi_exec python entrega/benchmark/verificar.py "$DEST" data/prof/prof_v1.jsonl "runs/profdev_${NOME}_r2/predicoes.jsonl" 10
lumi_exec python "$DEST/exemplo.py"
rm -rf "$DEST/__pycache__"

(cd "$DEST" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
chmod -R a+rX "$DEST"
du -sh "$DEST"
echo ">> pacote pronto: $DEST"
