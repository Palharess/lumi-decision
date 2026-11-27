#!/usr/bin/env bash
#SBATCH --job-name=lumi-bench-diag
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
# Diagnóstico da queda do cot do E3b no Lumi-Benchmark (analises/bench_diag.md). Mesma configuração do
# slurm/lumi_benchmark.sbatch (vLLM, bf16, temperatura 0, seed 42, 2048 tokens), em checkpoints que já existem,
# para separar as causas sem treinar nada.
#   sbatch slurm/benchmark_diag.sbatch SPLIT "VARIANTES" NOME...   (NOME = base ou uma pasta runs/<NOME>/adapter)
#   THINKING=1: template com o raciocínio ligado (padrão do Lumi) e 4096 tokens.
# Cada adaptador é fundido em $LUMI_RAID/modelos/diag/<NOME> (apagado no fim) com entrega/benchmark/fundir.py, que
# lê os alvos congelados (F2.2) do treino_meta.json ao lado do adaptador.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
source slurm/prof_config.sh
export HF_HUB_OFFLINE=1
SPLIT=${1:?SPLIT}; VARIANTES=${2:?VARIANTES}; shift 2
BENCH="$LUMI_RAID/Lumi-Benchmark"
REV=27b30f40b930fd05cee5e1c1f8790f70fa5b816d
SNAP="$HF_HOME/hub/models--ceia-br--lumi-4b/snapshots/$REV"
PORTA=$((18000 + ${SLURM_JOB_ID:-0} % 1000))
THINKING=${THINKING:-0}
if [ "$THINKING" = 1 ]; then
  TPL="$SNAP/chat_template.jinja"; SUF=com-thinking; MAXTOK=4096
else
  TPL="$LUMI_RAID/tmp/lumi_template_sem_thinking.jinja"; SUF=sem-thinking; MAXTOK=2048
  sed 's/enable_thinking = enable_thinking if enable_thinking is defined else True/enable_thinking = enable_thinking if enable_thinking is defined else False/' \
    "$SNAP/chat_template.jinja" > "$TPL"
  grep -q "is defined else False %}" "$TPL" || { echo ">> template não mudou"; exit 1; }
fi
mkdir -p "$LUMI_RAID/modelos/diag"

servir() {  # $1 = caminho do modelo, $2 = nome servido
  apptainer exec --nv -B "$LUMI_RAID:$LUMI_RAID" "$PROF_VLLM_IMAGE" \
    vllm serve "$1" --served-model-name "$2" --dtype bfloat16 --max-model-len 16384 --max-logprobs 20 \
    --chat-template "$TPL" --trust-remote-code --port "$PORTA" --gpu-memory-utilization 0.85 &
  VLLM_PID=$!
  for _ in $(seq 1 180); do
    curl -sf "http://localhost:$PORTA/v1/models" >/dev/null && return 0
    kill -0 "$VLLM_PID" 2>/dev/null || { echo ">> vLLM morreu"; return 1; }
    sleep 5
  done
  echo ">> vLLM não subiu"; return 1
}

RODADAS=(); FALHAS=()   # revisão de 27/11: qualquer rodada prevista que não sai faz o job terminar com erro
for nome in "$@"; do
  if [ "$nome" = base ]; then
    caminho="$SNAP"; servido="lumi-4b"
  else
    [ -f "runs/$nome/adapter/adapter_model.safetensors" ] || { echo ">> sem adaptador: runs/$nome"; FALHAS+=("$nome: sem adaptador"); continue; }
    caminho="$LUMI_RAID/modelos/diag/$nome"; servido="diag-$nome"
    if [ ! -f "$caminho/FUSAO.json" ]; then
      lumi_exec python entrega/benchmark/fundir.py "$SNAP" "runs/$nome/adapter" "$caminho" \
        || { echo ">> fusão de $nome falhou"; FALHAS+=("$nome: fusão"); rm -rf "$caminho"; continue; }
    fi
  fi
  servir "$caminho" "$servido" || { FALHAS+=("$nome: vLLM"); kill "$VLLM_PID" 2>/dev/null; continue; }
  for v in $VARIANTES; do
    run="diag-${nome}-${SUF}-${v}-${SPLIT}"
    cat > "$BENCH/configs/${run}.json" <<EOF
{
  "run_name": "$run",
  "backend": {"type": "openai_compat", "base_url": "http://localhost:$PORTA/v1", "model": "$servido", "want_probs": true, "timeout": 1800},
  "adapter": "default",
  "split": "$SPLIT",
  "prompt_variant": "$v",
  "max_new_tokens": $MAXTOK,
  "temperature": 0.0,
  "seed": 42,
  "quantization": "bf16 (vLLM 0.33, H100); $nome; diagnóstico, template $SUF",
  "output_dir": "runs",
  "data": {
    "conhecimento": [
      "data/benchmark/oab_exams.jsonl",
      "data/benchmark/drbodebench_inep.jsonl",
      "data/benchmark/bluex.jsonl",
      "data/benchmark/math_pt.jsonl"
    ],
    "decisao": "data/benchmark/bfcl_pt.jsonl"
  }
}
EOF
    apptainer exec --nv -B "$LUMI_RAID:$LUMI_RAID" "$PROF_VLLM_IMAGE" bash -c \
      'cd "$1" && python3 -m lumi_bench.run --config "configs/$2.json"' _ "$BENCH" "$run" \
      || { echo ">> $run falhou"; FALHAS+=("$run: avaliação"); }
    RODADAS+=("$run")
  done
  kill "$VLLM_PID"; wait "$VLLM_PID" 2>/dev/null
  [ "$nome" = base ] || rm -rf "$caminho"
done
mkdir -p analises/bench_diag
lumi_exec python3 scripts/analise_bench_diag.py "$BENCH/runs" "${RODADAS[@]}" \
  --saida "analises/bench_diag/${SUF}_${SPLIT}_${SLURM_JOB_ID:-local}.json" \
  || FALHAS+=("análise: rodada sem resultado completo")
if [ ${#FALHAS[@]} -gt 0 ]; then
  printf '>> FALHOU: %s\n' "${FALHAS[@]}"
  exit 1
fi
echo ">> diagnóstico concluído: $*"
