#!/usr/bin/env bash
#SBATCH --job-name=lumi-benchmark
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# Lumi-Benchmark (github.com/henriqueelj/Lumi-Benchmark, clonado em $LUMI_RAID/Lumi-Benchmark) no Lumi-4B
# congelado e no E3b semente 1 (LoRA fundido), servidos pelo vLLM, nas variantes direct e cot, em dev e test.
#   sbatch slurm/lumi_benchmark.sbatch [SPLITS] [MODELOS]      (padrão: "dev test" "base e3b")
# Configuração fixada antes de rodar, igual para os dois modelos:
#   - chat template do Lumi com enable_thinking=False por padrão: o adaptador foi treinado sem raciocínio, e o
#     runner do benchmark não passa argumentos ao template;
#   - temperature 0, seed 42, max_new_tokens 2048, adaptador "default" (como no guia do benchmark).
# Saída: $LUMI_RAID/Lumi-Benchmark/runs/<run_name>/
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
source slurm/prof_config.sh
export HF_HUB_OFFLINE=1
SPLITS=${1:-"dev test"}
MODELOS=${2:-"base e3b"}
BENCH="$LUMI_RAID/Lumi-Benchmark"
REV=27b30f40b930fd05cee5e1c1f8790f70fa5b816d
SNAP="$HF_HOME/hub/models--ceia-br--lumi-4b/snapshots/$REV"
FUNDIDO="$LUMI_RAID/modelos/lumi-decision-e3b-s1-fundido"
PORTA=$((18000 + ${SLURM_JOB_ID:-0} % 1000))
TPL="$LUMI_RAID/tmp/lumi_template_sem_thinking.jinja"

[ -d "$BENCH" ] || { echo ">> falta $BENCH"; exit 1; }
sed 's/enable_thinking = enable_thinking if enable_thinking is defined else True/enable_thinking = enable_thinking if enable_thinking is defined else False/' \
  "$SNAP/chat_template.jinja" > "$TPL"
grep -q "is defined else False %}" "$TPL" || { echo ">> template não mudou"; exit 1; }
sha256sum "$SNAP/chat_template.jinja" "$TPL"

if [[ " $MODELOS " == *" e3b "* && ! -f "$FUNDIDO/FUSAO.json" ]]; then
  lumi_exec python entrega/benchmark/fundir.py "$SNAP" runs/e3b_eikos_s1/adapter "$FUNDIDO" || exit 1
fi

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

for m in $MODELOS; do
  case $m in
    base) caminho="$SNAP"; nome="lumi-4b"; desc="ceia-research/lumi-4b@$REV" ;;
    e3b)  caminho="$FUNDIDO"; nome="lumi-decision-e3b-s1"; desc="lumi-decision/lumi-decision-e3b-s1 (LoRA fundido no lumi-4b@$REV)" ;;
    *) echo ">> modelo desconhecido: $m"; exit 1 ;;
  esac
  servir "$caminho" "$nome" || exit 1
  curl -s "http://localhost:$PORTA/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$nome\", \"messages\": [{\"role\": \"user\", \"content\": \"Quanto é 2+2? Responda no formato 'Resposta: X'.\"}], \"max_tokens\": 64, \"temperature\": 0}"
  echo
  for s in $SPLITS; do
    for v in direct cot; do
      run="${nome}-sem-thinking-${v}-${s}"
      cat > "$BENCH/configs/${run}.json" <<EOF
{
  "run_name": "$run",
  "backend": {"type": "openai_compat", "base_url": "http://localhost:$PORTA/v1", "model": "$nome", "want_probs": true, "timeout": 1800},
  "adapter": "default",
  "split": "$s",
  "prompt_variant": "$v",
  "max_new_tokens": 2048,
  "temperature": 0.0,
  "seed": 42,
  "quantization": "bf16 (vLLM 0.33, H100); $desc; chat template com enable_thinking=False",
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
        'cd "$1" && python3 -m lumi_bench.run --config "configs/$2.json"' _ "$BENCH" "$run" || echo ">> $run falhou"
      grep -m 12 -i "acur\|valid" "$BENCH/runs/$run/relatorio.md" 2>/dev/null
    done
  done
  kill "$VLLM_PID"; wait "$VLLM_PID" 2>/dev/null
done
echo ">> Lumi-Benchmark concluído: $MODELOS / $SPLITS"
