# Funções para servir o Lumi no vLLM e rodar o Lumi-Benchmark (usadas por slurm/guarda_corpo.sbatch).
# Mesma configuração do job 38177 (analises/lumi_benchmark.md): vLLM 0.33, bf16, template com o raciocínio
# desligado, temperatura 0, seed 42. Precisa de slurm/config.sh e slurm/prof_config.sh carregados antes.
BENCH="$LUMI_RAID/Lumi-Benchmark"
LUMI_REV=27b30f40b930fd05cee5e1c1f8790f70fa5b816d
SNAP="$HF_HOME/hub/models--ceia-br--lumi-4b/snapshots/$LUMI_REV"
PORTA=$((18000 + ${SLURM_JOB_ID:-0} % 1000))
TPL="$LUMI_RAID/tmp/lumi_template_sem_thinking_${SLURM_JOB_ID:-local}.jinja"

lumi_template_sem_thinking() {
  mkdir -p "$(dirname "$TPL")"
  sed 's/enable_thinking = enable_thinking if enable_thinking is defined else True/enable_thinking = enable_thinking if enable_thinking is defined else False/' \
    "$SNAP/chat_template.jinja" > "$TPL"
  grep -q "is defined else False %}" "$TPL" || { echo ">> template não mudou"; return 1; }
}

lumi_servir() {  # $1 = caminho do modelo, $2 = nome servido; deixa VLLM_PID
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

lumi_parar() {
  kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null
}

lumi_fundir() {  # $1 = pasta do adaptador, $2 = destino do modelo fundido (refaz se não houver FUSAO.json)
  [ -f "$2/FUSAO.json" ] && return 0
  lumi_exec python entrega/benchmark/fundir.py "$SNAP" "$1" "$2" || { rm -rf "$2"; return 1; }
}

lumi_rodar() {  # $1 = rodada, $2 = nome servido, $3 = variante, $4 = split, $5 = JSON de "data"; pula se já existir
  [ -f "$BENCH/runs/$1/pontuacao_conhecimento.jsonl" ] && { echo ">> $1 já existe"; return 0; }
  cat > "$BENCH/configs/$1.json" <<EOF
{
  "run_name": "$1",
  "backend": {"type": "openai_compat", "base_url": "http://localhost:$PORTA/v1", "model": "$2", "want_probs": true, "timeout": 1800},
  "adapter": "default",
  "split": "$4",
  "prompt_variant": "$3",
  "max_new_tokens": 2048,
  "temperature": 0.0,
  "seed": 42,
  "quantization": "bf16 (vLLM 0.33, H100); $2; guarda-corpo do E6, raciocínio desligado",
  "output_dir": "runs",
  "data": $5
}
EOF
  apptainer exec --nv -B "$LUMI_RAID:$LUMI_RAID" "$PROF_VLLM_IMAGE" bash -c \
    'cd "$1" && python3 -m lumi_bench.run --config "configs/$2.json"' _ "$BENCH" "$1"
}
