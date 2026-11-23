#!/usr/bin/env bash
#SBATCH --job-name=lumi-prof-gerar
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Reservadas com professor (analises/reservadas_v2.md, protocolo de 16/11/2026). Os passos, em ordem:
#   1. especificações e pedidos (código);  2. o escritor escreve;  3. filtro de forma;
#   4. o conferente responde às cegas, em 2 ordens;  5. montagem (650 grupos por família) e auditoria.
#   sbatch slurm/prof_gerar.sbatch [INICIO FIM [SUFIXO [FAMILIA...]]]   (padrão: 0 1000, todas as famílias;
#   SUFIXO separa rodadas e testes)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
source slurm/prof_config.sh
export HF_HUB_OFFLINE=1
INI=${1:-0}; FIM=${2:-1000}; SUF=${3:-}; FAMS=("${@:4}")
D=data/prof
mkdir -p "$D"
lumi_exec lumi-dados prof-pedidos --inicio "$INI" --fim "$FIM" --saida "$D/pedidos_escritor$SUF.jsonl" \
  ${FAMS[@]:+--familias "${FAMS[@]}"} || exit 1
# Job 37793: o Qwen3.8-27B estourou a memória na captura dos CUDA graphs (0.90, 512 sequências). Como a fila é
# longa, cada modelo tenta em ordem, cada vez num processo novo: só texto (sem perfilar o codificador de
# imagem); só texto sem CUDA graphs; sem CUDA graphs. A configuração que subiu fica no log e em cada resposta.
vllm_tentativas() {
  for extra in '{"limit_mm_per_prompt": {"image": 0, "video": 0}}' \
               '{"limit_mm_per_prompt": {"image": 0, "video": 0}, "enforce_eager": true}' \
               '{"enforce_eager": true}'; do
    prof_vllm_exec python3 scripts/prof_vllm.py "$@" --llm-kwargs "$extra" && return 0
    echo ">> vLLM falhou com $extra; próxima configuração"
  done
  return 1
}
vllm_tentativas --modelo "$PROF_ESCRITOR" --revisao "$PROF_ESCRITOR_REV" \
  --entrada "$D/pedidos_escritor$SUF.jsonl" --saida "$D/respostas_escritor$SUF.jsonl" \
  --chat-kwargs '{"enable_thinking": true}' --max-model-len 14336 --gpu-util 0.85 || { echo ">> escritor falhou"; exit 1; }
lumi_exec lumi-dados prof-candidatos --respostas "$D/respostas_escritor$SUF.jsonl" \
  --saida "$D/candidatos$SUF.jsonl" --pedidos-conferente "$D/pedidos_conferente$SUF.jsonl" || exit 1
vllm_tentativas --modelo "$PROF_CONFERENTE" --revisao "$PROF_CONFERENTE_REV" \
  --entrada "$D/pedidos_conferente$SUF.jsonl" --saida "$D/respostas_conferente$SUF.jsonl" --max-model-len 4096 \
  --gpu-util 0.92 || { echo ">> conferente falhou"; exit 1; }
lumi_exec lumi-dados prof-montar --candidatos "$D/candidatos$SUF.jsonl" \
  --conferencia "$D/respostas_conferente$SUF.jsonl" --saida "$D/prof_v1$SUF.jsonl" || exit 1
echo ">> geração concluída ($INI–$FIM$SUF)"
