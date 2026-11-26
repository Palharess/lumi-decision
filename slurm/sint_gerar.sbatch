#!/usr/bin/env bash
#SBATCH --job-name=lumi-sint-gerar
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# E5 (analises/e5_dados.md, piloto de 25/11/2026): sintético em PT-BR para TREINO, com os papéis das reservadas com
# professor trocados. O Gemma-4-31B escreve e o Qwen3.8-27B (sem raciocínio) confere às cegas, em 2 ordens.
#   sbatch slurm/sint_gerar.sbatch INICIO FIM SUFIXO [FAMILIA...]
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
source slurm/prof_config.sh
export HF_HUB_OFFLINE=1
INI=${1:?INICIO}; FIM=${2:?FIM}; SUF=${3:?SUFIXO}; FAMS=("${@:4}")
D=data/sint
mkdir -p "$D"
lumi_exec lumi-dados sint-pedidos --inicio "$INI" --fim "$FIM" --saida "$D/pedidos_escritor$SUF.jsonl" \
  ${FAMS[@]:+--familias "${FAMS[@]}"} || exit 1
SO_TEXTO='{"limit_mm_per_prompt": {"image": 0, "video": 0}}'
prof_vllm_exec python3 scripts/prof_vllm.py --modelo "$PROF_CONFERENTE" --revisao "$PROF_CONFERENTE_REV" \
  --entrada "$D/pedidos_escritor$SUF.jsonl" --saida "$D/respostas_escritor$SUF.jsonl" --max-model-len 4096 \
  --gpu-util 0.92 --llm-kwargs "$SO_TEXTO" || { echo ">> escritor falhou"; exit 1; }
lumi_exec lumi-dados sint-candidatos --respostas "$D/respostas_escritor$SUF.jsonl" \
  --saida "$D/candidatos$SUF.jsonl" --pedidos-conferente "$D/pedidos_conferente$SUF.jsonl" || exit 1
prof_vllm_exec python3 scripts/prof_vllm.py --modelo "$PROF_ESCRITOR" --revisao "$PROF_ESCRITOR_REV" \
  --entrada "$D/pedidos_conferente$SUF.jsonl" --saida "$D/respostas_conferente$SUF.jsonl" --max-model-len 4096 \
  --gpu-util 0.85 --chat-kwargs '{"enable_thinking": false}' --llm-kwargs "$SO_TEXTO" || { echo ">> conferente falhou"; exit 1; }
lumi_exec lumi-dados sint-montar --candidatos "$D/candidatos$SUF.jsonl" \
  --conferencia "$D/respostas_conferente$SUF.jsonl" --saida "$D/sint_pt$SUF.jsonl" || exit 1
echo ">> geração concluída ($INI–$FIM$SUF)"
