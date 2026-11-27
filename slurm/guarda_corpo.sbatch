#!/usr/bin/env bash
#SBATCH --job-name=lumi-guarda
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
# E6, guarda-corpo de preservação (analises/e6_protocolo.md): Lumi-Benchmark (A `direct` com a trilha B, A `cot`) e
# ENEM do VestBench em `cot`, com o LoRA fundido no Lumi-4B e servido no vLLM, contra o Lumi-4B.
#   sbatch slurm/guarda_corpo.sbatch ALVO...        ALVO = base | <run> | <run>:<pasta do adaptador>
#   ex.: sbatch slurm/guarda_corpo.sbatch base e3b_eikos_s0 e6_r2_s0:adapter_p50
#   PARTE=confirmar sbatch ...   confirmação final (Lumi-Benchmark test e ENEM `confirmar`), uma vez só.
# Rodadas que já existem são puladas (o Lumi-4B roda uma vez por parte). Veredito pela regra registrada em
# scripts/guarda_corpo.py, salvo em analises/guarda_corpo/. Qualquer rodada que falhe faz o job terminar com erro.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; [ -f slurm/config.sh ] || cd "/raid/$USER/lumi-decision"
source slurm/config.sh
source slurm/prof_config.sh
source slurm/lumi_vllm.sh
export HF_HUB_OFFLINE=1
PARTE=${PARTE:-dev}
case $PARTE in
  dev) SB=dev; SE=explorar ;;
  confirmar) SB=test; SE=confirmar ;;
  *) echo ">> PARTE desconhecida: $PARTE"; exit 1 ;;
esac
[ -d "$BENCH" ] || { echo ">> falta $BENCH"; exit 1; }
lumi_template_sem_thinking || exit 1
ENEM="$PWD/data/vestbench/enem_bench.jsonl"
[ -f "$ENEM" ] || lumi_exec lumi-dados exportar-enem-bench --saida data/vestbench/enem_bench.jsonl || exit 1
sha256sum "$ENEM"
A='["data/benchmark/oab_exams.jsonl", "data/benchmark/drbodebench_inep.jsonl", "data/benchmark/bluex.jsonl", "data/benchmark/math_pt.jsonl"]'
COM_B="{\"conhecimento\": $A, \"decisao\": \"data/benchmark/bfcl_pt.jsonl\"}"
SO_A="{\"conhecimento\": $A}"
SO_ENEM="{\"conhecimento\": [\"$ENEM\"]}"
mkdir -p "$LUMI_RAID/modelos/gc" analises/guarda_corpo
FALHAS=(); ROTULOS=()
for alvo in base "$@"; do
  nome=${alvo%%:*}; sub=${alvo#*:}; [ "$sub" = "$alvo" ] && sub=adapter
  if [ "$nome" = base ]; then
    rot=base; caminho="$SNAP"; servido=lumi-4b
  else
    rot="${nome}-${sub}"; caminho="$LUMI_RAID/modelos/gc/$rot"; servido="gc-$rot"; ROTULOS+=("$rot")
  fi
  rodadas=("gc-$rot-direct-$SB" "gc-$rot-cot-$SB" "gc-$rot-enemcot-$SE")
  falta=0
  for r in "${rodadas[@]}"; do [ -f "$BENCH/runs/$r/pontuacao_conhecimento.jsonl" ] || falta=1; done
  [ $falta = 0 ] && { echo ">> $rot: rodadas já existem"; continue; }
  if [ "$nome" != base ]; then
    [ -f "runs/$nome/$sub/adapter_model.safetensors" ] || { FALHAS+=("$rot: sem adaptador"); continue; }
    lumi_fundir "runs/$nome/$sub" "$caminho" || { FALHAS+=("$rot: fusão"); continue; }
  fi
  lumi_servir "$caminho" "$servido" || { FALHAS+=("$rot: vLLM"); lumi_parar; continue; }
  lumi_rodar "${rodadas[0]}" "$servido" direct "$SB" "$COM_B" || FALHAS+=("${rodadas[0]}")
  lumi_rodar "${rodadas[1]}" "$servido" cot "$SB" "$SO_A" || FALHAS+=("${rodadas[1]}")
  lumi_rodar "${rodadas[2]}" "$servido" cot "$SE" "$SO_ENEM" || FALHAS+=("${rodadas[2]}")
  lumi_parar
  [ "$nome" = base ] || rm -rf "$caminho"
done
if [ ${#ROTULOS[@]} -gt 0 ]; then
  lumi_exec python3 scripts/guarda_corpo.py "$BENCH/runs" --parte "$PARTE" "${ROTULOS[@]}" \
    --saida "analises/guarda_corpo/${PARTE}_${SLURM_JOB_ID:-local}.json" || FALHAS+=("veredito")
fi
if [ ${#FALHAS[@]} -gt 0 ]; then
  printf '>> FALHOU: %s\n' "${FALHAS[@]}"
  exit 1
fi
echo ">> guarda-corpo concluído ($PARTE): ${ROTULOS[*]:-só a base}"
