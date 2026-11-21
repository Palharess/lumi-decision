# Reservadas com professor (analises/reservadas_v2.md): modelos e imagem, com revisão fixada.
# Escolha do Gabriel em 16/11/2026: o Qwen3.8-27B escreve os textos e o Gemma-4-31B-it confere às cegas.
# Os dois são Apache-2.0 e cabem numa H100 em bf16.
export PROF_VLLM_DOCKER="${PROF_VLLM_DOCKER:-vllm/vllm-openai:v0.33.0-x86_64}"
export PROF_VLLM_IMAGE="${PROF_VLLM_IMAGE:-$LUMI_RAID/containers/vllm_v0.33.0.sif}"
export PROF_ESCRITOR="Qwen/Qwen3.8-27B"
export PROF_ESCRITOR_REV="37561816e66b5b9153b1b64c10968c18c1e6f4a0"
export PROF_CONFERENTE="google/gemma-4-31B-it"
export PROF_CONFERENTE_REV="842da3794eaa0b77d5f08bae87a17459d91ff475"

# Roda um comando no container do vLLM (GPU), com o cache do HF em /raid e a raiz do repo como diretório.
# Os argumentos vão como "$@" (não por "$*" dentro de uma string): JSON e espaços chegam intactos ao comando.
prof_vllm_exec() {
  apptainer exec --nv -B "$LUMI_RAID:$LUMI_RAID" "$PROF_VLLM_IMAGE" bash -c 'cd "$1" && shift && exec "$@"' _ "$LUMI_REPO" "$@"
}
