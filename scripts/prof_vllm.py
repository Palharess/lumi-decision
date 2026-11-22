"""Roda um lote de conversas num modelo com vLLM (reservadas com professor, analises/reservadas_v2.md).

Roda DENTRO da imagem do vLLM (slurm/prof_config.sh), sem importar o pacote do projeto.
Entrada: JSONL com {"id", "messages", "temperature", "top_p", "max_tokens", "seed"} (outros campos são copiados).
Saída: o mesmo JSONL sem "messages", com "texto", "finish_reason", "modelo" e "revisao".
"""

import argparse
import json

from vllm import LLM, SamplingParams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--revisao", required=True)
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--chat-kwargs", default="{}", help="JSON passado ao template de chat")
    ap.add_argument("--gpu-util", type=float, default=0.88)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--llm-kwargs", default="{}", help="JSON com argumentos extras do LLM (ex.: enforce_eager)")
    a = ap.parse_args()
    with open(a.entrada, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    extra = json.loads(a.llm_kwargs)
    llm = LLM(model=a.modelo, revision=a.revisao, tokenizer_revision=a.revisao, dtype="bfloat16",
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util, max_num_seqs=a.max_num_seqs,
              seed=0, **extra)
    print(f">> vLLM subiu: gpu_util={a.gpu_util} max_num_seqs={a.max_num_seqs} extra={extra}", flush=True)
    params = [SamplingParams(temperature=r["temperature"], top_p=r["top_p"], max_tokens=r["max_tokens"],
                             seed=r["seed"]) for r in rows]
    outs = llm.chat([r["messages"] for r in rows], sampling_params=params,
                    chat_template_kwargs=json.loads(a.chat_kwargs), use_tqdm=True)
    with open(a.saida, "w", encoding="utf-8", newline="\n") as f:
        for r, o in zip(rows, outs):
            r = {k: v for k, v in r.items() if k != "messages"}
            r.update(texto=o.outputs[0].text, finish_reason=o.outputs[0].finish_reason, modelo=a.modelo,
                     revisao=a.revisao, vllm_extra=extra)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f">> {len(rows)} respostas em {a.saida}")


if __name__ == "__main__":
    main()
