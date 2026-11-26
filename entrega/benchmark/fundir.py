"""Funde o adaptador LoRA no Lumi-4B e salva um modelo completo, para servir no vLLM (Lumi-Benchmark).

    python entrega/benchmark/fundir.py <base> <adaptador> <saida>

Confere a fusão: em prompts reais, compara a distribuição da próxima posição do modelo fundido (bf16) com a do
modelo com o LoRA sem fundir, como foi avaliado. Sai com erro se a letra mais provável mudar em mais de 1 de 20.
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base, adaptador, saida = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)


def carregar():
    return AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()


sem_fundir = PeftModel.from_pretrained(carregar(), adaptador, is_trainable=False).eval()
fundido = PeftModel.from_pretrained(carregar(), adaptador, is_trainable=False).merge_and_unload().eval()

casos = [json.loads(l) for l in Path("data/prof/prof_v1.jsonl").read_text(encoding="utf-8").splitlines()]
casos = [c for c in casos if c["split"] == "dev"][::97][:20]
iguais, max_kl = 0, 0.0
with torch.inference_mode():
    for c in casos:
        ops = "\n".join(f"{chr(65 + i)}. {d}" for i, (_, d) in enumerate(c["options"]))
        msg = (f"Contexto:\n{c['state']}\n\nPergunta: {c['question']}\n\nOpções:\n{ops}\n\n"
               "Responda apenas com a letra da opção escolhida.")
        p = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        ids = torch.tensor([tok(p, add_special_tokens=False)["input_ids"]], device="cuda")
        a = torch.log_softmax(sem_fundir(input_ids=ids).logits[0, -1].float(), -1)
        b = torch.log_softmax(fundido(input_ids=ids).logits[0, -1].float(), -1)
        iguais += int(a.argmax() == b.argmax())
        max_kl = max(max_kl, float(torch.sum(a.exp() * (a - b))))
print(f">> fusão: próxima posição igual em {iguais}/{len(casos)}; maior KL {max_kl:.5f}")
if iguais < len(casos) - 1:
    sys.exit(">> a fusão mudou o modelo demais")

saida.mkdir(parents=True, exist_ok=True)
fundido.save_pretrained(saida, safe_serialization=True)
tok.save_pretrained(saida)
for f in base.iterdir():  # código do modelo (trust_remote_code), template e configs que o save não copia
    if f.suffix in {".py", ".jinja"} or f.name == "generation_config.json":
        shutil.copy2(f, saida / f.name)
(saida / "FUSAO.json").write_text(json.dumps({"base": str(base), "adaptador": str(adaptador), "iguais": iguais,
                                              "n": len(casos), "max_kl": max_kl}, indent=1))
print(f">> salvo em {saida}")
