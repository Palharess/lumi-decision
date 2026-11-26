"""Funde o adaptador LoRA no Lumi-4B e salva um modelo completo, para servir no vLLM (Lumi-Benchmark).

    python entrega/benchmark/fundir.py <snapshot da base> <adaptador> <saida>

O adaptador foi treinado na implementação NATIVA do transformers (módulos `model.layers.*`). Com
`trust_remote_code=True` os módulos se chamam `backbone.layers.*`, o PEFT não acha as chaves e carrega o LoRA zerado
sem erro. Por isso: carrega nativo, confere que todo `lora_B` foi carregado e não é zero, funde, e grava os pesos com
os MESMOS nomes do `model.safetensors` original (que é o formato que o vLLM lê), mais os arquivos originais da base.

Conferências (o script sai com erro se alguma falhar):
- os 71 módulos com LoRA têm `lora_B` igual ao salvo no adaptador;
- fundido × LoRA sem fundir: a letra mais provável igual em pelo menos 19 de 20 prompts reais do dev;
- fundido × base: diferente em algum prompt (senão o LoRA não entrou);
- só os tensores dos módulos com LoRA mudam em relação ao original.
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

base, adaptador, saida = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
tok = AutoTokenizer.from_pretrained(base)


def carregar():
    return AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, trust_remote_code=False).to("cuda").eval()


def conferir_lora(m) -> int:
    salvo = load_file(str(adaptador / "adapter_model.safetensors"))
    n = 0
    for nome, p in m.named_parameters():
        if "lora_B" in nome:
            chave = nome.replace(".default", "")
            assert chave in salvo, f"chave do adaptador não encontrada: {chave}"
            assert torch.equal(p.detach().cpu(), salvo[chave].to(p.dtype)), f"lora_B diferente do salvo: {chave}"
            assert p.abs().sum() > 0, f"lora_B zerado: {chave}"
            n += 1
    assert n == len([k for k in salvo if "lora_B" in k]), f"{n} módulos com LoRA, o adaptador tem outro número"
    return n


puro = carregar()
sem_fundir = PeftModel.from_pretrained(carregar(), adaptador, is_trainable=False).eval()
n_lora = conferir_lora(sem_fundir)
print(f">> LoRA carregado em {n_lora} módulos")
fundido = PeftModel.from_pretrained(carregar(), adaptador, is_trainable=False).merge_and_unload().eval()

casos = [json.loads(l) for l in Path("data/prof/prof_v1.jsonl").read_text(encoding="utf-8").splitlines()]
casos = [c for c in casos if c["split"] == "dev"][::97][:20]
iguais, dif_base, max_kl = 0, 0, 0.0
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
        z = torch.log_softmax(puro(input_ids=ids).logits[0, -1].float(), -1)
        iguais += int(a.argmax() == b.argmax())
        dif_base += int(float(torch.sum(z.exp() * (z - b))) > 1e-3)
        max_kl = max(max_kl, float(torch.sum(a.exp() * (a - b))))
print(f">> fundido × sem fundir: mesma próxima posição em {iguais}/{len(casos)}, maior KL {max_kl:.5f}; "
      f"fundido × base: distribuição diferente em {dif_base}/{len(casos)}")
if iguais < len(casos) - 1 or dif_base == 0:
    sys.exit(">> fusão reprovada")

# Pesos com os nomes do checkpoint original. O estado nativo usa `model.` onde o original usa `backbone.`.
with safe_open(str(base / "model.safetensors"), "pt") as f:
    orig_keys = list(f.keys())
sd = {k: v.detach().cpu().contiguous() for k, v in fundido.state_dict().items()}
novo, mudou = {}, []
with safe_open(str(base / "model.safetensors"), "pt") as f:
    for k in orig_keys:
        cands = [k, k.replace("backbone.", "model.", 1)]
        nk = next((c for c in cands if c in sd), None)
        if nk is None:
            sys.exit(f">> sem correspondente no modelo fundido: {k}")
        t, o = sd[nk].to(f.get_tensor(k).dtype), f.get_tensor(k)
        assert t.shape == o.shape, f"forma diferente em {k}"
        if not torch.equal(t, o):
            mudou.append(k)
        novo[k] = t
alvos = {n.removesuffix(".lora_B.default.weight").split("base_model.model.", 1)[-1]
         for n, _ in sem_fundir.named_parameters() if "lora_B" in n}
esperado = {a.replace("model.", "backbone.", 1) + ".weight" for a in alvos}
if set(mudou) != esperado:
    sys.exit(f">> tensores alterados ≠ módulos com LoRA: {sorted(set(mudou) ^ esperado)[:10]}")
print(f">> {len(mudou)} tensores alterados, todos de módulos com LoRA; {len(orig_keys)} tensores gravados")

saida.mkdir(parents=True, exist_ok=True)
save_file(novo, str(saida / "model.safetensors"), metadata={"format": "pt"})
for f in base.iterdir():  # config, tokenizer, template e código originais, sem alteração
    if f.is_file() and f.name != "model.safetensors":
        shutil.copy2(f, saida / f.name)
(saida / "FUSAO.json").write_text(json.dumps({"base": str(base), "adaptador": str(adaptador), "modulos_lora": n_lora,
                                              "iguais_sem_fundir": iguais, "diferentes_da_base": dif_base,
                                              "n": len(casos), "max_kl": max_kl, "tensores_alterados": len(mudou)},
                                             indent=1))
print(f">> salvo em {saida}")
