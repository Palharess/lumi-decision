"""Lumi Decision E3b (semente 1): carrega o Lumi-4B + adaptador LoRA e decide entre opções.

    python exemplo.py                      # roda o exemplo embutido (baixa a base do Hub se não houver base/)
    python exemplo.py casos.jsonl > saida.jsonl

Cada linha de `casos.jsonl`: {"contexto": "...", "pergunta": "...", "opcoes": ["texto A", "texto B", ...]}.
Saída por linha: {"escolha": índice, "letra": "B", "probs": [...], "gerado": "B"}.

O modelo foi treinado e avaliado exatamente com este prompt e lendo a próxima letra (ver README.md). Precisa de GPU
CUDA com mamba-ssm e causal-conv1d instalados (o Lumi é híbrido Mamba-2 + atenção).
"""
from __future__ import annotations

import json
import string
import sys
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
# Pacote local (base/ e adapter/ ao lado deste arquivo) ou repositório do Hugging Face (adaptador na raiz e base
# baixada do Hub na mesma revisão usada no treino).
BASE_HUB, REV_HUB = "ceia-research/lumi-4b", "27b30f40b930fd05cee5e1c1f8790f70fa5b816d"
BASE = AQUI / "base" if (AQUI / "base").is_dir() else BASE_HUB
ADAPTADOR = AQUI / "adapter" if (AQUI / "adapter").is_dir() else AQUI
LETRAS = string.ascii_uppercase


def carregar(base=BASE, adaptador=ADAPTADOR, device: str = "cuda"):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = REV_HUB if base == BASE_HUB else None
    tok = AutoTokenizer.from_pretrained(base, revision=rev, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base, revision=rev, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(device)
    if adaptador is not None:
        # LoRA sem fundir, como na avaliação; `merge_and_unload()` também funciona (diferenças de arredondamento bf16).
        model = PeftModel.from_pretrained(model, adaptador, is_trainable=False).get_base_model()
    return tok, model.eval()


def montar_prompt(tok, contexto: str, pergunta: str, opcoes: list[str]) -> str:
    if not 2 <= len(opcoes) <= len(LETRAS):
        raise ValueError("de 2 a 26 opções")
    linhas = "\n".join(f"{LETRAS[i]}. {o}" for i, o in enumerate(opcoes))
    msg = (f"Contexto:\n{contexto}\n\nPergunta: {pergunta}\n\n"
           f"Opções:\n{linhas}\n\nResponda apenas com a letra da opção escolhida.")
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.inference_mode()
def decidir(tok, model, contexto: str, pergunta: str, opcoes: list[str], gerar: bool = True) -> dict:
    prompt = montar_prompt(tok, contexto, pergunta, opcoes)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    letras = []
    for L in LETRAS[: len(opcoes)]:  # cada letra precisa ser um único token logo depois do prompt
        x = tok(prompt + L, add_special_tokens=False)["input_ids"]
        assert len(x) == len(ids) + 1 and x[:-1] == ids, f"letra {L} não é um token na fronteira"
        letras.append(x[-1])
    t = torch.tensor([ids], device=model.device)
    # Leitura usada nos resultados: estado oculto final da última posição, projetado no LM head em FP32.
    h = model.base_model(input_ids=t, attention_mask=torch.ones_like(t))
    h = (h[0] if isinstance(h, tuple) else h.last_hidden_state)[0, -1].float()
    logp = torch.log_softmax(h @ model.get_output_embeddings().weight.float().T, dim=-1)[letras]
    probs = torch.softmax(logp, dim=0).tolist()
    out = {"escolha": int(max(range(len(probs)), key=probs.__getitem__)), "probs": probs}
    out["letra"] = LETRAS[out["escolha"]]
    if gerar:  # geração gulosa: quase sempre devolve a mesma letra que a leitura acima
        eos = [i for i in {model.config.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} if isinstance(i, int)]
        g = model.generate(input_ids=t, attention_mask=torch.ones_like(t), max_new_tokens=4, do_sample=False,
                           eos_token_id=eos, pad_token_id=tok.pad_token_id or 0)
        out["gerado"] = tok.decode(g[0, len(ids):], skip_special_tokens=True).strip()
    return out


def main():
    tok, model = carregar()
    if len(sys.argv) > 1:
        for linha in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
            if linha.strip():
                c = json.loads(linha)
                print(json.dumps(decidir(tok, model, c["contexto"], c["pergunta"], c["opcoes"]), ensure_ascii=False))
        return
    r = decidir(tok, model,
                "Política de reembolso: compras canceladas em até 7 dias são reembolsadas integralmente. "
                "Depois disso, não há reembolso. A cliente cancelou a compra 10 dias depois.",
                "A cliente tem direito a reembolso?",
                ["Sim, integral", "Sim, parcial", "Não"])
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
