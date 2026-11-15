"""Orçamento real de um treino: atualizações, exemplos vistos e tokens de conteúdo, refazendo os planos de lotes
exatamente como o `lora.train` (sem GPU). Uso: python scripts/contar_orcamento.py seed=1,data_seed=0 ...
(chave=valor em vez de JSON: as aspas não sobrevivem ao `lumi_exec` do cluster; listas com `+`, ex.:
sources=felhen+telepatia+traces+f3,source_caps=felhen=650+f3=1625)."""
import json
import sys

from transformers import AutoTokenizer

from lumi_decision import lora
from lumi_decision.data import ARTIFACTS_DIR
from lumi_decision.prompts import load_template



def parse(arg: str) -> dict:
    base = vars(lora.TrainConfig())
    out = {}
    for kv in filter(None, arg.split(",")):
        k, v = kv.split("=", 1)
        out[k] = v.split("+") if isinstance(base[k], list) else type(base[k])(v)  # listas: a+b
    return out


tok = None
for arg in sys.argv[1:]:
    kw = parse(arg)
    cfg = lora.TrainConfig(**kw)
    tok = tok or AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
    enc = lora.Encoder(lambda t: tok(t, add_special_tokens=False)["input_ids"],
                       load_template(ARTIFACTS_DIR / "lumi" / "chat_template.jinja"))
    cases = lora.training_set_for(cfg)
    steps = []
    for ep in range(cfg.epochs):
        s = cfg.seed * 1000 + ep + cfg.epoch_offset
        exs = [enc(c, f"shuffle{s}", cfg.soft) for c in cases]
        plan = lora.plan_microbatches(exs, cfg.token_budget, cfg.max_micro, s)
        for g in lora.group_steps(plan, cfg.batch_examples):
            steps.append((sum(len(exs[i].ids) for mi in g for i in plan[mi]), sum(len(plan[mi]) for mi in g)))
    if cfg.max_steps:
        steps = steps[: cfg.max_steps]
    print(json.dumps({"config": kw, "unicos": len(cases), "atualizacoes": len(steps),
                      "exemplos_vistos": sum(b for _, b in steps), "tokens_conteudo": sum(a for a, _ in steps)}),
          flush=True)
