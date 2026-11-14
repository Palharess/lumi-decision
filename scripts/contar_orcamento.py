"""Orçamento real de um treino: atualizações, exemplos vistos e tokens de conteúdo, refazendo os planos de lotes
exatamente como o `lora.train` (sem GPU). Uso: python scripts/contar_orcamento.py '{"seed": 1, "data_seed": 0}' ..."""
import json
import sys

from transformers import AutoTokenizer

from lumi_decision import lora
from lumi_decision.data import ARTIFACTS_DIR
from lumi_decision.prompts import load_template

tok = None
for arg in sys.argv[1:]:
    cfg = lora.TrainConfig(**json.loads(arg))
    tok = tok or AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
    enc = lora.Encoder(lambda t: tok(t, add_special_tokens=False)["input_ids"],
                       load_template(ARTIFACTS_DIR / "lumi" / "chat_template.jinja"))
    cases = lora.build_training_set(cfg.sources, cfg.cap_per_family, cfg.seed if cfg.data_seed < 0 else cfg.data_seed)
    steps = []
    for ep in range(cfg.epochs):
        s = cfg.seed * 1000 + ep + cfg.epoch_offset
        exs = [enc(c, f"shuffle{s}", cfg.soft) for c in cases]
        plan = lora.plan_microbatches(exs, cfg.token_budget, cfg.max_micro, s)
        for g in lora.group_steps(plan, cfg.batch_examples):
            steps.append((sum(len(exs[i].ids) for mi in g for i in plan[mi]), sum(len(plan[mi]) for mi in g)))
    if cfg.max_steps:
        steps = steps[: cfg.max_steps]
    print(json.dumps({"config": json.loads(arg), "unicos": len(cases), "atualizacoes": len(steps),
                      "exemplos_vistos": sum(b for _, b in steps), "tokens_conteudo": sum(a for a, _ in steps)}),
          flush=True)
