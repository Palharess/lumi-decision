"""Treino LoRA sobre o MESMO readout da avaliação (H2), usado pelo E2 (sanidade) e pelo E3 (treino).

Readout de treino: backbone -> estado da última posição real (padding à direita) -> projeção FP32 no LM head
congelado -> log-softmax no vocabulário inteiro -> log-prob das letras candidatas. É o caminho de
`HFBackend.readout_batch`, mas diferenciável (sem `inference_mode`).

Perda: entropia cruzada entre o alvo sobre as opções apresentadas e as log-probs das letras (vocabulário
inteiro, sem renormalizar: puxa a massa para as letras certas). Alvo duro = one-hot no gold; alvo suave =
`gold_dist` quando existe (Telepatia), senão one-hot.

Ordem das opções: sorteada por exemplo e por época (`shuffle<semente+época>`), contra o viés de posição.

Alvos do LoRA no Nemotron-H nativo: `in_proj` do Mamba, `up_proj`/`down_proj` do MLP, `q/k/v/o_proj` da
atenção. NÃO `out_proj` do Mamba: no treino o kernel fundido recebe `self.out_proj.weight` direto e um LoRA
ali não receberia gradiente (conferido no transformers 5.22.2; o E2 checa de novo).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import signal
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .data import load_cases, load_config, training_cases
from .prompts import LETTERS, candidate_tokens, order_options, render
from .schema import Case

LORA_TARGETS = ["in_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]


# ------------------------------------------------------------------ dados

def build_training_set(sources=("felhen", "telepatia", "traces"), cap_per_family: int = 4000,
                       seed: int = 0) -> list[Case]:
    """Casos de treino (só split train, só famílias vistas), com teto determinístico por família."""
    cfg = load_config()
    out = []
    for src in sources:
        cases = training_cases(load_cases(src, cfg))
        by = {}
        for c in cases:
            by.setdefault(c.family, []).append(c)
        for fam in sorted(by):
            items = sorted(by[fam], key=lambda c: hashlib.sha256(f"{seed}:{c.id}".encode()).hexdigest())
            out.extend(items[:cap_per_family] if cap_per_family else items)
    bad = [c.id for c in out if c.held_out or c.split != "train"]
    if bad:
        raise RuntimeError(f"casos fora do treino permitido: {bad[:5]}")
    return out


def target_vector(case: Case, options, soft: bool) -> list[float]:
    ids = [o[0] for o in options]
    if soft and case.gold_dist:
        q = [float(case.gold_dist.get(i, 0.0)) for i in ids]
        z = sum(q)
        if z > 0:
            return [x / z for x in q]
    return [1.0 if i == case.gold else 0.0 for i in ids]


@dataclass
class Example:
    case_id: str
    family: str
    ids: list
    cands: list
    target: list
    gold_pos: int


class Encoder:
    """Prompt -> ids e letras candidatas. Confere a fronteira das letras nos primeiros exemplos de cada
    número de opções e reaproveita os ids depois (o prompt sempre termina em `<think></think>`)."""

    def __init__(self, tokenize, template, n_checks: int = 20):
        self.tokenize, self.template, self.n_checks = tokenize, template, n_checks
        self.cache, self.checked = {}, Counter()

    def __call__(self, case: Case, order: str, soft: bool) -> Example:
        options = order_options(case, order)
        prompt = render(self.template, case, options)
        n = len(options)
        if self.checked[n] < self.n_checks:
            ids, cands = candidate_tokens(self.tokenize, prompt, n)
            if n in self.cache and self.cache[n] != cands:
                raise RuntimeError(f"ids das letras mudaram para n={n}")
            self.cache[n] = cands
            self.checked[n] += 1
        else:
            ids, cands = self.tokenize(prompt), self.cache[n]
        tgt = target_vector(case, options, soft)
        return Example(case.id, case.family, ids, cands, tgt, [o[0] for o in options].index(case.gold))


def plan_microbatches(examples: list[Example], token_budget: int, max_micro: int, seed: int,
                      pool: int = 256) -> list[list[int]]:
    """Índices em micro-lotes: embaralha, agrupa em blocos, ordena por comprimento dentro do bloco e
    empacota até `len(lote) * maior_comprimento <= token_budget`. Ordem final dos micro-lotes embaralhada."""
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    batches = []
    for s in range(0, len(idx), pool):
        block = sorted(idx[s:s + pool], key=lambda i: len(examples[i].ids))
        cur, cur_max = [], 0
        for i in block:
            L = len(examples[i].ids)
            m = max(cur_max, L)
            if cur and (m * (len(cur) + 1) > token_budget or len(cur) >= max_micro):
                batches.append(cur)
                cur, m = [], L
            cur.append(i)
            cur_max = m
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


# ------------------------------------------------------------------ modelo

def body_and_head(model):
    """(backbone, peso do LM head) de um NemotronHForCausalLM, com ou sem PEFT."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.base_model, base.get_output_embeddings().weight


def letter_logprobs(model, batch: list[Example], pad_id: int, head_fp32=None):
    """Log-probs (vocabulário inteiro) das letras candidatas, diferenciável. Devolve lista de tensores [n_i]."""
    import torch

    body, W = body_and_head(model)
    dev = W.device
    lens = torch.tensor([len(e.ids) for e in batch], device=dev)
    T = int(lens.max())
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long, device=dev)
    mask = torch.zeros_like(ids)
    for i, e in enumerate(batch):
        ids[i, : len(e.ids)] = torch.tensor(e.ids, device=dev)
        mask[i, : len(e.ids)] = 1
    out = body(input_ids=ids, attention_mask=mask)
    h = (out[0] if isinstance(out, tuple) else out.last_hidden_state)
    h = h[torch.arange(len(batch), device=dev), lens - 1].float()
    Wf = head_fp32 if head_fp32 is not None else W.float()
    logp = torch.log_softmax(h @ Wf.T, dim=-1)
    return [logp[i, e.cands] for i, e in enumerate(batch)]


def batch_loss(model, batch: list[Example], pad_id: int, head_fp32=None):
    """Soma das perdas por exemplo (CE com o alvo), acertos do argmax e massa média das letras."""
    import torch

    lps = letter_logprobs(model, batch, pad_id, head_fp32)
    losses, hits, mass = [], 0, 0.0
    for lp, e in zip(lps, batch):
        q = torch.tensor(e.target, device=lp.device, dtype=lp.dtype)
        losses.append(-(q * lp).sum())
        hits += int(int(lp.argmax()) == e.gold_pos)
        mass += float(lp.detach().exp().sum())
    return torch.stack(losses).sum(), hits, mass / len(batch)


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05, targets=None):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets or LORA_TARGETS,
                     bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def lora_modules(model) -> dict:
    """{nome do módulo: tipo (sufixo)} dos módulos com LoRA."""
    out = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            out[name] = name.rsplit(".", 1)[-1]
    return out


# ------------------------------------------------------------------ treino (E3)

@dataclass
class TrainConfig:
    model: str = "ceia-br/lumi-4b"
    revision: str = "27b30f40b930fd05cee5e1c1f8790f70fa5b816d"
    sources: list = field(default_factory=lambda: ["felhen", "telepatia", "traces"])
    cap_per_family: int = 4000
    soft: bool = True
    epochs: int = 1
    lr: float = 1e-4
    warmup_frac: float = 0.03
    weight_decay: float = 0.0
    batch_examples: int = 32
    token_budget: int = 16384
    max_micro: int = 32
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    seed: int = 0
    save_every: int = 100
    grad_clip: float = 1.0


class _Stop:
    flag = False


def _on_signal(signum, frame):  # Slurm manda SIGUSR1 antes do limite de tempo (--signal=USR1@...)
    _Stop.flag = True


def train(cfg: TrainConfig, out: Path, resume: bool = True, max_steps: int = 0) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .data import ARTIFACTS_DIR
    from .evaluate import git_commit
    from .prompts import load_template

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_signal)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint"
    log_path = out / "treino.jsonl"
    torch.manual_seed(cfg.seed)

    tok = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, revision=cfg.revision, dtype=torch.bfloat16)
    model.to("cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    model = apply_lora(model, cfg.r, cfg.alpha, cfg.dropout)
    mods = lora_modules(model)
    if any(t == "out_proj" for t in mods.values()):
        raise RuntimeError("LoRA em out_proj do Mamba não recebe gradiente no kernel fundido")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    _, W = body_and_head(model)
    head32 = W.detach().float()

    cases = build_training_set(cfg.sources, cfg.cap_per_family, cfg.seed)
    enc = Encoder(lambda t: tok(t, add_special_tokens=False)["input_ids"],
                  load_template(ARTIFACTS_DIR / "lumi" / "chat_template.jinja"))

    def epoch_plan(ep):
        exs = [enc(c, f"shuffle{cfg.seed * 1000 + ep}", cfg.soft) for c in cases]
        return exs, plan_microbatches(exs, cfg.token_budget, cfg.max_micro, cfg.seed * 1000 + ep)

    # passos de otimização por época (aproximado pelo nº de exemplos)
    steps_per_epoch = math.ceil(len(cases) / cfg.batch_examples)
    total = steps_per_epoch * cfg.epochs if not max_steps else max_steps
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))

    state = {"epoch": 0, "micro": 0, "step": 0}
    if resume and (ckpt / "estado.json").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        state = json.loads((ckpt / "estado.json").read_text())
        set_peft_model_state_dict(model, load_file(str(ckpt / "adapter_model.safetensors")))
        opt.load_state_dict(torch.load(ckpt / "otimizador.pt", map_location="cuda"))
        sched.load_state_dict(torch.load(ckpt / "agendador.pt"))
        print(f">> retomando de {state}", flush=True)

    meta = {"config": asdict(cfg), "git": git_commit(), "n_casos": len(cases),
            "por_familia": dict(Counter(f"{c.source}|{c.family}" for c in cases)),
            "modulos_lora": dict(Counter(mods.values())),
            "params_treinaveis": sum(p.numel() for p in params), "passos_total": total}
    (out / "treino_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)

    def save_ckpt():
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        ckpt.mkdir(exist_ok=True)
        save_file({k: v.contiguous() for k, v in get_peft_model_state_dict(model).items()},
                  str(ckpt / "adapter_model.safetensors.tmp"))
        os.replace(ckpt / "adapter_model.safetensors.tmp", ckpt / "adapter_model.safetensors")
        torch.save(opt.state_dict(), ckpt / "otimizador.pt")
        torch.save(sched.state_dict(), ckpt / "agendador.pt")
        (ckpt / "estado.json").write_text(json.dumps(state))

    model.train()
    t0, toks, acc_ex, run_loss, run_hits, run_n = time.time(), 0, 0, 0.0, 0, 0
    with log_path.open("a", encoding="utf-8") as log:
        for ep in range(state["epoch"], cfg.epochs):
            exs, plan = epoch_plan(ep)
            for mi in range(state["micro"] if ep == state["epoch"] else 0, len(plan)):
                batch = [exs[i] for i in plan[mi]]
                loss_sum, hits, mass = batch_loss(model, batch, pad_id, head32)
                (loss_sum / cfg.batch_examples).backward()
                acc_ex += len(batch)
                toks += sum(len(e.ids) for e in batch)
                run_loss, run_hits, run_n = run_loss + float(loss_sum), run_hits + hits, run_n + len(batch)
                last = mi == len(plan) - 1
                if acc_ex >= cfg.batch_examples or last:
                    gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    acc_ex = 0
                    state.update(step=state["step"] + 1, epoch=ep, micro=mi + 1)
                    if state["step"] % 10 == 0 or last:
                        rec = {"passo": state["step"], "epoca": ep, "perda": run_loss / run_n, "acc": run_hits / run_n,
                               "massa_letras": mass, "lr": sched.get_last_lr()[0], "norma_grad": float(gn),
                               "tokens_s": toks / (time.time() - t0),
                               "mem_gb": torch.cuda.max_memory_allocated() / 1e9}
                        log.write(json.dumps(rec) + "\n")
                        log.flush()
                        print(json.dumps(rec), flush=True)
                        run_loss, run_hits, run_n = 0.0, 0, 0
                    if state["step"] % cfg.save_every == 0 or _Stop.flag:
                        save_ckpt()
                    if _Stop.flag:
                        print(">> sinal recebido: checkpoint salvo, saindo para reenfileirar", flush=True)
                        return {"interrompido": True, **state}
                    if max_steps and state["step"] >= max_steps:
                        break
            if max_steps and state["step"] >= max_steps:
                break
            state.update(epoch=ep + 1, micro=0)
    model.save_pretrained(str(out / "adapter"))
    save_ckpt()
    (out / "CONCLUIDO").write_text(json.dumps(state))
    return {"interrompido": False, **state}


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(prog="lumi-treino", description="E3: LoRA sobre o readout das letras")
    d = TrainConfig()
    for k, v in asdict(d).items():
        if isinstance(v, bool):
            p.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "sim"), default=v)
        elif isinstance(v, list):
            p.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.split(","), default=v)
        else:
            p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--sem-retomar", action="store_true")
    a = vars(p.parse_args(argv))
    out, max_steps, no_resume = Path(a.pop("out")), a.pop("max_steps"), a.pop("sem_retomar")
    res = train(TrainConfig(**a), out, resume=not no_resume, max_steps=max_steps)
    print(json.dumps(res), flush=True)
    raise SystemExit(3 if res["interrompido"] else 0)


if __name__ == "__main__":
    main()
