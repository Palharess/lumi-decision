"""Treino LoRA sobre o MESMO readout da avaliação (H2), usado pelo E2 (sanidade) e pelo E3 (treino).

Readout de treino: backbone (use_cache=False, para o Mamba usar o kernel fundido de treino) -> estado da
última posição real (padding à direita) -> projeção FP32 no LM head congelado -> log-softmax no vocabulário
-> log-prob das letras candidatas. É o caminho de `HFBackend.readout_batch`, mas diferenciável.

Perda: entropia cruzada entre o alvo e a distribuição renormalizada entre as opções apresentadas (a mesma da
avaliação e da cabeça de ponteiro do E4). Alvo duro = one-hot no gold; alvo suave = `gold_dist` quando
existe (Telepatia), senão one-hot.

Lotes: micro-lotes por orçamento de tokens; as atualizações agrupam micro-lotes até >= batch_examples e a
perda é a média sobre o lote efetivo de cada atualização. O agendador usa o nº REAL de atualizações.
Checkpoint: pasta trocada de uma vez (pesos, otimizador, agendador, RNG, passo, config e hash dos dados);
retomar exige mesma config e mesmos dados.

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

from .data import LOADERS, TRAIN_ONLY, drop_eval_overlap, load_cases, load_config, training_cases
from .prompts import LETTERS, candidate_tokens, order_options, render
from .schema import Case

LORA_TARGETS = ["in_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]


# ------------------------------------------------------------------ dados

def build_training_set(sources=("felhen", "telepatia", "traces"), cap_per_family: int = 4000,
                       seed: int = 0) -> list[Case]:
    """Casos de treino (só split train, só famílias vistas), com teto determinístico por família."""
    cfg = load_config()
    out = []
    eval_cases = None
    for src in sources:
        loaded = load_cases(src, cfg)
        if src in TRAIN_ONLY:  # fonte externa: tira textos que aparecem em qualquer avaliação das nossas fontes
            if eval_cases is None:
                eval_cases = [c for s in LOADERS for c in load_cases(s, cfg)]
            drop_eval_overlap(loaded, eval_cases)
        cases = training_cases(loaded)
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




def group_steps(plan: list[list[int]], batch_examples: int) -> list[list[int]]:
    """Agrupa micro-lotes (na ordem do plano) em atualizações de >= batch_examples exemplos (a última pode ser
    menor). O número de atualizações e o tamanho efetivo de cada uma saem daqui, do plano real."""
    groups, cur, n = [], [], 0
    for mi, b in enumerate(plan):
        cur.append(mi)
        n += len(b)
        if n >= batch_examples:
            groups.append(cur)
            cur, n = [], 0
    if cur:
        groups.append(cur)
    return groups


# ------------------------------------------------------------------ modelo

def body_and_head(model):
    """(backbone, peso do LM head) de um NemotronHForCausalLM, com ou sem PEFT."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.base_model, base.get_output_embeddings().weight


def letter_logprobs(model, batch: list[Example], pad_id: int, head_fp32=None):
    """Log-probs (vocabulário inteiro) das letras candidatas, diferenciável. Devolve lista de tensores [n_i].

    `use_cache=False` é obrigatório: o config do Lumi tem use_cache=true e, com cache, o Mamba do Nemotron-H
    não usa o kernel fundido de treino (`self.training and cache_params is None`). O E2 confere o kernel.
    """
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
    out = body(input_ids=ids, attention_mask=mask, use_cache=False)
    h = (out[0] if isinstance(out, tuple) else out.last_hidden_state)
    h = h[torch.arange(len(batch), device=dev), lens - 1].float()
    Wf = head_fp32 if head_fp32 is not None else W.float()
    logp = torch.log_softmax(h @ Wf.T, dim=-1)
    return [logp[i, e.cands] for i, e in enumerate(batch)]


def batch_loss(model, batch: list[Example], pad_id: int, head_fp32=None):
    """(soma das perdas, acertos, massa média das letras no vocabulário).

    Perda = entropia cruzada entre o alvo e a distribuição RENORMALIZADA entre as opções apresentadas
    (o mesmo objeto que a avaliação usa e que a cabeça de ponteiro do E4 vai produzir). A massa das letras
    no vocabulário inteiro só é registrada.
    """
    import torch

    lps = letter_logprobs(model, batch, pad_id, head_fp32)
    losses, hits, mass = [], 0, 0.0
    for lp, e in zip(lps, batch):
        lp_opt = lp - torch.logsumexp(lp, dim=0)
        q = torch.tensor(e.target, device=lp.device, dtype=lp.dtype)
        losses.append(-(q * lp_opt).sum())
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
    token_budget: int = 8192   # E2 36812: 16384 -> pico 72,8 GB de 80; 8192 -> 35 GB, mesma vazão
    max_micro: int = 32
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    seed: int = 0
    save_every: int = 100
    grad_clip: float = 1.0
    max_steps: int = 0  # > 0: para depois desse nº de atualizações (E3 curto); o agendador usa esse total


class _Stop:
    flag = False


def _on_signal(signum, frame):  # Slurm manda SIGUSR1 antes do limite de tempo (--signal=B:USR1@...)
    _Stop.flag = True


def data_fingerprint(cases, plans) -> str:
    h = hashlib.sha256()
    for c in cases:
        h.update(f"{c.id}|{c.gold}\n".encode())
    h.update(json.dumps(plans).encode())
    return h.hexdigest()


def _rng_state():
    import torch
    st = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    import torch
    random.setstate(st["python"])
    torch.set_rng_state(st["torch"].cpu())  # estados de RNG são ByteTensor na CPU, mesmo os da CUDA
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])


def save_checkpoint(ckpt: Path, model, opt, sched, state: dict) -> None:
    """Checkpoint como UNIDADE: grava tudo numa pasta temporária e troca de uma vez (rename atômico)."""
    import shutil

    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    tmp, old = ckpt.with_name(ckpt.name + ".tmp"), ckpt.with_name(ckpt.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    save_file({k: v.detach().contiguous() for k, v in get_peft_model_state_dict(model).items()},
              str(tmp / "adapter_model.safetensors"))
    torch.save({"otimizador": opt.state_dict(), "agendador": sched.state_dict(), "rng": _rng_state()},
               tmp / "treino.pt")
    (tmp / "estado.json").write_text(json.dumps(state, indent=1))
    shutil.rmtree(old, ignore_errors=True)
    if ckpt.exists():
        os.replace(ckpt, old)
    os.replace(tmp, ckpt)
    shutil.rmtree(old, ignore_errors=True)


def train(cfg: TrainConfig, out: Path, resume: bool = True, *, model=None, tok=None, device: str = "cuda",
          cases=None, template=None, targets=None, _stop_after: int = 0) -> dict:
    """Treina e salva `out/adapter` + `out/CONCLUIDO`. Interrompido (SIGUSR1 ou arquivo `out/PARAR`): salva checkpoint e devolve
    interrompido=True. model/tok/cases/template/targets/_stop_after existem para os testes (modelo minúsculo)."""
    import shutil

    import torch

    from .data import ARTIFACTS_DIR
    from .evaluate import git_commit
    from .prompts import load_template

    _Stop.flag = False
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_signal)
    out.mkdir(parents=True, exist_ok=True)
    stop_file = out / "PARAR"  # pedido de parada: criado pelo sbatch ao receber SIGUSR1; consumido ao parar
    ckpt = out / "checkpoint"
    if not ckpt.exists() and ckpt.with_name("checkpoint.old").exists():  # queda no meio da troca
        os.replace(ckpt.with_name("checkpoint.old"), ckpt)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
        model = AutoModelForCausalLM.from_pretrained(cfg.model, revision=cfg.revision, dtype=torch.bfloat16)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model = apply_lora(model, cfg.r, cfg.alpha, cfg.dropout, targets)
    mods = lora_modules(model)
    if any(t == "out_proj" for t in mods.values()):
        raise RuntimeError("LoRA em out_proj do Mamba não recebe gradiente no kernel fundido")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    _, W = body_and_head(model)
    head32 = W.detach().float()

    cases = cases if cases is not None else build_training_set(cfg.sources, cfg.cap_per_family, cfg.seed)
    enc = Encoder(lambda t: tok(t, add_special_tokens=False)["input_ids"],
                  template or load_template(ARTIFACTS_DIR / "lumi" / "chat_template.jinja"))

    # Planos de TODAS as épocas antes de começar: o nº real de atualizações define o agendador.
    epochs = []
    for ep in range(cfg.epochs):
        s = cfg.seed * 1000 + ep
        exs = [enc(c, f"shuffle{s}", cfg.soft) for c in cases]
        plan = plan_microbatches(exs, cfg.token_budget, cfg.max_micro, s)
        epochs.append((exs, plan, group_steps(plan, cfg.batch_examples)))
    schedule = [(ep, gi) for ep, (_, _, groups) in enumerate(epochs) for gi in range(len(groups))]
    if cfg.max_steps:
        schedule = schedule[: cfg.max_steps]
    total = len(schedule)
    fp = data_fingerprint(cases, [p for _, p, _ in epochs])

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))

    state = {"passo": 0, "config": asdict(cfg), "dados_sha256": fp, "passos_total": total}
    if resume and (ckpt / "estado.json").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        old = json.loads((ckpt / "estado.json").read_text())
        if old.get("config") != asdict(cfg) or old.get("dados_sha256") != fp:
            raise SystemExit(f"checkpoint em {ckpt} é de outra configuração ou outros dados; use outra pasta.\n"
                             f"antes: {old.get('config')} {old.get('dados_sha256')}\nagora: {asdict(cfg)} {fp}")
        set_peft_model_state_dict(model, load_file(str(ckpt / "adapter_model.safetensors"), device=str(W.device)))
        # Na CPU: map_location=cuda levava o RNG junto e set_rng_state recusava (E3 36850). O otimizador
        # move o próprio estado para o dispositivo dos parâmetros no load_state_dict.
        tr = torch.load(ckpt / "treino.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(tr["otimizador"])
        sched.load_state_dict(tr["agendador"])
        _set_rng_state(tr["rng"])
        state = old
        print(f">> retomando do passo {state['passo']}/{total}", flush=True)
    elif ckpt.exists() and not resume:
        shutil.rmtree(ckpt)

    meta = {"config": asdict(cfg), "git": git_commit(), "n_casos": len(cases), "dados_sha256": fp,
            "por_familia": dict(Counter(f"{c.source}|{c.family}" for c in cases)),
            "modulos_lora": dict(Counter(mods.values())), "params_treinaveis": sum(p.numel() for p in params),
            "passos_total": total, "micro_lotes_por_epoca": [len(p) for _, p, _ in epochs],
            "maior_prompt": max(len(e.ids) for e in epochs[0][0])}
    (out / "treino_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)

    model.train()
    t0, toks = time.time(), 0
    with (out / "treino.jsonl").open("a", encoding="utf-8") as log:
        while state["passo"] < total:
            ep, gi = schedule[state["passo"]]
            exs, plan, groups = epochs[ep]
            micro = [[exs[i] for i in plan[mi]] for mi in groups[gi]]
            n_ex = sum(len(b) for b in micro)
            loss_t, hits_t, mass_t = 0.0, 0, 0.0
            for b in micro:
                loss_sum, hits, mass = batch_loss(model, b, pad_id, head32)
                (loss_sum / n_ex).backward()  # média sobre o lote EFETIVO desta atualização
                loss_t, hits_t, mass_t = loss_t + float(loss_sum.detach()), hits_t + hits, mass_t + mass * len(b)
                toks += sum(len(e.ids) for e in b)
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            state["passo"] += 1
            if state["passo"] % 10 == 0 or state["passo"] == total:
                rec = {"passo": state["passo"], "epoca": ep, "exemplos": n_ex, "perda": loss_t / n_ex,
                       "acc": hits_t / n_ex, "massa_letras": mass_t / n_ex, "lr": sched.get_last_lr()[0],
                       "norma_grad": float(gn), "tokens_s": toks / max(1e-9, time.time() - t0),
                       "mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            if _stop_after and state["passo"] == _stop_after:
                _Stop.flag = True
            if stop_file.exists():  # o Apptainer não repassa o SIGUSR1 ao processo (E3 36835): o sbatch cria o arquivo
                _Stop.flag = True
            if state["passo"] % cfg.save_every == 0 or _Stop.flag:
                save_checkpoint(ckpt, model, opt, sched, state)
            if _Stop.flag:
                stop_file.unlink(missing_ok=True)
                print(">> interrompido: checkpoint salvo; ressubmeta o mesmo comando para retomar", flush=True)
                return {"interrompido": True, "passo": state["passo"], "total": total}
    model.save_pretrained(str(out / "adapter"))
    save_checkpoint(ckpt, model, opt, sched, state)
    (out / "CONCLUIDO").write_text(json.dumps({"passo": state["passo"], "total": total}))
    return {"interrompido": False, "passo": state["passo"], "total": total}


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
    p.add_argument("--sem-retomar", action="store_true", help="apaga o checkpoint da pasta e começa do zero")
    a = vars(p.parse_args(argv))
    out, no_resume = Path(a.pop("out")), a.pop("sem_retomar")
    res = train(TrainConfig(**a), out, resume=not no_resume)
    print(json.dumps(res), flush=True)
    raise SystemExit(3 if res["interrompido"] else 0)


if __name__ == "__main__":
    main()
