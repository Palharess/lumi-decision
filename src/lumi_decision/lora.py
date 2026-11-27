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

from .data import LOADERS, PROTEGE_TAMBEM, TRAIN_ONLY, drop_eval_overlap, load_cases, load_config, training_cases
from .prompts import (FORMATO_COT, LETTERS, candidate_tokens, option_char_ends, option_token_positions, order_options,
                      render, render_formato)
from .schema import Case

LORA_TARGETS = ["in_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]


# ------------------------------------------------------------------ dados

def build_training_set(sources=("felhen", "telepatia", "traces"), cap_per_family: int = 4000,
                       seed: int = 0, source_caps: dict | None = None) -> list[Case]:
    """Casos de treino (só split train, só famílias vistas), com teto determinístico por família.
    `source_caps` troca o teto por família de uma fonte (F3: {"felhen": 650, "f3": 1625}). Com a mesma semente,
    o teto menor pega um prefixo da mesma ordenação: os 650 do Felhen estão entre os 1.300 do F2.1."""
    cfg = load_config()
    out = []
    eval_cases = None
    for src in sources:
        loaded = load_cases(src, cfg)
        if src in TRAIN_ONLY:  # fonte externa: tira textos que aparecem em qualquer avaliação das nossas fontes
            if eval_cases is None:
                eval_cases = [c for s in LOADERS for c in load_cases(s, cfg)]
            extra = [c for s in PROTEGE_TAMBEM.get(src, ()) for c in load_cases(s, cfg)]
            drop_eval_overlap(loaded, eval_cases + extra)
        cases = training_cases(loaded)
        by = {}
        for c in cases:
            by.setdefault(c.family, []).append(c)
        cap = (source_caps or {}).get(src, cap_per_family)
        for fam in sorted(by):
            items = sorted(by[fam], key=lambda c: hashlib.sha256(f"{seed}:{c.id}".encode()).hexdigest())
            out.extend(items[:cap] if cap else items)
    bad = [c.id for c in out if c.held_out or c.split != "train"]
    if bad:
        raise RuntimeError(f"casos fora do treino permitido: {bad[:5]}")
    return out


def parse_source_caps(items) -> dict:
    """["felhen=650", "f3=1625"] -> {"felhen": 650, "f3": 1625} (formato da linha de comando)."""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = int(v)
    return out


def training_set_for(cfg: "TrainConfig") -> list[Case]:
    """O conjunto de treino exatamente como o `train` monta (usado também pela contagem de orçamento)."""
    return build_training_set(cfg.sources, cfg.cap_per_family, cfg.seed if cfg.data_seed < 0 else cfg.data_seed,
                              parse_source_caps(cfg.source_caps))


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
    opt_pos: list | None = None  # E4: token do fim de cada opção (chaves da pointer head)
    formato: str = "v1"          # E6: formato de decisão (prompts.FORMATOS)
    pos_inicio: int = -1         # E6: posição cujo próximo token é o 1º da resposta (= a da decisão no v1)
    cot_ids: list | None = None  # E6: o mesmo caso no formato `cot`, só para a âncora do início da resposta


class Encoder:
    """Prompt -> ids e letras candidatas. Confere a fronteira das letras nos primeiros exemplos de cada
    (formato, nº de opções) e reaproveita os ids depois. No v1 o prompt termina em `<think></think>` e a letra é o
    1º token da resposta; nos formatos com prefixo, a letra (" A") vem depois de "Resposta:", e o prompt sem o
    prefixo precisa ser prefixo exato da tokenização (revisão de 27/11: o cache só por nº de opções não servia).
    Com `offsets` (texto -> (ids, offsets)), também localiza o token do fim de cada opção (E4, só v1)."""

    def __init__(self, tokenize, template, n_checks: int = 20, offsets=None):
        self.tokenize, self.template, self.n_checks = tokenize, template, n_checks
        self.offsets = offsets
        self.cache, self.checked = {}, Counter()

    def __call__(self, case: Case, order: str, soft: bool, formato: str = "v1") -> Example:
        options = order_options(case, order)
        n = len(options)
        if formato == "v1":
            prompt, prefixo = render(self.template, case, options), ""
        else:
            if self.offsets is not None:
                raise ValueError("formatos novos não valem com a pointer head (E4)")
            prompt, prefixo = render_formato(self.template, case, options, formato)
        full = prompt + prefixo
        key = (formato, n)
        espaco = " " if prefixo else ""
        if self.checked[key] < self.n_checks:
            ids, cands = candidate_tokens(self.tokenize, full, n, espaco)
            if key in self.cache and self.cache[key] != cands:
                raise RuntimeError(f"ids das letras mudaram para {key}")
            self.cache[key] = cands
            self.checked[key] += 1
        elif self.offsets is None:
            ids, cands = self.tokenize(full), self.cache[key]
        else:
            ids, cands = None, self.cache[key]
        pos = None
        if self.offsets is not None:
            ids2, offs = self.offsets(prompt)
            if ids is not None and list(ids2) != list(ids):
                raise RuntimeError(f"{case.id}: tokenização com offsets difere da tokenização do readout")
            ids = list(ids2)
            pos = option_token_positions(offs, prompt, option_char_ends(prompt, case, options))
        inicio = len(ids) - 1
        if prefixo:
            base = self.tokenize(prompt)
            if list(ids[:len(base)]) != list(base) or len(ids) <= len(base):
                raise RuntimeError(f"{case.id}: o prefixo {prefixo!r} muda a tokenização do prompt ({formato})")
            inicio = len(base) - 1
        tgt = target_vector(case, options, soft)
        return Example(case.id, case.family, ids, cands, tgt, [o[0] for o in options].index(case.gold), pos,
                       formato=formato, pos_inicio=inicio)

    def cot_ids(self, case: Case, order: str) -> list:
        """O caso no formato `cot` (instrução que deixa raciocinar), sem prefixo: a âncora é o 1º token da resposta."""
        prompt, _ = render_formato(self.template, case, order_options(case, order), FORMATO_COT)
        return self.tokenize(prompt)


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

    h, lens = hidden_states(model, batch, pad_id)
    _, W = body_and_head(model)
    h = h[torch.arange(len(batch), device=h.device), lens - 1].float()
    Wf = head_fp32 if head_fp32 is not None else W.float()
    logp = torch.log_softmax(h @ Wf.T, dim=-1)
    return [logp[i, e.cands] for i, e in enumerate(batch)]


def hidden_states(model, batch: list[Example], pad_id: int):
    """Estados finais do backbone [B, T, d] (padding à direita, `use_cache=False`) e os comprimentos."""
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
    return (out[0] if isinstance(out, tuple) else out.last_hidden_state), lens


def option_scores(model, batch: list[Example], pad_id: int, head_fp32=None, pointer=None):
    """Notas NÃO normalizadas entre as opções, um tensor [n_i] por exemplo: letras do LM head (A1), logits da
    pointer head (B, E4) ou letras + pointer residual (F2.3a), numa única passada do backbone."""
    if pointer is None:
        return letter_logprobs(model, batch, pad_id, head_fp32)
    import torch

    from .ponteiro import pointer_logits

    h, lens = hidden_states(model, batch, pad_id)
    z = pointer_logits(h, lens, [e.opt_pos for e in batch], pointer)
    if not getattr(pointer, "residual", False):
        return z
    _, W = body_and_head(model)
    hl = h[torch.arange(len(batch), device=h.device), lens - 1].float()
    Wf = head_fp32 if head_fp32 is not None else W.float()
    logp = torch.log_softmax(hl @ Wf.T, dim=-1)
    return [logp[i, e.cands] + z[i] for i, e in enumerate(batch)]


def batch_loss(model, batch: list[Example], pad_id: int, head_fp32=None, pointer=None):
    """(soma das perdas, acertos, massa média das letras no vocabulário).

    Perda = entropia cruzada entre o alvo e a distribuição RENORMALIZADA entre as opções apresentadas
    (o mesmo objeto que a avaliação usa e que a pointer head do E4 produz). A massa das letras no vocabulário
    inteiro só é registrada (com a pointer head é 1 por construção).
    """
    import torch

    lps = option_scores(model, batch, pad_id, head_fp32, pointer)
    losses, hits, mass = [], 0, 0.0
    for lp, e in zip(lps, batch):
        if pointer is not None and not getattr(pointer, "residual", False):
            lp = torch.log_softmax(lp, dim=0)
        lp_opt = lp - torch.logsumexp(lp, dim=0)
        q = torch.tensor(e.target, device=lp.device, dtype=lp.dtype)
        losses.append(-(q * lp_opt).sum())
        hits += int(int(lp.argmax()) == e.gold_pos)
        mass += float(lp.detach().exp().sum())
    return torch.stack(losses).sum(), hits, mass / len(batch)


# ------------------------------------------------------------------ âncoras KL (E6; LLM-as-Jev, 2610.02076, §3.4)

def _logprobs_em(model, seqs: list, posicoes: list, pad_id: int, head_fp32=None):
    """Log-probs (vocabulário inteiro, fp32) nas posições pedidas de cada sequência: lista de tensores [k_i, V]."""
    import torch

    body, W = body_and_head(model)
    dev = W.device
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long, device=dev)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, device=dev)
        mask[i, : len(s)] = 1
    out = body(input_ids=ids, attention_mask=mask, use_cache=False)
    h = out[0] if isinstance(out, tuple) else out.last_hidden_state
    bi = torch.tensor([i for i, ps in enumerate(posicoes) for _ in ps], device=dev)
    ti = torch.tensor([p for ps in posicoes for p in ps], device=dev)
    Wf = head_fp32 if head_fp32 is not None else W.float()
    lp = torch.log_softmax(h[bi, ti].float() @ Wf.T, dim=-1)
    res, k = [], 0
    for ps in posicoes:
        res.append(lp[k:k + len(ps)])
        k += len(ps)
    return res


def _referencia(model, seqs: list, posicoes: list, pad_id: int, head_fp32=None):
    """As mesmas log-probs no Lumi congelado: o mesmo modelo com o adaptador desligado, sem gradiente e SEM trocar
    o modo (no modo de avaliação o Mamba usa outro kernel; o Lumi não tem dropout fora do LoRA)."""
    import torch

    with torch.no_grad(), model.disable_adapter():
        return _logprobs_em(model, seqs, posicoes, pad_id, head_fp32)


def _kl_topk(lp0, lp, k: int):
    """KL(p0 ‖ p) com as k maiores probabilidades de p0 e um balde com o resto (o top-64 + resto do LLM-as-Jev)."""
    import torch

    v, idx = lp0.topk(min(k, lp0.shape[-1]))
    p0 = v.exp()
    r0 = (1.0 - p0.sum()).clamp_min(1e-12)
    lq = lp[idx]
    rq = (1.0 - lq.exp().sum()).clamp_min(1e-12)
    return (p0 * (v - lq)).sum() + r0 * (torch.log(r0) - torch.log(rq))


def _kl_bern(m0, m):
    """KL entre Bernoullis: massa nas letras contra o resto do vocabulário (o `L_mass`)."""
    import torch

    m0, m = m0.clamp(1e-6, 1 - 1e-6), m.clamp(1e-6, 1 - 1e-6)
    return m0 * (torch.log(m0) - torch.log(m)) + (1 - m0) * (torch.log1p(-m0) - torch.log1p(-m))


def _fora(lp, cands):
    """Log-probs renormalizadas fora das letras candidatas (o `L_out` compara esta parte)."""
    import torch

    x = lp.clone()
    x[cands] = float("-inf")
    return x - torch.logsumexp(x, dim=0)


def batch_loss_ancorado(model, batch: list[Example], pad_id: int, head_fp32=None, topk: int = 64):
    """E6 (R2): perda de decisão mais as âncoras, numa passada com o adaptador e outra sem, sobre as mesmas
    sequências (prompt + letra do gabarito; a letra acrescentada não muda a posição da decisão, que vem antes).

    Por exemplo:
    - decisão: a mesma entropia cruzada renormalizada do `batch_loss`, na posição da letra;
    - `L_mass`: KL de Bernoulli da massa nas letras, na posição da decisão;
    - `L_out`: KL da distribuição fora das letras, renormalizada, na posição da decisão;
    - `L_pos`: KL da distribuição inteira logo depois da letra e, nos formatos com prefixo, no início da resposta
      (pesos 0,5 e 0,5, como no artigo). No v1 o início da resposta é a própria decisão, coberta por `L_mass`/`L_out`.

    Devolve (soma das perdas de decisão, {"mass","out","pos"}: somas [tensores], acertos, massa média das letras)."""
    import torch

    seqs, pos = [], []
    for e in batch:
        d = len(e.ids) - 1
        ps = [d, d + 1]
        if e.pos_inicio not in (-1, d):
            ps.append(e.pos_inicio)
        seqs.append(list(e.ids) + [e.cands[e.gold_pos]])
        pos.append(ps)
    lps = _logprobs_em(model, seqs, pos, pad_id, head_fp32)
    ref = _referencia(model, seqs, pos, pad_id, head_fp32)
    dec, t_mass, t_out, t_pos = [], [], [], []
    hits, mass = 0, 0.0
    for e, lp, lp0 in zip(batch, lps, ref):
        c = e.cands
        letras = lp[0, c]
        q = torch.tensor(e.target, device=letras.device, dtype=letras.dtype)
        dec.append(-(q * (letras - torch.logsumexp(letras, dim=0))).sum())
        hits += int(int(letras.argmax()) == e.gold_pos)
        mass += float(letras.detach().exp().sum())
        t_mass.append(_kl_bern(lp0[0, c].exp().sum(), letras.exp().sum()))
        t_out.append(_kl_topk(_fora(lp0[0], c), _fora(lp[0], c), topk))
        depois = _kl_topk(lp0[1], lp[1], topk)
        t_pos.append(0.5 * _kl_topk(lp0[2], lp[2], topk) + 0.5 * depois if lp.shape[0] == 3 else depois)
    anc = {"mass": torch.stack(t_mass).sum(), "out": torch.stack(t_out).sum(), "pos": torch.stack(t_pos).sum()}
    return torch.stack(dec).sum(), anc, hits, mass / len(batch)


def batch_cot(model, batch: list[Example], pad_id: int, head_fp32=None, topk: int = 64):
    """E6 (R2): soma da KL do 1º token da resposta nos contextos `cot` (sem perda de decisão)."""
    import torch

    seqs = [e.cot_ids for e in batch]
    pos = [[len(s) - 1] for s in seqs]
    lps = _logprobs_em(model, seqs, pos, pad_id, head_fp32)
    ref = _referencia(model, seqs, pos, pad_id, head_fp32)
    return torch.stack([_kl_topk(lp0[0], lp[0], topk) for lp, lp0 in zip(lps, ref)]).sum()


def formato_do_exemplo(formatos: list, case_id: str, s: int) -> str:
    """E6: formato de um exemplo numa época, sorteado por hash (caso, semente da época), uniforme entre `formatos`."""
    h = int(hashlib.sha256(f"formato:{s}:{case_id}".encode()).hexdigest()[:12], 16)
    return formatos[h % len(formatos)]


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
    readout: str = "letras"  # letras (A1, LM head) | ponteiro (B, E4)
    head_dim: int = 256      # E4: dimensão das projeções da pointer head (como no kev)
    head_lr: float = 0.0     # E4: lr da cabeça; 0 = a mesma do LoRA (padrão do kev, `--head_lr 0`)
    head_seed: int = 1       # E4: semente do gerador PRÓPRIO da cabeça (não mexe no RNG do LoRA)
    data_seed: int = -1      # F1: semente da SELEÇÃO dos exemplos (teto por família); -1 = a mesma de `seed`
    init_adapter: str = ""   # F2.3a: começa do LoRA de um adaptador já treinado (ex.: o do E3)
    epoch_offset: int = 0    # F2.3a: desloca as ordens/planos (época 0 daqui = época `epoch_offset` do E3)
    frozen_targets: list = field(default_factory=list)  # F2.2: alvos cujo LoRA nasce (B = 0) e fica CONGELADO
    source_caps: list = field(default_factory=list)  # F3: teto por família de uma fonte, ex.: felhen=650,f3=1625
    # E6 (analises/e6_protocolo.md). Os padrões reproduzem o treino antigo (só v1, sem âncora, sem adaptadores
    # intermediários). Com formatos ou âncora, o plano de lotes é o do v1: mesmas atualizações e mesma ordem do R0.
    formatos: list = field(default_factory=list)  # formatos de decisão sorteados por exemplo (vazio = só v1)
    ancora_lambda: float = 0.0  # peso das âncoras KL contra o Lumi congelado (0 = sem âncora)
    ancora_cot: bool = False    # âncora do 1º token da resposta nos contextos `cot` (exige ancora_lambda > 0)
    ancora_topk: int = 64       # top-k da referência nas KL, mais um balde com o resto
    snapshots: list = field(default_factory=list)  # % do treino em que o adaptador é guardado (ex.: 25,50,75)


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


def save_checkpoint(ckpt: Path, model, opt, sched, state: dict, pointer=None) -> None:
    """Checkpoint como UNIDADE: grava tudo numa pasta temporária e troca de uma vez (rename atômico).
    Com a pointer head (E4), os pesos dela entram na mesma pasta."""
    import shutil

    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    tmp, old = ckpt.with_name(ckpt.name + ".tmp"), ckpt.with_name(ckpt.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    save_file({k: v.detach().contiguous() for k, v in get_peft_model_state_dict(model).items()},
              str(tmp / "adapter_model.safetensors"))
    if pointer is not None:
        save_file({k: v.detach().contiguous() for k, v in pointer.state_dict().items()}, str(tmp / "ponteiro.safetensors"))
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
    if cfg.readout not in ("letras", "ponteiro", "residual"):
        raise ValueError(f"readout desconhecido: {cfg.readout}")
    e6 = bool(cfg.formatos) or cfg.ancora_lambda > 0
    if e6 and cfg.readout != "letras":
        raise ValueError("formatos e âncoras do E6 só valem com a leitura pelas letras")
    if cfg.ancora_cot and not cfg.ancora_lambda > 0:
        raise ValueError("ancora_cot exige ancora_lambda > 0")
    from .prompts import FORMATOS
    if any(f not in FORMATOS for f in cfg.formatos):
        raise ValueError(f"formato desconhecido em {cfg.formatos}")
    if cfg.init_adapter:  # F2.3a: continua um LoRA já treinado (o otimizador e o cronograma recomeçam)
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        res = set_peft_model_state_dict(model, load_file(str(Path(cfg.init_adapter) / "adapter_model.safetensors"),
                                                         device=str(W.device)))
        if getattr(res, "unexpected_keys", None):
            raise RuntimeError(f"adaptador inicial não casa: {res.unexpected_keys[:3]}")
    frozen = {}
    if cfg.frozen_targets:
        # F2.2: o LoRA desses alvos é criado (mesma inicialização e mesmo consumo de RNG do dropout que o E3 da
        # mesma semente) mas não treina. Com B = 0, a saída desses módulos é exatamente a do modelo base.
        for name, mod in model.named_modules():
            if name.rsplit(".", 1)[-1] in cfg.frozen_targets and hasattr(mod, "lora_B") and "default" in mod.lora_B:
                for prm in list(mod.lora_A["default"].parameters()) + list(mod.lora_B["default"].parameters()):
                    prm.requires_grad_(False)
                frozen[name] = float(mod.lora_B["default"].weight.detach().abs().max())
        if not frozen:
            raise RuntimeError(f"nenhum módulo LoRA casa com frozen_targets={cfg.frozen_targets}")
        if not cfg.init_adapter and max(frozen.values()) != 0.0:
            raise RuntimeError("LoRA congelado com B != 0: o módulo não ficaria igual ao base")
    pointer = None
    if cfg.readout in ("ponteiro", "residual"):  # cabeça nova com gerador próprio; o LoRA nasce igual ao do E3
        from .ponteiro import new_head
        pointer = new_head(W.shape[1], cfg.head_dim, cfg.head_seed, residual=cfg.readout == "residual").to(W.device)

    cases = cases if cases is not None else training_set_for(cfg)

    def _offsets(t):
        e = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        return e["input_ids"], e["offset_mapping"]

    enc = Encoder(lambda t: tok(t, add_special_tokens=False)["input_ids"],
                  template or load_template(ARTIFACTS_DIR / "lumi" / "chat_template.jinja"),
                  offsets=_offsets if pointer is not None else None)

    # Planos de TODAS as épocas antes de começar: o nº real de atualizações define o agendador.
    epochs = []
    for ep in range(cfg.epochs):
        s = cfg.seed * 1000 + ep + cfg.epoch_offset
        exs = [enc(c, f"shuffle{s}", cfg.soft) for c in cases]
        plan = plan_microbatches(exs, cfg.token_budget, cfg.max_micro, s)
        if e6:  # E6: o plano é o do v1 (idêntico ao do R0); só o texto de cada exemplo muda
            exs = [enc(c, f"shuffle{s}", cfg.soft,
                       formato_do_exemplo(cfg.formatos, c.id, s) if cfg.formatos else "v1") for c in cases]
            if cfg.ancora_cot:
                for e, c in zip(exs, cases):
                    e.cot_ids = enc.cot_ids(c, f"shuffle{s}")
        epochs.append((exs, plan, group_steps(plan, cfg.batch_examples)))
    schedule = [(ep, gi) for ep, (_, _, groups) in enumerate(epochs) for gi in range(len(groups))]
    if cfg.max_steps:
        schedule = schedule[: cfg.max_steps]
    total = len(schedule)
    fp = data_fingerprint(cases, [p for _, p, _ in epochs])
    snaps = {}  # E6: passo -> % do treino em que o adaptador intermediário é guardado
    for p in cfg.snapshots:
        p = int(p)
        if not 0 < p < 100:
            raise ValueError(f"snapshot fora de (0, 100): {p}")
        snaps[max(1, math.ceil(total * p / 100))] = p

    params = [p for p in model.parameters() if p.requires_grad]
    if pointer is None:
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:  # mesmo cronograma (LambdaLR multiplica cada grupo); a cabeça pode ter lr própria
        hp = list(pointer.parameters())
        opt = torch.optim.AdamW([{"params": params, "lr": cfg.lr}, {"params": hp, "lr": cfg.head_lr or cfg.lr}],
                                weight_decay=cfg.weight_decay)
        params = params + hp
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
        if pointer is not None:
            pointer.load_state_dict(load_file(str(ckpt / "ponteiro.safetensors"), device=str(W.device)))
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
            "modulos_congelados": dict(Counter(n.rsplit(".", 1)[-1] for n in frozen)),
            "congelados_max_abs_B": max(frozen.values()) if frozen else None,
            "passos_total": total, "micro_lotes_por_epoca": [len(p) for _, p, _ in epochs],
            "maior_prompt": max(len(e.ids) for e in epochs[0][0])}
    if snaps:
        meta["snapshots_passo_pct"] = {str(k): v for k, v in sorted(snaps.items())}
    if e6:  # exposições e tokens por objetivo e por formato, nas atualizações que entram no cronograma
        vistos = [(ep, i) for ep, gi in schedule for mi in epochs[ep][2][gi] for i in epochs[ep][1][mi]]
        meta["e6"] = {
            "exposicoes_por_formato": dict(Counter(epochs[ep][0][i].formato for ep, i in vistos)),
            "tokens_decisao": sum(len(epochs[ep][0][i].ids) for ep, i in vistos),
            "tokens_cot": sum(len(epochs[ep][0][i].cot_ids or []) for ep, i in vistos),
            "formatos_sha256": hashlib.sha256(json.dumps(
                [[e.formato for e in exs] for exs, _, _ in epochs]).encode()).hexdigest()}
    if pointer is not None and pointer.residual and state["passo"] == 0:
        # F2.3a: com W_k = 0 a leitura residual tem de ser EXATAMENTE a das letras do adaptador inicial
        exs0, plan0, _ = epochs[0]
        b0 = [exs0[i] for i in plan0[0]]
        model.eval()
        with torch.no_grad():
            zr = option_scores(model, b0, pad_id, head32, pointer)
            zl = letter_logprobs(model, b0, pad_id, head32)
        dif = max(float((a - b).abs().max()) for a, b in zip(zr, zl))
        meta["residual_inicial_max_dif"] = dif
        if dif != 0.0:
            raise RuntimeError(f"leitura residual inicial difere das letras ({dif})")
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
            anc_t = Counter()
            for b in micro:
                if cfg.ancora_lambda > 0:
                    # E6 (R2): decisão + âncoras na mesma média sobre os exemplos de decisão da atualização; o
                    # contexto `cot` vai numa segunda passada (mesma atualização), para não dobrar a memória
                    loss_sum, anc, hits, mass = batch_loss_ancorado(model, b, pad_id, head32, cfg.ancora_topk)
                    ((loss_sum + cfg.ancora_lambda * (anc["mass"] + anc["out"] + anc["pos"])) / n_ex).backward()
                    anc_t.update({k: float(v.detach()) for k, v in anc.items()})
                    if cfg.ancora_cot:
                        cot = batch_cot(model, b, pad_id, head32, cfg.ancora_topk)
                        (cfg.ancora_lambda * cot / n_ex).backward()
                        anc_t["cot"] += float(cot.detach())
                else:
                    loss_sum, hits, mass = batch_loss(model, b, pad_id, head32, pointer)
                    (loss_sum / n_ex).backward()  # média sobre o lote EFETIVO desta atualização
                loss_t, hits_t, mass_t = loss_t + float(loss_sum.detach()), hits_t + hits, mass_t + mass * len(b)
                toks += sum(len(e.ids) for e in b)
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            state["passo"] += 1
            if state["passo"] in snaps:  # E6: adaptador intermediário (não é refeito numa retomada)
                d = out / f"adapter_p{snaps[state['passo']]}"
                if not (d / "adapter_model.safetensors").exists():
                    model.save_pretrained(str(d))
            if state["passo"] % 10 == 0 or state["passo"] == total:
                rec = {"passo": state["passo"], "epoca": ep, "exemplos": n_ex, "perda": loss_t / n_ex,
                       "acc": hits_t / n_ex, "massa_letras": mass_t / n_ex, "lr": sched.get_last_lr()[0],
                       "norma_grad": float(gn), "tokens_s": toks / max(1e-9, time.time() - t0),
                       "mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None}
                if anc_t:
                    rec.update({f"ancora_{k}": v / n_ex for k, v in sorted(anc_t.items())})
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            if _stop_after and state["passo"] == _stop_after:
                _Stop.flag = True
            if stop_file.exists():  # o Apptainer não repassa o SIGUSR1 ao processo (E3 36835): o sbatch cria o arquivo
                _Stop.flag = True
            if state["passo"] % cfg.save_every == 0 or _Stop.flag:
                save_checkpoint(ckpt, model, opt, sched, state, pointer)
            if _Stop.flag:
                stop_file.unlink(missing_ok=True)
                print(">> interrompido: checkpoint salvo; ressubmeta o mesmo comando para retomar", flush=True)
                return {"interrompido": True, "passo": state["passo"], "total": total}
    model.save_pretrained(str(out / "adapter"))
    if pointer is not None:
        from .ponteiro import save_head
        save_head(pointer, out / "adapter")
    save_checkpoint(ckpt, model, opt, sched, state, pointer)
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
