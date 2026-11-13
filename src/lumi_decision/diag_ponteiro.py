"""F0a — por que a pointer head do E4 errou em `factck_veracidade` (só avaliação, 1 GPU).

Para cada caso de dev de uma família e cada permutação das opções, registra:
  - letras do Lumi congelado (A0), do E3 (A1) e do adaptador do E4 lido pelas letras (sem a cabeça);
  - pointer head do E4: logits e as projeções q (estado final) e k_i (fim de cada opção), alinhadas pelo rótulo.
Cruzamento: no mesmo caso, combina o q de uma ordem com os k de outra e mede a diferença de logit entre duas
opções (padrão: "impreciso" - "falso"). Fixar q e trocar os k isola o efeito dos estados das opções; o inverso,
o do estado final.

    python -m lumi_decision.diag_ponteiro --e3 runs/e3_lora_r16/adapter --e4 runs/e4_ponteiro_s0/adapter \
        --source felhen --family factck_veracidade --out runs/f0_ponteiro_<job>
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from .data import ARTIFACTS_DIR, SOURCES, load_cases, load_config
from .prompts import candidate_tokens, load_template, option_char_ends, option_token_positions, order_options, render


def perm_orders(n: int, max_orders: int) -> list[str]:
    perms = ["perm" + "".join(map(str, p)) for p in itertools.permutations(range(n))]
    return perms if len(perms) <= max_orders else perms[:max_orders]


def run(args) -> dict:
    import numpy as np
    import torch

    from .backends.hf import HFBackend
    from .evaluate import git_commit

    cases = [c for c in load_cases(args.source, load_config())
             if c.split == "dev" and c.family == args.family]
    n_opts = {len(c.options) for c in cases}
    if len(n_opts) != 1:
        raise SystemExit(f"{args.family}: número de opções variável {n_opts}")
    n = n_opts.pop()
    orders = perm_orders(n, args.max_orders)
    template = load_template(ARTIFACTS_DIR / "lumi" / "chat_template.jinja")
    rep = {"meta": {"git": git_commit(), "args": vars(args), "inicio": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "n_casos": len(cases), "ordens": orders}}
    rows = []  # uma linha por (leitura, caso, ordem)

    def prompts_for(be):
        for c in cases:
            for o in orders:
                opts = order_options(c, o)
                prompt = render(template, c, opts)
                ids, cands = candidate_tokens(be.tokenize, prompt, len(opts))
                yield c, o, opts, prompt, ids, cands

    def letters(be, name):
        for c, o, opts, _, ids, cands in prompts_for(be):
            lps = be.readout(ids, cands)[0]
            m = max(lps)
            w = [np.exp(x - m) for x in lps]
            rows.append({"leitura": name, "case_id": c.id, "order": o, "gold": c.gold,
                         "presented": [x[0] for x in opts], "logits": lps,
                         "probs": {opts[i][0]: float(w[i] / sum(w)) for i in range(len(opts))}})

    m = dict(model=args.model, revision=args.revision, dtype="bfloat16", trust_remote_code=False)
    be = HFBackend(**m)
    letters(be, "A0_congelado")
    del be
    torch.cuda.empty_cache()
    be = HFBackend(**m, adapter=args.e3)
    letters(be, "A1_E3")
    del be
    torch.cuda.empty_cache()
    be = HFBackend(**m, adapter=args.e4)
    if be.pointer is None:
        raise SystemExit(f"{args.e4} sem pointer head")
    letters(be, "E4_pelas_letras")
    head = be.pointer
    qk = {}  # (case_id, order) -> (q [dp], {rótulo: k [dp]})
    with torch.inference_mode():
        for c, o, opts, prompt, ids, _ in prompts_for(be):
            ids2, offs = be.offsets(prompt)
            pos = option_token_positions(offs, prompt, option_char_ends(prompt, c, opts))
            out = be._body()(input_ids=torch.tensor([ids], device=be.device),
                             attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=be.device))
            h = (out[0] if isinstance(out, tuple) else out.last_hidden_state)[0]
            q = head.q(h[-1].float())
            k = head.k(h[torch.tensor(pos, device=be.device)].float())
            z = ((k @ q) * head.scale).tolist()
            mx = max(z)
            w = [np.exp(x - mx) for x in z]
            rows.append({"leitura": "B_E4_ponteiro", "case_id": c.id, "order": o, "gold": c.gold,
                         "presented": [x[0] for x in opts], "logits": z,
                         "probs": {opts[i][0]: float(w[i] / sum(w)) for i in range(len(opts))}})
            qk[(c.id, o)] = (q.cpu().numpy(), {opts[i][0]: k[i].cpu().numpy() for i in range(len(opts))})

    # resumo por leitura e ordem
    labels = [x[0] for x in cases[0].options]
    a, b = args.alvo, args.contra
    resumo = {}
    for name in sorted({r["leitura"] for r in rows}):
        rr = [r for r in rows if r["leitura"] == name]
        by_case = defaultdict(dict)
        for r in rr:
            by_case[r["case_id"]][r["order"]] = r
        per_order = {}
        for o in orders:
            xs = [v[o] for v in by_case.values()]
            per_order[o] = {"acc": float(np.mean([max(x["probs"], key=x["probs"].get) == x["gold"] for x in xs])),
                            f"p_{a}": float(np.mean([x["probs"][a] for x in xs])),
                            "escolhas": dict(Counter(max(x["probs"], key=x["probs"].get) for x in xs))}
        flips = np.mean([len({max(x["probs"], key=x["probs"].get) for x in v.values()}) > 1 for v in by_case.values()])
        resumo[name] = {"por_ordem": per_order, "muda_entre_ordens": float(flips),
                        "acc_media": float(np.mean([p["acc"] for p in per_order.values()]))}

    # cruzamento q x k no E4: média sobre os casos de [logit(a) - logit(b)] com q da ordem i e k da ordem j
    cross = np.zeros((len(orders), len(orders)))
    for i, oq in enumerate(orders):
        for j, ok in enumerate(orders):
            ds = []
            for c in cases:
                q = qk[(c.id, oq)][0]
                ks = qk[(c.id, ok)][1]
                ds.append(float((ks[a] - ks[b]) @ q) * head.scale)
            cross[i, j] = np.mean(ds)
    ref = 0
    rep.update(resumo=resumo, rotulos=labels, alvo=a, contra=b, cruzamento={
        "ordens": orders, "matriz_q_linha_k_coluna": cross.round(4).tolist(),
        "so_k_muda": {o: float(cross[ref, j] - cross[ref, ref]) for j, o in enumerate(orders)},
        "so_q_muda": {o: float(cross[i, ref] - cross[ref, ref]) for i, o in enumerate(orders)},
        "ambos_mudam": {o: float(cross[i, i] - cross[ref, ref]) for i, o in enumerate(orders)}})
    out = Path(args.out)
    with (out / "linhas.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.savez_compressed(out / "qk.npz", **{f"{cid}|{o}|q": v[0] for (cid, o), v in qk.items()},
                        **{f"{cid}|{o}|k|{lab}": kk for (cid, o), v in qk.items() for lab, kk in v[1].items()})
    return rep


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="lumi-diag-ponteiro")
    p.add_argument("--model", default=SOURCES["lumi"]["repo"])
    p.add_argument("--revision", default=SOURCES["lumi"]["revision"])
    p.add_argument("--e3", required=True)
    p.add_argument("--e4", required=True)
    p.add_argument("--source", default="felhen")
    p.add_argument("--family", default="factck_veracidade")
    p.add_argument("--alvo", default="impreciso")
    p.add_argument("--contra", default="falso")
    p.add_argument("--max-orders", type=int, default=6)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rep = run(a)
    (Path(a.out) / "diag.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("resumo", "cruzamento")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
