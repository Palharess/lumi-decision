"""E0 — sanidade do checkpoint oficial no backend HF (rodar no cluster, 1 GPU).

Verifica, nos mesmos casos do piloto local:
  1. nosso template == tokenizer.apply_chat_template;
  2. readout (backbone + linhas do LM head) == forward completo;
  3. lote com padding à direita == lote de 1;
  4. geração: token de parada, 1º token == argmax do readout, cache == sem cache;
  5. concordância com uma execução de referência (ex.: GGUF local), se fornecida.

    lumi-e0 --model ceia-br/lumi-4b --revision 27b30f4... --source casos.json --ref-run runs/piloto40_q4_vulkan --out runs/e0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .data import ARTIFACTS_DIR, SOURCES, load_cases, load_config
from .evaluate import git_commit, load_rows, predict_one
from .prompts import candidate_tokens, load_template, order_options, render, user_message


def run(args) -> dict:
    import torch

    from .backends.hf import HFBackend

    be = HFBackend(model=args.model, revision=args.revision, dtype=args.dtype,
                   trust_remote_code=not args.native)
    tpl = load_template(Path(args.template))
    cases = load_cases(args.source, load_config())
    if args.split != "all":
        cases = [c for c in cases if c.split == args.split] or cases
    cases = cases[: args.n]
    orders = args.orders.split(",")
    rep = {"meta": {"git": git_commit(), "backend": be.describe(), "args": vars(args),
                    "inicio": time.strftime("%Y-%m-%dT%H:%M:%S")}}

    # 1. template
    mism = []
    for c in cases:
        opts = order_options(c, orders[0])
        ours = render(tpl, c, opts)
        theirs = be.tok.apply_chat_template([{"role": "user", "content": user_message(c, opts)}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        if ours != theirs:
            mism.append(c.id)
    rep["1_template_igual_ao_tokenizer"] = {"ok": not mism, "divergentes": mism[:10]}

    # 2. readout vs forward completo, e 3. padding
    prompts, diffs, argmax_ok = [], [], 0
    for c in cases:
        opts = order_options(c, orders[0])
        ids, cands = candidate_tokens(be.tokenize, render(tpl, c, opts), len(opts))
        prompts.append((ids, cands))
        lp, _ = be.readout(ids, cands)
        full = be.full_forward_last_logprobs(ids)[cands].tolist()
        diffs.append(max(abs(a - b) for a, b in zip(lp, full)))
        argmax_ok += int(max(range(len(lp)), key=lp.__getitem__) == max(range(len(full)), key=full.__getitem__))
    rep["2_readout_vs_forward"] = {"max_abs_dlogprob": max(diffs), "argmax_igual": f"{argmax_ok}/{len(cases)}",
                                   "ok": argmax_ok == len(cases) and max(diffs) < args.tol}
    pad_diffs = []
    for i in range(0, len(prompts), args.batch):
        chunk = prompts[i:i + args.batch]
        batched, _ = be.readout_batch([p[0] for p in chunk], [p[1] for p in chunk])
        for (ids, cands), lb in zip(chunk, batched):
            ls, _ = be.readout(ids, cands)
            pad_diffs.append(max(abs(a - b) for a, b in zip(lb, ls)))
    rep["3_padding_direita_vs_lote_1"] = {"max_abs_dlogprob": max(pad_diffs), "ok": max(pad_diffs) < args.tol}

    # 4. geração
    stops, first_ok, cache_ok, last_tokens = [], 0, 0, set()
    for ids, cands in prompts[: args.n_gen]:
        new, text, info = be.generate(ids, 16)
        stops.append(info["stop_type"])
        last_tokens.add(new[-1] if new else None)
        lp, _ = be.readout(ids, cands)
        first_ok += int(bool(new) and new[0] == cands[max(range(len(lp)), key=lp.__getitem__)])
        nc = be.greedy_no_cache(ids, n_tokens=min(4, len(new)))
        cache_ok += int(nc == new[: len(nc)])
    rep["4_geracao"] = {"paradas": {s: stops.count(s) for s in set(stops)}, "ultimo_token": sorted(map(str, last_tokens)),
                        "eos_aceitos": be.eos_ids, "primeiro_token_igual_readout": f"{first_ok}/{len(stops)}",
                        "cache_igual_sem_cache": f"{cache_ok}/{len(stops)}",
                        "ok": first_ok == len(stops) and cache_ok == len(stops)}

    # 5. concordância com execução de referência
    if args.ref_run:
        ref = {(r["case_id"], r["order"]): r for r in load_rows(Path(args.ref_run))}
        agree, same_tokens, dps, n = 0, 0, [], 0
        rows = []
        for c in cases:
            for o in orders:
                r = predict_one(be, tpl, c, o, generate=False)
                rows.append(r)
                k = (c.id, o)
                if k in ref:
                    n += 1
                    agree += r["pred"] == ref[k]["pred"]
                    same_tokens += r["prompt_sha1"] == ref[k].get("prompt_sha1")
                    dps.append(max(abs(r["probs"][x] - ref[k]["probs"][x]) for x in r["probs"]))
        out = Path(args.out)
        (out / "predicoes_hf.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        rep["5_vs_referencia"] = {"n": n, "decisao_igual": f"{agree}/{n}", "tokens_do_prompt_iguais": f"{same_tokens}/{n}",
                                  "max_abs_dp": max(dps) if dps else None,
                                  "mediana_abs_dp": sorted(dps)[len(dps) // 2] if dps else None}
    rep["memoria_pico_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    return rep


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="lumi-e0")
    p.add_argument("--model", default=SOURCES["lumi"]["repo"])
    p.add_argument("--revision", default=SOURCES["lumi"]["revision"])
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--native", action="store_true", help="usar a implementação nativa do transformers (sem remote code)")
    p.add_argument("--template", default=str(ARTIFACTS_DIR / "lumi" / "chat_template.jinja"))
    p.add_argument("--source", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--orders", default="canonical,reversed")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--n-gen", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--tol", type=float, default=0.05, help="tolerância em log-prob (BF16)")
    p.add_argument("--ref-run", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rep = run(a)
    (Path(a.out) / "e0.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
