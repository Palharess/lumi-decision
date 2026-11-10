"""Execução e relatório.

    lumi-eval run --backend llamacpp --model X.gguf --server-bin llama-server.exe \\
        --source felhen --split dev --orders canonical,reversed --out runs/h1_gguf
    lumi-eval report runs/h1_gguf
    lumi-eval baselines --source felhen --split dev --out runs/baselines_felhen_dev
    lumi-eval comparar runs/A runs/B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import REPO_ROOT, metrics
from .data import ARTIFACTS_DIR, check_integrity, load_cases, load_config
from .prompts import LETTERS, PROMPT_VERSION, candidate_tokens, load_template, order_options, render

GEN_RE = re.compile(r"\s*(?:\*\*)?([A-Z])(?:\*\*)?[.)]?\s*")


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True)
        return (out.stdout.strip() or "sem-commit") + ("+alteracoes" if dirty.stdout.strip() else "")
    except OSError:
        return None


def select_cases(args) -> list:
    cases = load_cases(args.source, load_config())
    check_integrity(cases)
    cases = [c for c in cases if c.split == args.split] if args.split != "all" else cases
    if args.families:
        fams = set(args.families.split(","))
        cases = [c for c in cases if c.family in fams]
    if args.held_out == "exclude":
        cases = [c for c in cases if not c.held_out]
    elif args.held_out == "only":
        cases = [c for c in cases if c.held_out]
    if args.per_family:
        by = defaultdict(list)
        for c in cases:
            by[c.family].append(c)
        cases = []
        for fam in sorted(by):
            items = sorted(by[fam], key=lambda c: hashlib.sha256(f"{args.seed}:{c.id}".encode()).hexdigest())
            cases.extend(items[: args.per_family])
    if args.limit:
        cases = cases[: args.limit]
    return cases


def predict_one(backend, template, case, order, generate=True, max_new_tokens=16) -> dict:
    options = order_options(case, order)
    prompt = render(template, case, options)
    ids, cands = candidate_tokens(backend.tokenize, prompt, len(options))
    lps, rinfo = backend.readout(ids, cands)
    m = max(lps)
    w = [math.exp(x - m) for x in lps]
    z = sum(w)
    probs = {options[i][0]: w[i] / z for i in range(len(options))}
    row = {
        "case_id": case.id, "group_id": case.group_id, "source": case.source, "family": case.family,
        "split": case.split, "held_out": case.held_out, "qtype": case.qtype, "order": order,
        "presented": [o[0] for o in options], "gold": case.gold, "probs": probs,
        "pred": max(probs, key=probs.get), "cand_mass": float(sum(math.exp(x) for x in lps)),
        "prompt_tokens": len(ids), "prompt_sha1": hashlib.sha1(json.dumps(ids).encode()).hexdigest(),
        "readout_ms": rinfo.get("latency_ms"),
    }
    if case.gold_dist:
        row["gold_dist"] = case.gold_dist
    if generate:
        toks, text, ginfo = backend.generate(ids, max_new_tokens)
        mm = GEN_RE.fullmatch(text)
        letter = mm[1] if mm and LETTERS.index(mm[1]) < len(options) else None
        row.update({
            "gen_text": text, "gen_token_ids": toks, "gen_stop": ginfo.get("stop_type"),
            "gen_pred": options[LETTERS.index(letter)][0] if letter else None,
            "gen_first_is_readout_argmax": bool(toks) and toks[0] == cands[int(np.argmax(lps))],
            "gen_ms": ginfo.get("latency_ms"),
        })
    return row


def cmd_run(args):
    from .backends import make_backend

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = select_cases(args)
    orders = args.orders.split(",")
    template = load_template(Path(args.template))
    kw = {k: v for k, v in {"model": args.model, "server_bin": args.server_bin, "url": args.url,
                            "revision": args.revision, "dtype": args.dtype}.items() if v is not None}
    if args.backend == "llamacpp":
        kw.pop("revision", None), kw.pop("dtype", None)
        kw["log_path"] = str(out / "llama_server.log")
        kw["ctx"] = args.ctx
    else:
        kw["trust_remote_code"] = not args.native
    backend = make_backend(args.backend, **kw)
    pred_path = out / "predicoes.jsonl"
    done = set()
    if pred_path.exists() and not args.overwrite:
        done = {(r["case_id"], r["order"]) for r in map(json.loads, pred_path.read_text(encoding="utf-8").splitlines())}
    meta = {
        "args": {k: v for k, v in vars(args).items() if k != "func"}, "prompt_version": PROMPT_VERSION, "git": git_commit(), "inicio": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "maquina": platform.node(), "plataforma": platform.platform(), "backend": backend.describe(),
        "n_casos": len(cases), "familias": dict(Counter(c.family for c in cases)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    t0 = time.time()
    try:
        with pred_path.open("w" if args.overwrite else "a", encoding="utf-8") as f:
            for i, case in enumerate(cases):
                for order in orders:
                    if (case.id, order) in done:
                        continue
                    row = predict_one(backend, template, case, order, generate=not args.no_generate)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                if (i + 1) % 25 == 0 or i + 1 == len(cases):
                    print(f"{i + 1}/{len(cases)} casos  {time.time() - t0:.0f}s", flush=True)
    finally:
        if hasattr(backend, "close"):
            backend.close()
    cmd_report(argparse.Namespace(run=str(out)))


def load_rows(run: Path) -> list[dict]:
    return [json.loads(x) for x in (run / "predicoes.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def _latency(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))} if v else None


def build_report(rows) -> dict:
    rep = {"geral": {}, "por_familia": {}, "permutacao": {}, "latencia_ms": {}}
    by_src = defaultdict(list)
    for r in rows:
        by_src[(r["source"], r["held_out"])].append(r)
    for (src, held), rs in sorted(by_src.items()):
        key = f"{src}|{'reservadas' if held else 'vistas'}"
        fams = defaultdict(list)
        for r in rs:
            fams[r["family"]].append(r)
        per_fam = {f: metrics.summarize(v) for f, v in sorted(fams.items())}
        rep["por_familia"][key] = per_fam
        rep["geral"][key] = {
            **metrics.summarize(rs),
            "acuracia_balanceada_media_familias": float(np.mean([m["acuracia_balanceada"] for m in per_fam.values()])),
        }
        rep["permutacao"][key] = metrics.permutation_stats(rs)
        rep["latencia_ms"][key] = {"readout": _latency(rs, "readout_ms"), "geracao": _latency(rs, "gen_ms")}
    return rep


def cmd_report(args):
    run = Path(args.run)
    rep = build_report(load_rows(run))
    (run / "relatorio.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["| grupo | n | acc | acc bal. média fam. | NLL | ECE | AURC | risco@cob80 | muda c/ ordem | geração = readout |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, g in rep["geral"].items():
        p = rep["permutacao"].get(k) or {}
        lines.append(f"| {k} | {g['n']} | {g['acuracia']:.3f} | {g['acuracia_balanceada_media_familias']:.3f} | "
                     f"{g['nll']:.3f} | {g['ece']:.3f} | {g['aurc']:.3f} | {g['risco@cob80']:.3f} | "
                     f"{p.get('decisao_muda_entre_ordens', float('nan')):.3f} | "
                     f"{g.get('geracao_concorda_readout', float('nan')):.3f} |")
    lines += ["", "| grupo | família | n | acc | acc bal. | acaso |", "|---|---|---:|---:|---:|---:|"]
    for k, fams in rep["por_familia"].items():
        for f, m in fams.items():
            lines.append(f"| {k} | {f} | {m['n']} | {m['acuracia']:.3f} | {m['acuracia_balanceada']:.3f} | {m['acaso']:.3f} |")
    (run / "relatorio.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_baselines(args):
    """Maioria (por família, a partir do treino) e TF-IDF supervisionado (só famílias vistas)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    cases = load_cases(args.source, load_config())
    check_integrity(cases)
    train = defaultdict(list)
    for c in cases:
        if c.split == "train" and not c.held_out:
            train[c.family].append(c)
    evalset = [c for c in cases if c.split == args.split]
    out = Path(args.out)
    for name in ("maioria", "tfidf"):
        rows = []
        for fam in sorted({c.family for c in evalset}):
            items = [c for c in evalset if c.family == fam]
            tr = train.get(fam, [])
            if not tr:
                continue  # família reservada: referência supervisionada não se aplica
            if name == "maioria":
                top = Counter(c.gold for c in tr).most_common(1)[0][0]
                probas = [{o[0]: float(o[0] == top) for o in c.options} for c in items]
            else:
                clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                                    max_features=200000, sublinear_tf=True),
                                    LogisticRegression(max_iter=1000))
                clf.fit([f"{c.question}\n{c.state}" for c in tr], [c.gold for c in tr])
                P = clf.predict_proba([f"{c.question}\n{c.state}" for c in items])
                probas = [{o[0]: (float(P[i][list(clf.classes_).index(o[0])]) if o[0] in clf.classes_ else 0.0)
                           for o in c.options} for i, c in enumerate(items)]
            for c, p in zip(items, probas):
                s = sum(p.values()) or 1.0
                p = {k: v / s for k, v in p.items()}
                rows.append({"case_id": c.id, "group_id": c.group_id, "source": c.source, "family": c.family,
                             "split": c.split, "held_out": c.held_out, "qtype": c.qtype, "order": "canonical",
                             "gold": c.gold, "probs": p, "pred": max(p, key=p.get),
                             **({"gold_dist": c.gold_dist} if c.gold_dist else {})})
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "predicoes.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        (d / "meta.json").write_text(json.dumps({"baseline": name, "args": {k: v for k, v in vars(args).items() if k != "func"},
                                                "git": git_commit()}, indent=2), encoding="utf-8")
        print(f"\n## {name}")
        cmd_report(argparse.Namespace(run=str(d)))


def cmd_comparar(args):
    a, b = load_rows(Path(args.a)), load_rows(Path(args.b))
    groups = defaultdict(lambda: ([], []))
    for r in a:
        groups[(r["source"], r["held_out"])][0].append(r)
    for r in b:
        groups[(r["source"], r["held_out"])][1].append(r)
    res = {f"{s}|{'reservadas' if h else 'vistas'}": metrics.paired_bootstrap(ra, rb)
           for (s, h), (ra, rb) in sorted(groups.items()) if ra and rb}
    print(json.dumps(res, indent=2, ensure_ascii=False))


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="lumi-eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_selection(sp):
        sp.add_argument("--source", required=True, help="felhen | telepatia | traces | caminho de arquivo de casos")
        sp.add_argument("--split", default="dev", help="train | dev | calib | test | all")
        sp.add_argument("--families", default=None)
        sp.add_argument("--held-out", choices=("include", "exclude", "only"), default="include")
        sp.add_argument("--per-family", type=int, default=0, help="amostra determinística por família")
        sp.add_argument("--limit", type=int, default=0)
        sp.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("run")
    add_selection(r)
    r.add_argument("--backend", choices=("llamacpp", "hf"), required=True)
    r.add_argument("--model", default=None)
    r.add_argument("--server-bin", default=None)
    r.add_argument("--url", default=None)
    r.add_argument("--revision", default=None)
    r.add_argument("--dtype", default=None)
    r.add_argument("--native", action="store_true", help="HF: implementação nativa do transformers")
    r.add_argument("--ctx", type=int, default=8192)
    r.add_argument("--template", default=str(ARTIFACTS_DIR / "lumi" / "chat_template.jinja"))
    r.add_argument("--orders", default="canonical,reversed")
    r.add_argument("--no-generate", action="store_true")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report")
    rp.add_argument("run")
    rp.set_defaults(func=cmd_report)

    b = sub.add_parser("baselines")
    b.add_argument("--source", required=True)
    b.add_argument("--split", default="dev")
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_baselines)

    c = sub.add_parser("comparar")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(func=cmd_comparar)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
