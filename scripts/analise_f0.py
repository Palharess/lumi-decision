"""Resumo do F0a (juristcu/invoice em várias ordens) e do F0b (ENEM), com peso igual por ORDEM ÚNICA.

    python scripts/analise_f0.py <pasta do F0a> <pasta com f0_enem_*>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(p):
    return [json.loads(l) for l in open(Path(p) / "predicoes.jsonl", encoding="utf-8")]


def per_case(rows):
    """{case_id: {sequência apresentada: acerto}}: ordens que repetem a mesma sequência contam uma vez."""
    by = defaultdict(dict)
    for r in rows:
        by[r["case_id"]][tuple(r["presented"])] = (r["pred"] == r["gold"], r["pred"])
    return by


def resumo(rows):
    by = per_case(rows)
    acc = np.array([np.mean([a for a, _ in v.values()]) for v in by.values()])
    flip = np.array([len({p for _, p in v.values()}) > 1 for v in by.values()])
    return by, acc, flip


def boot(d, n=4000):
    rng = np.random.default_rng(0)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [round(100 * x, 1) for x in np.percentile(b, [2.5, 97.5])]


def compare(dirs, ref="A0"):
    """Acertos/total (ordens únicas) e diferença contra `ref` na MESMA base de contagem; IC reamostrando casos
    (soma de acertos / soma de respostas)."""
    res = {k: per_case(load(p)) for k, p in dirs.items()}
    ids = sorted(res[ref])
    out = {}
    for k, by in res.items():
        ok = np.array([sum(a for a, _ in by[c].values()) for c in ids], dtype=float)
        tot = np.array([len(by[c]) for c in ids], dtype=float)
        ok0 = np.array([sum(a for a, _ in res[ref][c].values()) for c in ids], dtype=float)
        flips = int(sum(len({p for _, p in by[c].values()}) > 1 for c in ids))
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(4000):
            s = rng.integers(0, len(ids), len(ids))
            bs.append((ok[s].sum() - ok0[s].sum()) / tot[s].sum())
        out[k] = {"acertos": int(ok.sum()), "respostas": int(tot.sum()), "acc": round(100 * ok.sum() / tot.sum(), 1),
                  "casos": len(ids), "mudam_com_ordem": flips,
                  f"delta_vs_{ref}": round(100 * (ok.sum() - ok0.sum()) / tot.sum(), 1),
                  "ic95": [round(100 * x, 1) for x in np.percentile(bs, [2.5, 97.5])] if k != ref else None}
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    f0a, f0b = Path(sys.argv[1]), Path(sys.argv[2])
    rep = {}
    for fam in ("juristcu_area", "invoice_processing"):
        rep[fam] = compare({k: f0a / f"{k}_{fam}" for k in ("A0", "A1", "E4letras", "B")})
        if fam == "invoice_processing":  # por pergunta
            for q in ("discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"):
                sub = {}
                for k in ("A0", "A1", "E4letras", "B"):
                    rows = [r for r in load(f0a / f"{k}_{fam}") if r["case_id"].endswith(":" + q)]
                    _, acc, flip = resumo(rows)
                    sub[k] = (round(100 * acc.mean(), 1), round(100 * flip.mean(), 1))
                rep[f"{fam}:{q}"] = sub
    rep["enem"] = compare({k: f0b / f"f0_enem_{k}" for k in ("A0", "A1", "B", "E4letras")})
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
