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
    res = {k: resumo(load(p)) for k, p in dirs.items()}
    ids = sorted(res[ref][0])
    out = {}
    for k, (by, acc, flip) in res.items():
        a = np.array([np.mean([x for x, _ in by[c].values()]) for c in ids])
        f = np.array([len({p for _, p in by[c].values()}) > 1 for c in ids], dtype=float)
        a0 = np.array([np.mean([x for x, _ in res[ref][0][c].values()]) for c in ids])
        out[k] = {"acc": round(100 * a.mean(), 1), "muda_ordem": round(100 * f.mean(), 1),
                  f"delta_vs_{ref}": round(100 * (a - a0).mean(), 1), "ic95": boot(a - a0) if k != ref else None}
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
