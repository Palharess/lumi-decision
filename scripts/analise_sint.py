"""Controles do piloto sintético do E5: acertos/total do Lumi congelado por família e condição, todas as rotações.

    python3 scripts/analise_sint.py --sufixo _teste2    (lê runs/sintdif_<condição><sufixo>/predicoes.jsonl)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

CONDICOES = ("completo", "sem_texto", "so_versao_2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sufixo", required=True)
    a = ap.parse_args()
    out = {}
    for cond in CONDICOES:
        p = Path(f"runs/sintdif_{cond}{a.sufixo}/predicoes.jsonl")
        if not p.exists():
            continue
        n, k, acaso = defaultdict(int), defaultdict(int), defaultdict(float)
        for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()):
            n[r["family"]] += 1
            k[r["family"]] += r["pred"] == r["gold"]
            acaso[r["family"]] += 1 / len(r["presented"])
        out[cond] = {f: {"acertos": f"{k[f]}/{n[f]}", "acc": round(k[f] / n[f], 3),
                         "acaso": round(acaso[f] / n[f], 3)} for f in sorted(n)}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    dest = Path(f"analises/sint_controles{a.sufixo}.json")
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
