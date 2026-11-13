"""F0c: anatomia da distância entre famílias vistas e reservadas (E3 contra o Lumi congelado do E1).

Por pergunta (família + chave da pergunta): acurácia média das duas ordens no A0 e no A1, ganho, tipo, nº de
opções, e se o treino tem exemplos do mesmo tipo e do mesmo nº de opções. Lê predições já salvas.

    python scripts/analise_lacuna.py --e1 runs/e1_lumi_27b30f4_nativa --e3 <pasta com e3_lora_r16_*_dev>
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from lumi_decision import lora


def load(p):
    return [json.loads(l) for l in open(Path(p) / "predicoes.jsonl", encoding="utf-8")]


def pergunta(r):
    if r["source"] == "telepatia":
        return f"{r['family']}:{r['case_id'].rsplit(':', 1)[-1]}"
    return r["family"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True)
    ap.add_argument("--e3", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    train = lora.build_training_set()
    tr_tipo_n = Counter((c.qtype, len(c.options)) for c in train)
    tr_tipo = Counter(c.qtype for c in train)
    linhas = []
    for src in ("felhen", "telepatia", "traces"):
        base = load(f"{a.e1}_{src}_dev_completo")
        novo = load(Path(a.e3) / f"e3_lora_r16_{src}_dev")
        acc = {}
        for nome, rows in (("A0", base), ("A1", novo)):
            by = defaultdict(list)
            for r in rows:
                by[pergunta(r)].append(r)
            acc[nome] = {k: (float(np.mean([x["pred"] == x["gold"] for x in v])), v) for k, v in by.items()}
        for k, (a1, rows) in acc["A1"].items():
            a0 = acc["A0"][k][0]
            r0 = rows[0]
            n = len(r0["presented"])
            maj = Counter(x["gold"] for x in rows).most_common(1)[0][1] / len(rows)
            linhas.append({"fonte": src, "pergunta": k, "reservada": r0["held_out"], "tipo": r0["qtype"], "n_opcoes": n,
                           "n_linhas": len(rows), "maioria": maj, "A0": a0, "A1": a1, "ganho": a1 - a0,
                           "treino_mesmo_tipo_e_n": tr_tipo_n.get((r0["qtype"], n), 0),
                           "treino_mesmo_tipo": tr_tipo.get(r0["qtype"], 0)})
    linhas.sort(key=lambda x: (x["reservada"], x["fonte"], x["pergunta"]))
    print(f"{'pergunta':42s} {'res':>3s} {'tipo':>6s} {'n':>3s} {'maior':>6s} {'A0':>6s} {'A1':>6s} {'ganho':>6s} {'treino(tipo,n)':>14s}")
    for x in linhas:
        print(f"{x['fonte'][:4]}:{x['pergunta'][:37]:37s} {'R' if x['reservada'] else 'V':>3s} {x['tipo']:>6s} {x['n_opcoes']:>3d} "
              f"{x['maioria']:6.2f} {x['A0']:6.3f} {x['A1']:6.3f} {x['ganho']:+6.3f} {x['treino_mesmo_tipo_e_n']:>14d}")
    if a.out:
        Path(a.out).write_text(json.dumps(linhas, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
