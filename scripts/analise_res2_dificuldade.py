"""Filtro de dificuldade das reservadas v2 (analises/reservadas_v2.md): só o Lumi congelado, fatia `dificuldade`.

    python scripts/analise_res2_dificuldade.py <pasta com res2dif_<família>>

Regra registrada: a família fica se acertos/respostas estiver entre o acaso + 5 pontos e 90%.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(sys.argv[1])
    out = {}
    for d in sorted(root.glob("res2dif_*")):
        fam = d.name.removeprefix("res2dif_")
        rows = [json.loads(x) for x in (d / "predicoes.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        n_op = {len(r["presented"]) for r in rows}
        assert len(n_op) == 1, n_op
        acaso = 1 / n_op.pop()
        hit = sum(r["pred"] == r["gold"] for r in rows)
        acc = hit / len(rows)
        por_ordem, por_classe, pos = Counter(), defaultdict(lambda: [0, 0]), Counter()
        preds = defaultdict(set)
        for r in rows:
            por_ordem[r["order"]] += r["pred"] == r["gold"]
            por_classe[r["gold"]][0] += r["pred"] == r["gold"]
            por_classe[r["gold"]][1] += 1
            pos["ABCD"[r["presented"].index(r["pred"])]] += 1
            preds[r["case_id"]].add(r["pred"])
        n_ord = len(rows) // len(preds)
        fica = acaso + 0.05 <= acc <= 0.90
        out[fam] = {
            "acertos": f"{hit}/{len(rows)}", "acc": round(acc, 3), "acaso": round(acaso, 3),
            "faixa": [round(acaso + 0.05, 3), 0.90], "veredito": "fica" if fica else "falha",
            "por_rotacao": {o: f"{v}/{len(preds)}" for o, v in sorted(por_ordem.items())},
            "posicao_escolhida": dict(sorted(pos.items())),
            "muda_com_a_rotacao": f"{sum(len(v) > 1 for v in preds.values())}/{len(preds)}",
            "por_classe": ({k: f"{a}/{b}" for k, (a, b) in sorted(por_classe.items())}
                           if fam == "regra_beneficio" else None),
            "rotacoes": n_ord,
        }
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
