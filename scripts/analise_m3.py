"""M3 (analises/reservadas_v2.md, leitura registrada): checkpoints no dev das 4 famílias com professor.

    python scripts/analise_m3.py <pasta>

<pasta> tem `profdev_<nome>_r3` e `profdev_<nome>_r2` (slurm/m3_prof.sbatch) e, para as 11 reservadas, o que a M2
usa (`res2dev_<nome>_r4/_r3`, `factck6_<nome>`, `<nome>_felhen_dev`, `<nome>_telepatia_dev`). Sem esses, as linhas das
11 reservadas saem como "faltam predições".

Pergunta principal: E3b − controle na média com peso igual das 4 com professor (dev, todas as rotações, IC com
famílias fixas reamostrando grupos). "Transfere longe": IC acima de zero nas duas sementes; "atrapalha longe": abaixo
de zero nas duas; qualquer outro caso: inconclusivo.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from lumi_decision import metrics

sys.path.insert(0, str(Path(__file__).parent))
from analise_m1 import COMPARACOES, load  # noqa: E402
from analise_m2 import ATUAIS, NOVAS, dois_estagios, fmt, por_familia_grupos, rows_atuais, rows_novas  # noqa: E402

PROF = ["postura_alvo", "inferencia_pt", "resposta_indireta", "correferencia"]
PRINCIPAL = ["E3b s0 − controle", "E3b s1 − controle (E3b-r)"]
REGISTRADAS = PRINCIPAL + [r for r, _, _ in COMPARACOES if r.startswith(("H2 ", "E3c "))]


def rows_prof(root: Path, n: str) -> list[dict]:
    return load(root / f"profdev_{n}_r3") + load(root / f"profdev_{n}_r2")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(sys.argv[1])
    nomes = sorted(p.name.removeprefix("profdev_").removesuffix("_r3") for p in root.glob("profdev_*_r3"))
    prof = {n: rows_prof(root, n) for n in nomes}
    out = {"checkpoints": {}, "comparacoes": {}, "pergunta_principal": None}
    for n in nomes:
        fam = {}
        for f in PROF:
            rs = [r for r in prof[n] if r["family"] == f]
            preds = defaultdict(set)
            for r in rs:
                preds[r["case_id"]].add(r["pred"])
            fam[f] = {"acertos": f"{sum(r['pred'] == r['gold'] for r in rs)}/{len(rs)}",
                      "muda_com_rotacao": f"{sum(len(v) > 1 for v in preds.values())}/{len(preds)}"}
        out["checkpoints"][n] = fam
    principal = {}
    for rot, a, b in COMPARACOES:
        if rot not in REGISTRADAS:
            continue
        if a not in prof or b not in prof:
            out["comparacoes"][rot] = "faltam predições"
            continue
        ra, rb = prof[a], prof[b]
        res = {f: fmt(metrics.family_mean_bootstrap([r for r in ra if r["family"] == f],
                                                    [r for r in rb if r["family"] == f], require_families=[f]))
               for f in PROF}
        m4 = metrics.family_mean_bootstrap(ra, rb, require_families=PROF)
        res["media_4_professor"] = fmt(m4)
        if rot in PRINCIPAL:
            principal[rot] = m4["ic95"]
        try:
            ta = ra + rows_novas(root, a) + rows_atuais(root, a)
            tb = rb + rows_novas(root, b) + rows_atuais(root, b)
            m11 = metrics.family_mean_bootstrap(ta, tb, require_families=PROF + NOVAS + ATUAIS)
            lo, hi = dois_estagios(por_familia_grupos(ta, tb))
            res["media_11_familias_fixas"] = fmt(m11)
            res["media_11_dois_estagios"] = (f"{100 * m11['delta_media_familias']:+.1f} "
                                             f"[{100 * lo:+.1f}; {100 * hi:+.1f}]")
            pf = m11["por_familia"]
            res["familias_ganha_perde_11"] = (f"{sum(v > 0 for v in pf.values())} ganha, "
                                              f"{sum(v < 0 for v in pf.values())} perde")
        except (FileNotFoundError, ValueError, AssertionError) as err:
            res["media_11"] = f"faltam predições ({type(err).__name__})"
        out["comparacoes"][rot] = res
    if len(principal) == 2:
        if all(lo > 0 for lo, _ in principal.values()):
            out["pergunta_principal"] = "transfere longe (IC acima de zero nas duas sementes)"
        elif all(hi < 0 for _, hi in principal.values()):
            out["pergunta_principal"] = "atrapalha longe (IC abaixo de zero nas duas sementes)"
        else:
            out["pergunta_principal"] = "inconclusivo"
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
