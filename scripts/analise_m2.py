"""M2 (analises/reservadas_v2.md): checkpoints existentes no dev das reservadas v2, e o critério com 7 reservadas.

    python scripts/analise_m2.py <pasta>

<pasta> tem `res2dev_<nome>_r4` e `res2dev_<nome>_r3` (slurm/m2_res2.sbatch) e, para as 3 reservadas atuais,
o que a M1 usa: `factck6_<nome>`, `<nome>_felhen_dev` e `<nome>_telepatia_dev`.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from lumi_decision import metrics

sys.path.insert(0, str(Path(__file__).parent))
from analise_m1 import COMPARACOES, ORIGINAL, load  # noqa: E402

NOVAS = ["restricoes_oferta", "regra_beneficio", "custo_plano", "ordem_eventos"]
ATUAIS = ["factck_veracidade", "juristcu_area", "invoice_processing"]


def rows_novas(root: Path, n: str) -> list[dict]:
    return load(root / f"res2dev_{n}_r4") + load(root / f"res2dev_{n}_r3")


def rows_atuais(root: Path, n: str) -> list[dict]:
    """factck nas 6 permutações (regra da M1); juristcu e invoice nas 2 ordens das avaliações originais."""
    out = list(load(root / f"factck6_{n}"))
    for src, fam in (("felhen", "juristcu_area"), ("telepatia", "invoice_processing")):
        base = ORIGINAL.get(n, f"{n}_felhen_dev").replace("_felhen_", f"_{src}_")
        out += [r for r in load(root / base) if r["family"] == fam]
    return out


def por_familia_grupos(a: list[dict], b: list[dict]) -> dict[str, np.ndarray]:
    """{família: [[soma das diferenças do grupo, nº de respostas], ...]}, pareado por (caso, ordem)."""
    ia = {(r["case_id"], r["order"]): r for r in a}
    ib = {(r["case_id"], r["order"]): r for r in b}
    assert set(ia) == set(ib), "casos/ordens diferentes"
    g = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for k, x in ia.items():
        y = ib[k]
        cell = g[x["family"]][x["group_id"]]
        cell[0] += float(y["pred"] == y["gold"]) - float(x["pred"] == x["gold"])
        cell[1] += 1
    return {f: np.array(list(v.values())) for f, v in g.items()}


def dois_estagios(arr: dict[str, np.ndarray], n_boot: int = 2000, seed: int = 0) -> list[float]:
    """IC da média sobre famílias tratando a família como amostra: reamostra famílias, depois grupos."""
    rng = np.random.default_rng(seed)
    fams = list(arr)
    boots = []
    for _ in range(n_boot):
        ds = []
        for f in rng.choice(fams, len(fams)):
            x = arr[f][rng.integers(0, len(arr[f]), len(arr[f]))]
            ds.append(x[:, 0].sum() / x[:, 1].sum())
        boots.append(np.mean(ds))
    return [float(v) for v in np.percentile(boots, [2.5, 97.5])]


def fmt(m: dict) -> str:
    return f"{100 * m['delta_media_familias']:+.1f} [{100 * m['ic95'][0]:+.1f}; {100 * m['ic95'][1]:+.1f}]"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(sys.argv[1])
    nomes = sorted(p.name.removeprefix("res2dev_").removesuffix("_r4") for p in root.glob("res2dev_*_r4"))
    novas = {n: rows_novas(root, n) for n in nomes}
    out = {"checkpoints": {}, "comparacoes": {}}
    for n in nomes:
        fam = {}
        for f in NOVAS:
            rs = [r for r in novas[n] if r["family"] == f]
            preds = defaultdict(set)
            for r in rs:
                preds[r["case_id"]].add(r["pred"])
            item = {"acertos": f"{sum(r['pred'] == r['gold'] for r in rs)}/{len(rs)}",
                    "muda_com_rotacao": f"{sum(len(v) > 1 for v in preds.values())}/{len(preds)}"}
            if f == "regra_beneficio":
                h, t = Counter(), Counter()
                for r in rs:
                    t[r["gold"]] += 1
                    h[r["gold"]] += r["pred"] == r["gold"]
                item["por_classe"] = {k: f"{h[k]}/{t[k]}" for k in sorted(t)}
            fam[f] = item
        out["checkpoints"][n] = fam
    for rot, a, b in COMPARACOES:
        if a not in novas or b not in novas:
            out["comparacoes"][rot] = "faltam predições"
            continue
        ra, rb = novas[a], novas[b]
        res = {f: fmt(metrics.family_mean_bootstrap([r for r in ra if r["family"] == f],
                                                    [r for r in rb if r["family"] == f], require_families=[f]))
               for f in NOVAS}
        res["media_4_novas"] = fmt(metrics.family_mean_bootstrap(ra, rb, require_families=NOVAS))
        ta, tb = ra + rows_atuais(root, a), rb + rows_atuais(root, b)
        m7 = metrics.family_mean_bootstrap(ta, tb, require_families=NOVAS + ATUAIS)
        res["media_7_familias_fixas"] = fmt(m7)
        arr = por_familia_grupos(ta, tb)
        lo, hi = dois_estagios(arr)
        res["media_7_dois_estagios"] = f"{100 * m7['delta_media_familias']:+.1f} [{100 * lo:+.1f}; {100 * hi:+.1f}]"
        pf = m7["por_familia"]
        res["familias_ganha_perde"] = f"{sum(v > 0 for v in pf.values())} ganha, {sum(v < 0 for v in pf.values())} perde"
        res["por_familia_7"] = {f: round(100 * v, 1) for f, v in pf.items()}
        out["comparacoes"][rot] = res
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
