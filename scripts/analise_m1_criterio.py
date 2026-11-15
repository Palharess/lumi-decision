"""M1, leitura exploratória: o critério principal (3 reservadas, peso igual) com a regra nova (factck nas
6 permutações; juristcu e invoice nas 2 ordens) aplicada aos pares já julgados, e a deriva numérica das
respostas em perm012/perm210 contra as avaliações originais.

    python scripts/analise_m1_criterio.py <pasta com factck6_*, <nome>_felhen_dev e <nome>_telepatia_dev>
"""
import json
import sys
from pathlib import Path

from lumi_decision import metrics

sys.path.insert(0, str(Path(__file__).parent))
from analise_m1 import COMPARACOES, DUAS, FAM, ORIGINAL, load  # noqa: E402

RES = ["factck_veracidade", "juristcu_area", "invoice_processing"]


def orig(root, n, src):
    base = ORIGINAL.get(n, f"{n}_felhen_dev").replace("_felhen_", f"_{src}_")
    return load(root / base)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(sys.argv[1])
    rows, deriva = {}, {}
    for p in sorted(root.glob("factck6_*")):
        n = p.name.removeprefix("factck6_")
        f6 = load(p)
        o = [r for s in ("felhen", "telepatia") for r in orig(root, n, s) if r["held_out"] and r["source"] != "traces"]
        rows[n] = (f6, [r for r in o if r["family"] != FAM], [r for r in o if r["family"] == FAM])
        ob = {(r["case_id"], r["order"]): r for r in rows[n][2]}
        mud = []
        for r in f6:
            if r["order"] in DUAS:
                x = ob[(r["case_id"], DUAS[r["order"]])]
                if x["pred"] != r["pred"]:
                    pr = sorted(x["probs"].values(), reverse=True)
                    mud.append(round(pr[0] - pr[1], 3))
        deriva[n] = mud
    print("respostas que mudaram em perm012/perm210 (margem top1-top2 na avaliação original):")
    for n, m in deriva.items():
        if m:
            print(f"  {n}: {len(m)} {sorted(m)}")
    print("\ncritério principal: original (2 ordens) | regra nova (factck em 6)")
    for rot, a, b in COMPARACOES:
        if rot.startswith("semente"):
            continue
        res = []
        for novo in (False, True):
            fa = rows[a][0] if novo else rows[a][2]
            fb = rows[b][0] if novo else rows[b][2]
            m = metrics.family_mean_bootstrap(rows[a][1] + fa, rows[b][1] + fb, require_families=RES)
            pf = m["por_familia"]
            res.append(f"{100 * m['delta_media_familias']:+.1f} [{100 * m['ic95'][0]:+.1f}; {100 * m['ic95'][1]:+.1f}]"
                       f" (f {100 * pf[RES[0]]:+.1f}, j {100 * pf[RES[1]]:+.1f}, i {100 * pf[RES[2]]:+.1f})")
        print(f"{rot:32s} {res[0]} | {res[1]}")


if __name__ == "__main__":
    main()
