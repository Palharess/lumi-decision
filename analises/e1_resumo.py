"""Resumo do E1 completo (jobs 36807/36808): Lumi x Nemotron x baselines, por fonte/grupo/família, + calibração.

Uso: python analises/e1_resumo.py <raiz do repo>  (runs/ copiados do cluster; baselines em runs/bl_v2_<fonte>_dev)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / "src"))
from lumi_decision import metrics  # noqa: E402

R = REPO / "runs"
M = {"lumi": "e1_lumi_27b30f4_nativa", "nemo": "e1_nemotron_dfaf35d_nativa"}


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def grp(r):
    return "reservadas" if r["held_out"] else "vistas"


def bal_mean_fam(rs):
    by = defaultdict(list)
    for r in rs:
        by[r["family"]].append(r)
    return float(np.mean([metrics.balanced_accuracy(v) for v in by.values()]))


for src in ("felhen", "telepatia", "traces"):
    print(f"\n################ {src}")
    llm = {k: rows(R / f"{v}_{src}_dev_completo" / "predicoes.jsonl") for k, v in M.items()}
    bl = {}
    for b in ("maioria", "tfidf", "lexical"):
        p = R / f"bl_v2_{src}_dev" / b / "predicoes.jsonl"
        if p.exists():
            bl[b] = rows(p)
    for g in ("vistas", "reservadas"):
        line = []
        for k, rs in llm.items():
            can = [r for r in rs if grp(r) == g and r["order"] == "canonical"]
            allr = [r for r in rs if grp(r) == g]
            if not can:
                continue
            perm = metrics.permutation_stats(allr)
            line.append(f"{k}: acc {metrics.accuracy(can):.3f} bal {bal_mean_fam(can):.3f} "
                        f"(media-ordens {perm.get('acuracia_media_das_ordens', float('nan')):.3f}, "
                        f"todas-ordens {perm.get('acerta_em_todas_as_ordens', float('nan')):.3f})")
        for b, rs in bl.items():
            sel = [r for r in rs if grp(r) == g]
            if sel:
                line.append(f"{b}: acc {metrics.accuracy(sel):.3f} bal {bal_mean_fam(sel):.3f}")
        print(f"[{g}] n_casos={sum(1 for r in llm['lumi'] if grp(r) == g and r['order'] == 'canonical')}")
        for x in line:
            print("   ", x)
        a = [r for r in llm["nemo"] if grp(r) == g]
        b_ = [r for r in llm["lumi"] if grp(r) == g]
        if a:
            pb = metrics.paired_bootstrap(a, b_, n_boot=2000)
            print(f"    lumi - nemotron (2 ordens, bootstrap por grupo): {pb['delta']:+.3f} IC95 [{pb['ic95'][0]:+.3f}, {pb['ic95'][1]:+.3f}] n_grupos={pb['n_grupos']}")
        if "tfidf" in bl:
            t = {r["case_id"]: r for r in bl["tfidf"] if grp(r) == g}
            if t:
                lc = [r for r in llm["lumi"] if grp(r) == g and r["order"] == "canonical" and r["case_id"] in t]
                pb = metrics.paired_bootstrap([t[r["case_id"]] for r in lc], lc, n_boot=2000)
                print(f"    lumi - tfidf (canônica): {pb['delta']:+.3f} IC95 [{pb['ic95'][0]:+.3f}, {pb['ic95'][1]:+.3f}]")
    # por família (canônica)
    print("  por família (acc bal., canônica): família | lumi | nemotron | tfidf | maioria | acaso")
    fams = sorted({(grp(r), r["family"]) for r in llm["lumi"]})
    for g, f in fams:
        vals = []
        for rs in (llm["lumi"], llm["nemo"], bl.get("tfidf", []), bl.get("maioria", [])):
            sel = [r for r in rs if r["family"] == f and r.get("order", "canonical") == "canonical"]
            vals.append(f"{metrics.balanced_accuracy(sel):.3f}" if sel else "  -  ")
        sel = [r for r in llm["lumi"] if r["family"] == f and r["order"] == "canonical"]
        print(f"   {g[:4]} {f:32s} n={len(sel):4d} | " + " | ".join(vals) + f" | {np.mean([1/len(r['probs']) for r in sel]):.3f}")
    for k, v in M.items():
        c = json.load(open(R / f"{v}_{src}_dev_completo" / "calibracao.json", encoding="utf-8"))
        out = [f"T={c['temperatura']:.2f} limiar(10%)={c['limiar']}"]
        for g in ("vistas", "reservadas"):
            key = f"{src}|{g}"
            if key in c:
                s = c[key]["abstencao_com_limiar_fixo"]
                out.append(f"{g}: ECE {c[key]['ece_antes']:.3f}->{c[key]['ece_depois']:.3f} NLL {c[key]['nll_antes']:.3f}->{c[key]['nll_depois']:.3f} "
                           f"cob {s['cobertura']:.2f} risco {s['risco_aceitos']} vs acaso {s['risco_aleatorio']:.3f}")
        print(f"  calibração {k}: " + " | ".join(out))
    # limiares alternativos (fixados no calib, aplicados no dev) só como diagnóstico
    for k, v in M.items():
        cal = rows(R / f"{v}_{src}_calib_completo" / "predicoes.jsonl")
        c = json.load(open(R / f"{v}_{src}_dev_completo" / "calibracao.json", encoding="utf-8"))
        calT = metrics.with_temperature(cal, c["temperatura"])
        dev = metrics.with_temperature(llm[k], c["temperatura"])
        out = []
        for alvo in (0.10, 0.15, 0.20):
            tau = metrics.threshold_for_risk(calT, alvo)
            for g in ("vistas", "reservadas"):
                sel = [r for r in dev if grp(r) == g]
                if sel:
                    s = metrics.selective_at(sel, tau)
                    rk = "-" if s["risco_aceitos"] is None else f"{s['risco_aceitos']:.3f}"
                    out.append(f"alvo {alvo:.2f} {g[:4]}: cob {s['cobertura']:.2f} risco {rk} (acaso {s['risco_aleatorio']:.3f})")
        print(f"  abstenção {k}: " + "; ".join(out))
