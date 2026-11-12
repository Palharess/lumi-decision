"""Análise do E1 (H1): Lumi x Nemotron x referências no dev, por fonte e grupo (vistas/reservadas).

Lê as pastas de runs/ produzidas no cluster (copiadas para cá) e as referências de `lumi-eval baselines`.
Calibração: temperatura e limiar ajustados SÓ no calib (famílias vistas) e aplicados fixos no dev.

    python scripts/analise_e1.py [--escopo completo] [--json runs/analise_e1.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lumi_decision import metrics
from lumi_decision.evaluate import load_rows

MODELOS = {"lumi": "e1_lumi_27b30f4_nativa", "nemotron": "e1_nemotron_dfaf35d_nativa"}
FONTES = ("felhen", "telepatia", "traces")
REFS = ("maioria", "tfidf", "lexical")


def grupo(r):
    return "reservadas" if r.get("held_out") else "vistas"


def fam_bal(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["family"]].append(r)
    return float(np.mean([metrics.balanced_accuracy(v) for v in by.values()])) if by else float("nan")


def bloco(rows):
    if not rows:
        return {}
    rc = metrics.risk_coverage(rows, (0.10, 0.20))
    return {"n": len(rows), "acc": metrics.accuracy(rows), "acc_bal_fam": fam_bal(rows), "nll": metrics.nll(rows),
            "ece": metrics.ece(rows), "aurc": rc["aurc"], "cob@r10": rc["cobertura@risco10"],
            "cob@r20": rc["cobertura@risco20"], "risco_global": rc["risco_global"]}


def ensemble_ordens(rows):
    """Média das PROBABILIDADES entre as ordens de cada caso, depois o argmax (um "ensemble" de ordens).

    Não é a métrica principal: a principal é a média das acurácias por ordem (metrics.order_mean_accuracy).
    """
    by = defaultdict(list)
    for r in rows:
        by[r["case_id"]].append(r)
    out = []
    for v in by.values():
        if len(v) < 2:
            continue
        keys = v[0]["probs"].keys()
        p = {k: float(np.mean([x["probs"][k] for x in v])) for k in keys}
        base = next((x for x in v if x["order"] == "canonical"), v[0])
        out.append({**base, "order": "ensemble", "probs": p, "pred": max(p, key=p.get)})
    return out


def calib_fixa(calib_rows, dev_rows, risco):
    T = metrics.fit_temperature(calib_rows)
    cal = metrics.with_temperature(calib_rows, T)
    tau = metrics.threshold_for_risk(cal, risco)
    res = {"T": T, "limiar": tau}
    for g in ("vistas", "reservadas"):
        d = metrics.with_temperature([r for r in dev_rows if grupo(r) == g], T)
        if d:
            res[g] = metrics.selective_at(d, tau)
            res[g]["ece_T"] = metrics.ece(d)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--escopo", default="completo")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    R = Path(a.runs)
    out = {}
    for src in FONTES:
        o = out[src] = {}
        dev = {m: load_rows(R / f"{p}_{src}_dev_{a.escopo}") for m, p in MODELOS.items()}
        cal = {m: load_rows(R / f"{p}_{src}_calib_{a.escopo}") for m, p in MODELOS.items()}
        refs = {b: load_rows(R / f"bl_e1_{src}_dev" / b) for b in REFS}
        ids = {r["case_id"] for r in dev["lumi"]}
        for g in ("vistas", "reservadas"):
            og = o[g] = {}
            for m, rows in dev.items():
                rg = [r for r in rows if grupo(r) == g]
                can = [r for r in rg if r["order"] == "canonical"]
                og[m] = {"acc_media_das_ordens": metrics.order_mean_accuracy(rg), "todas_as_ordens": bloco(rg),
                         "canonica": bloco(can), "ensemble_prob_ordens": bloco(ensemble_ordens(rg)),
                         "ordem": metrics.permutation_stats(rg)}
                gen = [r for r in can if "gen_pred" in r]
                if gen:
                    og[m]["geracao_acc"] = float(np.mean([r["gen_pred"] == r["gold"] for r in gen]))
                    og[m]["geracao_invalida"] = float(np.mean([r["gen_pred"] is None for r in gen]))
            for b, rows in refs.items():
                og[b] = {"canonica": bloco([r for r in rows if grupo(r) == g and r["case_id"] in ids])}
            can = {m: [r for r in dev[m] if grupo(r) == g and r["order"] == "canonical"] for m in dev}
            if can["lumi"]:
                todas = {m: [r for r in dev[m] if grupo(r) == g] for m in dev}
                og["lumi_menos_nemotron"] = metrics.paired_bootstrap(todas["nemotron"], todas["lumi"])
                og["lumi_menos_nemotron_canonica"] = metrics.paired_bootstrap(can["nemotron"], can["lumi"])
                tf = [dict(r, order="canonical") for r in refs["tfidf"] if grupo(r) == g]
                if tf:
                    og["lumi_menos_tfidf"] = metrics.paired_bootstrap(tf, can["lumi"])
            # por família (canônica, acc balanceada)
            fams = defaultdict(dict)
            for m in list(dev) + ["tfidf"]:
                rows = dev[m] if m in dev else refs[m]
                by = defaultdict(list)
                for r in rows:
                    if grupo(r) == g and r["order"] == "canonical" and r["case_id"] in ids:
                        by[r["family"]].append(r)
                for f, v in by.items():
                    fams[f][m] = round(metrics.balanced_accuracy(v), 3)
                    fams[f]["n"] = len(v)
            og["familias"] = dict(fams)
        for m in dev:
            calib_seen = [r for r in cal[m] if grupo(r) == "vistas"]
            o[f"calibracao_{m}"] = {f"risco{int(k * 100)}": calib_fixa(calib_seen, dev[m], k) for k in (0.10, 0.20)}
    # Forma do critério da H2 (média com peso igual das famílias reservadas de Felhen e Telepatia), aqui
    # aplicada a Lumi x Nemotron só como ilustração do cálculo; no E3 a comparação é LoRA x Lumi congelado.
    res = {m: [r for src in ("felhen", "telepatia")
               for r in load_rows(R / f"{p}_{src}_dev_{a.escopo}") if r.get("held_out")] for m, p in MODELOS.items()}
    out["h2_forma_lumi_menos_nemotron_reservadas"] = metrics.family_mean_bootstrap(res["nemotron"], res["lumi"])
    txt = json.dumps(out, indent=1, ensure_ascii=False, default=float)
    if a.json:
        Path(a.json).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
