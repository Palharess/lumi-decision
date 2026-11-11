"""Métricas sobre linhas de predição (uma linha por caso × ordem de opções).

Campos usados de cada linha: case_id, group_id, family, order, gold, probs{opção: p},
pred, e opcionalmente gold_dist, gen_pred.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

EPS = 1e-12


def accuracy(rows) -> float:
    return float(np.mean([r["pred"] == r["gold"] for r in rows])) if rows else float("nan")


def balanced_accuracy(rows) -> float:
    by_gold = defaultdict(list)
    for r in rows:
        by_gold[r["gold"]].append(r["pred"] == r["gold"])
    return float(np.mean([np.mean(v) for v in by_gold.values()])) if by_gold else float("nan")


def nll(rows) -> float:
    return float(np.mean([-math.log(max(r["probs"].get(r["gold"], 0.0), EPS)) for r in rows]))


def brier(rows) -> float:
    """Brier multiclasse: soma sobre as opções de (p - y)^2, média sobre itens (intervalo 0..2)."""
    return float(np.mean([sum((p - (k == r["gold"])) ** 2 for k, p in r["probs"].items()) for r in rows]))


def ece(rows, bins: int = 15) -> float:
    """ECE sobre a probabilidade máxima, bins de largura igual."""
    conf = np.array([max(r["probs"].values()) for r in rows])
    hit = np.array([r["pred"] == r["gold"] for r in rows], dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def soft_metrics(rows) -> dict:
    """Aderência à distribuição de referência (só onde há gold_dist)."""
    rows = [r for r in rows if r.get("gold_dist")]
    if not rows:
        return {}
    ce = [-sum(q * math.log(max(r["probs"].get(k, 0.0), EPS)) for k, q in r["gold_dist"].items()) for r in rows]
    tv = [0.5 * sum(abs(r["probs"].get(k, 0.0) - r["gold_dist"].get(k, 0.0))
                    for k in set(r["probs"]) | set(r["gold_dist"])) for r in rows]
    return {"n_soft": len(rows), "cross_entropy_vs_ref": float(np.mean(ce)), "tv_vs_ref": float(np.mean(tv))}


def risk_coverage(rows, risk_targets=(0.05, 0.10, 0.20)) -> dict:
    """Curva risco × cobertura aceitando os itens de maior confiança primeiro.

    `risco_aleatorio` é o erro esperado ao aceitar a mesma fração ao acaso (= erro global).
    """
    conf = np.array([max(r["probs"].values()) for r in rows])
    err = np.array([r["pred"] != r["gold"] for r in rows], dtype=float)
    idx = np.argsort(-conf, kind="stable")
    cum_risk = np.cumsum(err[idx]) / np.arange(1, len(rows) + 1)
    cov = np.arange(1, len(rows) + 1) / len(rows)
    out = {"risco_global": float(err.mean()), "aurc": float(cum_risk.mean())}
    for c in (0.5, 0.8):
        out[f"risco@cob{int(c * 100)}"] = float(cum_risk[max(0, math.ceil(c * len(rows)) - 1)])
    for t in risk_targets:
        ok = np.nonzero(cum_risk <= t)[0]
        out[f"cobertura@risco{int(t * 100)}"] = float(cov[ok.max()]) if ok.size else 0.0
    return out


def summarize(rows) -> dict:
    if not rows:
        return {}
    out = {
        "n": len(rows),
        "acuracia": accuracy(rows),
        "acuracia_balanceada": balanced_accuracy(rows),
        "nll": nll(rows),
        "brier": brier(rows),
        "ece": ece(rows),
        "acaso": float(np.mean([1 / len(r["probs"]) for r in rows])),
        **risk_coverage(rows),
        **soft_metrics(rows),
    }
    gen = [r for r in rows if "gen_pred" in r]
    if gen:
        out["geracao_acuracia"] = float(np.mean([r["gen_pred"] == r["gold"] for r in gen]))
        out["geracao_invalida"] = float(np.mean([r["gen_pred"] is None for r in gen]))
        out["geracao_concorda_readout"] = float(np.mean([r["gen_pred"] == r["pred"] for r in gen]))
    return out


def permutation_stats(rows) -> dict:
    """Sensibilidade à ordem: compara as ordens de cada caso (exige ≥2 ordens por caso)."""
    by_case = defaultdict(dict)
    for r in rows:
        by_case[r["case_id"]][r["order"]] = r
    orders = sorted({r["order"] for r in rows})
    cases = [v for v in by_case.values() if len(v) == len(orders)]
    if len(orders) < 2 or not cases:
        return {}
    ref = "canonical" if "canonical" in orders else orders[0]
    flips = [len({v[o]["pred"] for o in orders}) > 1 for v in cases]
    ref_ok = [v for v in cases if v[ref]["pred"] == v[ref]["gold"]]
    keep = [all(v[o]["pred"] == v[o]["gold"] for o in orders) for v in ref_ok]
    avg_hits = []
    for v in cases:
        keys = v[ref]["probs"].keys()
        mean = {k: np.mean([v[o]["probs"][k] for o in orders]) for k in keys}
        avg_hits.append(max(mean, key=mean.get) == v[ref]["gold"])
    return {
        "ordens": orders,
        "n_casos": len(cases),
        "decisao_muda_entre_ordens": float(np.mean(flips)),
        "retencao_dos_acertos": float(np.mean(keep)) if keep else float("nan"),
        "acuracia_por_ordem": {o: accuracy([v[o] for v in cases]) for o in orders},
        "acuracia_media_das_ordens": float(np.mean(avg_hits)),
        "acerta_em_todas_as_ordens": float(np.mean([all(v[o]["pred"] == v[o]["gold"] for o in orders) for v in cases])),
    }


def paired_bootstrap(rows_a, rows_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """IC 95% de acc(B) - acc(A) nos mesmos (caso, ordem), reamostrando grupos inteiros."""
    a = {(r["case_id"], r["order"]): r for r in rows_a}
    b = {(r["case_id"], r["order"]): r for r in rows_b}
    keys = sorted(set(a) & set(b))
    groups = defaultdict(list)
    for k in keys:
        groups[a[k]["group_id"]].append((a[k]["pred"] == a[k]["gold"], b[k]["pred"] == b[k]["gold"]))
    g = list(groups.values())
    sums = np.array([[sum(x[1] - x[0] for x in grp), len(grp)] for grp in g], dtype=float)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        s = sums[rng.integers(0, len(g), len(g))]
        deltas.append(s[:, 0].sum() / s[:, 1].sum())
    point = sums[:, 0].sum() / sums[:, 1].sum()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"n_itens": len(keys), "n_grupos": len(g), "delta": float(point), "ic95": [float(lo), float(hi)]}


# ---------------------------------------------------------------- calibração (ajuste em calib, uso fixo em dev/test)

def with_temperature(rows, T: float) -> list[dict]:
    """Reescala as probabilidades: p_k ∝ p_k^(1/T). Não muda o argmax."""
    out = []
    for r in rows:
        logp = {k: math.log(max(p, EPS)) / T for k, p in r["probs"].items()}
        m = max(logp.values())
        w = {k: math.exp(v - m) for k, v in logp.items()}
        z = sum(w.values())
        out.append({**r, "probs": {k: v / z for k, v in w.items()}})
    return out


def fit_temperature(rows) -> float:
    """Temperatura que minimiza a NLL (busca em grade log-espaçada e refinamento local)."""
    grid = np.exp(np.linspace(math.log(0.2), math.log(10), 61))
    best = min(grid, key=lambda t: nll(with_temperature(rows, t)))
    fine = np.linspace(best / 1.1, best * 1.1, 21)
    return float(min(fine, key=lambda t: nll(with_temperature(rows, t))))


def threshold_for_risk(rows, target: float, min_accept: int = 20) -> float | None:
    """Menor limiar de confiança cujo risco entre os aceitos (no conjunto dado) fica <= alvo."""
    conf = np.array([max(r["probs"].values()) for r in rows])
    err = np.array([r["pred"] != r["gold"] for r in rows], dtype=float)
    idx = np.argsort(-conf, kind="stable")
    cum_risk = np.cumsum(err[idx]) / np.arange(1, len(rows) + 1)
    sc = conf[idx]
    # Só posições onde a confiança muda são cortes realizáveis: com um limiar fixo, empates entram juntos.
    cuts = [i for i in range(min_accept - 1, len(rows)) if (i == len(rows) - 1 or sc[i] > sc[i + 1])]
    ok = [i for i in cuts if cum_risk[i] <= target]
    return float(sc[max(ok)]) if ok else None


def selective_at(rows, tau: float | None) -> dict:
    """Aplica um limiar FIXO: cobertura, risco entre os aceitos e o risco de aceitar a mesma fração ao acaso."""
    err = np.array([r["pred"] != r["gold"] for r in rows], dtype=float)
    if tau is None or not len(rows):
        return {"limiar": tau, "cobertura": 0.0, "risco_aceitos": None, "risco_aleatorio": float(err.mean()) if len(rows) else None}
    acc = np.array([max(r["probs"].values()) >= tau for r in rows])
    return {"limiar": tau, "cobertura": float(acc.mean()),
            "risco_aceitos": float(err[acc].mean()) if acc.any() else None,
            "risco_aleatorio": float(err.mean()), "n": len(rows)}
