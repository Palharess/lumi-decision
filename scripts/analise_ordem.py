"""Sensibilidade à ordem das opções em todos os modelos avaliados no dev (E1, E3, E3b, E4).

Cada caso foi lido duas vezes, com as opções na ordem original (canônica) e invertida. Para cada modelo e
grupo: acurácia em cada ordem, a diferença entre elas, a fração de casos em que a escolha muda e a fração
acertada nas duas ordens. Também a taxa de mudança por família reservada e a média com peso igual delas
(a mesma conta do desfecho de estabilidade do E4). Lê `runs/` (fora do git).

    python scripts/analise_ordem.py            # imprime a tabela em Markdown e grava analises/ordem_resumo.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
MODELOS = {  # nome -> prefixo das pastas de run (dev)
    "Nemotron base congelado": ("e1_nemotron_dfaf35d_nativa", "_completo"),
    "Lumi congelado (E1)": ("e1_lumi_27b30f4_nativa", "_completo"),
    "LoRA + letras (E3)": ("e3_lora_r16", ""),
    "Controle, treino mais longo (E3b)": ("e3b_ctrl_s0", ""),
    "LoRA + Eikos (E3b)": ("e3b_eikos_s0", ""),
    "LoRA + pointer head (E4)": ("e4_ponteiro_s0", ""),
}
GRUPOS = [("felhen", False), ("felhen", True), ("telepatia", False), ("telepatia", True), ("traces", False)]
RESERVADAS = ("factck_veracidade", "juristcu_area", "invoice_processing")


def load(prefix, suf, src):
    p = RUNS / f"{prefix}_{src}_dev{suf}" / "predicoes.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else None


def by_case(rows):
    d = defaultdict(dict)
    for r in rows:
        d[r["case_id"]][r["order"]] = r
    return [v for v in d.values() if {"canonical", "reversed"} <= set(v)]


def stats(cases):
    can = np.mean([v["canonical"]["pred"] == v["canonical"]["gold"] for v in cases])
    rev = np.mean([v["reversed"]["pred"] == v["reversed"]["gold"] for v in cases])
    return {"n": len(cases), "acc_canonica": float(can), "acc_invertida": float(rev), "diferenca": float(rev - can),
            "muda_com_ordem": float(np.mean([v["canonical"]["pred"] != v["reversed"]["pred"] for v in cases])),
            "acerta_nas_duas": float(np.mean([all(v[o]["pred"] == v[o]["gold"] for o in ("canonical", "reversed"))
                                              for v in cases]))}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = {}
    for nome, (prefix, suf) in MODELOS.items():
        res, fam = {}, {}
        for src, held in GRUPOS:
            rows = load(prefix, suf, src)
            if rows is None:
                continue
            cases = by_case([r for r in rows if r["held_out"] == held])
            if cases:
                res[f"{src}|{'reservadas' if held else 'vistas'}"] = stats(cases)
            if held:
                for f in RESERVADAS:
                    cf = [v for v in cases if v["canonical"]["family"] == f]
                    if cf:
                        fam[f] = stats(cf)
        if len(fam) == len(RESERVADAS):
            res["reservadas_media_familias"] = {
                k: float(np.mean([fam[f][k] for f in RESERVADAS])) for k in ("muda_com_ordem", "acerta_nas_duas")}
        res["por_familia_reservada"] = fam
        out[nome] = res
    (ROOT / "analises" / "ordem_resumo.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    print("| Modelo | Grupo | Acurácia original | Acurácia invertida | Muda com a ordem | Acerta nas duas |")
    print("|---|---|---|---|---|---|")
    for nome, res in out.items():
        for g, s in res.items():
            if "|" in g:
                print(f"| {nome} | {g} | {s['acc_canonica']:.2f} | {s['acc_invertida']:.2f} | {pct(s['muda_com_ordem'])} | "
                      f"{pct(s['acerta_nas_duas'])} |")
    print("\nReservadas, por família (muda com a ordem):")
    for nome, res in out.items():
        fam = res["por_familia_reservada"]
        m = res.get("reservadas_media_familias", {})
        print(nome, {f: pct(s["muda_com_ordem"]) for f, s in fam.items()}, "média", pct(m.get("muda_com_ordem", float("nan"))))


if __name__ == "__main__":
    main()
