"""Diagnóstico da queda do cot (analises/bench_diag.md): por rodada do Lumi-Benchmark e por dataset da trilha A,
acertos/total, tokens gerados, respostas diretas (até 8 tokens) e respostas que começam pela letra da opção.

    python3 scripts/analise_bench_diag.py <pasta runs do Lumi-Benchmark> <rodada>... [--saida arquivo.json]
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

COMECA_PELA_OPCAO = re.compile(r"^\s*(?:\*\*)?[A-E][\).:]")


def resumo(pasta: Path) -> dict:
    pred = {r["id"]: r for r in map(json.loads, (pasta / "predicoes_conhecimento.jsonl").read_text(
        encoding="utf-8").splitlines())}
    por = defaultdict(lambda: defaultdict(int))
    for r in map(json.loads, (pasta / "pontuacao_conhecimento.jsonl").read_text(encoding="utf-8").splitlines()):
        p = pred[r["id"]]
        for d in (r["dataset"], "_total"):
            x = por[d]
            x["n"] += 1
            x["acertos"] += bool(r["correct"])
            x["validas"] += bool(r["valid"])
            x["tokens"] += p["n_gen_tokens"]
            direto = p["n_gen_tokens"] <= 8
            x["diretas"] += direto
            x["acertos_diretas"] += direto and bool(r["correct"])
            x["comeca_pela_opcao"] += bool(COMECA_PELA_OPCAO.match(p["raw_output"] or ""))
    out = {}
    for d, x in sorted(por.items()):
        n = x["n"]
        out[d] = {"acertos": f"{x['acertos']}/{n}", "validas": f"{x['validas']}/{n}",
                  "tokens_medio": round(x["tokens"] / n, 1), "diretas": f"{x['diretas']}/{n}",
                  "acertos_nas_diretas": f"{x['acertos_diretas']}/{x['diretas']}",
                  "comeca_pela_opcao": f"{x['comeca_pela_opcao']}/{n}"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("rodadas", nargs="+")
    ap.add_argument("--saida")
    a = ap.parse_args()
    # Revisão de 27/11: uma rodada prevista sem resultado, ou com menos itens que as outras, não pode passar calada.
    faltam = [r for r in a.rodadas if not (Path(a.runs) / r / "pontuacao_conhecimento.jsonl").exists()]
    out = {r: resumo(Path(a.runs) / r) for r in a.rodadas if r not in faltam}
    maior = max((int(d["_total"]["acertos"].split("/")[1]) for d in out.values()), default=0)
    incompletas = [r for r, d in out.items() if int(d["_total"]["acertos"].split("/")[1]) != maior]
    print("| rodada | acertos | tokens médio | diretas | começa pela opção | MATH-PT |")
    print("|---|---:|---:|---:|---:|---:|")
    for r, d in out.items():
        t, m = d["_total"], d.get("math_pt", {})
        print(f"| {r} | {t['acertos']} | {t['tokens_medio']} | {t['diretas']} | {t['comeca_pela_opcao']} | "
              f"{m.get('acertos', '-')} |")
    if a.saida:
        Path(a.saida).write_text(json.dumps({**out, "_faltam": faltam, "_incompletas": incompletas}, indent=1,
                                            ensure_ascii=False) + "\n", encoding="utf-8")
    if faltam or incompletas:
        print(f">> rodadas sem resultado: {faltam}; com menos itens que {maior}: {incompletas}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
