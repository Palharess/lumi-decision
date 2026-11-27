"""E6, regra de preservação (analises/e6_protocolo.md, aceita pelo Gabriel em 27/11/2026): cada candidato contra o
Lumi-4B, nas mesmas questões.

- A `direct` e A `cot` (Lumi-Benchmark) e ENEM `cot`: passa se a diferença pontual for de no mínimo −2 pontos e o
  IC 95% pareado não ficar inteiro abaixo de zero. Bootstrap reamostrando questões, 10.000 réplicas, semente 0.
- B (chamada exata): passa se a diferença pontual for de no mínimo 0.
- O candidato passa se passar nas quatro medidas.

    python3 scripts/guarda_corpo.py <pasta runs do Lumi-Benchmark> --parte dev|confirmar ROTULO... [--saida x.json]

As rodadas se chamam `gc-<rótulo>-<variante>-<split>` (slurm/guarda_corpo.sbatch); o Lumi-4B é o rótulo `base`.
"""
import argparse
import json
import random
import sys
from pathlib import Path

MARGEM = 2.0  # pontos percentuais
N_BOOT, SEMENTE = 10_000, 0
SPLITS = {"dev": ("dev", "explorar"), "confirmar": ("test", "confirmar")}


def medidas(parte: str) -> dict:
    b, e = SPLITS[parte]
    return {"A_direct": (f"direct-{b}", "conhecimento", "correct"),
            "A_cot": (f"cot-{b}", "conhecimento", "correct"),
            "ENEM_cot": (f"enemcot-{e}", "conhecimento", "correct"),
            "B": (f"direct-{b}", "decisao", "exact")}


def carregar(runs: Path, rodada: str, trilha: str, campo: str) -> tuple[dict, dict]:
    pont = runs / rodada / f"pontuacao_{trilha}.jsonl"
    if not pont.exists():
        raise SystemExit(f">> falta a rodada {rodada} ({pont})")
    acertos = {r["id"]: bool(r[campo]) for r in map(json.loads, pont.read_text(encoding="utf-8").splitlines())}
    pred = runs / rodada / f"predicoes_{trilha}.jsonl"
    tokens = {r["id"]: r.get("n_gen_tokens") or 0 for r in map(json.loads, pred.read_text(encoding="utf-8").splitlines())}
    return acertos, tokens


def comparar(base: dict, cand: dict) -> dict:
    if base.keys() != cand.keys():
        raise SystemExit(f">> questões diferentes: {len(base.keys() ^ cand.keys())} fora da interseção")
    ids = sorted(base)
    d = [int(cand[i]) - int(base[i]) for i in ids]
    n = len(d)
    rng = random.Random(SEMENTE)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]
    return {"base": f"{sum(base.values())}/{n}", "candidato": f"{sum(cand.values())}/{n}",
            "diferenca": round(100 * sum(d) / n, 2), "ic95": [round(100 * lo, 2), round(100 * hi, 2)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("rotulos", nargs="+")
    ap.add_argument("--parte", choices=sorted(SPLITS), default="dev")
    ap.add_argument("--saida")
    a = ap.parse_args()
    runs, med = Path(a.runs), medidas(a.parte)
    base = {m: carregar(runs, f"gc-base-{suf}", tr, campo) for m, (suf, tr, campo) in med.items()}
    out = {}
    for rot in a.rotulos:
        res, passa = {}, True
        for m, (suf, tr, campo) in med.items():
            acertos, tokens = carregar(runs, f"gc-{rot}-{suf}", tr, campo)
            r = comparar(base[m][0], acertos)
            if m == "B":
                r["passa"] = r["diferenca"] >= 0
            else:
                r["passa"] = r["diferenca"] >= -MARGEM and r["ic95"][1] >= 0
                r["tokens_medio"] = {"base": round(sum(base[m][1].values()) / len(base[m][1]), 1),
                                     "candidato": round(sum(tokens.values()) / len(tokens), 1)}
            res[m] = r
            passa = passa and r["passa"]
        out[rot] = {"medidas": res, "passa": passa}
        print(f"| {rot} | " + " | ".join(f"{m} {r['candidato']} ({r['diferenca']:+.1f} [{r['ic95'][0]:+.1f}; "
                                         f"{r['ic95'][1]:+.1f}]) {'ok' if r['passa'] else 'REPROVA'}"
                                         for m, r in res.items()) + f" | {'PASSA' if passa else 'REPROVA'} |")
    if a.saida:
        Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
        Path(a.saida).write_text(json.dumps({"parte": a.parte, "margem": MARGEM, "candidatos": out}, indent=1,
                                            ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
