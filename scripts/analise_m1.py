"""Medida M1 (analises/plano_fase2.md): factck_veracidade nas 6 permutações, para todos os checkpoints.

    python scripts/analise_m1.py <pasta>

<pasta> tem `factck6_<nome>` (saída de slurm/m1_factck6.sbatch) e, para a sanidade, as avaliações originais de
2 ordens: `<nome>_felhen_dev` (congelado: `e1_lumi_27b30f4_nativa_felhen_dev_completo`).
"""
import json
import sys
from collections import Counter
from pathlib import Path

from lumi_decision import metrics

FAM = "factck_veracidade"
ORDENS = ["perm012", "perm021", "perm102", "perm120", "perm201", "perm210"]
DUAS = {"perm012": "canonical", "perm210": "reversed"}
ORIGINAL = {"congelado": "e1_lumi_27b30f4_nativa_felhen_dev_completo"}
COMPARACOES = [  # (rótulo, base, novo): os mesmos pares e sentidos dos critérios registrados
    *[(f"H2 s{s}: E3 − congelado", "congelado", e3) for s, e3 in enumerate(("e3_lora_r16", "e3_lora_r16_s1", "e3_lora_r16_s2"))],
    *[(f"H3 s{s}: E4 − E3", e3, f"e4_ponteiro_s{s}") for s, e3 in enumerate(("e3_lora_r16", "e3_lora_r16_s1", "e3_lora_r16_s2"))],
    ("E3b s0 − controle", "e3b_ctrl_s0", "e3b_eikos_s0"),
    ("E3b s1 − controle (E3b-r)", "e3b_ctrl_s1", "e3b_eikos_s1"),
    *[(f"E3c s{s}: próximo − resto", f"e3c_resto_s{s}", f"e3c_proximo_s{s}") for s in (0, 1)],
    *[(f"E3c s{s}: próximo − controle", f"e3c_ctrl_s{s}", f"e3c_proximo_s{s}") for s in (0, 1)],
    *[(f"E3c s{s}: resto − controle", f"e3c_ctrl_s{s}", f"e3c_resto_s{s}") for s in (0, 1)],
    *[(f"F2.3a s{s}: residual − controle", f"f23_controle_s{s}", f"f23_residual_s{s}") for s in (0, 1)],
    *[(f"F2.1 s{s}: balanceado − E3", e3, f"f21_balanceado_s{s}") for s, e3 in enumerate(("e3_lora_r16", "e3_lora_r16_s1"))],
    *[(f"F2.2 s{s}: sem in_proj − E3", e3, f"f22_sem_inproj_s{s}") for s, e3 in enumerate(("e3_lora_r16", "e3_lora_r16_s1"))],
    *[(f"F3 s{s}: diverso − controle", f"f21_balanceado_s{s}", f"f3p_diverso_s{s}") for s in (0, 1)],
    # ruído de semente no mesmo braço (descritivo)
    ("semente: E3 s1 − s0", "e3_lora_r16", "e3_lora_r16_s1"),
    ("semente: E3 s2 − s0", "e3_lora_r16", "e3_lora_r16_s2"),
    ("semente: controle F3 s1 − s0", "f21_balanceado_s0", "f21_balanceado_s1"),
]


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in (p / "predicoes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def resumo(rows: list[dict]) -> dict:
    assert Counter(r["order"] for r in rows) == {o: 102 for o in ORDENS}, Counter(r["order"] for r in rows)
    hit, tot = Counter(), Counter()
    pos, preds = Counter(), {}
    for r in rows:
        tot[r["gold"]] += 1
        hit[r["gold"]] += r["pred"] == r["gold"]
        pos["ABC"[r["presented"].index(r["pred"])]] += 1
        preds.setdefault(r["case_id"], set()).add(r["pred"])
    duas = [r for r in rows if r["order"] in DUAS]
    return {"acertos": sum(hit.values()), "respostas": len(rows),
            "por_classe": {k: f"{hit[k]}/{tot[k]}" for k in sorted(tot)},
            "previstos": dict(Counter(r["pred"] for r in rows)),
            "muda_entre_6_ordens": sum(len(v) > 1 for v in preds.values()),
            "posicao": dict(sorted(pos.items())),
            "duas_ordens": f"{sum(r['pred'] == r['gold'] for r in duas)}/{len(duas)}"}


def sanidade(rows: list[dict], orig_dir: Path) -> str:
    """As respostas em perm012/perm210 têm de repetir as da avaliação original (canonical/reversed)."""
    if not (orig_dir / "predicoes.jsonl").exists():
        return "sem avaliação original para conferir"
    orig = {(r["case_id"], r["order"]): r for r in load(orig_dir) if r["family"] == FAM}
    novo = {(r["case_id"], DUAS[r["order"]]): r for r in rows if r["order"] in DUAS}
    if set(orig) != set(novo):
        return f"CASOS DIFERENTES ({len(set(orig) ^ set(novo))})"
    dif = sum(orig[k]["pred"] != novo[k]["pred"] for k in orig)
    dp = max(abs(orig[k]["probs"][o] - novo[k]["probs"][o]) for k in orig for o in orig[k]["probs"])
    return f"{len(orig) - dif}/{len(orig)} respostas iguais; maior diferença de prob {dp:.2e}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(sys.argv[1])
    nomes = sorted(p.name.removeprefix("factck6_") for p in root.glob("factck6_*"))
    rows = {n: load(root / f"factck6_{n}") for n in nomes}
    out = {"checkpoints": {}, "comparacoes": {}}
    for n in nomes:
        r = resumo(rows[n])
        r["sanidade"] = sanidade(rows[n], root / ORIGINAL.get(n, f"{n}_felhen_dev"))
        out["checkpoints"][n] = r
    for rot, a, b in COMPARACOES:
        if a not in rows or b not in rows:
            out["comparacoes"][rot] = "faltam predições"
            continue
        seis = metrics.family_mean_bootstrap(rows[a], rows[b], require_families=[FAM])
        da = [r for r in rows[a] if r["order"] in DUAS]
        db = [r for r in rows[b] if r["order"] in DUAS]
        duas = metrics.family_mean_bootstrap(da, db, require_families=[FAM])
        ha, hb = (sum(r["pred"] == r["gold"] for r in x) for x in (rows[a], rows[b]))
        out["comparacoes"][rot] = {
            "6_ordens": f"{hb}/612 contra {ha}/612: {100 * seis['delta_media_familias']:+.1f} "
                        f"[{100 * seis['ic95'][0]:+.1f}; {100 * seis['ic95'][1]:+.1f}]",
            "2_ordens": f"{100 * duas['delta_media_familias']:+.1f} "
                        f"[{100 * duas['ic95'][0]:+.1f}; {100 * duas['ic95'][1]:+.1f}]"}
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
