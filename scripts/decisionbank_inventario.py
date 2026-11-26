"""Inventário do DecisionBank (analises/e5_dados.md): baixa a config 625k numa revisão fixa e conta as linhas por
parte, família, idioma, licença, origem do alvo e tipo de pergunta. Não grava nenhuma linha, só contagens.

    python scripts/decisionbank_inventario.py <saida.json>
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download

REPO, REVISAO = "jgeuter/DecisionBank", "fbeee67758966fe78bfb008b28fc771924c0bc71"


def main():
    saida = Path(sys.argv[1])
    raiz = Path(snapshot_download(REPO, repo_type="dataset", revision=REVISAO,
                                  allow_patterns=["data/DecisionBank-625k/*", "README.md", "SHA256SUMS"]))
    cont = defaultdict(Counter)
    familias = defaultdict(lambda: defaultdict(Counter))
    exemplos = {}
    n = 0
    for arq in sorted((raiz / "data" / "DecisionBank-625k").glob("*.jsonl")):
        with arq.open(encoding="utf-8") as f:
            for linha in f:
                r = json.loads(linha)
                n += 1
                for k in ("part", "language", "source_license", "target_origin", "qtype", "state_is_json", "tier"):
                    cont[k][str(r.get(k))] += 1
                cont["part×language"][f"{r['part']}|{r.get('language')}"] += 1
                fam = familias[r["family"]]
                fam["part"][r["part"]] += 1
                fam["licenca"][r["source_license"]] += 1
                fam["qtype"][r["qtype"]] += 1
                fam["n_opcoes"][len(r["options"])] += 1
                fam["fonte"][r["source_dataset"]] += 1
                if r["family"] not in exemplos:
                    exemplos[r["family"]] = {"question": r["question"][:300], "state": r["state"][:300],
                                             "options": [o["description"][:80] for o in r["options"]][:6]}
    out = {"repo": REPO, "revisao": REVISAO, "linhas": n,
           "contagens": {k: dict(v.most_common()) for k, v in cont.items()},
           "familias": {f: {k: dict(v.most_common()) for k, v in d.items()} for f, d in sorted(familias.items())},
           "exemplo_por_familia": exemplos}
    saida.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f">> {n} linhas, {len(familias)} famílias -> {saida}")


if __name__ == "__main__":
    main()
