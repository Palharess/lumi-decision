"""Confere que o pacote reproduz a avaliação do lumi-decision (não vai na entrega).

    python verificar.py <pacote> <casos.jsonl> <predicoes.jsonl> [n_por_familia]

Para n casos por família (ordem canônica), compara com as predições salvas pelo `lumi-eval`: mesmo prompt
(sha1 dos ids), mesma escolha e probabilidades próximas. Sai com erro se alguma coisa divergir.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

pacote = Path(sys.argv[1])
sys.path.insert(0, str(pacote))
from exemplo import carregar, decidir, montar_prompt  # noqa: E402

casos = {c["id"]: c for c in map(json.loads, Path(sys.argv[2]).read_text(encoding="utf-8").splitlines())}
n = int(sys.argv[4]) if len(sys.argv) > 4 else 10
por_fam = defaultdict(list)
for r in map(json.loads, Path(sys.argv[3]).read_text(encoding="utf-8").splitlines()):
    if r["order"] == "canonical" and len(por_fam[r["family"]]) < n:
        por_fam[r["family"]].append(r)

tok, model = carregar(pacote / "base", pacote / "adapter")
falhas, max_dp, gen_ok, total = 0, 0.0, 0, 0
for fam, rows in por_fam.items():
    for r in rows:
        c = casos[r["case_id"]]
        opcoes = [d for _, d in c["options"]]
        ids = tok(montar_prompt(tok, c["state"], c["question"], opcoes), add_special_tokens=False)["input_ids"]
        sha = hashlib.sha1(json.dumps(ids).encode()).hexdigest()
        o = decidir(tok, model, c["state"], c["question"], opcoes)
        pred = c["options"][o["escolha"]][0]
        dp = max(abs(o["probs"][i] - r["probs"][k]) for i, (k, _) in enumerate(c["options"]))
        max_dp = max(max_dp, dp)
        total += 1
        gen_ok += o["gerado"] == o["letra"]
        ok = sha == r["prompt_sha1"] and (pred == r["pred"] or dp < 0.02)
        falhas += not ok
        print(f"{fam:20s} {r['case_id']:32s} prompt {'=' if sha == r['prompt_sha1'] else '≠'} "
              f"pred {pred}/{r['pred']} dp {dp:.4f} gerado {o['gerado']!r}")
print(f">> {total - falhas}/{total} iguais à avaliação; maior diferença de prob {max_dp:.4f}; "
      f"geração = leitura em {gen_ok}/{total}")
sys.exit(1 if falhas else 0)
