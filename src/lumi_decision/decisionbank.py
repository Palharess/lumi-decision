"""DecisionBank (jgeuter/DecisionBank, revisão fbeee677, CC BY-SA 4.0) para o braço DecisionBank do E5
(analises/e5_dados.md). Só TREINO; nunca é avaliado.

Só entra a parte `j11_generator`: questões geradas e rotuladas por código, sem modelo no texto nem no alvo. O resto
do DecisionBank fica fora porque o alvo é, no todo ou pela metade, o Qwen3.8-27B, o mesmo escritor das reservadas
com professor. Dentro da `j11_generator`, saem os tipos que colidem com famílias reservadas (lista fixada em 26/11,
antes de qualquer treino, a partir dos modelos de pergunta contados no cluster):
- `j11_composite_gen` inteira: regras compostas de elegibilidade com severidade (`regra_beneficio`, `restricoes_oferta`);
- `j11_jaggedness`: prazos e datas de devolução (`ordem_eventos`), cupons (`custo_plano`) e soma de itens contra o
  total declarado (`invoice`);
- `j11_negation_pair_gen`: fica só "qual afirmação (não) está correta segundo o registro" de reserva, envio e
  funcionário, e "a resposta é consistente com a folha de referência". Os modelos que aplicam uma política
  (liberar o carro, pagar a bolsa, aplicar o desconto, reembolsar, deixar entrar...) e o resumo de política saem.
A sobreposição de texto com qualquer avaliação é tirada depois, em `lora.build_training_set`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .data import DISCARDED, sha256_file, source_dir, text_group
from .schema import Case

REPO = "jgeuter/DecisionBank"
REVISAO = "fbeee67758966fe78bfb008b28fc771924c0bc71"
DB_FILTER_VERSION = "v1"
# sha256 das partes, como no SHA256SUMS do próprio repositório na revisão fixada
SHARDS = {
    "data/DecisionBank-625k/train-00000-of-00012.jsonl": "31fac127734f40bba69c0212b1168c086e9554d09d02682c2e1db0abfcc7846d",
    "data/DecisionBank-625k/train-00001-of-00012.jsonl": "8451fd2acefda730f1f19f8dfb0b02abbe060d56244b39c3d573959c5a7c2b48",
    "data/DecisionBank-625k/train-00002-of-00012.jsonl": "315944313cde7b031e4cf6d6947d96374fdacc20985bc759e2dfd1eb5930e97a",
    "data/DecisionBank-625k/train-00003-of-00012.jsonl": "9724877f02a5b69f257e0fbe536e75742a5446ca40d2334b9e43053df940b356",
    "data/DecisionBank-625k/train-00004-of-00012.jsonl": "cb3bcfddba416a0c5d2cae9ebca5c0b0485cd353c62528acc40879cf8aa428fa",
    "data/DecisionBank-625k/train-00005-of-00012.jsonl": "5b4dc17482d4a22377755f400db2bdbd2e5f194d3e7e0058d1bb3683f4904c9d",
    "data/DecisionBank-625k/train-00006-of-00012.jsonl": "ef3c5753a852f24e3705e3ca94a4fbbbf8febcccd2d34385fac6cbda3733b64e",
    "data/DecisionBank-625k/train-00007-of-00012.jsonl": "4c128d11c642ff8acd625fa284fd79ac1ccd070ff5c78d24dce4841c3cdc03d8",
    "data/DecisionBank-625k/train-00008-of-00012.jsonl": "13a7af61fdf92544abb318a757bd4a93364ad413e6c1a999d0da2e82eb5be2c1",
    "data/DecisionBank-625k/train-00009-of-00012.jsonl": "67fb230f2a8f3fea885acd9e89acdbfb5785225a76931922b0f9081dc5ca74c4",
    "data/DecisionBank-625k/train-00010-of-00012.jsonl": "360bde8255bf37876a3137488e8beaaacb1efd67b0aaa4118cee690a9d276c46",
    "data/DecisionBank-625k/train-00011-of-00012.jsonl": "8ffff644d4e2101d57a88eeb3722f94fc3437b5ad01b4c47466f8a7b75e77ef4",
}
PARTE = "j11_generator"
LICENCA = "cc-by-sa-4.0 (ours)"
ORIGEM = "computed by code"
FAMILIAS_FORA = {"j11_composite_gen": "família: regras compostas (regra_beneficio, restricoes_oferta)"}
JAGGED_FORA = re.compile(r"deadline comes first|return (?:request|window)|after delivery|line items|stated total|coupon",
                         re.I)
NEGACAO_DENTRO = re.compile(r"according to the (?:booking|shipment|employee) record|reference sheet", re.I)


def motivo(row: dict) -> str | None:
    """Por que a linha fica fora (ou None). A ordem dos testes fixa o motivo contado na auditoria."""
    if row.get("part") != PARTE:
        return "parte"
    if row.get("source_license") != LICENCA or row.get("target_origin") != ORIGEM:
        return "licença ou origem do alvo"
    fam, q = row["family"], row["question"]
    if fam in FAMILIAS_FORA:
        return FAMILIAS_FORA[fam]
    if fam == "j11_jaggedness" and JAGGED_FORA.search(q):
        return "jaggedness: datas, cupons ou soma de itens"
    if fam == "j11_negation_pair_gen" and not NEGACAO_DENTRO.search(q):
        return "negation_pair: aplicação de política"
    if not 2 <= len(row["options"]) <= 26:
        return "nº de opções"
    dist = row.get("target_distribution")
    if not dist or len(dist) != len(row["options"]) or sum(dist) <= 0:
        return "alvo sem distribuição"
    return None


def caminhos_padrao() -> list[Path]:
    base = source_dir("decisionbank")
    out = []
    for rel, sha in SHARDS.items():
        p = base / rel
        if sha256_file(p) != sha:
            raise ValueError(f"{p} não é a revisão fixada do DecisionBank (sha256 diferente)")
        out.append(p)
    return out


def decisionbank_cases(cfg: dict | None = None, paths: list[Path] | None = None) -> list[Case]:
    """Linhas da `j11_generator` como casos de treino; as excluídas viram `descartado` com o motivo em
    `meta["descarte"]`. As outras partes nem viram caso (são lidas só pela auditoria do inventário)."""
    cases = []
    for p in paths if paths is not None else caminhos_padrao():
        with Path(p).open(encoding="utf-8") as f:
            for line in f:
                if f'"part": "{PARTE}"' not in line:  # atalho: só a parte limpa é decodificada
                    continue
                row = json.loads(line)
                if row.get("part") != PARTE:
                    continue
                reason = motivo(row)
                ids = [o["id"] for o in row["options"]]
                dist = row.get("target_distribution") or []
                z = sum(dist)
                meta = {"tier": row.get("tier"), "use_case": row.get("use_case"), "language": row.get("language"),
                        "filtro": DB_FILTER_VERSION}
                if reason:
                    meta["descarte"] = reason
                cases.append(Case(
                    id=f"decisionbank:{row['id']}", group_id=text_group("decisionbank", row["state"]),
                    source="decisionbank", family=row["family"], split=DISCARDED if reason else "train",
                    held_out=False, qtype=row["qtype"], state=row["state"], question=row["question"],
                    options=[(o["id"], o["description"]) for o in row["options"]], gold=str(ids[row["label"]]),
                    gold_dist={k: x / z for k, x in zip(ids, dist)} if z > 0 and len(dist) == len(ids) else None,
                    meta=meta))
    return cases
