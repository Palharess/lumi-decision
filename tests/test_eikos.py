"""Conversão e filtros do Eikos (E3b), com linhas sintéticas: o arquivo real fica fora do git."""

import json

from lumi_decision import data
from lumi_decision.schema import Case


def row(i, instructions="Should the transfer be approved?", state="Wire of USD 900 to a known payee.", **kw):
    r = {"id": f"r{i}", "family": "rules_permit", "topic": "rules: payments", "lang": "English",
         "source": "prog_rules:permit:payments", "question_type": "noul", "state": state,
         "instructions": instructions,
         "options": [{"label": "yes", "description": "Approve: within the limit."},
                     {"label": "no", "description": "Reject: above the limit."}],
         "expected": "yes", "target_probs": [0.9, 0.3]}
    r.update(kw)
    return r


def write(tmp_path, rows):
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_conversion_keeps_labels_descriptions_and_normalized_target(tmp_path):
    (c,) = data.eikos_cases({}, path=write(tmp_path, [row(0)]))
    assert c.split == "train" and not c.held_out and c.source == "eikos" and c.family == "rules_permit"
    assert c.options == [("yes", "Approve: within the limit."), ("no", "Reject: above the limit.")]
    assert c.gold == "yes" and abs(sum(c.gold_dist.values()) - 1) < 1e-9 and c.gold_dist["yes"] == 0.75


def test_task_filters_drop_held_out_lookalikes(tmp_path):
    rows = [row(0),
            row(1, instructions="Is invoice INV-22 a duplicate of an earlier one?"),
            row(2, state="Pedido de compra 881 e nota fiscal 12."),
            row(3, source="prog_fin:t_three_way"),
            row(4, instructions="Segundo a checagem, a notícia é verdadeira?"),
            row(5, state="Acórdão do Tribunal de Contas da União sobre licitação."),
            row(6, options=[{"label": str(k), "description": str(k)} for k in range(27)], target_probs=[1] * 27,
                expected="0")]
    cs = data.eikos_cases({}, path=write(tmp_path, rows))
    kept = [c.id for c in cs if c.split == "train"]
    assert kept == ["eikos:r0"]
    reasons = {c.id: c.meta["descarte"] for c in cs if c.split != "train"}
    assert reasons["eikos:r1"] == reasons["eikos:r2"] == reasons["eikos:r3"] == "tarefa: invoice_processing"
    assert reasons["eikos:r4"] == "tarefa: factck_veracidade"
    assert reasons["eikos:r5"] == "tarefa: juristcu_area"
    assert reasons["eikos:r6"] == "nº de opções"


def test_language_filter(tmp_path):
    cs = data.eikos_cases({}, langs={"Brazilian Portuguese"},
                          path=write(tmp_path, [row(0), row(1, lang="Brazilian Portuguese")]))
    assert [c.id for c in cs if c.split == "train"] == ["eikos:r1"]


def _case(i, state, split="dev", held=False):
    return Case(id=f"x{i}", group_id=f"g{i}", source="felhen", family="f", split=split, held_out=held,
                qtype="choice", state=state, question="q", options=[("a", "A"), ("b", "B")], gold="a")


def test_drop_eval_overlap_exact_and_ngram(tmp_path):
    long = "um dois tres quatro cinco seis sete oito nove dez onze doze treze quatorze"
    others = [_case(0, "Texto  IGUAL ao do dev."), _case(1, f"prefixo {long} sufixo", split="train", held=True),
              _case(2, "texto só de treino de outra fonte", split="train")]
    cs = data.eikos_cases({}, path=write(tmp_path, [
        row(0, state="texto igual ao do dev."), row(1, state=f"outro começo {long}"),
        row(2, state="texto só de treino de outra fonte"), row(3)]))
    assert data.drop_eval_overlap(cs, others) == 2
    assert [c.id for c in cs if c.split == "train"] == ["eikos:r2", "eikos:r3"]
    assert {c.meta["descarte"] for c in cs if c.split != "train"} == {"texto em avaliação"}
