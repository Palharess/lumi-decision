"""Conversão e filtros do Eikos (E3b), com linhas sintéticas: o arquivo real fica fora do git."""

import json
from collections import Counter

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


def test_tipo_separa_proximo_e_resto_com_o_mesmo_tamanho(tmp_path):
    fams = ["rules_permit"] * 3 + ["temporal_numeric"] * 2 + ["judge"] * 6 + ["trap"] * 4
    rows = [row(i, family=f, state=f"estado {i}") for i, f in enumerate(fams)]
    rows.append(row(99, family="judge", instructions="Is invoice INV-9 a duplicate?"))  # filtro do E3b vale antes
    p = write(tmp_path, rows)
    prox = data.eikos_tipo_cases({}, "proximo", path=p)
    resto = data.eikos_tipo_cases({}, "resto", path=p)
    kp = [c for c in prox if c.split == "train"]
    kr = [c for c in resto if c.split == "train"]
    assert {c.family for c in kp} == {"rules_permit", "temporal_numeric"} and len(kp) == 5
    assert {c.family for c in kr} <= {"judge", "trap"} and len(kr) == 5
    assert "eikos:r99" not in {c.id for c in kr}
    motivos = Counter(c.meta["descarte"] for c in resto if c.split != "train")
    assert motivos == {"tipo: fora do grupo": 5, "amostra do resto": 5, "tarefa: invoice_processing": 1}
    # determinístico
    assert [c.id for c in kr] == [c.id for c in data.eikos_tipo_cases({}, "resto", path=p) if c.split == "train"]


def test_resto_estratificado_por_idioma(tmp_path):
    pt = "Brazilian Portuguese"
    rows = ([row(i, family="rules_book", lang=pt if i < 2 else "English", state=f"p{i}") for i in range(4)]
            + [row(10 + i, family="judge", lang=pt if i < 1 else "English", state=f"r{i}") for i in range(8)])
    resto = [c for c in data.eikos_tipo_cases({}, "resto", path=write(tmp_path, rows)) if c.split == "train"]
    # o próximo tem 2 PT e 2 EN; o resto só tem 1 PT: entra ele e a falta é completada em inglês
    assert len(resto) == 4 and Counter(c.meta["lang"] for c in resto) == {pt: 1, "English": 3}
