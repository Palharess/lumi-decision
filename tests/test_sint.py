"""Sintético em PT-BR do E5 (src/lumi_decision/sint.py): especificações, filtro de forma e montagem, sem modelo."""

import json
from collections import Counter

import pytest

from lumi_decision import prof, sint
from lumi_decision.prompts import LETTERS, order_options


def _doc(e):
    a, b = e["itens"]
    ta, tb = e["tabela"][a], e["tabela"][b]
    fa = "; ".join(f"{c}: {v}" for c, v in ta.items())
    fb = "; ".join(f"{c}: {v}" for c, v in tb.items())
    return (f"Registro do setor. O item {a} saiu do depósito com os dados {fa}. Já o item {b} ficou na bancada "
            f"principal, com os dados {fb}. Os dois voltam para a revisão na semana que vem, conforme o combinado.")


def test_especificacao_deterministica_e_sem_vazamento():
    for fam in sint.SINT_FAMILIES:
        assert sint.especificacao(fam, 3) == sint.especificacao(fam, 3)
    for i in range(200):
        e = sint.especificacao("extracao_campo", i)
        vals = [v for t in e["tabela"].values() for v in t.values()]
        assert len(set(vals)) == 6 and all(len(v) == 7 and v[2] == "-" for v in vals)  # mesmo formato
        e = sint.especificacao("campo_alterado", i)
        assert e["novo_valor"] not in e["versao_1"].values() and e["alterado"] in e["campos"]
    alvos = Counter(sint.especificacao("campo_alterado", i)["alterado"] for i in range(700))
    assert len(alvos) == len(sint.CAMPOS_ALTERADO) and max(alvos.values()) < 160  # campo sorteado, sem classe dominante
    # domínios disjuntos dos temas das reservadas
    assert not set(sint.DOMINIOS) & set(prof.TEMAS)


def test_extracao_campo():
    e = sint.especificacao("extracao_campo", 0)
    cs = sint.ler_resposta("extracao_campo", 0, e, json.dumps({"documento": _doc(e)}, ensure_ascii=False))
    assert len(cs) == 2 and all(c.split == "train" and not c.held_out for c in cs)
    for c, (it, campo) in zip(cs, e["perguntas"]):
        outro = next(x for x in e["itens"] if x != it)
        vals = [o[0] for o in c.options]
        assert c.gold == e["tabela"][it][campo] and e["tabela"][outro][campo] in vals and len(set(vals)) == 4
    v = next(iter(e["tabela"][e["itens"][0]].values()))
    with pytest.raises(prof.Rejeitado, match="repetido"):
        sint.ler_resposta("extracao_campo", 0, e, json.dumps({"documento": _doc(e) + f" Confira {v}."}))
    with pytest.raises(prof.Rejeitado, match="a mais"):
        sint.ler_resposta("extracao_campo", 0, e, json.dumps({"documento": _doc(e) + " Ver também QQ-1234."}))


def test_campo_alterado():
    e = sint.especificacao("campo_alterado", 5)
    v1 = e["versao_1"]
    v2 = dict(v1, **{e["alterado"]: e["novo_valor"]})
    d = {"versao_1": "Primeiro registro: " + "; ".join(f"{c} {v}" for c, v in v1.items()) + ".",
         "versao_2": "No dia seguinte: " + "; ".join(f"{c} {v2[c]}" for c in e["ordem_2"]) + "."}
    (c,) = sint.ler_resposta("campo_alterado", 5, e, json.dumps(d, ensure_ascii=False))
    assert c.gold == e["alterado"] and len(c.options) == 4
    with pytest.raises(prof.Rejeitado, match="denuncia"):
        sint.ler_resposta("campo_alterado", 5, e, json.dumps(dict(d, versao_2=d["versao_2"] + " Dado alterado."),
                                                             ensure_ascii=False))
    with pytest.raises(prof.Rejeitado, match="antigo"):
        antigo = v1[e["alterado"]]
        sint.ler_resposta("campo_alterado", 5, e, json.dumps(dict(d, versao_2=d["versao_2"] + f" (era {antigo})"),
                                                             ensure_ascii=False))


def test_candidatos_e_montar():
    resp = []
    for i in range(6):
        e = sint.especificacao("extracao_campo", i)
        resp.append({"id": f"sint:extracao_campo:{i}", "familia": "extracao_campo", "indice": i, "especificacao": e,
                     "finish_reason": "stop", "texto": json.dumps({"documento": _doc(e)}, ensure_ascii=False)})
    resp[0]["finish_reason"] = "length"
    casos, motivos, pedidos = sint.candidatos(resp)
    assert motivos[("extracao_campo", "passou")] == 5 and motivos[("extracao_campo", "resposta cortada")] == 1
    assert len(pedidos) == 2 * len(casos)
    conf = []
    for c in casos:
        for o in prof.ORDENS_CONFERENCIA:
            ids = [op[0] for op in order_options(c, o)]
            alvo = c.gold if c.group_id != casos[0].group_id else next(x for x in ids if x != c.gold)
            conf.append({"id": f"{c.id}|{o}", "texto": f"Resposta: {LETTERS[ids.index(alvo)]}"})
    final, rel = sint.montar(casos, conf)
    assert rel["extracao_campo"]["grupos_lidos"] == 5 and rel["extracao_campo"]["grupos_aprovados"] == 4
    assert casos[0].group_id not in {c.group_id for c in final}
    aud = sint.auditoria(final)
    assert aud["extracao_campo"]["casos"] == 8
