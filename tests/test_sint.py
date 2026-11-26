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
    # v2: campos do próprio domínio; o campo que muda é sorteado, sem classe dominante dentro do domínio
    por_dom = {}
    for i in range(1600):
        e = sint.especificacao("campo_alterado", i)
        nomes = [c for c, _ in sint.REGISTROS_ALTERADO[e["dominio"]][1]]
        assert set(e["campos"]) <= set(nomes) and e["registro"] == sint.REGISTROS_ALTERADO[e["dominio"]][0]
        por_dom.setdefault(e["dominio"], Counter())[e["alterado"]] += 1
    for dom, alvos in por_dom.items():
        n = sum(alvos.values())
        assert len(alvos) == len(sint.REGISTROS_ALTERADO[dom][1]) and max(alvos.values()) < 0.3 * n
    assert sint.especificacao("extracao_campo", 0)["itens"] == ["lupa de bancada", "multímetro"]  # v1 intacta
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
    with pytest.raises(prof.Rejeitado, match="rótulo"):
        c0 = e["campos"][0]
        sint.ler_resposta("campo_alterado", 5, e, json.dumps(dict(d, versao_1=d["versao_1"] + f" {c0}: ok."),
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


def test_controles():
    casos = []
    for fam, i in (("extracao_campo", 0), ("campo_alterado", 5)):
        e = sint.especificacao(fam, i)
        if fam == "extracao_campo":
            casos += sint.ler_resposta(fam, i, e, json.dumps({"documento": _doc(e)}, ensure_ascii=False))
        else:
            v1 = e["versao_1"]
            v2 = dict(v1, **{e["alterado"]: e["novo_valor"]})
            d = {"versao_1": "Primeiro registro: " + "; ".join(f"{c} {v}" for c, v in v1.items()) + ".",
                 "versao_2": "No dia seguinte: " + "; ".join(f"{c} {v2[c]}" for c in e["ordem_2"]) + "."}
            casos += sint.ler_resposta(fam, i, e, json.dumps(d, ensure_ascii=False))
    k = sint.controles(casos)
    assert [len(k[c]) for c in sint.CONDICOES] == [3, 3, 1]
    assert all(c.state == "(texto não disponível)" for c in k["sem_texto"])
    (so,) = k["so_versao_2"]
    assert so.state.startswith("Versão 2: No dia seguinte") and "Versão 1" not in so.state
    assert casos[-1].state.startswith("Versão 1")  # o original fica intacto
