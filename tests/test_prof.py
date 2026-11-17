"""Reservadas com professor (src/lumi_decision/prof.py): especificações, filtro de forma, conferente e montagem.
Sem modelo: as respostas do escritor e do conferente são escritas à mão."""

import json
from collections import Counter

import pytest

from lumi_decision import prof
from lumi_decision.data import check_integrity, load_config
from lumi_decision.prompts import LETTERS, order_options


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _spec(cfg, fam, **troca):
    """Primeira especificação da família que satisfaz `troca` (para os exemplos escritos à mão)."""
    for i in range(500):
        e = prof.especificacao(fam, i, cfg["prof"]["salt"])
        if all(e.get(k) == v for k, v in troca.items()):
            return i, e
    raise AssertionError("nenhuma especificação com essas características")


def test_especificacao_deterministica_e_equilibrada(cfg):
    salt = cfg["prof"]["salt"]
    for fam in prof.PROF_FAMILIES:
        assert prof.especificacao(fam, 7, salt) == prof.especificacao(fam, 7, salt)
    posturas = Counter(p for i in range(600) for p in prof.especificacao("postura_alvo", i, salt)["posturas"])
    assert all(350 <= n <= 450 for n in posturas.values())
    neg = Counter(k for i in range(600) for k, v in prof.especificacao("inferencia_pt", i, salt)["negacao"].items() if v)
    assert all(170 <= n <= 250 for n in neg.values())  # ~35% em cada classe, independente da classe
    refs = Counter(prof.especificacao("correferencia", i, salt)["referente_v1"] for i in range(600))
    assert abs(refs[0] - refs[1]) < 80


def test_pedidos_tem_semente_e_sistema(cfg):
    rows = prof.pedidos_escritor(cfg, 0, 3)
    assert len(rows) == 3 * len(cfg["prof"]["familias"])
    assert len({r["seed"] for r in rows}) == len(rows)
    assert all(r["messages"][0]["content"] == prof.SISTEMA_ESCRITOR for r in rows)


def test_postura(cfg):
    i, e = _spec(cfg, "postura_alvo", posturas=["a_favor", "sem_posicao"])
    a, b = e["alvos"]
    texto = (f"Gente, sobre {a}: acho ótimo, finalmente alguém pensou nisso e vai melhorar a vida de todo mundo. "
             f"Agora, quanto a {b}, sinceramente ainda não sei, tem pontos bons e ruins e vou esperar a reunião.")
    cs = prof.ler_resposta("postura_alvo", i, e, json.dumps({"comentario": texto}, ensure_ascii=False))
    assert [c.gold for c in cs] == ["a_favor", "sem_posicao"]
    assert a in cs[0].question and b in cs[1].question
    with pytest.raises(prof.Rejeitado, match="proibida"):
        prof.ler_resposta("postura_alvo", i, e, json.dumps({"comentario": texto + " Sou contra o resto."}))


def test_inferencia_negacao_segue_a_especificacao(cfg):
    i, e = _spec(cfg, "inferencia_pt", negacao={"decorre": False, "contradiz": True, "neutra": False})
    d = {"texto": "Marta chegou à reunião do condomínio às 19h e votou pela pintura da fachada, que custa R$ 40 mil.",
         "decorre": "Marta participou da reunião do condomínio naquela noite.",
         "contradiz": "Marta não foi à reunião do condomínio naquela noite.",
         "neutra": "Marta mora no prédio há mais de dez anos com a família."}
    cs = prof.ler_resposta("inferencia_pt", i, e, json.dumps(d, ensure_ascii=False))
    assert [c.gold for c in cs] == ["decorre", "contradiz", "neutra"]
    assert len({c.group_id for c in cs}) == 1
    d2 = dict(d, decorre="Marta não faltou à reunião do condomínio naquela noite.")
    with pytest.raises(prof.Rejeitado, match="negação"):
        prof.ler_resposta("inferencia_pt", i, e, json.dumps(d2, ensure_ascii=False))


def test_resposta_indireta(cfg):
    i, e = _spec(cfg, "resposta_indireta", se_quando={"sim": True, "nao": False, "condicional": True})
    a, b = e["nomes"]
    d = {"situacao": f"{a} e {b} conversam na saída do prédio numa sexta à tarde.",
         "pergunta": "Você vai ao churrasco no sábado?",
         "sim": "Se tem uma coisa que eu adoro é churrasco, já separei a roupa.",
         "nao": "Vou passar o sábado inteiro de plantão na loja do meu tio.",
         "condicional": "Vou, se o meu carro sair da oficina até amanhã cedo."}
    cs = prof.ler_resposta("resposta_indireta", i, e, "```json\n" + json.dumps(d, ensure_ascii=False) + "\n```")
    assert [c.gold for c in cs] == ["sim", "nao", "condicional"]
    assert f'{b} responde: "Vou passar' in cs[1].state
    with pytest.raises(prof.Rejeitado, match="proibida"):
        prof.ler_resposta("resposta_indireta", i, e, json.dumps(dict(d, nao="Não dá, tenho plantão no sábado.")))
    with pytest.raises(prof.Rejeitado, match="se/quando"):
        prof.ler_resposta("resposta_indireta", i, e, json.dumps(dict(d, nao="Vou trabalhar quando der o sábado.")))


def test_correferencia_par_minimo(cfg):
    i, e = _spec(cfg, "correferencia", genero="f", referente_v1=0)
    n1, n2 = e["nomes"]
    d = {"versao_1": f"{n1} emprestou o livro para {n2} porque ela já tinha lido.",
         "versao_2": f"{n1} emprestou o livro para {n2} porque ela ainda não tinha lido."}
    cs = prof.ler_resposta("correferencia", i, e, json.dumps(d, ensure_ascii=False))
    assert [c.gold for c in cs] == ["pessoa_1", "pessoa_2"]
    assert cs[0].options == [("pessoa_1", n1), ("pessoa_2", n2)]
    longe = dict(d, versao_2=f"{n1} emprestou o livro para {n2} porque ela queria muito ler a história toda.")
    with pytest.raises(prof.Rejeitado, match="3 palavras"):
        prof.ler_resposta("correferencia", i, e, json.dumps(longe, ensure_ascii=False))
    duas = dict(d, versao_2=f"{n1} emprestou o livro para {n2} porque ela disse que ela não leu.")
    with pytest.raises(prof.Rejeitado, match="pronome"):
        prof.ler_resposta("correferencia", i, e, json.dumps(duas, ensure_ascii=False))


def _casos_falsos(cfg, n):
    """`n` grupos por família, montados pelo filtro de forma a partir de respostas escritas à mão."""
    casos = []
    salt = cfg["prof"]["salt"]
    for j in range(n):
        e = prof.especificacao("correferencia", j, salt)
        n1, n2 = e["nomes"]
        p = e["pronome"]
        d = {"versao_1": f"{n1} ajudou {n2} na mudança porque {p} tinha um carro grande número {j}.",
             "versao_2": f"{n1} ajudou {n2} na mudança porque {p} precisava de um carro grande número {j}."}
        casos += prof.ler_resposta("correferencia", j, e, json.dumps(d, ensure_ascii=False))
    return casos


def _conferencia(casos, errar=()):
    out = []
    for c in casos:
        for o in prof.ORDENS_CONFERENCIA:
            ids = [op[0] for op in order_options(c, o)]
            alvo = c.gold if c.group_id not in errar else next(x for x in ids if x != c.gold)
            out.append({"id": f"{c.id}|{o}", "texto": f"Pensei bem.\nResposta: {LETTERS[ids.index(alvo)]}"})
    return out


def test_conferente_le_a_letra_na_ordem_certa(cfg):
    c = _casos_falsos(cfg, 1)[0]
    for o in prof.ORDENS_CONFERENCIA:
        conf = _conferencia([c])
        texto = next(r["texto"] for r in conf if r["id"].endswith(o))
        assert prof.ler_conferente(texto, c, o) == c.gold
    assert prof.ler_conferente("sem resposta", c, "canonical") is None
    assert len(prof.pedido_conferente(c, "rot1")) == 2


def test_montar_filtra_seleciona_e_divide(cfg):
    conf = json.loads(json.dumps(cfg))
    conf["prof"].update(familias=["correferencia"], grupos_por_familia=20, grupos_dificuldade=4)
    casos = _casos_falsos(conf, 30)
    errados = {casos[0].group_id, casos[2].group_id}
    final, rel = prof.montar(conf, casos, _conferencia(casos, errar=errados))
    r = rel["correferencia"]
    assert r["grupos_lidos"] == 30 and r["passam_conferencia"] == 28 and r["escolhidos"] == 20
    assert not errados & {c.group_id for c in final}
    check_integrity(final)
    grupos = Counter(c.split for c in final if c.id.endswith(":0"))
    assert grupos["dificuldade"] == 4 and grupos["dev"] + grupos["test"] == 16
    rep = prof.prof_audit(conf, final)["familias"]["correferencia"]
    assert rep["atalhos_de_leitura_parcial"]["ignora_a_palavra_trocada"] == "20/40"
