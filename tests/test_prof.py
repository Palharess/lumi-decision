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
    inf = [prof.especificacao("inferencia_pt", i, salt) for i in range(600)]
    assert 170 <= sum(e["negacao_hipotese"] for e in inf) <= 250 and 170 <= sum(e["negacao_textos"] for e in inf) <= 250
    assert len({e["mecanismo"] for e in inf}) == len(prof.MEC_INFERENCIA)
    versoes = Counter(prof.especificacao("resposta_indireta", i, salt)["sim_na_versao"] for i in range(600))
    assert abs(versoes[1] - versoes[2]) < 80
    refs = Counter(prof.especificacao("correferencia", i, salt)["referente_v1"] for i in range(600))
    assert abs(refs[0] - refs[1]) < 80


def test_pedidos_tem_semente_e_sistema(cfg):
    rows = prof.pedidos_escritor(cfg, 0, 3)
    assert len(rows) == 3 * len(cfg["prof"]["familias"])
    assert len({r["seed"] for r in rows}) == len(rows)
    assert all(r["messages"][0]["content"] == prof.SISTEMA_ESCRITOR for r in rows)
    assert all(r["top_k"] == 20 and r["max_tokens"] == 12000 for r in rows)
    for r in rows:  # emenda 2: os nomes da inferência vêm da especificação e aparecem no pedido
        if r["familia"] == "inferencia_pt":
            assert all(n in r["messages"][1]["content"] for n in r["especificacao"]["nomes"])
    # emenda 4: postura e correferência continuam com as especificações da v1 (as já geradas valem)
    salt = cfg["prof"]["salt"]
    assert prof.especificacao("correferencia", 3, salt)["nomes"] == prof.especificacao("correferencia", 3, salt)["nomes"]
    assert prof.VERSAO_FAMILIA["correferencia"] == "v1" and prof.VERSAO_FAMILIA["postura_alvo"] == "v2"
    for r in rows:  # emenda 6: o modo de cada posição aparece no pedido da postura
        if r["familia"] == "postura_alvo":
            assert all(m in r["messages"][1]["content"] for m in r["especificacao"]["modos"])


def test_postura(cfg):
    i, e = _spec(cfg, "postura_alvo", posturas=["a_favor", "sem_posicao"])
    a, b = e["alvos"]
    texto = (f"Gente, sobre {a}: finalmente alguém pensou nisso, e a vida de todo mundo vai melhorar já no primeiro mês. "
             f"Agora, quanto a {b}, sinceramente ainda não sei, tem pontos bons e ruins e vou esperar a reunião.")
    cs = prof.ler_resposta("postura_alvo", i, e, json.dumps({"comentario": texto}, ensure_ascii=False))
    assert [c.gold for c in cs] == ["a_favor", "sem_posicao"]
    assert a in cs[0].question and b in cs[1].question
    with pytest.raises(prof.Rejeitado, match="proibida"):
        prof.ler_resposta("postura_alvo", i, e, json.dumps({"comentario": texto + " Sou contra o resto."}))
    with pytest.raises(prof.Rejeitado, match="proibida"):  # emenda 6: adjetivo de avaliação direta
        prof.ler_resposta("postura_alvo", i, e, json.dumps({"comentario": texto + " Achei ótima a ideia."}))


def test_inferencia_v2_uma_hipotese_tres_textos(cfg):
    i, e = _spec(cfg, "inferencia_pt", negacao_hipotese=False, negacao_textos=False)
    n1, n2 = e["nomes"]
    base = f"{n1} e {n2} foram à reunião do condomínio às 19h, que votou a pintura da fachada por R$ 40 mil."
    d = {"hipotese": f"{n1} participou da votação sobre a fachada naquela noite.",
         "texto_decorre": base + f" {n1} votou a favor da proposta e {n2} anotou o resultado.",
         "texto_contradiz": base + f" {n1} saiu antes da votação e {n2} anotou o resultado.",
         "texto_neutra": base + f" {n1} levou um bolo para a reunião e {n2} anotou o resultado."}
    cs = prof.ler_resposta("inferencia_pt", i, e, json.dumps(d, ensure_ascii=False))
    assert [c.gold for c in cs] == ["decorre", "contradiz", "neutra"]
    assert len({c.state.split("Hipótese: ")[1] for c in cs}) == 1  # a mesma hipótese nos 3 casos
    assert cs[0].meta["versao"] == "v2"
    with pytest.raises(prof.Rejeitado, match="negação dos textos"):
        prof.ler_resposta("inferencia_pt", i, e, json.dumps(
            dict(d, texto_contradiz=base + f" {n1} não votou e {n2} anotou o resultado."), ensure_ascii=False))
    with pytest.raises(prof.Rejeitado, match="sobreposição com a hipótese"):  # emenda 5: a neutra foge das palavras
        prof.ler_resposta("inferencia_pt", i, e, json.dumps(dict(d, hipotese=(
            f"{n1} votou a favor da proposta da fachada e {n2} anotou o resultado.")), ensure_ascii=False))
    with pytest.raises(prof.Rejeitado, match="diferem demais"):
        prof.ler_resposta("inferencia_pt", i, e, json.dumps(dict(d, texto_neutra=(
            f"Na assembleia de domingo, {n2} cuidou da lista de presença enquanto {n1} servia café e bolo para os "
            "vizinhos que chegavam para discutir a reforma da piscina e o novo regulamento do salão.")),
            ensure_ascii=False))


def test_resposta_indireta_v2_uma_resposta_duas_perguntas(cfg):
    i, e = _spec(cfg, "resposta_indireta", sim_na_versao=2)
    a, b = e["nomes"]
    d = {"situacao": f"{a} e {b} conversam na saída do prédio numa sexta à tarde.",
         "pergunta_1": "Você vai de carro ao churrasco amanhã?",
         "pergunta_2": "Você vai de ônibus ao churrasco amanhã?",
         "resposta": "Meu Gol está na oficina até segunda."}
    raciocinio = "Preciso de duas perguntas {sem} sim nem não...\n</think>\n\n"  # o raciocínio pode ter chaves
    cs = prof.ler_resposta("resposta_indireta", i, e,
                           raciocinio + "```json\n" + json.dumps(d, ensure_ascii=False) + "\n```")
    assert [c.gold for c in cs] == ["nao", "sim"]
    assert all(f'{b} responde: "Meu Gol está na oficina' in c.state for c in cs)
    assert cs[0].options == [("sim", "Sim."), ("nao", "Não.")]
    with pytest.raises(prof.Rejeitado, match="sem fim"):
        prof.ler_resposta("resposta_indireta", i, e, "<think>ainda pensando {")
    with pytest.raises(prof.Rejeitado, match="proibida"):
        prof.ler_resposta("resposta_indireta", i, e, json.dumps(dict(d, resposta="Claro, o Gol está na oficina.")))
    with pytest.raises(prof.Rejeitado, match="palavra trocada"):  # emenda 5
        prof.ler_resposta("resposta_indireta", i, e, json.dumps(dict(d, resposta="Meu carro está na oficina.")))
    with pytest.raises(prof.Rejeitado, match="se/quando"):
        prof.ler_resposta("resposta_indireta", i, e, json.dumps(dict(d, resposta="Vou se o carro ficar pronto.")))
    with pytest.raises(prof.Rejeitado, match="4 palavras"):
        prof.ler_resposta("resposta_indireta", i, e, json.dumps(
            dict(d, pergunta_2="Quer que eu passe na sua casa para irmos juntos de ônibus?"), ensure_ascii=False))


def test_auditoria_v2_controles(cfg):
    conf = json.loads(json.dumps(cfg))
    conf["prof"].update(familias=["resposta_indireta"])
    casos = []
    for j in range(40):
        i, e = j, prof.especificacao("resposta_indireta", j, conf["prof"]["salt"])
        d = {"situacao": "Os dois conversam na portaria do prédio no fim da tarde.",
             "pergunta_1": f"Você vai de carro ao evento {j}?", "pergunta_2": f"Você vai de ônibus ao evento {j}?",
             "resposta": f"Meu Gol está na oficina até a semana {j}."}
        casos += prof.ler_resposta("resposta_indireta", i, e, json.dumps(d, ensure_ascii=False))
    rep = prof.prof_audit(conf, casos)["familias"]["resposta_indireta"]
    assert rep["acaso"] == 0.5 and rep["limites"]["forma"] == 0.6
    assert {"tfidf_so_resposta", "tfidf_so_pergunta"} <= set(rep["atalhos_de_forma"])
    assert int(rep["atalhos_de_forma"]["tfidf_so_resposta"].split("/")[0]) <= 48  # mesma resposta nos 2 casos


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


def test_montar_descarta_conferente_cortado(cfg):
    conf = json.loads(json.dumps(cfg))
    conf["prof"].update(familias=["correferencia"], grupos_por_familia=20, grupos_dificuldade=4)
    casos = _casos_falsos(conf, 3)
    resp = _conferencia(casos)
    alvo = casos[0].group_id
    for r in resp:
        if r["id"].startswith(alvo + ":"):
            r["finish_reason"] = "length"  # texto com "Resposta: X" certo, mas cortado: não vale
    final, rel = prof.montar(conf, casos, resp)
    r = rel["correferencia"]
    assert r["passam_conferencia"] == 2 and r["conferente_cortado"] == 4 and r["faltam"] == 18
    assert alvo not in {c.group_id for c in final}


def test_carregar_exige_registro(cfg, tmp_path):
    conf = json.loads(json.dumps(cfg))
    conf["prof"].update(familias=["correferencia"], grupos_por_familia=20, grupos_dificuldade=4, sha256="")
    final, _ = prof.montar(conf, _casos_falsos(conf, 20), _conferencia(_casos_falsos(conf, 20)))
    p = tmp_path / "prof.jsonl"
    sha = prof.gravar_casos(final, p)
    with pytest.raises(ValueError, match="sem registro"):
        prof.prof_cases(conf, p)
    assert len(prof.prof_cases(conf, p, antes_do_registro=True)) == 40
    conf["prof"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256 diferente"):
        prof.prof_cases(conf, p)
    conf["prof"]["sha256"] = sha
    assert len(prof.prof_cases(conf, p)) == 40


def test_revisao_e_registro(cfg):
    conf = json.loads(json.dumps(cfg))
    conf["prof"].update(familias=["correferencia"], grupos_por_familia=40, grupos_dificuldade=4)
    casos = _casos_falsos(conf, 40)
    final, rel = prof.montar(conf, casos, _conferencia(casos))
    assert rel["correferencia"]["faltam"] == 0
    md, gab = prof.revisao_humana(conf, final, 30)
    assert len(gab) == 60
    # preenche tudo certo, menos 2 itens de grupos diferentes e 1 em branco
    letras = {k: g["letra"] for k, g in gab.items()}
    letras["1"] = "B" if letras["1"] == "A" else "A"
    letras["7"] = "?"
    letras["12"] = ""
    feito, k = [], 0
    for linha in md.split("\n"):
        if linha.startswith("**Item "):
            k = linha.split()[1].rstrip("*")
        if linha.strip() == "Resposta:":
            linha = f"   Resposta: {letras[k]}"
        feito.append(linha)
    corr = prof.corrigir_revisao("\n".join(feito), gab)["correferencia"]
    assert corr["itens"] == 60 and corr["itens_errados"] == 3 and corr["grupos_revisados"] == 30
    assert len(corr["grupos_errados"]) == len({gab[i]["grupo"] for i in ("1", "7", "12")})
    aud = prof.prof_audit(conf, final)
    aud["familias"]["correferencia"]["acima_do_limite"] = []  # os casos falsos são todos iguais; aqui só o fluxo
    assert prof.checar_registro(conf, final, {"correferencia": corr}, aud, {"felhen": 0}) == []
    probs = prof.checar_registro(conf, final[:-1], {"correferencia": dict(corr, grupos_errados=4)}, aud, {"eikos": 2})
    assert any("1 grupos incompletos" in p for p in probs) and any("4 grupos errados" in p for p in probs)
    assert any("eikos" in p for p in probs)
    assert prof.checar_registro(conf, final, {}, aud, {}) == ["correferencia: sem revisão humana adjudicada"]
