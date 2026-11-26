"""Sintético em PT-BR para o E5 (analises/e5_dados.md, piloto de 25/11/2026): dados de TREINO, não de avaliação.

Mesmo desenho das reservadas com professor (prof.py), com os papéis trocados: o Gemma-4-31B escreve e o
Qwen3.8-27B confere às cegas. Domínios e gêneros disjuntos dos `TEMAS` das reservadas.

- `extracao_campo`: um documento não financeiro sobre dois itens, com três campos cada, todos no mesmo formato de
  código. Pergunta: qual é o campo X do item Y. As opções são valores do próprio documento (o mesmo campo do outro
  item e outros campos), então sem o documento não dá para acertar. 2 perguntas por documento.
- `campo_alterado`: duas versões do mesmo registro, com a ordem dos campos diferente; um campo mudou. O registro e a
  mudança são sorteados por código, então o gabarito vem da diferença entre as versões.

Fluxo: `pedidos_escritor` → scripts/prof_vllm.py → `candidatos` (filtro de forma, pedidos do conferente) →
scripts/prof_vllm.py → `montar` (fica o grupo em que o conferente acerta tudo nas 2 ordens; split `train`).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import string
from collections import Counter, defaultdict

from .prof import ORDENS_CONFERENCIA, Rejeitado, _campos, _json, ler_conferente, pedido_conferente
from .schema import Case

SINT_VERSION = "v1"
SINT_FAMILIES = ("extracao_campo", "campo_alterado")
SALT = "lumi-decision-sint-pt-v1"

SISTEMA_ESCRITOR = ("Você escreve documentos curtos em português do Brasil para um conjunto de treino de modelos de "
                    "linguagem. Siga a especificação à risca e copie cada valor exatamente como está escrito. "
                    "Responda só com um objeto JSON válido, sem texto antes ou depois e sem blocos de código.")

# Domínios disjuntos dos TEMAS das reservadas (condomínio, bairro, escola, trabalho, clube, eventos, transporte, casa).
DOMINIOS = {
    "oficina mecânica": ["elevador hidráulico", "scanner automotivo", "compressor de ar", "macaco jacaré",
                         "balanceadora de rodas", "carregador de bateria"],
    "clínica veterinária": ["autoclave", "balança de consultório", "monitor cardíaco", "concentrador de oxigênio",
                            "aparelho de raio X portátil", "bomba de infusão"],
    "biblioteca": ["leitor de código de barras", "scanner de livros", "impressora de etiquetas", "carrinho de livros",
                   "totem de autoatendimento", "estante deslizante"],
    "laboratório de análises": ["centrífuga", "microscópio", "estufa de secagem", "pipeta automática",
                                "agitador magnético", "geladeira de amostras"],
    "almoxarifado de hospital": ["cadeira de rodas", "maca de transporte", "suporte de soro", "desfibrilador",
                                 "carrinho de emergência", "oxímetro de mesa"],
    "gráfica": ["guilhotina", "plotter de recorte", "impressora digital", "laminadora", "grampeador elétrico",
                "dobradeira de papel"],
    "cooperativa agrícola": ["roçadeira", "pulverizador costal", "balança de sacas", "secador de grãos",
                             "trator compacto", "debulhadora"],
    "assistência técnica": ["estação de solda", "fonte de bancada", "multímetro", "separadora de telas",
                            "osciloscópio", "lupa de bancada"],
}
DOCUMENTOS = ["uma ficha de cadastro de equipamentos", "uma ordem de serviço de manutenção",
              "um registro de empréstimo entre setores", "um laudo de vistoria", "um e-mail de transferência de itens",
              "uma nota de inventário escrita por um funcionário"]
CAMPOS_EXTRACAO = ["número de série", "código de patrimônio", "número do lote de compra", "registro interno",
                   "código do contrato de manutenção", "número da última ordem de serviço"]
TIPOS_REGISTRO = {
    "clínica veterinária": "a ficha de um animal atendido",
    "biblioteca": "o cadastro de um leitor",
    "oficina mecânica": "a ordem de um conserto",
    "laboratório de análises": "o pedido de um exame",
    "almoxarifado de hospital": "a requisição de um material",
    "gráfica": "o pedido de um trabalho de impressão",
    "cooperativa agrícola": "o cadastro de uma entrega de safra",
    "assistência técnica": "a ordem de um reparo",
}
NOMES = ["Adriana", "Bernardo", "Cecília", "Douglas", "Elaine", "Fabiano", "Gisele", "Heitor", "Ingrid", "Jonas",
         "Kátia", "Leandro", "Mônica", "Nelson", "Olga", "Pedro", "Rosana", "Sérgio", "Tânia", "Wagner"]
BAIRROS = ["Vila Esperança", "Jardim Primavera", "Alto da Serra", "Parque das Águas", "Boa Vista", "Santa Luzia",
           "Morada do Sol", "Vila Nova", "Recanto Verde", "Centro Histórico"]
CATEGORIAS = ["básica", "padrão", "prioritária", "especial", "empresarial", "simples"]
GENEROS_ALTERADO = ["a ficha original e a ficha reescrita depois de uma atualização",
                    "o registro no sistema e um e-mail que confirma os dados depois de uma correção",
                    "a primeira versão do formulário e a versão corrigida no dia seguinte",
                    "as anotações do atendimento e o resumo enviado ao cliente depois de uma mudança"]


def _codigo(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_uppercase, k=2)) + "-" + "".join(rng.choices(string.digits, k=4))


def _valor(tipo: str, rng: random.Random) -> str:
    if tipo == "data":
        return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}"
    if tipo == "horario":
        return f"{rng.randint(7, 19)}h{rng.choice(['00', '15', '30', '45'])}"
    if tipo == "quantidade":
        return f"{rng.randint(2, 95)} unidades"
    if tipo == "codigo":
        return _codigo(rng)
    if tipo == "nome":
        return rng.choice(NOMES)
    if tipo == "bairro":
        return rng.choice(BAIRROS)
    if tipo == "categoria":
        return rng.choice(CATEGORIAS)
    raise ValueError(tipo)


CAMPOS_ALTERADO = [("data de entrega", "data"), ("horário combinado", "horario"), ("quantidade", "quantidade"),
                   ("código de referência", "codigo"), ("responsável", "nome"), ("bairro de entrega", "bairro"),
                   ("categoria do atendimento", "categoria")]


def especificacao(fam: str, i: int, salt: str = SALT) -> dict:
    rng = random.Random(f"{salt}:{SINT_VERSION}:{fam}:{i}")
    dominio = rng.choice(sorted(DOMINIOS))
    if fam == "extracao_campo":
        itens = rng.sample(DOMINIOS[dominio], 2)
        campos = rng.sample(CAMPOS_EXTRACAO, 3)
        valores = set()
        while len(valores) < 6:
            valores.add(_codigo(rng))
        valores = sorted(valores)
        rng.shuffle(valores)
        tabela = {it: dict(zip(campos, valores[3 * k:3 * k + 3])) for k, it in enumerate(itens)}
        perguntas = rng.sample([(it, c) for it in itens for c in campos], 2)
        return {"dominio": dominio, "documento": rng.choice(DOCUMENTOS), "itens": itens, "campos": campos,
                "tabela": tabela, "perguntas": [list(p) for p in perguntas]}
    if fam == "campo_alterado":
        campos = rng.sample(CAMPOS_ALTERADO, 5)
        v1 = {c: _valor(t, rng) for c, t in campos}
        while len(set(v1.values())) < 5:
            v1 = {c: _valor(t, rng) for c, t in campos}
        alvo, tipo = rng.choice(campos)
        novo = _valor(tipo, rng)
        while novo in v1.values():
            novo = _valor(tipo, rng)
        ordem2 = [c for c, _ in campos]
        rng.shuffle(ordem2)
        return {"dominio": dominio, "registro": TIPOS_REGISTRO[dominio], "genero": rng.choice(GENEROS_ALTERADO),
                "campos": [c for c, _ in campos], "versao_1": v1, "ordem_2": ordem2, "alterado": alvo,
                "novo_valor": novo}
    raise ValueError(fam)


def pedido_escritor(fam: str, e: dict) -> str:
    if fam == "extracao_campo":
        linhas = "\n".join(f"- {it}: " + "; ".join(f"{c} {v}" for c, v in e["tabela"][it].items())
                           for it in e["itens"])
        return f"""Escreva {e['documento']} de 4 a 7 frases, no contexto de "{e['dominio']}", sobre dois itens e os seus dados:
{linhas}

Regras:
- Cada um dos seis códigos aparece exatamente uma vez, escrito exatamente como acima, e fica claro a qual item e a qual campo ele pertence.
- Escreva como um documento real (frases, não uma tabela): o nome do campo pode vir com outras palavras ("n.º de série", "patrimônio", "lote da compra").
- Não acrescente outros códigos nem números com o mesmo formato.
- Os dois itens aparecem pelo nome.

JSON: {{"documento": "..."}}"""
    if fam == "campo_alterado":
        v1 = "\n".join(f"- {c}: {v}" for c, v in e["versao_1"].items())
        v2 = dict(e["versao_1"], **{e["alterado"]: e["novo_valor"]})
        v2 = "\n".join(f"- {c}: {v2[c]}" for c in e["ordem_2"])
        return f"""Escreva {e['genero']}, sobre {e['registro']}, no contexto de "{e['dominio']}". São duas versões do mesmo registro.

Dados da versão 1:
{v1}

Dados da versão 2:
{v2}

Regras:
- Cada versão tem de 2 a 4 frases e menciona todos os seus cinco dados, cada valor escrito exatamente como acima e uma única vez.
- As duas versões são textos diferentes, com outras palavras e na ordem indicada de cada uma; não diga qual dado mudou nem use palavras como "alterado", "corrigido para" ou "antes era".
- Não acrescente outros dados.

JSON: {{"versao_1": "...", "versao_2": "..."}}"""
    raise ValueError(fam)


def pedidos_escritor(inicio: int, fim: int, familias: list[str] | None = None) -> list[dict]:
    out = []
    for fam in familias or SINT_FAMILIES:
        for i in range(inicio, fim):
            e = especificacao(fam, i)
            out.append({"id": f"sint:{fam}:{i}", "familia": fam, "indice": i, "especificacao": e,
                        "messages": [{"role": "system", "content": SISTEMA_ESCRITOR},
                                     {"role": "user", "content": pedido_escritor(fam, e)}],
                        "temperature": 0.8, "top_p": 0.95, "max_tokens": 1500,
                        "seed": int(hashlib.sha256(f"{SALT}:seed:{fam}:{i}".encode()).hexdigest()[:8], 16)})
    return out


def _vezes(texto: str, valor: str) -> int:
    return len(re.findall(rf"(?<![\w-]){re.escape(valor)}(?![\w-])", texto, re.I))


def ler_resposta(fam: str, i: int, e: dict, texto: str) -> list[Case]:
    """Filtro de forma. Devolve os casos do grupo ou levanta `Rejeitado` com o motivo."""
    d = _json(texto)
    g = f"sint:{fam}:{i}"
    meta = {"especificacao": e, "versao": SINT_VERSION, "escritor": d}

    def caso(k: int, estado: str, pergunta: str, opcoes, gold: str) -> Case:
        return Case(id=f"{g}:{k}", group_id=g, source="sint_pt", family=fam, split="train", held_out=False,
                    qtype="choice", state=estado, question=pergunta, options=list(opcoes), gold=gold, meta=meta)

    if fam == "extracao_campo":
        _campos(d, ["documento"], 200, 1500)
        doc = d["documento"]
        for it in e["itens"]:
            if it.casefold() not in doc.casefold():
                raise Rejeitado("item ausente")
            for v in e["tabela"][it].values():
                if _vezes(doc, v) != 1:
                    raise Rejeitado("código ausente ou repetido")
        if len(re.findall(r"\b[A-Z]{2}-\d{4}\b", doc)) != 6:
            raise Rejeitado("código a mais")
        casos = []
        for k, (it, c) in enumerate(e["perguntas"]):
            gold = e["tabela"][it][c]
            outro = next(x for x in e["itens"] if x != it)
            resto = sorted(v for x in e["itens"] for cc, v in e["tabela"][x].items()
                           if v not in (gold, e["tabela"][outro][c]))
            rng = random.Random(f"{SALT}:opcoes:{g}:{k}")
            vals = [gold, e["tabela"][outro][c]] + rng.sample(resto, 2)
            rng.shuffle(vals)
            casos.append(caso(k, f"Documento: {doc}", f"No documento, qual é o {c} do item \"{it}\"?",
                              [(v, v) for v in vals], gold))
        return casos
    if fam == "campo_alterado":
        _campos(d, ["versao_1", "versao_2"], 60, 900)
        v1t, v2t = d["versao_1"], d["versao_2"]
        v2 = dict(e["versao_1"], **{e["alterado"]: e["novo_valor"]})
        for t, vals in ((v1t, e["versao_1"]), (v2t, v2)):
            for v in vals.values():
                if _vezes(t, v) != 1:
                    raise Rejeitado("valor ausente ou repetido")
        if _vezes(v2t, e["versao_1"][e["alterado"]]) or _vezes(v1t, e["novo_valor"]):
            raise Rejeitado("valor antigo na versão 2 ou novo na versão 1")
        if re.search(r"\b(alterad|corrigid|mudou|mudança|antes era|atualizad[oa] para)", v1t + " " + v2t, re.I):
            raise Rejeitado("texto denuncia a mudança")
        rng = random.Random(f"{SALT}:opcoes:{g}")
        outros = rng.sample([c for c in e["campos"] if c != e["alterado"]], 3)
        ops = [e["alterado"]] + outros
        rng.shuffle(ops)
        return [caso(0, f"Versão 1: {v1t}\n\nVersão 2: {v2t}",
                     "Entre a versão 1 e a versão 2 do registro, qual dado mudou?", [(c, c) for c in ops],
                     e["alterado"])]
    raise ValueError(fam)


def candidatos(respostas: list[dict]) -> tuple[list[Case], Counter, list[dict]]:
    casos, motivos, pedidos = [], Counter(), []
    for r in respostas:
        fam, i = r["familia"], r["indice"]
        e = especificacao(fam, i)
        if e != r["especificacao"]:
            raise AssertionError(f"{r['id']}: especificação do arquivo difere da do código")
        if r.get("finish_reason") != "stop":
            motivos[(fam, "resposta cortada")] += 1
            continue
        try:
            cs = ler_resposta(fam, i, e, r["texto"])
        except Rejeitado as err:
            motivos[(fam, str(err))] += 1
            continue
        motivos[(fam, "passou")] += 1
        casos += cs
        for c in cs:
            for o in ORDENS_CONFERENCIA:
                pedidos.append({"id": f"{c.id}|{o}", "messages": pedido_conferente(c, o), "temperature": 0.0,
                                "top_p": 1.0, "max_tokens": 400, "seed": 0})
    return casos, motivos, pedidos


def montar(casos: list[Case], conferencia: list[dict]) -> tuple[list[Case], dict]:
    """Fica com os grupos em que o conferente acerta todos os casos nas 2 ordens (resposta completa)."""
    resp = {r["id"]: r["texto"] for r in conferencia if r.get("finish_reason", "stop") == "stop"}
    por_grupo = defaultdict(list)
    for c in casos:
        por_grupo[c.group_id].append(c)
    final, rel = [], defaultdict(lambda: Counter())
    for g, cs in sorted(por_grupo.items()):
        acertos = []
        for c in cs:
            for o in ORDENS_CONFERENCIA:
                t = resp.get(f"{c.id}|{o}")
                esc = None if t is None else ler_conferente(t, c, o)
                rel[c.family]["ilegível" if esc is None else ("acerta" if esc == c.gold else "erra")] += 1
                acertos.append(esc == c.gold)
        rel[cs[0].family]["grupos_lidos"] += 1
        if all(acertos):
            rel[cs[0].family]["grupos_aprovados"] += 1
            final += cs
    return final, {f: dict(v) for f, v in rel.items()}


def auditoria(casos: list[Case]) -> dict:
    """Controles do lote pequeno (sem modelo). Por construção, sem o documento não há informação: em
    `extracao_campo` as opções são códigos sorteados no mesmo formato; em `campo_alterado` o campo que muda é
    sorteado e, com só uma versão, nenhum valor denuncia a mudança. O que pode vazar é a posição do gabarito, a
    classe mais comum e, no `campo_alterado`, o valor novo aparecer de um jeito diferente dos outros; isso fica para
    a leitura dos textos e para o lote de teste com o Lumi congelado só com a versão 2."""
    out = {}
    for fam in SINT_FAMILIES:
        cs = [c for c in casos if c.family == fam]
        if not cs:
            continue
        pos = Counter([o[0] for o in c.options].index(c.gold) for c in cs)
        item = {"casos": len(cs), "acaso": round(sum(1 / len(c.options) for c in cs) / len(cs), 3),
                "posicao_do_gabarito": dict(sorted(pos.items())),
                "melhor_posicao": f"{max(pos.values())}/{len(cs)}"}
        if fam == "campo_alterado":
            classes = Counter(c.gold for c in cs)
            item["classe_mais_comum"] = f"{max(classes.values())}/{len(cs)}"
        out[fam] = item
    return out


def gravar(casos: list[Case], path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for c in sorted(casos, key=lambda c: c.id):
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()
