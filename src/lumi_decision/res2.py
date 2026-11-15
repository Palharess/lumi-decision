"""Reservadas v2 (analises/reservadas_v2.md): famílias reservadas novas com gabarito por construção.

Quatro famílias programáticas, escritas aqui em PT-BR, com sorteios determinísticos (sal do `splits.toml [res2]`):
- `restricoes_oferta`: qual das 4 ofertas cumpre todas as condições do pedido (cada distratora viola uma);
- `regra_beneficio`: aplicar um regulamento curto, gerado por caso, a um pedido (conceder / negar / pedir
  documento / encaminhar à comissão), com ordem de precedência explícita;
- `custo_plano`: qual dos 3 planos sai mais barato para um consumo dado (mensalidade, franquia, excedente);
- `ordem_eventos`: num relato com datas absolutas e relativas, qual dos 4 acontecimentos veio primeiro ou por
  último (a ordem do relato não é a cronológica).

Todas são reservadas: nunca entram em treino, seleção ou calibração. Splits: `dificuldade` (50 grupos, só para o
filtro de dificuldade com o Lumi congelado, descartados depois), `dev` e `test` (metade cada). Cada caso guarda em
`meta` os dados estruturados, e `verificar_gabarito` recalcula o gabarito por um caminho independente.
"""

from __future__ import annotations

import random
import re
from collections import Counter

from .data import bucket
from .schema import Case

RES2_VERSION = "v1"
RES2_FAMILIES = ("restricoes_oferta", "regra_beneficio", "custo_plano", "ordem_eventos")


def _reais(v: float) -> str:
    """R$ 2.350 ou R$ 4,50 (formato brasileiro)."""
    if abs(v - round(v)) < 1e-9:
        return "R$ " + f"{int(round(v)):,}".replace(",", ".")
    return "R$ " + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _lista(itens: list[str]) -> str:
    return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1]


# ------------------------------------------------------------------ restricoes_oferta

# Atributos: (chave, tipo, parâmetros). Tipos: "max" (valor até X), "min" (pelo menos X), "bool" (precisa ter),
# "cat" (precisa ser X). Cada atributo tem o texto do pedido e o texto da oferta, com palavras diferentes.
OFERTA_DOMINIOS = {
    "apartamento": {
        "abertura": ["Estou procurando um apartamento para alugar.", "Preciso alugar um apartamento."],
        "extra": ["Vou me mudar no mês que vem.", "Trabalho em casa quase todos os dias.", "Moro com mais uma pessoa."],
        "nome": "Apartamento",
        "attrs": {
            "aluguel": ("max", {"lo": 1200, "hi": 4000, "passo": 50,
                                "pedido": "o aluguel pode ser de no máximo {x}", "oferta": "aluguel de {v}"}),
            "quartos": ("min", {"valores": [1, 2, 3, 4],
                                "pedido": "preciso de pelo menos {x} quartos", "oferta": "{v} quartos",
                                "oferta_1": "1 quarto"}),
            "pets": ("bool", {"pedido": "tenho um cachorro, então o prédio precisa permitir animais",
                              "sim": "aceita animais de estimação", "nao": "não aceita animais de estimação"}),
            "vaga": ("bool", {"pedido": "tenho carro e preciso de lugar para estacionar",
                              "sim": "com vaga de garagem", "nao": "sem vaga de garagem"}),
            "bairro": ("cat", {"valores": ["Centro", "Jardim América", "Vila Nova", "Setor Sul", "Parque das Flores"],
                               "pedido": "quero morar no bairro {x}", "oferta": "bairro {v}"}),
        },
    },
    "notebook": {
        "abertura": ["Quero comprar um notebook para o trabalho.", "Estou escolhendo um notebook novo."],
        "extra": ["Vou usar para planilhas e reuniões.", "Viajo bastante a trabalho.", "O antigo parou de ligar."],
        "nome": "Notebook",
        "attrs": {
            "preco": ("max", {"lo": 2500, "hi": 7000, "passo": 100,
                              "pedido": "posso gastar até {x}", "oferta": "preço de {v}"}),
            "ram": ("min", {"valores": [8, 16, 32, 64],
                            "pedido": "a memória precisa ter pelo menos {x} GB", "oferta": "{v} GB de RAM"}),
            "bateria": ("min", {"valores": [6, 8, 10, 12],
                                "pedido": "a bateria tem de durar no mínimo {x} horas",
                                "oferta": "bateria de {v} horas"}),
            "numerico": ("bool", {"pedido": "digito muitos números, então quero teclado com a parte numérica",
                                  "sim": "teclado com teclado numérico", "nao": "teclado sem teclado numérico"}),
            "tela": ("cat", {"valores": ["13 polegadas", "14 polegadas", "15,6 polegadas", "16 polegadas"],
                             "pedido": "prefiro uma tela de {x}", "oferta": "tela de {v}"}),
        },
    },
    "curso": {
        "abertura": ["Quero fazer um curso de inglês.", "Estou buscando um curso de programação."],
        "extra": ["Tenho as noites livres.", "A empresa vai reembolsar parte do valor.", "Já fiz um curso básico."],
        "nome": "Curso",
        "attrs": {
            "mensalidade": ("max", {"lo": 150, "hi": 900, "passo": 10,
                                    "pedido": "consigo pagar no máximo {x} por mês", "oferta": "mensalidade de {v}"}),
            "carga": ("min", {"valores": [40, 60, 80, 120, 160],
                              "pedido": "o curso precisa ter pelo menos {x} horas", "oferta": "{v} horas de aula"}),
            "certificado": ("bool", {"pedido": "preciso comprovar a formação para a empresa",
                                     "sim": "com certificado", "nao": "sem certificado"}),
            "sabado": ("bool", {"pedido": "só tenho tempo livre aos sábados",
                                "sim": "com turmas aos sábados", "nao": "sem turmas aos sábados"}),
            "modalidade": ("cat", {"valores": ["online", "presencial", "híbrida"],
                                   "pedido": "quero a modalidade {x}", "oferta": "modalidade {v}"}),
        },
    },
    "salao": {
        "abertura": ["Estou procurando um espaço para uma festa de formatura.",
                     "Preciso de um salão para a confraternização da empresa."],
        "extra": ["A data ainda pode mudar.", "Vamos contratar o bufê à parte.", "Haverá música ao vivo."],
        "nome": "Espaço",
        "attrs": {
            "diaria": ("max", {"lo": 1500, "hi": 9000, "passo": 100,
                               "pedido": "o orçamento para o aluguel é de até {x}", "oferta": "diária de {v}"}),
            "capacidade": ("min", {"valores": [50, 80, 120, 150, 200, 300],
                                   "pedido": "vamos receber {x} convidados", "oferta": "capacidade para {v} pessoas"}),
            "estacionamento": ("bool", {"pedido": "muitos convidados vão de carro",
                                        "sim": "com estacionamento", "nao": "sem estacionamento"}),
            "cozinha": ("bool", {"pedido": "o bufê precisa de onde preparar a comida",
                                 "sim": "com cozinha equipada", "nao": "sem cozinha"}),
            "regiao": ("cat", {"valores": ["zona norte", "zona sul", "zona leste", "zona oeste", "região central"],
                               "pedido": "o espaço deve ficar na {x}", "oferta": "na {v}"}),
        },
    },
}
OFERTA_PERGUNTA = "Qual das opções atende a todas as condições do pedido?"


def _oferta_valor(rng, tipo, p, restr, viola):
    """Valor de um atributo numa oferta. `restr`: o limite do pedido (None se o atributo não é condição);
    `viola`: se esta oferta deve descumprir a condição (por pouco)."""
    if tipo == "max":
        if restr is None:
            return rng.randrange(p["lo"], p["hi"] + 1, p["passo"])
        if viola:
            return restr + p["passo"] * rng.randint(1, max(1, int(0.15 * restr / p["passo"])))
        return restr - p["passo"] * rng.randint(0, max(1, int(0.3 * restr / p["passo"])))
    if tipo == "min":
        vals = p["valores"]
        if restr is None:
            return rng.choice(vals)
        i = vals.index(restr)
        return vals[i - 1] if viola else rng.choice(vals[i:i + 2])
    if tipo == "bool":
        return (not viola) if restr is not None else rng.random() < 0.5
    if restr is None:
        return rng.choice(p["valores"])
    return rng.choice([v for v in p["valores"] if v != restr]) if viola else restr


def _cumpre(tipo, restr, v) -> bool:
    return {"max": lambda: v <= restr, "min": lambda: v >= restr, "bool": lambda: bool(v),
            "cat": lambda: v == restr}[tipo]()


def _gera_restricoes(rng: random.Random) -> dict:
    dom_nome = rng.choice(sorted(OFERTA_DOMINIOS))
    dom = OFERTA_DOMINIOS[dom_nome]
    attrs = dom["attrs"]
    k = rng.choice([3, 4])
    cond = rng.sample(sorted(attrs), k)
    restr = {}
    for a in cond:
        tipo, p = attrs[a]
        if tipo == "max":
            restr[a] = rng.randrange(p["lo"] + 3 * p["passo"], p["hi"] - 3 * p["passo"] + 1, p["passo"])
        elif tipo == "min":
            restr[a] = rng.choice(p["valores"][1:])
        elif tipo == "bool":
            restr[a] = True
        else:
            restr[a] = rng.choice(p["valores"])
    violadas = rng.sample(cond, 3)  # cada distratora descumpre uma condição diferente
    ofertas = []
    for i, viola in enumerate([None] + violadas):
        ofertas.append({a: _oferta_valor(rng, attrs[a][0], attrs[a][1], restr.get(a), a == viola) for a in attrs})
    ids = ["o1", "o2", "o3", "o4"]
    ordem = list(range(4))
    rng.shuffle(ordem)  # posição canônica da correta: uniforme
    return {"dominio": dom_nome, "condicoes": restr, "ofertas": [ofertas[j] for j in ordem], "ids": ids,
            "gabarito": ids[ordem.index(0)], "violadas": violadas,
            "abertura": rng.choice(dom["abertura"]), "extra": rng.choice(dom["extra"]) if rng.random() < 0.5 else None,
            "ordem_condicoes": rng.sample(cond, k)}


def _texto_restricoes(d: dict) -> tuple[str, list[tuple[str, str]]]:
    dom = OFERTA_DOMINIOS[d["dominio"]]
    attrs = dom["attrs"]

    def fmt(a, v):
        tipo, p = attrs[a]
        if tipo == "bool":
            return p["sim"] if v else p["nao"]
        if v == 1 and "oferta_1" in p:
            return p["oferta_1"]
        return p["oferta"].format(v=_reais(v) if tipo == "max" else v)

    partes = []
    for a in d["ordem_condicoes"]:
        tipo, p = attrs[a]
        x = d["condicoes"][a]
        partes.append(p["pedido"].format(x=_reais(x) if tipo == "max" else x))
    frase = _lista(partes)
    pedido = d["abertura"] + " " + frase[0].upper() + frase[1:] + "."
    if d["extra"]:
        pedido += " " + d["extra"]
    opcoes = []
    for oid, of in zip(d["ids"], d["ofertas"]):
        txt = ", ".join(fmt(a, of[a]) for a in attrs)
        opcoes.append((oid, f"{dom['nome']}: {txt}."))
    return f"Pedido: {pedido}", opcoes


def _verifica_restricoes(d: dict) -> str:
    attrs = OFERTA_DOMINIOS[d["dominio"]]["attrs"]
    ok = [oid for oid, of in zip(d["ids"], d["ofertas"])
          if all(_cumpre(attrs[a][0], x, of[a]) for a, x in d["condicoes"].items())]
    if len(ok) != 1:
        raise ValueError(f"restricoes_oferta: {len(ok)} ofertas cumprem tudo")
    return ok[0]


# ------------------------------------------------------------------ regra_beneficio

BENEFICIO_PROGRAMAS = {
    "bolsa": {
        "titulo": "Programa de bolsas de permanência da faculdade",
        "req": [("semestres", "min", "estar matriculado há pelo menos {x} semestres", "Tempo de matrícula",
                 "{v} semestres", (1, 6)),
                ("renda", "max", "ter renda familiar por pessoa de até {x}", "Renda familiar por pessoa",
                 "{v}", (800, 2500, 50))],
        "docs": ["comprovante de matrícula", "comprovante de renda", "histórico escolar", "documento de identidade",
                 "comprovante de residência"],
        "valor": ("bolsa mensal", (300, 1500, 50)),
    },
    "capacitacao": {
        "titulo": "Política de apoio a cursos de capacitação da empresa",
        "req": [("meses", "min", "trabalhar na empresa há pelo menos {x} meses", "Tempo de empresa", "{v} meses",
                 (3, 24)),
                ("nota", "min", "ter tirado nota igual ou maior que {x} na última avaliação de desempenho",
                 "Nota na última avaliação", "{v}", (5, 9))],
        "docs": ["proposta do curso", "orçamento da instituição", "autorização do gestor", "plano de estudos",
                 "currículo atualizado"],
        "valor": ("custo do curso", (800, 12000, 100)),
    },
    "edital": {
        "titulo": "Edital de apoio a projetos culturais do município",
        "req": [("anos", "min", "morar no município há pelo menos {x} anos", "Tempo de residência no município",
                 "{v} anos", (1, 5)),
                ("idade", "min", "ter pelo menos {x} anos de idade", "Idade", "{v} anos", (18, 21))],
        "docs": ["portfólio", "orçamento detalhado", "comprovante de residência", "carta de apresentação",
                 "cronograma do projeto"],
        "valor": ("valor do apoio", (2000, 30000, 500)),
    },
}
TETO_VALOR = {"nota": 10}  # a nota de desempenho vai de 0 a 10
NOMES = ["Ana", "Bruno", "Carla", "Diego", "Elisa", "Fábio", "Gabriela", "Heitor", "Isabela", "João", "Larissa",
         "Marcos", "Natália", "Otávio", "Paula", "Rafael", "Sílvia", "Tiago", "Vanessa", "Wagner"]
BENEFICIO_OPCOES = [("conceder", "Conceder o benefício."), ("negar", "Negar o pedido."),
                    ("pedir_documento", "Pedir o documento que falta antes de decidir."),
                    ("encaminhar", "Encaminhar o pedido à comissão.")]
BENEFICIO_PERGUNTA = "Pelo regulamento, o que deve ser feito com este pedido?"


def _faixa(rng, faixa):
    lo, hi, *passo = faixa
    return rng.randrange(lo, hi + 1, passo[0]) if passo else rng.randint(lo, hi)


def _gera_beneficio(rng: random.Random) -> dict:
    prog_nome = rng.choice(sorted(BENEFICIO_PROGRAMAS))
    prog = BENEFICIO_PROGRAMAS[prog_nome]
    alvo = rng.choice([o for o, _ in BENEFICIO_OPCOES])  # gabarito sorteado antes: classes equilibradas
    limites = {}
    for chave, _tipo, *_resto, faixa in prog["req"]:
        limites[chave] = _faixa(rng, faixa)
    exigidos = rng.sample(prog["docs"], 2)
    vlo, vhi, vpasso = prog["valor"][1]
    teto = rng.randrange(vlo + 4 * vpasso, vhi - 4 * vpasso + 1, vpasso)

    def req_val(chave, tipo, faixa, cumpre):
        x = limites[chave]
        passo = faixa[2] if len(faixa) == 3 else 1
        if tipo == "min":
            return min(x + passo * rng.randint(0, 3), TETO_VALOR.get(chave, 10**9)) if cumpre                 else max(0, x - passo * rng.randint(1, 2))
        return x - passo * rng.randint(0, 6) if cumpre else x + passo * rng.randint(1, 4)

    if alvo == "negar":
        falha = rng.choice([[0], [1], [0, 1]])
    else:
        falha = []
    vals = {}
    for i, (chave, tipo, *_r, faixa) in enumerate(prog["req"]):
        vals[chave] = req_val(chave, tipo, faixa, i not in falha)
    # documentos: no "pedir_documento" falta ao menos um exigido; nos outros, depois de negar, vale qualquer coisa
    falta_doc = alvo == "pedir_documento" or (alvo == "negar" and rng.random() < 0.6)
    outros = [x for x in prog["docs"] if x not in exigidos]
    if falta_doc:
        anexos = rng.sample(exigidos, 1) + rng.sample(outros, rng.randint(0, 2))
    else:
        anexos = list(exigidos) + rng.sample(outros, rng.randint(0, 1))
    rng.shuffle(anexos)
    acima = alvo == "encaminhar" or (alvo == "negar" and rng.random() < 0.6) \
        or (alvo == "pedir_documento" and rng.random() < 0.75)
    valor = teto + vpasso * rng.randint(1, 6) if acima else teto - vpasso * rng.randint(0, 6)
    return {"programa": prog_nome, "limites": limites, "exigidos": exigidos, "teto": teto, "valores": vals,
            "anexos": anexos, "valor": valor, "nome": rng.choice(NOMES), "gabarito": alvo,
            "ordem_opcoes": [o for o, _ in BENEFICIO_OPCOES]}


def _texto_beneficio(d: dict) -> tuple[str, list[tuple[str, str]]]:
    prog = BENEFICIO_PROGRAMAS[d["programa"]]
    (c1, t1, p1, r1, v1, f1), (c2, t2, p2, r2, v2, f2) = prog["req"]

    def fmtv(chave, fmt, tipo):
        v = d["valores"][chave]
        return fmt.format(v=_reais(v) if tipo == "max" else v)

    def fmtx(chave, frase, tipo):
        x = d["limites"][chave]
        return frase.format(x=_reais(x) if tipo == "max" else x)

    nome_valor = prog["valor"][0]
    regulamento = (
        f"{prog['titulo']}:\n"
        f"1. Para receber, é preciso {fmtx(c1, p1, t1)}.\n"
        f"2. Para receber, é preciso {fmtx(c2, p2, t2)}.\n"
        f"3. O pedido deve vir com {d['exigidos'][0]} e {d['exigidos'][1]}.\n"
        f"4. Pedidos com {nome_valor} acima de {_reais(d['teto'])} precisam ser analisados pela comissão.\n"
        "Como decidir: se o pedido descumpre a regra 1 ou a regra 2, negue. Se cumpre as duas, mas falta algum "
        "documento da regra 3, peça o documento. Se está completo e o valor passa do limite da regra 4, "
        "encaminhe à comissão. Nos demais casos, conceda.")
    pedido = (f"Pedido:\nSolicitante: {d['nome']}\n{r1}: {fmtv(c1, v1, t1)}\n{r2}: {fmtv(c2, v2, t2)}\n"
              f"{nome_valor.capitalize()}: {_reais(d['valor'])}\nDocumentos anexados: {', '.join(d['anexos'])}")
    opcoes = dict(BENEFICIO_OPCOES)
    return f"{regulamento}\n\n{pedido}", [(o, opcoes[o]) for o in d["ordem_opcoes"]]


def _verifica_beneficio(d: dict) -> str:
    prog = BENEFICIO_PROGRAMAS[d["programa"]]
    for chave, tipo, *_r in prog["req"]:
        v, x = d["valores"][chave], d["limites"][chave]
        if (tipo == "min" and v < x) or (tipo == "max" and v > x):
            return "negar"
    if not set(d["exigidos"]) <= set(d["anexos"]):
        return "pedir_documento"
    return "encaminhar" if d["valor"] > d["teto"] else "conceder"


# ------------------------------------------------------------------ custo_plano

PLANO_DOMINIOS = {
    "celular": {"consumo": "Uso previsto: {u} GB de internet no mês.", "unidade": "GB",
                "plano": "{f} por mês, com {g} GB incluídos; cada GB a mais custa {p}",
                "u": (5, 60), "f": (20, 150), "g": (0, 40, 5), "p": (2.0, 12.0)},
    "coworking": {"consumo": "Uso previsto: {u} horas de sala no mês.", "unidade": "horas",
                  "plano": "{f} por mês, com {g} horas incluídas; cada hora a mais custa {p}",
                  "u": (10, 120), "f": (100, 900), "g": (0, 80, 10), "p": (8.0, 40.0)},
    "frete": {"consumo": "Envios previstos: {u} kg de mercadoria no mês.", "unidade": "kg",
              "plano": "taxa fixa de {f} por mês, com {g} kg incluídos; cada kg a mais custa {p}",
              "u": (20, 400), "f": (50, 600), "g": (0, 150, 25), "p": (1.0, 9.0)},
}
PLANO_ABERTURA = {"celular": "Estou escolhendo um plano de celular.",
                  "coworking": "Minha equipe vai alugar sala num coworking.",
                  "frete": "Minha loja virtual vai contratar uma transportadora."}
PLANO_PERGUNTA = "Qual plano sai mais barato no mês para esse uso?"


def _custo(pl: dict, u: int) -> float:
    return round(pl["f"] + max(0, u - pl["g"]) * pl["p"], 2)


def _gera_plano(rng: random.Random) -> dict:
    dom = rng.choice(sorted(PLANO_DOMINIOS))
    c = PLANO_DOMINIOS[dom]
    # tipo do gabarito sorteado antes, para os atalhos "menor mensalidade" e "menor preço extra" não acertarem
    quer_menor_f = rng.random() < 1 / 3
    quer_menor_p = rng.random() < 1 / 3
    quer_maior_g = rng.random() < 1 / 3
    for _ in range(5000):
        u = rng.randint(*c["u"])
        planos = [{"f": rng.randrange(c["f"][0], c["f"][1] + 1, 5), "g": rng.randrange(*c["g"][:2], c["g"][2]),
                   "p": round(rng.uniform(*c["p"]) * 2) / 2} for _ in range(3)]
        if len({p["f"] for p in planos}) < 3 or len({p["p"] for p in planos}) < 3:
            continue
        custos = [_custo(p, u) for p in planos]
        o = sorted(range(3), key=lambda i: custos[i])
        if custos[o[1]] < 1.05 * custos[o[0]]:  # sem quase empate
            continue
        g = o[0]
        menor_f = min(range(3), key=lambda i: planos[i]["f"]) == g
        menor_p = min(range(3), key=lambda i: planos[i]["p"]) == g
        gs = sorted(p["g"] for p in planos)
        if gs[1] == gs[2]:  # maior franquia sem empate
            continue
        maior_g = max(range(3), key=lambda i: planos[i]["g"]) == g
        if menor_f == quer_menor_f and menor_p == quer_menor_p and maior_g == quer_maior_g:
            ids = ["p1", "p2", "p3"]
            nomes = rng.sample(["Básico", "Leve", "Mais", "Livre", "Flex", "Total", "Pro", "Essencial"], 3)
            return {"dominio": dom, "uso": u, "planos": planos, "nomes": nomes, "ids": ids,
                    "gabarito": ids[g], "custos": custos}
    raise RuntimeError("custo_plano: não achei planos com as restrições")


PLANO_SEM_FRANQUIA = {"celular": "{f} por mês, sem franquia; cada GB custa {p}",
                      "coworking": "{f} por mês, sem horas incluídas; cada hora custa {p}",
                      "frete": "taxa fixa de {f} por mês, sem franquia; cada kg custa {p}"}


def _texto_plano(d: dict) -> tuple[str, list[tuple[str, str]]]:
    c = PLANO_DOMINIOS[d["dominio"]]
    estado = PLANO_ABERTURA[d["dominio"]] + " " + c["consumo"].format(u=d["uso"])
    ops = []
    for oid, nome, pl in zip(d["ids"], d["nomes"], d["planos"]):
        molde = PLANO_SEM_FRANQUIA[d["dominio"]] if pl["g"] == 0 else c["plano"]
        txt = molde.format(f=_reais(pl["f"]), g=pl["g"], p=_reais(pl["p"]))
        ops.append((oid, f"Plano {nome}: {txt}."))
    return estado, ops


def _verifica_plano(d: dict) -> str:
    custos = [pl["f"] + max(0, d["uso"] - pl["g"]) * pl["p"] for pl in d["planos"]]
    m = min(custos)
    if sum(abs(x - m) < 1e-9 for x in custos) != 1:
        raise ValueError("custo_plano: empate")
    return d["ids"][custos.index(m)]


# ------------------------------------------------------------------ ordem_eventos

EVENTOS = {  # (artigo, evento): só singulares, para a contração "do/da" ficar certa
    "mudança de casa": [("a", "pintura do apartamento novo"), ("a", "entrega das caixas de papelão"),
                        ("o", "transporte dos móveis"), ("a", "instalação da internet"),
                        ("a", "limpeza do apartamento antigo"), ("a", "troca da fechadura"),
                        ("a", "visita do síndico")],
    "festa de aniversário": [("a", "compra dos balões"), ("o", "envio dos convites"), ("a", "encomenda do bolo"),
                             ("o", "aluguel do salão"), ("a", "reunião com o bufê"), ("a", "prova da roupa"),
                             ("o", "ensaio da banda")],
    "reforma da cozinha": [("a", "troca do piso"), ("a", "instalação da pia"), ("a", "compra dos armários"),
                           ("o", "conserto do encanamento"), ("a", "pintura das paredes"),
                           ("a", "visita do arquiteto"), ("a", "entrega do fogão")],
    "evento da empresa": [("a", "reserva do auditório"), ("o", "convite aos palestrantes"),
                          ("a", "impressão dos crachás"), ("o", "teste do som"), ("a", "abertura das inscrições"),
                          ("a", "reunião de planejamento"), ("o", "envio do programa")],
    "lançamento de um livro": [("a", "revisão do texto"), ("a", "escolha da capa"),
                               ("a", "impressão dos exemplares"), ("a", "sessão de autógrafos"),
                               ("o", "envio às livrarias"), ("a", "entrevista na rádio"), ("o", "contrato com a editora")],
}
NUM = {1: "um", 2: "dois", 3: "três", 4: "quatro", 5: "cinco", 6: "seis", 7: "sete", 8: "oito", 9: "nove",
       10: "dez", 11: "onze", 12: "doze"}
ORDEM_PERGUNTAS = {"primeiro": "Qual destes acontecimentos veio primeiro?",
                   "ultimo": "Qual destes acontecimentos veio por último?"}


def _de(art: str) -> str:
    return "do" if art == "o" else "da"


def _gera_ordem(rng: random.Random) -> dict:
    cen = rng.choice(sorted(EVENTOS))
    evs = rng.sample(EVENTOS[cen], 4)
    dias = rng.sample(range(0, 13), 4)  # sem empate
    base = rng.randint(2, 14)
    pergunta = rng.choice(["primeiro", "ultimo"])
    alvo = min(range(4), key=lambda i: dias[i]) if pergunta == "primeiro" else max(range(4), key=lambda i: dias[i])
    pos_alvo = rng.randrange(4)  # posição do correto no relato: uniforme
    resto = [i for i in range(4) if i != alvo]
    rng.shuffle(resto)
    relato = resto[:pos_alvo] + [alvo] + resto[pos_alvo:]
    frases = []
    for k, i in enumerate(relato):
        if k == 0 or rng.random() < 0.3:
            frases.append({"evento": i, "tipo": "data", "dia": base + dias[i]})
        else:
            # relativo a um evento já citado, com diferença de até 12 dias
            ref = rng.choice([j for j in relato[:k] if abs(dias[j] - dias[i]) <= 12])
            frases.append({"evento": i, "tipo": "rel", "ref": ref, "dif": dias[i] - dias[ref]})
    ids = ["e1", "e2", "e3", "e4"]
    ordem_opcoes = list(range(4))
    rng.shuffle(ordem_opcoes)
    return {"cenario": cen, "eventos": evs, "dias": dias, "pergunta": pergunta, "frases": frases, "ids": ids,
            "ordem_opcoes": ordem_opcoes, "gabarito": ids[alvo], "pos_no_relato": pos_alvo}


def _texto_ordem(d: dict) -> tuple[str, list[tuple[str, str]]]:
    evs = d["eventos"]
    out = [f"Anotações sobre a {d['cenario']}:"] if d["cenario"].startswith(("mudança", "festa", "reforma")) \
        else [f"Anotações sobre o {d['cenario']}:"]
    for f in d["frases"]:
        art, ev = evs[f["evento"]]
        suj = f"{art.upper()} {ev}"
        if f["tipo"] == "data":
            out.append(f"{suj} foi no dia {f['dia']}.")
        else:
            ra, rev = evs[f["ref"]]
            n = abs(f["dif"])
            quando = "um dia" if n == 1 else f"{NUM[n]} dias"
            out.append(f"{suj} foi {quando} {'depois' if f['dif'] > 0 else 'antes'} {_de(ra)} {rev}.")
    ops = [(d["ids"][i], f"{evs[i][0].upper()} {evs[i][1]}.") for i in d["ordem_opcoes"]]
    return "\n".join(out), ops


def _verifica_ordem(d: dict) -> str:
    """Reconstrói as datas só a partir das frases do relato (não usa `dias`)."""
    data = {}
    for f in d["frases"]:
        data[f["evento"]] = f["dia"] if f["tipo"] == "data" else data[f["ref"]] + f["dif"]
    if len(set(data.values())) != 4:
        raise ValueError("ordem_eventos: datas repetidas")
    alvo = (min if d["pergunta"] == "primeiro" else max)(data, key=data.get)
    return d["ids"][alvo]


# ------------------------------------------------------------------ casos

GERADORES = {
    "restricoes_oferta": (_gera_restricoes, _texto_restricoes, _verifica_restricoes, lambda d: OFERTA_PERGUNTA),
    "regra_beneficio": (_gera_beneficio, _texto_beneficio, _verifica_beneficio, lambda d: BENEFICIO_PERGUNTA),
    "custo_plano": (_gera_plano, _texto_plano, _verifica_plano, lambda d: PLANO_PERGUNTA),
    "ordem_eventos": (_gera_ordem, _texto_ordem, _verifica_ordem, lambda d: ORDEM_PERGUNTAS[d["pergunta"]]),
}


def verificar_gabarito(case: Case) -> bool:
    return GERADORES[case.family][2](case.meta["dados"]) == case.gold


def res2_cases(cfg: dict) -> list[Case]:
    conf = cfg["res2"]
    salt, n, n_dif = conf["salt"], int(conf["grupos_por_familia"]), int(conf["grupos_dificuldade"])
    cases = []
    for fam in conf["familias"]:
        gera, texto, verifica, pergunta = GERADORES[fam]
        fam_cases = []
        for i in range(n):
            rng = random.Random(f"{salt}:{RES2_VERSION}:{fam}:{i}")
            d = gera(rng)
            if verifica(d) != d["gabarito"]:
                raise AssertionError(f"{fam}:{i}: gabarito do gerador difere da verificação")
            estado, opcoes = texto(d)
            g = f"res2:{fam}:{i}"
            fam_cases.append(Case(
                id=f"res2:{fam}:{i}", group_id=g, source="res2", family=fam, split="", held_out=True,
                qtype="choice", state=estado, question=pergunta(d), options=opcoes, gold=d["gabarito"],
                meta={"dados": d, "versao": RES2_VERSION}))
        # fatia de dificuldade: os `n_dif` grupos de menor sorteio; o resto, metade dev e metade teste
        ranking = sorted(fam_cases, key=lambda c: bucket(f"dificuldade:{c.group_id}", salt))
        for j, c in enumerate(ranking):
            c.split = "dificuldade" if j < n_dif else ("dev" if bucket(f"split:{c.group_id}", salt) < 0.5 else "test")
        cases += fam_cases
    return cases


# ------------------------------------------------------------------ auditoria

def _palavras(t: str) -> set[str]:
    return {w for w in re.findall(r"\w+", t.casefold()) if len(w) > 2}


def _heuristicas(c: Case) -> tuple[dict[str, str], dict[str, str]]:
    """Regras que não resolvem o problema; cada uma devolve o id que escolheria.
    `forma`: não lê o problema (posição, tamanho, sobreposição de palavras, um atributo só).
    `parcial`: lê só parte do problema (ignora uma regra, um termo do custo ou as datas relativas)."""
    ops = c.options
    lens = [len(desc) for _, desc in ops]
    sw = _palavras(c.state)
    forma = {"mais_longa": ops[lens.index(max(lens))][0], "mais_curta": ops[lens.index(min(lens))][0],
             "mais_palavras_do_contexto": max(ops, key=lambda o: len(_palavras(o[1]) & sw))[0],
             "primeira": ops[0][0]}
    parcial = {}
    d = c.meta["dados"]
    if c.family == "restricoes_oferta":
        attrs = OFERTA_DOMINIOS[d["dominio"]]["attrs"]
        num = [a for a, (t, _) in attrs.items() if t == "max"][0]
        forma["mais_barata"] = d["ids"][min(range(4), key=lambda i: d["ofertas"][i][num])]
        for tipo in ("max", "min", "bool", "cat"):  # ignora as condições de um tipo: fica a primeira que cumpre o resto
            cond = {a: x for a, x in d["condicoes"].items() if attrs[a][0] != tipo}
            ok = [i for i in range(4) if all(_cumpre(attrs[a][0], x, d["ofertas"][i][a]) for a, x in cond.items())]
            parcial[f"ignora_condicoes_{tipo}"] = d["ids"][ok[0]]
    elif c.family == "custo_plano":
        pl, u = d["planos"], d["uso"]
        forma["menor_mensalidade"] = d["ids"][min(range(3), key=lambda i: pl[i]["f"])]
        forma["menor_preco_extra"] = d["ids"][min(range(3), key=lambda i: pl[i]["p"])]
        forma["maior_franquia"] = d["ids"][max(range(3), key=lambda i: pl[i]["g"])]
        parcial["ignora_franquia"] = d["ids"][min(range(3), key=lambda i: pl[i]["f"] + u * pl[i]["p"])]
        parcial["ignora_mensalidade"] = d["ids"][min(range(3), key=lambda i: max(0, u - pl[i]["g"]) * pl[i]["p"])]
    elif c.family == "ordem_eventos":
        rel = [f["evento"] for f in d["frases"]]
        forma["citado_primeiro_ou_ultimo"] = d["ids"][rel[0] if d["pergunta"] == "primeiro" else rel[-1]]
        com_data = [f for f in d["frases"] if f["tipo"] == "data"]
        alvo = (min if d["pergunta"] == "primeiro" else max)(com_data, key=lambda f: f["dia"])["evento"]
        parcial["so_datas_explicitas"] = d["ids"][alvo]
    elif c.family == "regra_beneficio":
        req = BENEFICIO_PROGRAMAS[d["programa"]]["req"]
        elegivel = all((d["valores"][k] >= d["limites"][k]) if t == "min" else (d["valores"][k] <= d["limites"][k])
                       for k, t, *_r in req)
        completo = set(d["exigidos"]) <= set(d["anexos"])
        acima = d["valor"] > d["teto"]
        parcial["sem_precedencia"] = ("encaminhar" if acima else "pedir_documento" if not completo
                                      else "conceder" if elegivel else "negar")
        parcial["ignora_elegibilidade"] = ("pedir_documento" if not completo else "encaminhar" if acima
                                           else "conceder")
        parcial["ignora_documentos"] = "negar" if not elegivel else "encaminhar" if acima else "conceder"
        parcial["ignora_valor"] = "negar" if not elegivel else "pedir_documento" if not completo else "conceder"
    return forma, parcial


LIMITE_PARCIAL = 0.80  # cada parte ignorada decide pelo menos 1/5 dos casos


def res2_audit(cfg: dict, cases: list[Case] | None = None) -> dict:
    """Registro antes da fatia de dificuldade: contagens, gabarito verificado, posição do correto na ordem
    canônica e o acerto de cada atalho. Limites: de forma, min(60%, acaso + 20 pontos); de leitura parcial, 80%."""
    cases = res2_cases(cfg) if cases is None else cases
    out = {"versao": RES2_VERSION, "familias": {}}
    for fam in cfg["res2"]["familias"]:
        cs = [c for c in cases if c.family == fam]
        n = len(cs)
        n_op = Counter(len(c.options) for c in cs)
        acaso = sum(k_n / k for k, k_n in n_op.items()) / n
        limite = min(0.60, acaso + 0.20)
        maj = Counter(c.gold for c in cs).most_common(1)[0][0]
        hf, hp = Counter({"classe_majoritaria": sum(c.gold == maj for c in cs)}), Counter()
        for c in cs:
            forma, parcial = _heuristicas(c)
            for k, v in forma.items():
                hf[k] += v == c.gold
            for k, v in parcial.items():
                hp[k] += v == c.gold
        out["familias"][fam] = {
            "por_split": dict(sorted(Counter(c.split for c in cs).items())),
            "opcoes": dict(n_op),
            "gabarito_verificado": sum(verificar_gabarito(c) for c in cs),
            "posicao_canonica_do_correto": dict(sorted(Counter(
                [o[0] for o in c.options].index(c.gold) for c in cs).items())),
            "classes": dict(Counter(c.gold for c in cs).most_common()) if fam == "regra_beneficio" else None,
            "acaso": round(acaso, 3),
            "atalhos_de_forma": {k: f"{v}/{n}" for k, v in sorted(hf.items())},
            "atalhos_de_leitura_parcial": {k: f"{v}/{n}" for k, v in sorted(hp.items())},
            "acima_do_limite": sorted([k for k, v in hf.items() if v / n > limite]
                                      + [k for k, v in hp.items() if v / n > LIMITE_PARCIAL]),
            "limites": {"forma": round(limite, 3), "parcial": LIMITE_PARCIAL},
            "texto_medio_chars": round(sum(len(c.state) for c in cs) / n),
        }
    return out


def res2_sobreposicao(cfg: dict, cases: list[Case] | None = None) -> dict:
    """Casos de treino ou calib das fontes atuais (e do f3) que repetem texto das reservadas v2: texto
    normalizado igual ou 13 palavras seguidas. Tem de dar zero em todas."""
    from .data import LOADERS, NEW_SOURCES, _shingles
    cases = res2_cases(cfg) if cases is None else cases
    exact = {" ".join(c.state.casefold().split()) for c in cases}
    grams = set().union(*(_shingles(c.state) for c in cases))
    fontes = {**{s: LOADERS[s](cfg) for s in LOADERS}, "f3": NEW_SOURCES["f3"](cfg)}
    return {s: sum(1 for c in cs if c.split in ("train", "calib") and (
        " ".join(c.state.casefold().split()) in exact or not grams.isdisjoint(_shingles(c.state))))
        for s, cs in fontes.items()}
