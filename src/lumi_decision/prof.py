"""Reservadas com professor (analises/reservadas_v2.md, protocolo de 16/11/2026): 4 famílias de tipos que nem as
vistas nem o Eikos cobrem, com o texto escrito por um modelo (o escritor) para um gabarito fixado antes.

- `postura_alvo`: posição do autor de um comentário sobre um alvo (a favor / contra / sem posição); o mesmo
  comentário fala de 2 alvos com posturas diferentes.
- `inferencia_pt` (v2, emenda 4): relação texto × hipótese (decorre / contradiz / não dá para concluir); uma
  hipótese e 3 versões do texto que só diferem no trecho que a decide. Ler só a hipótese dá o acaso por construção.
- `resposta_indireta` (v2, emenda 4): o que uma resposta indireta quer dizer (sim / não); uma resposta e 2 versões da
  pergunta que diferem em 1 a 4 palavras e invertem o sentido dela. Ler só a resposta dá o acaso por construção.
- `correferencia`: a quem um pronome se refere; par de textos que diferem em 1 a 3 palavras e trocam o referente.

Fluxo (cada passo grava um arquivo em `data/prof/`, fora do git):
1. `especificacao` sorteia, por código, tudo o que define o gabarito; `pedido_escritor` monta o pedido;
2. o escritor responde (scripts/prof_vllm.py); `ler_resposta` aplica o filtro de forma e devolve os casos;
3. o conferente responde às cegas a cada caso em duas ordens (`pedido_conferente`, `ler_conferente`);
4. `montar` fica com os grupos que passam, escolhe 650 por família e divide os splits.
O arquivo final é fixado pelo sha256 no `splits.toml [prof]`.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .data import REPO_ROOT as ROOT, bucket, read_jsonl
from .prompts import order_options, user_message
from .schema import Case

PROF_VERSION = "v1"
PROF_FAMILIES = ("postura_alvo", "inferencia_pt", "resposta_indireta", "correferencia")
# Emenda 4 (23/11/2026): inferência e resposta indireta reprovaram na auditoria registrada (só a hipótese 1378/1950;
# só a resposta 1676/1950) e foram redesenhadas. A versão entra na semente da especificação e no meta de cada caso;
# postura e correferência continuam na v1, com as mesmas especificações.
VERSAO_FAMILIA = {"postura_alvo": "v1", "inferencia_pt": "v2", "resposta_indireta": "v2", "correferencia": "v1"}
PROF_DIR = ROOT / "data" / "prof"

# ------------------------------------------------------------------ material comum

# Domínios do protocolo; cada tema serve de alvo (postura) e de semente de cena (as outras três famílias).
TEMAS = {
    "condomínio": ["a instalação de câmeras nos corredores", "a proibição de festas depois das 22h",
                   "a reforma da piscina com taxa extra", "a troca do porteiro por portaria remota",
                   "a liberação de cachorros na área de lazer", "a pintura da fachada neste ano",
                   "o novo horário da academia do prédio", "a cobrança pelo uso do salão de festas"],
    "bairro e prefeitura": ["a ciclovia na avenida principal", "o estacionamento rotativo pago no centro",
                            "a feira de rua aos domingos", "o fechamento da rua para pedestres no fim de semana",
                            "a mudança do ponto de ônibus para a outra quadra", "a nova iluminação da praça",
                            "o novo horário da coleta de lixo", "a pista de skate na praça"],
    "escola e universidade": ["o uso obrigatório de uniforme", "a proibição de celular em sala",
                              "as provas aos sábados", "o fim do intervalo de 30 minutos",
                              "a avaliação por projetos em grupo", "as aulas de reforço no contraturno",
                              "a nova biblioteca digital", "a mudança do horário de entrada para as 7h"],
    "trabalho": ["o retorno ao escritório três dias por semana", "o banco de horas", "as metas por equipe",
                 "o horário flexível", "o ponto registrado por aplicativo", "as reuniões só pela manhã",
                 "o rodízio de mesas no escritório", "a confraternização no sábado"],
    "clube e esporte amador": ["o aumento da mensalidade do clube", "a reserva de quadras pelo aplicativo",
                               "o torneio interno obrigatório", "a abertura do clube para não sócios",
                               "a troca do treinador do time amador", "o novo horário da piscina",
                               "a taxa para convidados", "a reforma do campo de futebol"],
    "cultura e eventos": ["a mudança do festival para o estádio", "o ingresso pago na festa junina",
                          "o show na praça no domingo à noite", "a feira do livro no shopping",
                          "o cineclube às segundas", "o concurso de bandas da escola",
                          "o sarau mensal na biblioteca", "o desfile de carnaval no bairro"],
    "cidade e transporte": ["a tarifa zero aos domingos", "os patinetes elétricos nas calçadas",
                            "o rodízio de carros", "a faixa exclusiva de ônibus", "o fim da linha de ônibus noturna",
                            "o novo radar na avenida", "o aumento da passagem do metrô",
                            "as bicicletas compartilhadas"],
    "casa e família": ["a divisão das tarefas por escala", "a compra de um carro novo",
                       "a mudança para um apartamento menor", "a adoção de um cachorro",
                       "o almoço de domingo na casa da avó", "o fim da TV a cabo", "a reforma da cozinha",
                       "a mesada ligada às tarefas"],
}
NOMES_F = ["Ana", "Beatriz", "Camila", "Daniela", "Eduarda", "Fernanda", "Gabriela", "Helena", "Isabela", "Juliana",
           "Larissa", "Mariana", "Natália", "Patrícia", "Renata", "Sofia", "Tatiana", "Vanessa"]
NOMES_M = ["André", "Bruno", "Carlos", "Diego", "Eduardo", "Felipe", "Gustavo", "Henrique", "Igor", "João",
           "Lucas", "Marcelo", "Nicolas", "Otávio", "Paulo", "Rafael", "Thiago", "Vinícius"]

SISTEMA_ESCRITOR = ("Você escreve textos em português do Brasil para um conjunto de avaliação de modelos de linguagem. "
                    "Siga a especificação à risca. Raciocine de forma objetiva: monte a resposta, confira as regras "
                    "uma vez e responda. As regras sobre palavras valem para palavras inteiras (\"reservado\" não "
                    "contém a palavra \"se\"). Responda só com um objeto JSON válido, sem texto antes ou depois "
                    "e sem blocos de código.")
SISTEMA_CONFERENTE = ("Você resolve questões de múltipla escolha com cuidado. Raciocine em no máximo 3 frases e "
                      "termine com uma linha no formato \"Resposta: <letra>\".")

NEGACAO = re.compile(r"\b(não|nunca|nenhum|nenhuma|nem|jamais|ninguém|nada)\b", re.I)
HESITACAO = re.compile(r"\b(talvez|provavelmente|possivelmente|pode ser|quem sabe)\b", re.I)
POSTURA_PROIBIDAS = re.compile(r"(a favor|\bcontra\b|\bcontrári|neutr|concordo|discordo)", re.I)
# Emenda de 22/11 (lote de teste 5000–5011): "não", "nunca" e "jamais" saem da lista; a negação passa a ser sorteada
# por resposta, independente da classe, como na inferência (a proibição total rejeitava 8 de 12 grupos).
RESPOSTA_PROIBIDAS = re.compile(
    r"\b(sim|claro|claramente|certeza|certamente|óbvio|obviamente|lógico|talvez|"
    r"pode ser|beleza|fechado|combinado|ok|okay|nem pensar|de jeito nenhum|negativo|positivo|aham|uhum|"
    r"pode contar|conta comigo|desde que|contanto que|caso|dependendo|depende|a não ser|a menos que)\b", re.I)
SE_QUANDO = re.compile(r"\b(se|quando)\b", re.I)
PRONOMES = {"f": "ela", "m": "ele"}


def _frases(t: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s])


def _fracao_comum(hip: str, texto: str) -> float:
    """Fração das palavras de conteúdo da hipótese que aparecem no texto (a sobreposição da auditoria)."""
    ph = {w for w in re.findall(r"\w+", hip.casefold()) if len(w) > 2}
    return len(ph & {w for w in re.findall(r"\w+", texto.casefold()) if len(w) > 2}) / max(1, len(ph))


def _tem(texto: str, nome: str) -> bool:
    return re.search(rf"\b{re.escape(nome)}\b", texto) is not None


# ------------------------------------------------------------------ especificações (gabarito antes do texto)

POSTURAS = {"a_favor": "a favor", "contra": "contra", "sem_posicao": "sem posição"}
GENEROS_TEXTO = ["um comentário numa rede social", "uma mensagem num grupo de conversa",
                 "uma carta de leitor para o jornal do bairro", "a fala de uma pessoa numa reunião, transcrita",
                 "um comentário num fórum da comunidade"]
# Emenda 4 (23/11): um mecanismo por grupo, o mesmo para as três classes. Na v1 cada classe tinha a sua lista
# ("alguém", "pelo menos um" só em "decorre"; opinião, motivo e futuro só em "neutra"), e o vocabulário do mecanismo
# entregava o rótulo.
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
         "novembro", "dezembro"]
MEC_INFERENCIA = ["um número ou uma conta simples com números", "a ordem no tempo de dois acontecimentos",
                  "quem fez o quê (o papel de cada pessoa)",
                  "um atributo de uma pessoa ou de uma coisa (cor, lugar, tamanho, estado)",
                  "uma generalização (\"alguém\", \"pelo menos um\", \"todos\")",
                  "a opinião ou a preferência de uma pessoa", "o motivo de uma ação", "algo combinado para depois"]
# Na v2 não há classe condicional; a pergunta sobre algo que já aconteceu volta a caber.
TIPOS_PERGUNTA = ["um convite", "um pedido de ajuda", "uma pergunta sobre preferência",
                  "uma pergunta sobre um plano da pessoa", "um pedido de permissão ou de favor",
                  "uma pergunta sobre algo que já aconteceu"]
RELACOES = ["colegas de trabalho", "vizinhos", "amigos", "irmãos", "colegas de turma", "chefe e funcionário",
            "mãe e filho", "membros do mesmo clube"]


def especificacao(fam: str, i: int, salt: str) -> dict:
    rng = random.Random(f"{salt}:{VERSAO_FAMILIA[fam]}:{fam}:{i}")
    dominio = rng.choice(sorted(TEMAS))
    if fam == "postura_alvo":
        a, b = rng.sample(TEMAS[dominio], 2)
        pa, pb = rng.sample(list(POSTURAS), 2)
        return {"dominio": dominio, "alvos": [a, b], "posturas": [pa, pb], "genero": rng.choice(GENEROS_TEXTO),
                "frases": rng.choice(["3 a 4", "5 a 6"])}
    if fam == "inferencia_pt":
        # v2: uma hipótese, três versões do texto. A negação é sorteada para a hipótese (35%) e para os textos (35%,
        # a mesma nas três versões): ela não separa as classes.
        return {"dominio": dominio, "tema": rng.choice(TEMAS[dominio]), "nomes": rng.sample(NOMES_F + NOMES_M, 2),
                "mecanismo": rng.choice(MEC_INFERENCIA), "negacao_hipotese": rng.random() < 0.35,
                "negacao_textos": rng.random() < 0.35,
                # Emenda 5: a data vem da especificação (o lote 29000 voltou a usar "12 de março" e "12 de maio").
                "data": f"{rng.randint(1, 28)} de {rng.choice(MESES)}"}
    if fam == "resposta_indireta":
        genero = rng.choice("fm")
        nomes = rng.sample(NOMES_F if genero == "f" else NOMES_M, 1) + rng.sample(NOMES_F + NOMES_M, 1)
        while nomes[1] == nomes[0]:
            nomes[1] = rng.choice(NOMES_F + NOMES_M)
        # v2: uma resposta, duas versões da pergunta; sorteia qual versão quer dizer "sim".
        return {"dominio": dominio, "tema": rng.choice(TEMAS[dominio]), "relacao": rng.choice(RELACOES),
                "nomes": nomes, "tipo_pergunta": rng.choice(TIPOS_PERGUNTA), "sim_na_versao": rng.choice([1, 2])}
    if fam == "correferencia":
        genero = rng.choice("fm")
        nomes = rng.sample(NOMES_F if genero == "f" else NOMES_M, 2)
        return {"dominio": dominio, "tema": rng.choice(TEMAS[dominio]), "genero": genero, "nomes": nomes,
                "pronome": PRONOMES[genero], "referente_v1": rng.randrange(2)}
    raise ValueError(fam)


def pedido_escritor(fam: str, e: dict) -> str:
    if fam == "postura_alvo":
        (a, b), (pa, pb) = e["alvos"], e["posturas"]
        return f"""Escreva {e['genero']} ({e['frases']} frases), no domínio "{e['dominio']}", em que o autor fala de dois assuntos:
- assunto 1: {a}. Posição do autor: {POSTURAS[pa]}.
- assunto 2: {b}. Posição do autor: {POSTURAS[pb]}.

Regras:
- Cite os dois assuntos de forma reconhecível.
- "a favor" e "contra": a posição aparece pelos argumentos, pelo tom ou pelo que o autor propõe, nunca dita com essas palavras.
- "sem posição": o autor menciona o assunto, mas não toma partido (pesa os dois lados, diz que tanto faz ou que ainda vai se informar).
- Não use as palavras "a favor", "contra", "contrário", "neutro", "concordo" nem "discordo".
- Linguagem natural, como a de uma pessoa comum.

JSON: {{"comentario": "..."}}"""
    if fam == "inferencia_pt":
        n1, n2 = e["nomes"]
        neg_h = ("contém uma negação (não, nunca, nenhum, nem, ninguém ou nada)" if e["negacao_hipotese"]
                 else "não contém negação (nada de não, nunca, nenhum, nem, jamais, ninguém ou nada)")
        neg_t = ("cada versão contém pelo menos uma negação (não, nunca, nenhum, nem, ninguém ou nada)"
                 if e["negacao_textos"] else
                 "nenhuma versão contém negação (nada de não, nunca, nenhum, nem, jamais, ninguém ou nada)")
        return f"""Escreva uma hipótese de uma frase e três versões de um texto curto (2 a 4 frases) sobre uma situação do cotidiano no domínio "{e['dominio']}" (tema: {e['tema']}), com {n1} e {n2} (pode haver outras pessoas), com números concretos escolhidos por você para esta situação e, se houver uma data, {e['data']}.

As três versões contam a mesma situação, com as mesmas pessoas e quase as mesmas frases. Elas só diferem no trecho que decide a hipótese, que é sobre {e['mecanismo']}:
- "texto_decorre": o texto garante que a hipótese é verdadeira.
- "texto_contradiz": o texto garante que a hipótese é falsa. Ele traz um fato incompatível com a hipótese; não basta pôr um "não" na frase que decide.
- "texto_neutra": o texto não permite saber se a hipótese é verdadeira ou falsa. No lugar do trecho que decide, ponha outra informação do mesmo assunto e de tamanho parecido, que não decide a hipótese.

Regras:
- Antes de responder, confira cada versão: em "texto_decorre" a hipótese é verdadeira em qualquer leitura razoável; em "texto_contradiz", falsa em qualquer leitura razoável; em "texto_neutra", o texto não dá base nem para afirmar nem para negar. Se uma ordem no tempo, um número ou um papel não estiver escrito no texto, ele não garante nada.
- As três versões têm tamanho parecido e usam quase as mesmas palavras: a diferença entre elas cabe num trecho de no máximo uma frase. {n1} e {n2} aparecem nas três.
- A hipótese {neg_h}. Ela não copia frases de nenhuma versão: diga com outras palavras.
- Negação nos textos: {neg_t}.
- Não use "talvez", "provavelmente", "possivelmente", "pode ser" nem "quem sabe".

JSON: {{"hipotese": "...", "texto_decorre": "...", "texto_contradiz": "...", "texto_neutra": "..."}}"""
    if fam == "resposta_indireta":
        a, b = e["nomes"]
        sentido = {1: ("sim", "não"), 2: ("não", "sim")}[e["sim_na_versao"]]
        return f"""Crie uma cena curta entre {a} e {b} ({e['relacao']}), no domínio "{e['dominio']}" (tema: {e['tema']}). {a} faz a {b} {e['tipo_pergunta']} que se responde com sim ou não, e {b} responde de forma indireta: conta um fato, um plano ou um sentimento, sem dizer sim nem não.

Escreva duas versões da pergunta, "pergunta_1" e "pergunta_2", que diferem em 1 a 4 palavras, e uma única "resposta" de {b} que serve para as duas. Com "pergunta_1", a resposta quer dizer {sentido[0]}; com "pergunta_2", a mesma resposta quer dizer {sentido[1]}. As palavras trocadas na pergunta é que invertem o sentido da resposta.

Esquema (não use este exemplo; invente uma situação do tema): resposta "Torci o tornozelo no treino de ontem." Para "Você vai a pé para a festa amanhã?", quer dizer não; para "Você vai de carro para a festa amanhã?", quer dizer sim.

Regras:
- Com cada versão da pergunta, uma pessoa comum entende sem dúvida o que {b} quis dizer.
- A resposta tem 1 ou 2 frases e não põe condição: não use "se" nem "quando". Também não use: sim, claro, certeza, certamente, óbvio, lógico, talvez, pode ser, beleza, fechado, combinado, ok, nem pensar, de jeito nenhum, pode contar, conta comigo, desde que, contanto que, caso, depende, dependendo, a não ser, a menos que.
- As duas perguntas terminam com "?" e as palavras trocadas não são nomes de pessoas.
- A resposta não usa nenhuma das palavras trocadas entre as perguntas: quem lê liga a resposta à pergunta pelo sentido, não por uma palavra repetida.
- "situacao" diz em 1 ou 2 frases onde estão e o que está acontecendo, sem antecipar a resposta, e serve para as duas versões.

JSON: {{"situacao": "...", "pergunta_1": "...", "pergunta_2": "...", "resposta": "..."}}"""
    if fam == "correferencia":
        n1, n2 = e["nomes"]
        r1 = e["nomes"][e["referente_v1"]]
        r2 = e["nomes"][1 - e["referente_v1"]]
        return f"""Escreva duas versões de um texto curto (1 ou 2 frases) no domínio "{e['dominio']}" (tema: {e['tema']}) com {n1} e {n2}, nessa ordem de aparição, e o pronome "{e['pronome']}" uma única vez.
- Versão 1: "{e['pronome']}" se refere a {r1}.
- Versão 2: igual à versão 1, trocando de 1 a 3 palavras, e com isso "{e['pronome']}" passa a se referir a {r2}.

Como montar (é o esquema de Winograd):
- Uma oração principal em que {n1} e {n2} têm papéis diferentes numa ação (por exemplo, quem faz e quem recebe: "X ajudou Y", "X emprestou algo a Y", "X avisou Y"), nunca os dois juntos no mesmo papel ("X e Y fizeram...").
- Depois, uma oração com o pronome, ligada por "porque", "pois", "mas" ou "então", dizendo algo que, pelo senso comum, só combina com um dos dois papéis.
- A palavra trocada inverte qual papel combina. Esquema, com X e Y no lugar dos nomes: "X emprestou dinheiro a Y porque {e['pronome']} tinha de sobra" (X) / "...porque {e['pronome']} estava sem nenhum" (Y).
- Antes de responder, confira: em cada versão, uma pessoa comum aponta o referente sem hesitar. Se der para defender os dois, refaça.

Regras:
- O referente se decide pelo sentido (o que faz sentido no mundo), nunca pela gramática: {n1} e {n2} têm o mesmo gênero, e a frase com o pronome poderia, gramaticalmente, falar de qualquer um dos dois.
- Os dois nomes aparecem nas duas versões, nessa ordem, escritos exatamente assim, e as palavras trocadas não são nomes.
- Não use o esquema de exemplo (dinheiro emprestado); invente uma situação do tema.
- O pronome "{e['pronome']}" aparece exatamente uma vez em cada versão, e nenhuma outra forma dele ("{e['pronome']}s", "d{e['pronome']}", "n{e['pronome']}").

JSON: {{"versao_1": "...", "versao_2": "..."}}"""
    raise ValueError(fam)


def pedidos_escritor(cfg: dict, inicio: int, fim: int, familias: list[str] | None = None) -> list[dict]:
    conf = cfg["prof"]
    out = []
    for fam in familias or conf["familias"]:
        if fam not in conf["familias"]:
            raise ValueError(f"família desconhecida: {fam}")
        for i in range(inicio, fim):
            e = especificacao(fam, i, conf["salt"])
            out.append({"id": f"prof:{fam}:{i}", "familia": fam, "indice": i, "especificacao": e,
                        "messages": [{"role": "system", "content": SISTEMA_ESCRITOR},
                                     {"role": "user", "content": pedido_escritor(fam, e)}],
                        # O escritor raciocina antes do JSON (emenda de 22/11): a amostragem recomendada para o modo com
                        # raciocínio (0,6, top_p 0,95, top_k 20). Sem raciocínio eram 1200 tokens e 0,8. A emenda 2
                        # (lote 5100–5123, 17 de 96 respostas cortadas em 8000) sobe para 10000 e acrescenta top_k; a
                        # emenda 3 (lote 5200–5223, 11 de 96 cortadas em 10000) sobe para 12000.
                        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 12000,
                        "seed": int(hashlib.sha256(f"{conf['salt']}:seed:{fam}:{i}".encode()).hexdigest()[:8], 16)})
    return out


# ------------------------------------------------------------------ leitura do escritor (filtro de forma)

OPCOES = {
    "postura_alvo": [("a_favor", "É a favor."), ("contra", "É contra."), ("sem_posicao", "Não toma posição.")],
    "inferencia_pt": [("decorre", "O texto garante que a hipótese é verdadeira."),
                      ("contradiz", "O texto garante que a hipótese é falsa."),
                      ("neutra", "O texto não permite concluir se a hipótese é verdadeira ou falsa.")],
    "resposta_indireta": [("sim", "Sim."), ("nao", "Não.")],  # v2: sem a classe condicional
}


class Rejeitado(ValueError):
    pass


def _json(texto: str) -> dict:
    t = texto.strip()
    if "</think>" in t:  # escritor com raciocínio (emenda de 22/11): o JSON vem depois do fim do raciocínio
        t = t.rsplit("</think>", 1)[1].strip()
    elif "<think>" in t:
        raise Rejeitado("raciocínio sem fim")
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    ini, fim = t.find("{"), t.rfind("}")
    if ini < 0 or fim < ini:
        raise Rejeitado("sem JSON")
    try:
        d = json.loads(t[ini:fim + 1])
    except json.JSONDecodeError as err:
        raise Rejeitado("JSON inválido") from err
    if not isinstance(d, dict):
        raise Rejeitado("JSON não é objeto")
    return {k: " ".join(str(v).split()) if isinstance(v, str) else v for k, v in d.items()}


def _campos(d: dict, nomes: list[str], minimo: int, maximo: int) -> None:
    for k in nomes:
        v = d.get(k)
        if not isinstance(v, str) or not minimo <= len(v) <= maximo:
            raise Rejeitado(f"campo {k} ausente ou fora do tamanho")


def _diferenca_palavras(a: str, b: str) -> tuple[int, list[str]]:
    """Maior número de palavras trocadas (de um dos lados) e as palavras envolvidas."""
    ta, tb = re.findall(r"\w+|[^\w\s]", a), re.findall(r"\w+|[^\w\s]", b)
    n, envolvidas = 0, []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes():
        if op != "equal":
            n += max(i2 - i1, j2 - j1)
            envolvidas += ta[i1:i2] + tb[j1:j2]
    return n, envolvidas


def ler_resposta(fam: str, i: int, e: dict, texto: str) -> list[Case]:
    """Filtro de forma do protocolo. Devolve os casos do grupo ou levanta `Rejeitado` com o motivo."""
    d = _json(texto)
    g = f"prof:{fam}:{i}"
    meta = {"especificacao": e, "versao": VERSAO_FAMILIA[fam], "escritor": d}

    def caso(k: int, estado: str, pergunta: str, opcoes, gold: str) -> Case:
        return Case(id=f"{g}:{k}", group_id=g, source="prof", family=fam, split="", held_out=True, qtype="choice",
                    state=estado, question=pergunta, options=list(opcoes), gold=gold, meta=meta)

    if fam == "postura_alvo":
        _campos(d, ["comentario"], 150, 1200)
        c = d["comentario"]
        if POSTURA_PROIBIDAS.search(c):
            raise Rejeitado("palavra proibida")
        for alvo in e["alvos"]:  # cada alvo citado: pelo menos uma palavra de conteúdo dele no comentário
            chave = [w for w in re.findall(r"\w+", alvo.casefold()) if len(w) > 4]
            if chave and not any(re.search(rf"\b{re.escape(w[:-1])}", c.casefold()) for w in chave):
                raise Rejeitado("alvo não citado")
        return [caso(k, f"Comentário: {c}", f"Qual é a posição do autor do comentário sobre {alvo}?",
                     OPCOES[fam], e["posturas"][k]) for k, alvo in enumerate(e["alvos"])]
    if fam == "inferencia_pt":
        _campos(d, ["hipotese"], 15, 220)
        _campos(d, ["texto_decorre", "texto_contradiz", "texto_neutra"], 80, 800)
        h = d["hipotese"]
        ts = {k: d[f"texto_{k}"] for k in ("decorre", "contradiz", "neutra")}
        lens = [len(t) for t in ts.values()]
        if max(lens) > 1.5 * min(lens):
            raise Rejeitado("versões de tamanhos muito diferentes")
        if HESITACAO.search(h) or any(HESITACAO.search(t) for t in ts.values()):
            raise Rejeitado("hesitação")
        if bool(NEGACAO.search(h)) != e["negacao_hipotese"]:
            raise Rejeitado("negação da hipótese fora da especificação")
        if any(bool(NEGACAO.search(t)) != e["negacao_textos"] for t in ts.values()):
            raise Rejeitado("negação dos textos fora da especificação")
        if not all(_tem(t, n) for t in ts.values() for n in e["nomes"]):
            raise Rejeitado("nomes ausentes")
        if any(h.rstrip(".").casefold() in t.casefold() for t in ts.values()):
            raise Rejeitado("hipótese copiada do texto")
        for x, y in (("decorre", "contradiz"), ("decorre", "neutra"), ("contradiz", "neutra")):
            if not 1 <= _diferenca_palavras(ts[x], ts[y])[0] <= 30:
                raise Rejeitado("versões diferem demais (ou em nada)")
        # Emenda 5 (lote 29000): a versão que decorre tendia a repetir a hipótese e a neutra a fugir das palavras dela.
        sob = [_fracao_comum(h, t) for t in ts.values()]
        if max(sob) - min(sob) > 0.2:
            raise Rejeitado("sobreposição com a hipótese desigual entre as versões")
        return [caso(k, f"Texto: {ts[gold]}\n\nHipótese: {h}",
                     "Considerando só o texto, qual é a relação entre o texto e a hipótese?", OPCOES[fam], gold)
                for k, gold in enumerate(("decorre", "contradiz", "neutra"))]
    if fam == "resposta_indireta":
        _campos(d, ["situacao"], 30, 400)
        _campos(d, ["pergunta_1", "pergunta_2"], 8, 200)
        _campos(d, ["resposta"], 10, 260)
        a, b = e["nomes"]
        r = d["resposta"].strip().strip('"“”')
        ps = [d[k].strip().strip('"“”') for k in ("pergunta_1", "pergunta_2")]
        if RESPOSTA_PROIBIDAS.search(r):
            raise Rejeitado("palavra proibida na resposta")
        if SE_QUANDO.search(r):
            raise Rejeitado("se/quando na resposta")
        if not all(p.endswith("?") for p in ps):
            raise Rejeitado("pergunta sem interrogação")
        n, envolvidas = _diferenca_palavras(*ps)
        if not 1 <= n <= 4:
            raise Rejeitado("perguntas diferem em mais de 4 palavras (ou em nenhuma)")
        if {a, b} & set(envolvidas):
            raise Rejeitado("troca envolve nome")
        # Emenda 5 (lote 29000): quando a palavra trocada aparecia na resposta, quase sempre era a da versão "sim".
        trocadas = {w.casefold() for w in envolvidas if len(w) >= 3 or w.isdigit()}
        if trocadas & {w.casefold() for w in re.findall(r"\w+", r)}:
            raise Rejeitado("palavra trocada aparece na resposta")
        golds = ["sim", "nao"] if e["sim_na_versao"] == 1 else ["nao", "sim"]
        return [caso(k, f"Situação: {d['situacao']}\n{a} pergunta: \"{p}\"\n{b} responde: \"{r}\"",
                     f"Pela resposta, o que {b} quis dizer?", OPCOES[fam], golds[k]) for k, p in enumerate(ps)]
    if fam == "correferencia":
        _campos(d, ["versao_1", "versao_2"], 30, 400)
        n1, n2 = e["nomes"]
        p = e["pronome"]
        for v in (d["versao_1"], d["versao_2"]):
            if not (_tem(v, n1) and _tem(v, n2)) or v.index(n1) > v.index(n2):
                raise Rejeitado("nomes ausentes ou fora de ordem")
            if len(re.findall(rf"\b{p}\b", v, re.I)) != 1 or re.search(rf"\b(d|n){p}s?\b|\b{p}s\b", v, re.I):
                raise Rejeitado("pronome ausente, repetido ou em outra forma")
        n, envolvidas = _diferenca_palavras(d["versao_1"], d["versao_2"])
        if not 1 <= n <= 3:
            raise Rejeitado("versões diferem em mais de 3 palavras (ou em nenhuma)")
        if {n1, n2, p} & set(envolvidas):
            raise Rejeitado("troca envolve nome ou pronome")
        opcoes = [("pessoa_1", n1), ("pessoa_2", n2)]
        refs = [e["referente_v1"], 1 - e["referente_v1"]]
        return [caso(k, f"Texto: {v}", f"No texto, a quem se refere a palavra \"{p}\"?", opcoes,
                     opcoes[refs[k]][0]) for k, v in enumerate((d["versao_1"], d["versao_2"]))]
    raise ValueError(fam)


# ------------------------------------------------------------------ conferente às cegas

ORDENS_CONFERENCIA = ("canonical", "rot1")


def pedido_conferente(c: Case, ordem: str) -> list[dict]:
    msg = user_message(c, order_options(c, ordem))
    fim = "Responda apenas com a letra da opção escolhida."
    assert msg.endswith(fim)
    msg = msg[: -len(fim)] + "Raciocine em no máximo 3 frases e termine com \"Resposta: <letra>\"."
    return [{"role": "system", "content": SISTEMA_CONFERENTE}, {"role": "user", "content": msg}]


def ler_conferente(texto: str, c: Case, ordem: str) -> str | None:
    """Id da opção que o conferente escolheu (ou None, se não deu para ler)."""
    m = re.findall(r"Resposta:\s*\**\s*\(?([A-Z])\b", texto)
    if not m:
        return None
    ops = order_options(c, ordem)
    k = ord(m[-1]) - ord("A")
    return ops[k][0] if 0 <= k < len(ops) else None


def candidatos(cfg: dict, respostas: list[dict]) -> tuple[list[Case], Counter, list[dict]]:
    """Lê as respostas do escritor. Devolve os casos que passaram no filtro de forma, os motivos de rejeição e os
    pedidos do conferente (um por caso e ordem)."""
    conf = cfg["prof"]
    casos, motivos, pedidos = [], Counter(), []
    for r in respostas:
        fam, i = r["familia"], r["indice"]
        e = especificacao(fam, i, conf["salt"])
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


def montar(cfg: dict, casos: list[Case], conferencia: list[dict]) -> tuple[list[Case], dict]:
    """Fica com os grupos em que o conferente acerta todos os casos nas duas ordens; escolhe os 650 primeiros por
    sorteio em cada família e divide: 50 de dificuldade, o resto meio a meio entre dev e teste."""
    conf = cfg["prof"]
    salt, n, n_dif = conf["salt"], int(conf["grupos_por_familia"]), int(conf["grupos_dificuldade"])
    # Resposta do conferente cortada (sem "stop") não vale: o "Resposta: X" poderia ser de um raciocínio
    # incompleto. Fica como ilegível e o grupo sai.
    resp = {r["id"]: r["texto"] for r in conferencia if r.get("finish_reason", "stop") == "stop"}
    cortadas = Counter(r["id"].split(":")[1] for r in conferencia if r.get("finish_reason", "stop") != "stop")
    por_grupo = defaultdict(list)
    for c in casos:
        por_grupo[c.group_id].append(c)
    rel, final = {}, []
    for fam in conf["familias"]:
        grupos = sorted(g for g, cs in por_grupo.items() if cs[0].family == fam)
        ok, cont = [], Counter()
        for g in grupos:
            acertos = []
            for c in por_grupo[g]:
                for o in ORDENS_CONFERENCIA:
                    t = resp.get(f"{c.id}|{o}")
                    esc = None if t is None else ler_conferente(t, c, o)
                    cont["ilegível" if esc is None else ("acerta" if esc == c.gold else "erra")] += 1
                    acertos.append(esc == c.gold)
            if all(acertos):
                ok.append(g)
        escolhidos = sorted(ok, key=lambda g: bucket(f"selecao:{g}", salt))[:n]
        ranking = sorted(escolhidos, key=lambda g: bucket(f"dificuldade:{g}", salt))
        for j, g in enumerate(ranking):
            sp = "dificuldade" if j < n_dif else ("dev" if bucket(f"split:{g}", salt) < 0.5 else "test")
            for c in por_grupo[g]:
                c.split = sp
                final.append(c)
        rel[fam] = {"grupos_lidos": len(grupos), "passam_conferencia": len(ok), "escolhidos": len(escolhidos),
                    "faltam": max(0, n - len(escolhidos)), "respostas_do_conferente": dict(cont),
                    "conferente_cortado": cortadas.get(fam, 0),
                    "por_split": dict(Counter(por_grupo[g][0].split for g in escolhidos))}
    return final, rel


# ------------------------------------------------------------------ fonte registrada

def sha256_arquivo(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prof_cases(cfg: dict, path: Path | None = None, antes_do_registro: bool = False) -> list[Case]:
    """Casos do arquivo registrado. Sem sha256 no splits.toml só abre com `antes_do_registro=True` (auditoria,
    revisão humana e filtro de dificuldade); com sha256, o arquivo tem de ser exatamente o registrado."""
    conf = cfg["prof"]
    path = path or ROOT / conf["arquivo"]
    esperado = conf.get("sha256")
    if not path.exists():
        raise FileNotFoundError(f"{path}: arquivo das reservadas com professor (gerado por slurm/prof_gerar.sbatch)")
    if esperado:
        if sha256_arquivo(path) != esperado:
            raise ValueError(f"{path} não é o arquivo registrado (sha256 diferente)")
    elif not antes_do_registro:
        raise ValueError("reservadas com professor ainda sem registro (sha256 vazio em splits.toml [prof])")
    return [Case.from_dict(r) for r in read_jsonl(path)]


TAMANHO_GRUPO = {"postura_alvo": 2, "inferencia_pt": 3, "resposta_indireta": 2, "correferencia": 2}
MAX_GRUPOS_ERRADOS_REVISAO = 3  # protocolo de 16/11: a família fica com até 3 de 30 grupos errados


def corrigir_revisao(planilha: str, gabarito: dict) -> dict:
    """Compara a planilha preenchida às cegas com o gabarito. Um grupo conta como errado se qualquer item dele
    tiver letra diferente, `?` ou ficar em branco. Saída por família, para a adjudicação."""
    letras = dict(re.findall(r"\*\*Item (\d+)\*\*.*?Resposta:[ \t]*([A-Z?]?)", planilha, re.S))
    out = {}
    for k, g in gabarito.items():
        f = out.setdefault(g["familia"], {"itens": 0, "itens_errados": 0, "grupos": set(), "grupos_errados": set()})
        f["itens"] += 1
        f["grupos"].add(g["grupo"])
        if letras.get(k, "") != g["letra"]:
            f["itens_errados"] += 1
            f["grupos_errados"].add(g["grupo"])
    return {fam: {"itens": f["itens"], "itens_errados": f["itens_errados"], "grupos_revisados": len(f["grupos"]),
                  "grupos_errados": sorted(f["grupos_errados"])} for fam, f in out.items()}


def checar_registro(cfg: dict, casos: list[Case], adjudicacao: dict, auditoria: dict, sobreposicao: dict) -> list[str]:
    """Tudo o que o protocolo exige antes de pôr o sha256 no splits.toml. Devolve a lista de problemas (vazia = pode
    registrar). `adjudicacao`: {familia: {"grupos_revisados": n, "grupos_errados": [ids] ou n}} depois da
    adjudicação da revisão humana."""
    conf = cfg["prof"]
    n, n_dif = int(conf["grupos_por_familia"]), int(conf["grupos_dificuldade"])
    probs = []
    por_g = defaultdict(list)
    for c in casos:
        por_g[c.group_id].append(c)
    for fam in conf["familias"]:
        gs = [g for g, cs in por_g.items() if cs[0].family == fam]
        if len(gs) != n:
            probs.append(f"{fam}: {len(gs)} grupos, o protocolo pede {n}")
        ruins = [g for g in gs if len(por_g[g]) != TAMANHO_GRUPO[fam] or len({c.split for c in por_g[g]}) != 1]
        if ruins:
            probs.append(f"{fam}: {len(ruins)} grupos incompletos ou com split misturado (ex.: {ruins[0]})")
        sp = Counter(por_g[g][0].split for g in gs)
        if sp["dificuldade"] != n_dif or sp["dev"] + sp["test"] != n - n_dif or set(sp) - {"dificuldade", "dev", "test"}:
            probs.append(f"{fam}: splits {dict(sp)}")
        a = auditoria["familias"].get(fam)
        if a is None:
            probs.append(f"{fam}: sem auditoria")
        elif a["acima_do_limite"]:
            probs.append(f"{fam}: atalhos acima do limite {a['acima_do_limite']}")
        r = adjudicacao.get(fam)
        if r is None:
            probs.append(f"{fam}: sem revisão humana adjudicada")
        else:
            errados = r["grupos_errados"] if isinstance(r["grupos_errados"], int) else len(r["grupos_errados"])
            if r["grupos_revisados"] < 30:
                probs.append(f"{fam}: só {r['grupos_revisados']} grupos revisados (o protocolo pede 30)")
            if errados > MAX_GRUPOS_ERRADOS_REVISAO:
                probs.append(f"{fam}: {errados} grupos errados na revisão (máximo {MAX_GRUPOS_ERRADOS_REVISAO})")
    probs += [f"sobreposição com {s}: {k} casos" for s, k in sobreposicao.items() if k]
    return probs


def gravar_casos(casos: list[Case], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for c in sorted(casos, key=lambda c: (PROF_FAMILIES.index(c.family), int(c.id.split(":")[2]),
                                              int(c.id.split(":")[3]))):
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    return sha256_arquivo(path)


# ------------------------------------------------------------------ auditoria (antes do registro)

def _palavras(t: str) -> set[str]:
    return {w for w in re.findall(r"\w+", t.casefold()) if len(w) > 2}


def _desempate(c: Case) -> str:
    """Desempate neutro das heurísticas de grupo: os casos vêm na ordem do gabarito, e um sort estável acertaria."""
    return hashlib.sha256(c.id.encode()).hexdigest()


def _sobreposicao(c: Case) -> float:
    texto, hip = c.state.split("\n\nHipótese: ")
    return len(_palavras(hip) & _palavras(texto)) / max(1, len(_palavras(hip)))


# Partes do item que, sozinhas, não bastam para decidir (entradas do classificador TF-IDF). Emenda 4: na v2 a
# hipótese e a resposta são as mesmas dentro do grupo, e as leituras que importam passam a ser "só o texto" (com a
# faixa de sobreposição com a hipótese) e "só a pergunta"; "só a hipótese" e "só a resposta" ficam como controle,
# perto do acaso por construção.
PARTES = {
    "inferencia_pt": {
        "so_hipotese": lambda c: c.state.split("\n\nHipótese: ")[1],
        "so_texto": lambda c: (c.state.split("\n\nHipótese: ")[0]
                               + f" __sobreposicao_{min(4, int(_sobreposicao(c) * 5))}__"),
    },
    "resposta_indireta": {
        "so_resposta": lambda c: c.state.rsplit(" responde: ", 1)[1],
        "so_pergunta": lambda c: c.state.split(" pergunta: ", 1)[1].split("\n", 1)[0],
    },
}


def _tfidf_cv(casos: list[Case], parte=lambda c: c.state) -> int:
    """Acertos de um TF-IDF + regressão logística em `parte(caso)`, em validação cruzada (5 partes, por grupo)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    x, y, g = [parte(c) for c in casos], [c.gold for c in casos], [c.group_id for c in casos]
    modelo = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                           LogisticRegression(max_iter=2000, C=1.0))
    pred = cross_val_predict(modelo, x, y, groups=g, cv=GroupKFold(n_splits=5))
    return int(sum(p == t for p, t in zip(pred, y)))


def _heuristicas(c: Case) -> dict[str, str]:
    """Atalhos de forma de um caso: cada um devolve o id que escolheria."""
    ops = c.options
    lens = [len(desc) for _, desc in ops]
    sw = _palavras(c.state)
    forma = {"mais_longa": ops[lens.index(max(lens))][0], "mais_curta": ops[lens.index(min(lens))][0],
             "mais_palavras_do_contexto": max(ops, key=lambda o: len(_palavras(o[1]) & sw))[0],
             "primeira": ops[0][0]}
    if c.family == "correferencia":
        texto = c.state.removeprefix("Texto: ")
        pron = re.search(r"\"(\w+)\"", c.question).group(1)
        pos_p = re.search(rf"\b{pron}\b", texto, re.I).start()
        pos = {oid: texto.index(nome) for oid, nome in ops}
        forma["primeira_citada"] = min(pos, key=pos.get)
        antes = {oid: p for oid, p in pos.items() if p < pos_p} or pos
        forma["citada_mais_perto_antes_do_pronome"] = max(antes, key=antes.get)
    return forma


def prof_audit(cfg: dict, casos: list[Case]) -> dict:
    """Contagens, classes e o acerto de cada atalho (limites de 15/11: forma min(60%, acaso + 20 pontos), leitura
    parcial 80%). As leituras parciais que precisam do grupo inteiro são calculadas aqui."""
    out = {"versao": PROF_VERSION, "familias": {}}
    for fam in cfg["prof"]["familias"]:
        cs = [c for c in casos if c.family == fam]
        if not cs:
            continue
        n = len(cs)
        acaso = sum(1 / len(c.options) for c in cs) / n
        limite = min(0.60, acaso + 0.20)
        maj = Counter(c.gold for c in cs).most_common(1)[0][0]
        hf, hp = Counter({"classe_majoritaria": sum(c.gold == maj for c in cs)}), Counter()
        for c in cs:
            for k, v in _heuristicas(c).items():
                hf[k] += v == c.gold
        for nome, parte in PARTES.get(fam, {}).items():
            hf["tfidf_" + nome] = _tfidf_cv(cs, parte)
        if fam == "inferencia_pt":  # no grupo: mais sobreposição com a hipótese -> decorre, menos -> neutra
            por_g = defaultdict(list)
            for c in cs:
                por_g[c.group_id].append(c)
            hf["sobreposicao_no_grupo"] = 0
            for gcs in por_g.values():
                ordem = sorted(gcs, key=lambda c: (-_sobreposicao(c), _desempate(c)))
                hf["sobreposicao_no_grupo"] += sum(c.gold == r for c, r in zip(ordem, ("decorre", "contradiz", "neutra")))
        if fam in ("postura_alvo", "correferencia"):  # ignora o alvo / a palavra trocada: uma resposta por grupo
            por_g = defaultdict(list)
            for c in cs:
                por_g[c.group_id].append(c.gold)
            chave = "ignora_o_alvo" if fam == "postura_alvo" else "ignora_a_palavra_trocada"
            hp[chave] = sum(Counter(gs).most_common(1)[0][1] for gs in por_g.values())
        if fam == "postura_alvo":
            hp["tfidf_so_comentario"] = _tfidf_cv(cs)
        out["familias"][fam] = {
            "casos": n, "grupos": len({c.group_id for c in cs}),
            "por_split_grupos": dict(sorted(Counter(c.split for c in cs if c.id.endswith(":0")).items())),
            "classes": dict(Counter(c.gold for c in cs).most_common()),
            "acaso": round(acaso, 3),
            "atalhos_de_forma": {k: f"{v}/{n}" for k, v in sorted(hf.items())},
            "atalhos_de_leitura_parcial": {k: f"{v}/{n}" for k, v in sorted(hp.items())},
            "acima_do_limite": sorted([k for k, v in hf.items() if v / n > limite]
                                      + [k for k, v in hp.items() if v / n > 0.80]),
            "limites": {"forma": round(limite, 3), "parcial": 0.80},
            "texto_medio_chars": round(sum(len(c.state) for c in cs) / n),
        }
    return out


def prof_sobreposicao(cfg: dict, casos: list[Case]) -> dict:
    """Casos de treino ou calib (Felhen, Telepatia, Traces, F3, Eikos) que repetem texto das reservadas com
    professor: texto normalizado igual ou 13 palavras seguidas. Tem de dar zero em todas."""
    from .data import LOADERS, NEW_SOURCES, TRAIN_ONLY, _shingles
    exact = {" ".join(c.state.casefold().split()) for c in casos}
    grams = set().union(*(_shingles(c.state) for c in casos))
    fontes = {**{s: LOADERS[s](cfg) for s in LOADERS}, "f3": NEW_SOURCES["f3"](cfg),
              "eikos": TRAIN_ONLY["eikos"](cfg)}
    return {s: sum(1 for c in cs if c.split in ("train", "calib") and (
        " ".join(c.state.casefold().split()) in exact or not grams.isdisjoint(_shingles(c.state))))
        for s, cs in fontes.items()}


# ------------------------------------------------------------------ revisão humana (às cegas)

def revisao_humana(cfg: dict, casos: list[Case], por_familia: int = 30) -> tuple[str, dict]:
    """Planilha às cegas (markdown) com `por_familia` grupos de dev+teste por família, e o gabarito à parte."""
    from .prompts import LETTERS
    salt = cfg["prof"]["salt"]
    por_g = defaultdict(list)
    for c in casos:
        if c.split in ("dev", "test"):
            por_g[c.group_id].append(c)
    linhas = ["# Revisão humana das reservadas com professor (às cegas)", "",
              "Para cada item, escreva a letra que você escolheria depois de `Resposta:`. Se nenhuma opção servir, "
              "escreva `?` e um comentário curto na mesma linha.", ""]
    gabarito, k = {}, 0
    for fam in cfg["prof"]["familias"]:
        gs = sorted((g for g, cs in por_g.items() if cs[0].family == fam),
                    key=lambda g: bucket(f"revisao:{g}", salt))[:por_familia]
        linhas += [f"## {fam}", ""]
        for g in gs:
            for c in sorted(por_g[g], key=lambda c: c.id):
                k += 1
                ops = "\n".join(f"   {LETTERS[j]}. {d}" for j, (_, d) in enumerate(c.options))
                estado = c.state.replace("\n", "\n   > ")
                linhas += [f"**Item {k}** ({g})", "", f"   > {estado}", "", f"   {c.question}", "", ops, "",
                           "   Resposta: ", ""]
                gabarito[str(k)] = {"id": c.id, "grupo": g, "familia": fam,
                                    "letra": LETTERS[[o[0] for o in c.options].index(c.gold)]}
    return "\n".join(linhas), gabarito
