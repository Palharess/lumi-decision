"""Montagem do prompt e das opções apresentadas.

A decisão é lida exatamente na posição em que a geração do Lumi escreve a letra:
logo após `<|im_start|>assistant\\n<think></think>`, com letras sem espaço antes.
Qualquer mudança no texto abaixo muda PROMPT_VERSION.
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .schema import Case

PROMPT_VERSION = "v1"
LETTERS = string.ascii_uppercase
MAX_OPTIONS = len(LETTERS)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def load_template(path: Path):
    """Mesmo ambiente Jinja que o transformers usa em `apply_chat_template`."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = _tojson
    return env.from_string(path.read_text(encoding="utf-8"))


def order_options(case: Case, order: str) -> list[tuple[str, str]]:
    """`canonical` (ordem do dataset), `reversed`, `shuffle<seed>` (determinística por caso), `rot<k>` (rotação
    de k posições, módulo n) ou `perm<índices>` (permutação explícita, ex.: `perm201`; exige n = nº de dígitos)."""
    opts = list(case.options)
    if order == "canonical":
        return opts
    if order == "reversed":
        return opts[::-1]
    if order.startswith("shuffle"):
        seed = int(order.removeprefix("shuffle") or 0)
        random.Random(f"{seed}:{case.id}").shuffle(opts)
        return opts
    if order.startswith("rot"):
        k = int(order.removeprefix("rot")) % len(opts)
        return opts[k:] + opts[:k]
    if order.startswith("perm"):
        idx = [int(ch) for ch in order.removeprefix("perm")]
        if sorted(idx) != list(range(len(opts))):
            raise ValueError(f"{case.id}: {order} não é permutação de {len(opts)} opções")
        return [opts[i] for i in idx]
    raise ValueError(f"ordem desconhecida: {order}")


def user_message(case: Case, options: list[tuple[str, str]]) -> str:
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"{case.id}: {len(options)} opções (suportado: 2..{MAX_OPTIONS})")
    choices = "\n".join(f"{LETTERS[i]}. {desc}" for i, (_, desc) in enumerate(options))
    return (f"Contexto:\n{case.state}\n\nPergunta: {case.question}\n\n"
            f"Opções:\n{choices}\n\nResponda apenas com a letra da opção escolhida.")


def option_char_ends(prompt: str, case: Case, options: list[tuple[str, str]]) -> list[int]:
    """Índice (no prompt renderizado) do último caractere não branco de cada linha de opção "X. descrição".
    Recalcula a mensagem com `user_message` (o prompt NÃO muda) e exige que ela apareça uma única vez."""
    msg = user_message(case, options)
    if prompt.count(msg) != 1:
        raise ValueError(f"{case.id}: mensagem do usuário não aparece exatamente uma vez no prompt")
    head = f"Contexto:\n{case.state}\n\nPergunta: {case.question}\n\nOpções:\n"
    if not msg.startswith(head):
        raise ValueError(f"{case.id}: formato da mensagem mudou")
    pos, ends = prompt.index(msg) + len(head), []
    for i, (_, desc) in enumerate(options):
        line = f"{LETTERS[i]}. {desc}"
        ends.append(pos + len(line.rstrip()) - 1)
        pos += len(line) + 1
    return ends


def option_token_positions(offsets, prompt: str, char_ends: list[int]) -> list[int]:
    """Token que contém o último caractere de cada opção (E4: chave da pointer head). O token pode avançar
    sobre a quebra de linha seguinte, mas só sobre espaço/quebra de linha; posições estritamente crescentes."""
    import bisect

    spans = sorted((s, e, t) for t, (s, e) in enumerate(offsets) if e > s)  # tokens especiais podem ter (0, 0)
    starts = [s for s, _, _ in spans]
    out = []
    for c in char_ends:
        s, e, t = spans[max(0, bisect.bisect_right(starts, c) - 1)]
        if not s <= c < e:
            raise ValueError(f"caractere {c} sem token (offsets {(s, e)})")
        if prompt[c + 1:e].strip():
            raise ValueError(f"token da opção avança sobre texto não branco: {prompt[s:e]!r}")
        if out and t <= out[-1]:
            raise ValueError("duas opções no mesmo token")
        out.append(t)
    if out and out[-1] >= len(offsets) - 1:
        raise ValueError("opção na última posição do prompt")
    return out


def render(template, case: Case, options: list[tuple[str, str]]) -> str:
    return template.render(messages=[{"role": "user", "content": user_message(case, options)}],
                           add_generation_prompt=True, enable_thinking=False)


# E6 (analises/e6_protocolo.md): formatos de decisão do treino. "v1" é o prompt de sempre, o mesmo da avaliação das
# reservadas, com a letra no 1º token da resposta. Nos outros, a resposta começa preenchida com o prefixo
# "Resposta:" e a letra (com espaço antes) é lida na posição seguinte, como o LLM-as-Jev lê depois de
# "Best answer: [". Assim o 1º token da resposta não é onde se decide e pode ser ancorado no Lumi. Nenhum formato
# copia o texto dos prompts do Lumi-Benchmark, que é o guarda-corpo. Mudar um texto daqui muda o protocolo do E6.
FORMATOS = {
    "v1": {"cab": ("Contexto", "Pergunta", "Opções"), "rotulo": "{L}. ",
           "instrucao": "Responda apenas com a letra da opção escolhida.", "prefixo": ""},
    "p1": {"cab": ("Contexto", "Pergunta", "Opções"), "rotulo": "{L}) ",
           "instrucao": "Responda no formato 'Resposta: <letra>'.", "prefixo": "Resposta:"},
    "p2": {"cab": ("Texto", "Questão", "Alternativas"), "rotulo": "({L}) ",
           "instrucao": "Indique a alternativa correta no formato 'Resposta: <letra>', sem justificar.",
           "prefixo": "Resposta:"},
    "p3": {"cab": ("Situação", "O que decidir", "Escolhas"), "rotulo": "{L} - ",
           "instrucao": "Escolha uma das opções e responda só com 'Resposta: <letra>'.", "prefixo": "Resposta:"},
}
# Contexto de âncora `cot`: o mesmo caso com uma instrução que deixa raciocinar. Sem perda de decisão; só a
# distribuição do 1º token da resposta é ancorada no Lumi (é a posição em que o E3b passou a escrever a letra).
FORMATO_COT = {"cab": ("Contexto", "Pergunta", "Opções"), "rotulo": "{L}) ",
               "instrucao": "Se ajudar, pense passo a passo antes de responder. Termine com uma linha "
                            "'Resposta: <letra>'.", "prefixo": ""}


def user_message_formato(case: Case, options: list[tuple[str, str]], formato: str | dict) -> str:
    """Mensagem do usuário num formato de decisão. "v1" devolve exatamente `user_message`."""
    if formato == "v1":
        return user_message(case, options)
    f = FORMATOS[formato] if isinstance(formato, str) else formato
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"{case.id}: {len(options)} opções (suportado: 2..{MAX_OPTIONS})")
    c, p, o = f["cab"]
    choices = "\n".join(f["rotulo"].format(L=LETTERS[i]) + desc for i, (_, desc) in enumerate(options))
    return f"{c}:\n{case.state}\n\n{p}: {case.question}\n\n{o}:\n{choices}\n\n{f['instrucao']}"


def render_formato(template, case: Case, options: list[tuple[str, str]], formato: str | dict) -> tuple[str, str]:
    """(prompt até o início da resposta, prefixo preenchido na resposta). Em "v1" o prompt é o de `render`."""
    f = FORMATOS[formato] if isinstance(formato, str) else formato
    prompt = template.render(messages=[{"role": "user", "content": user_message_formato(case, options, formato)}],
                             add_generation_prompt=True, enable_thinking=False)
    return prompt, f["prefixo"]


def candidate_tokens(tokenize, prompt: str, n: int, espaco: str = "") -> tuple[list[int], list[int]]:
    """Tokeniza o prompt e devolve (ids do prompt, id de cada letra candidata).

    Exige que cada letra (precedida de `espaco`, " " depois do prefixo "Resposta:") seja exatamente um token novo
    sem alterar a tokenização do prompt; caso contrário o readout não corresponderia à geração.
    """
    base = tokenize(prompt)
    cands = []
    for letter in LETTERS[:n]:
        ids = tokenize(prompt + espaco + letter)
        if len(ids) != len(base) + 1 or ids[:-1] != base:
            raise ValueError(f"letra {espaco + letter!r} não é um único token na fronteira do prompt")
        cands.append(ids[-1])
    if len(set(cands)) != n:
        raise ValueError("letras candidatas com ids repetidos")
    return base, cands
