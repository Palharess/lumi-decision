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
    """`canonical` (ordem do dataset), `reversed`, ou `shuffle<seed>` (determinística por caso)."""
    opts = list(case.options)
    if order == "canonical":
        return opts
    if order == "reversed":
        return opts[::-1]
    if order.startswith("shuffle"):
        seed = int(order.removeprefix("shuffle") or 0)
        random.Random(f"{seed}:{case.id}").shuffle(opts)
        return opts
    raise ValueError(f"ordem desconhecida: {order}")


def user_message(case: Case, options: list[tuple[str, str]]) -> str:
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"{case.id}: {len(options)} opções (suportado: 2..{MAX_OPTIONS})")
    choices = "\n".join(f"{LETTERS[i]}. {desc}" for i, (_, desc) in enumerate(options))
    return (f"Contexto:\n{case.state}\n\nPergunta: {case.question}\n\n"
            f"Opções:\n{choices}\n\nResponda apenas com a letra da opção escolhida.")


def render(template, case: Case, options: list[tuple[str, str]]) -> str:
    return template.render(messages=[{"role": "user", "content": user_message(case, options)}],
                           add_generation_prompt=True, enable_thinking=False)


def candidate_tokens(tokenize, prompt: str, n: int) -> tuple[list[int], list[int]]:
    """Tokeniza o prompt e devolve (ids do prompt, id de cada letra candidata).

    Exige que cada letra seja exatamente um token novo sem alterar a tokenização do prompt;
    caso contrário o readout não corresponderia à geração.
    """
    base = tokenize(prompt)
    cands = []
    for letter in LETTERS[:n]:
        ids = tokenize(prompt + letter)
        if len(ids) != len(base) + 1 or ids[:-1] != base:
            raise ValueError(f"letra {letter} não é um único token na fronteira do prompt")
        cands.append(ids[-1])
    if len(set(cands)) != n:
        raise ValueError("letras candidatas com ids repetidos")
    return base, cands
