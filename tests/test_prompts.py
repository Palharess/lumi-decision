"""Testes de prompt com o tokenizer real do Lumi (precisa de `lumi-dados baixar lumi`)."""

from pathlib import Path

import pytest
from jinja2 import Environment

from lumi_decision.data import ARTIFACTS_DIR, check_integrity, load_cases, load_config
from lumi_decision.prompts import LETTERS, candidate_tokens, load_template, order_options, render, user_message
from lumi_decision.schema import Case

TEMPLATE = ARTIFACTS_DIR / "lumi" / "chat_template.jinja"
TOKENIZER = ARTIFACTS_DIR / "lumi" / "tokenizer.json"
pytestmark = pytest.mark.skipif(not TOKENIZER.exists(), reason="artefatos do Lumi não baixados")


@pytest.fixture(scope="module")
def tok():
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(str(TOKENIZER))
    return lambda s: t.encode(s, add_special_tokens=False).ids


def make_case(n=26):
    return Case(id="x", group_id="x", source="t", family="f", split="dev", held_out=False, qtype="choice",
                state="Cliente pede reembolso de um pedido.", question="O que fazer?",
                options=[(f"op{i}", f"Opção número {i}") for i in range(n)], gold="op0")


def test_prompt_ends_at_generation_position():
    p = render(load_template(TEMPLATE), make_case(3), make_case(3).options)
    assert p.endswith("<|im_start|>assistant\n<think></think>")


def test_hf_style_render_vs_default_jinja():
    """O piloto usou o Environment padrão do Jinja, que deixa um '\\n' antes de <|im_start|>
    (sobra da macro do template). O transformers usa trim/lstrip_blocks e não deixa.
    Seguimos o transformers; a diferença é só esse primeiro caractere."""
    case = make_case(4)
    msgs = [{"role": "user", "content": user_message(case, case.options)}]
    default = Environment().from_string(TEMPLATE.read_text(encoding="utf-8")).render(
        messages=msgs, add_generation_prompt=True, enable_thinking=False)
    ours = render(load_template(TEMPLATE), case, case.options)
    assert ours.startswith("<|im_start|>system\n")
    assert default == "\n" + ours


def test_all_26_letters_are_single_tokens(tok):
    case = make_case(26)
    prompt = render(load_template(TEMPLATE), case, case.options)
    base, cands = candidate_tokens(tok, prompt, 26)
    assert len(cands) == 26 and len(set(cands)) == 26


def test_every_dev_case_fits_letter_readout(tok):
    """Todos os casos de dev das três fontes passam na checagem de fronteira (amostra)."""
    tpl = load_template(TEMPLATE)
    cfg = load_config()
    for src in ("felhen", "telepatia", "traces"):
        cases = [c for c in load_cases(src, cfg) if c.split == "dev"][:60]
        for c in cases:
            for order in ("canonical", "reversed", "shuffle1"):
                opts = order_options(c, order)
                candidate_tokens(tok, render(tpl, c, opts), len(opts))


def test_splits_integrity():
    cfg = load_config()
    for src in ("felhen", "telepatia", "traces"):
        cases = load_cases(src, cfg)
        check_integrity(cases)
        assert not any(c.held_out and c.split == "calib" for c in cases)


def test_order_options_deterministic():
    c = make_case(6)
    assert order_options(c, "shuffle3") == order_options(c, "shuffle3")
    assert order_options(c, "reversed") == c.options[::-1]
    assert sorted(order_options(c, "shuffle3")) == sorted(c.options)


def test_rot_and_perm_orders():
    from lumi_decision.prompts import order_options
    from lumi_decision.schema import Case
    c = Case(id="x", group_id="g", source="s", family="f", split="dev", held_out=False, qtype="choice", state="s",
             question="q", options=[("a", "A"), ("b", "B"), ("c", "C")], gold="a")
    ids = lambda o: [x[0] for x in order_options(c, o)]  # noqa: E731
    assert ids("rot1") == ["b", "c", "a"] and ids("rot3") == ["a", "b", "c"]
    assert ids("perm201") == ["c", "a", "b"]
    import pytest
    with pytest.raises(ValueError):
        order_options(c, "perm01")
