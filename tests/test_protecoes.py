"""Proteções que a revisão de 11/11 pediu: splits por texto, referências dentro do menu,
retomada só com a mesma configuração e E0 que reprova de verdade."""

import argparse
import json
from collections import defaultdict

import pytest

from lumi_decision import e0_sanidade, evaluate
from lumi_decision.data import ARTIFACTS_DIR, DISCARDED, load_cases, load_config

pytestmark = pytest.mark.skipif(not (ARTIFACTS_DIR / "lumi" / "tokenizer.json").exists(),
                                reason="dados não baixados")


@pytest.mark.parametrize("src", ["felhen", "telepatia", "traces"])
def test_no_text_in_two_active_splits(src):
    splits = defaultdict(set)
    for c in load_cases(src, load_config()):
        if c.split != DISCARDED and not (c.held_out and c.split == "train"):
            splits[" ".join(c.state.casefold().split())].add(c.split)
    assert not [t for t, s in splits.items() if len(s) > 1]


@pytest.mark.parametrize("src", ["felhen", "telepatia", "traces"])
def test_baselines_pick_inside_menu(src):
    cases = load_cases(src, load_config())
    dev = [c for c in cases if c.split == "dev"]
    menus = {c.id: {o[0] for o in c.options} for c in dev}
    for name in ("maioria", "lexical"):
        rows, _ = evaluate.baseline_probs(name, cases, dev)
        assert rows and all(r["pred"] in menus[r["case_id"]] for r in rows)
        assert all(abs(sum(r["probs"].values()) - 1) < 1e-6 for r in rows)


def test_resume_refuses_other_configuration(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "predicoes.jsonl").write_text('{"case_id": "x", "order": "canonical"}\n', encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"impressao": {"model": "outro"}}), encoding="utf-8")
    args = argparse.Namespace(backend="hf", model="ceia-br/lumi-4b", revision="abc", dtype="bfloat16", native=False,
                              ctx=0, template=str(ARTIFACTS_DIR / "lumi" / "chat_template.jinja"), source="felhen",
                              split="dev", no_generate=True, overwrite=False, out=str(out))
    with pytest.raises(SystemExit, match="outra configuração"):
        evaluate.cmd_run(args)


def test_truncated_last_line_is_dropped(tmp_path):
    p = tmp_path / "predicoes.jsonl"
    p.write_text('{"case_id": "a", "order": "canonical"}\n{"case_id": "b", "ord', encoding="utf-8")
    assert evaluate.read_existing(p) == {("a", "canonical")}
    assert p.read_text(encoding="utf-8").count("\n") == 1


def test_e0_verdict_requires_every_mandatory_check():
    ok = {k: {"ok": True} for k in e0_sanidade.REQUIRED}
    assert e0_sanidade.verdict(ok)
    for k in e0_sanidade.REQUIRED:
        bad = dict(ok)
        bad[k] = {"ok": False}
        assert not e0_sanidade.verdict(bad)
    assert not e0_sanidade.verdict({k: v for k, v in ok.items() if k != "4_geracao"})
