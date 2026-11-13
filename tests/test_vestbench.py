"""Carregador das provas (VestBench-Jev). O JSONL convertido fica fora do git; sem ele, o teste é pulado."""

import pytest

from lumi_decision import data

ENEM = data.DATA_DIR / "vestbench" / "enem.jsonl"


@pytest.mark.skipif(not ENEM.exists(), reason="ENEM convertido não está em data/")
def test_enem_cases_are_eval_only_and_split_in_halves():
    cs = data.load_cases("vestbench_enem")
    data.check_integrity(cs)
    assert all(c.held_out and c.qtype == "choice" and len(c.options) == 5 for c in cs)
    splits = {c.split for c in cs}
    assert splits == {"explorar", "confirmar", data.DISCARDED}
    assert {c.id for c in cs if c.split == data.DISCARDED} == {"vestbench:enem-enem-2015-2-144",
                                                               "vestbench:enem-enem-2016-2-2-177"}
    assert "vestbench_enem" not in data.LOADERS  # nunca entra nas auditorias nem no treino


def test_positional_reference_regex():
    rx = data._POSICIONAL
    assert rx.search("A e C.") and rx.search("Todas as alternativas estão corretas") and rx.search("nenhuma das anteriores")
    assert not rx.search("a invenção do tear propiciou novas relações") and not rx.search("o período anterior foi longo")
