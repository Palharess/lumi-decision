"""Reservadas v2 (src/lumi_decision/res2.py): determinismo, gabarito por construção, splits e atalhos."""

from collections import Counter

import pytest

from lumi_decision.data import check_integrity, load_cases, load_config
from lumi_decision.prompts import order_options, user_message
from lumi_decision.res2 import RES2_FAMILIES, res2_audit, res2_cases, verificar_gabarito


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def cases(cfg):
    return res2_cases(cfg)


def test_deterministic_and_loaded_by_source(cfg, cases):
    again = load_cases("res2", cfg)
    assert [(c.id, c.state, c.options, c.gold, c.split) for c in cases] == \
           [(c.id, c.state, c.options, c.gold, c.split) for c in again]


def test_all_held_out_with_verified_gold(cases):
    check_integrity(cases)
    assert {c.family for c in cases} == set(RES2_FAMILIES)
    assert all(c.held_out and c.source == "res2" for c in cases)
    assert all(verificar_gabarito(c) for c in cases)
    assert len({c.group_id for c in cases}) == len(cases)


def test_splits(cfg, cases):
    conf = cfg["res2"]
    for fam in RES2_FAMILIES:
        sp = Counter(c.split for c in cases if c.family == fam)
        assert set(sp) == {"dificuldade", "dev", "test"}
        assert sp["dificuldade"] == conf["grupos_dificuldade"]
        assert sum(sp.values()) == conf["grupos_por_familia"]
        assert sp["dev"] >= 250 and sp["test"] >= 250


def test_fixed_option_count_and_rotations_cover_positions(cases):
    for fam in RES2_FAMILIES:
        fc = [c for c in cases if c.family == fam]
        n = {len(c.options) for c in fc}
        assert len(n) == 1, fam  # rotações rot0..rot(n-1) valem para a família inteira
        n = n.pop()
        c = fc[0]
        pos = {[o[0] for o in order_options(c, "canonical" if k == 0 else f"rot{k}")].index(c.gold)
               for k in range(n)}
        assert pos == set(range(n))
        user_message(c, c.options)  # o prompt monta sem erro


def test_gold_breaks_if_data_changes(cases):
    """O verificador lê os dados de verdade: trocar um valor que decide o caso muda o gabarito."""
    c = next(c for c in cases if c.family == "regra_beneficio" and c.gold == "conceder")
    d = dict(c.meta["dados"])
    d["valor"] = d["teto"] + 1
    from lumi_decision.res2 import _verifica_beneficio
    assert _verifica_beneficio(d) == "encaminhar"
    o = next(c for c in cases if c.family == "ordem_eventos")
    d = dict(o.meta["dados"])
    d["pergunta"] = "ultimo" if d["pergunta"] == "primeiro" else "primeiro"
    from lumi_decision.res2 import _verifica_ordem
    assert _verifica_ordem(d) != o.gold


def test_no_shortcut_above_limits(cfg, cases):
    rep = res2_audit(cfg, cases)
    for fam, r in rep["familias"].items():
        assert r["acima_do_limite"] == [], (fam, r["acima_do_limite"])
        assert r["gabarito_verificado"] == cfg["res2"]["grupos_por_familia"]
        pos = r["posicao_canonica_do_correto"]
        assert min(pos.values()) > 0.6 * max(pos.values()), (fam, pos)
