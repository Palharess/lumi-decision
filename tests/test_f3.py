"""F3: as famílias novas respeitam as regras de split e não mexem nas fontes atuais."""

from collections import Counter

import pytest

from lumi_decision import data
from lumi_decision.f3 import GOEMO_OPTIONS, RAW, f3_cases

pytestmark = pytest.mark.skipif(not (RAW / "massive" / "pt-PT.jsonl").exists(), reason="brutos do F3 não baixados")


@pytest.fixture(scope="module")
def cases():
    return f3_cases(data.load_config(), protect=[])


def test_f3_is_not_in_current_loaders():
    """Os dados e as auditorias dos treinos já feitos (E3, E3b, F2) não podem mudar."""
    assert "f3" not in data.LOADERS and "f3" in data.NEW_SOURCES


def test_integrity_and_held_out(cases):
    data.check_integrity(cases)
    held = data.load_config()["f3"]["held_out_families"]
    assert held == ["pss_mais_simples"]
    assert all(c.split in ("dev", "test") for c in cases if c.family in held)
    assert not [c for c in data.training_cases(cases) if c.family in held]


def test_menus_and_labels(cases):
    assert all(len(c.options) >= 2 and c.gold in {o for o, _ in c.options} for c in cases)
    pss = Counter(c.gold for c in cases if c.family == "pss_mais_simples")
    assert abs(pss["1"] - pss["2"]) < 0.05 * sum(pss.values())  # a frase simples cai nas duas posições
    emo = [c for c in cases if c.family == "goemo_emocao"]
    assert all(abs(sum(c.gold_dist.values()) - 1) < 1e-9 and max(c.gold_dist, key=c.gold_dist.get) == c.gold
               for c in emo)
    assert {o for o, _ in GOEMO_OPTIONS} == {c.gold for c in emo}
    assert not {c.meta["cenario"] for c in cases if c.family == "massive_intencao"} & {"transport", "general",
                                                                                         "cooking", "news", "weather"}
