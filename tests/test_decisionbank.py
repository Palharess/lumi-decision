"""DecisionBank no E5 (src/lumi_decision/decisionbank.py): filtro da `j11_generator` e proteção das avaliações."""

import json

from lumi_decision import data, decisionbank as db, lora
from lumi_decision.schema import Case


def row(i, family="j11_calibration_gen", question="Which outcome will the new case have?", **kw):
    r = {"id": f"j11g-{i}", "family": family, "part": "j11_generator", "source_license": "cc-by-sa-4.0 (ours)",
         "target_origin": "computed by code", "state": f"New lead number {i} from Quillon Markets.",
         "question": question, "qtype": "choice", "label": 1, "target_distribution": [0.2, 0.5, 0.3],
         "options": [{"id": k, "description": f"{k}: the case ends as {k}."} for k in ("a", "b", "c")],
         "tier": "easy", "language": "en"}
    r.update(kw)
    return r


def write(tmp_path, rows):
    p = tmp_path / "shard.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return [p]


def test_caso_e_alvo(tmp_path):
    (c,) = db.decisionbank_cases({}, paths=write(tmp_path, [row(0)]))
    assert c.split == "train" and c.source == "decisionbank" and not c.held_out
    assert c.gold == "b" and abs(c.gold_dist["b"] - 0.5) < 1e-9 and c.options[0] == ("a", "a: the case ends as a.")


def test_filtros(tmp_path):
    rows = [
        row(1),
        row(2, family="j11_composite_gen"),
        row(3, family="j11_jaggedness", question="Which deadline comes first?"),
        row(4, family="j11_jaggedness", question="Which coupon gives the lower final price?"),
        row(5, family="j11_jaggedness", question="How many entries in the list are fruits?"),
        row(6, family="j11_negation_pair_gen", question="Should the car be released to the renter?"),
        row(7, family="j11_negation_pair_gen", question="Which statement is NOT correct according to the booking record?"),
        row(8, family="j11_negation_pair_gen", question="Which statement is correct according to the policy summary?"),
        row(9, target_origin="50% decider label + 50% Qwen3.8-27B"),
        row(10, target_distribution=None),
    ]
    other = dict(row(11), part="j11_written")  # outras partes nem viram caso
    cs = {c.id.split("-")[-1]: c for c in db.decisionbank_cases({}, paths=write(tmp_path, rows + [other]))}
    assert "11" not in cs
    assert {k for k, c in cs.items() if c.split == "train"} == {"1", "5", "7"}
    assert cs["2"].meta["descarte"].startswith("família") and cs["9"].meta["descarte"] == "licença ou origem do alvo"
    assert cs["8"].meta["descarte"].startswith("negation_pair") and cs["10"].meta["descarte"] == "alvo sem distribuição"


def test_protege_reservadas_no_treino(monkeypatch):
    """O DecisionBank sai do treino se repetir texto das reservadas v2/professor, além das avaliações de LOADERS."""
    texto = "Um relato reservado com palavras suficientes para formar treze palavras seguidas sem nenhuma pausa aqui"
    def caso(i, src, split, held, state):
        return Case(id=f"{src}:{i}", group_id=f"{src}:{i}", source=src, family="f", split=split, held_out=held,
                    qtype="choice", state=state, question="?", options=[("a", "a"), ("b", "b")], gold="a")
    fontes = {"decisionbank": [caso(0, "decisionbank", "train", False, texto), caso(1, "decisionbank", "train", False, "outro")],
              "res2": [caso(0, "res2", "dev", True, texto)]}
    monkeypatch.setattr(lora, "LOADERS", {})
    monkeypatch.setattr(lora, "PROTEGE_TAMBEM", {"decisionbank": ("res2",)})
    monkeypatch.setattr(lora, "TRAIN_ONLY", {"decisionbank": None})
    monkeypatch.setattr(lora, "load_cases", lambda s, cfg: [Case.from_dict(c.to_dict()) for c in fontes[s]])
    tr = lora.build_training_set(("decisionbank",), cap_per_family=0)
    assert [c.id for c in tr] == ["decisionbank:1"]
    assert set(data.PROTEGE_TAMBEM["decisionbank"]) == {"f3", "res2", "prof", "vestbench_enem"}
