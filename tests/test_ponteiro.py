"""E4: posições das opções, pointer head no treino (GPT-2 minúsculo em CPU), recarga e critério da H3."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from lumi_decision import lora, metrics, ponteiro  # noqa: E402
from lumi_decision.data import ARTIFACTS_DIR  # noqa: E402
from lumi_decision.prompts import (load_template, option_char_ends, option_token_positions,  # noqa: E402
                                   order_options, render)
from lumi_decision.schema import Case  # noqa: E402

TOKENIZER = ARTIFACTS_DIR / "lumi" / "tokenizer.json"
needs_tok = pytest.mark.skipif(not TOKENIZER.exists(), reason="artefatos do Lumi não baixados")
TEMPLATE = ARTIFACTS_DIR / "lumi" / "chat_template.jinja"


def case(i, n=3, gold_dist=None):
    return Case(id=f"c{i}", group_id=f"g{i}", source="t", family="f", split="train", held_out=False,
                qtype="choice", state="estado " * (i + 1), question="qual?",
                options=[(f"o{k}", f"opção {k}" + (" longa." if k % 2 else "")) for k in range(n)], gold="o1",
                gold_dist=gold_dist)


@pytest.fixture(scope="module")
def tok():
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER), eos_token="<|im_end|>", pad_token="<unk>")


def _offsets(tok):
    def f(t):
        e = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        return e["input_ids"], e["offset_mapping"]
    return f


@needs_tok
def test_positions_point_at_end_of_each_option(tok):
    tpl = load_template(TEMPLATE)
    c = case(3, n=4)
    for order in ("canonical", "reversed", "shuffle5"):
        opts = order_options(c, order)
        prompt = render(tpl, c, opts)
        ids, offs = _offsets(tok)(prompt)
        ends = option_char_ends(prompt, c, opts)
        pos = option_token_positions(offs, prompt, ends)
        assert len(pos) == 4 and pos == sorted(pos) and pos[-1] < len(ids) - 1
        for (_, desc), ch, t in zip(opts, ends, pos):
            assert prompt[ch] == desc.rstrip()[-1]
            s, e = offs[t]
            assert s <= ch < e and not prompt[ch + 1:e].strip()


def test_positions_refuse_token_crossing_into_next_text():
    prompt = "A. sim\nB. não"
    with pytest.raises(ValueError, match="não branco"):
        option_token_positions([(0, 3), (3, 8), (8, 13)], prompt, [5, 12])


def test_new_head_keeps_global_rng(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda s: calls.append(s))
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda s: calls.append(s))
    before = torch.get_rng_state()
    h1 = ponteiro.new_head(16, 8, seed=1)
    assert calls == []  # não resemeia a CUDA (o dropout do treino na GPU fica igual ao do E3)
    assert torch.equal(before, torch.get_rng_state())
    h2 = ponteiro.new_head(16, 8, seed=1)
    assert all(torch.equal(a, b) for a, b in zip(h1.state_dict().values(), h2.state_dict().values()))


def _tiny():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=131072, n_embd=16, n_layer=1, n_head=2, n_positions=2048,
                                      eos_token_id=11)).float()


def _train(tmp, tok, cfg, **kw):
    cases = [case(i % 7, n=2 + i % 3, gold_dist={"o0": .3, "o1": .7} if i % 4 == 0 else None) for i in range(24)]
    for i, c in enumerate(cases):
        c.id = f"c{i}"
    return lora.train(cfg, tmp, model=_tiny(), tok=tok, device="cpu", cases=cases,
                      template=load_template(TEMPLATE), targets=["c_attn"], **kw)


@needs_tok
def test_pointer_training_resume_reload_and_backend(tok, tmp_path):
    from safetensors.torch import load_file

    from lumi_decision.backends.hf import HFBackend

    cfg = lora.TrainConfig(batch_examples=6, token_budget=900, max_micro=4, r=4, alpha=8, dropout=0.1, lr=5e-3,
                           epochs=2, save_every=100, seed=3, readout="ponteiro", head_dim=8)
    full = _train(tmp_path / "direto", tok, cfg)
    assert not full["interrompido"]
    adir = tmp_path / "direto" / "adapter"
    assert (adir / ponteiro.FILE).exists() and (adir / ponteiro.CONFIG).exists()
    part = _train(tmp_path / "parado", tok, cfg, _stop_after=3)
    assert part["interrompido"] and (tmp_path / "parado" / "checkpoint" / "ponteiro.safetensors").exists()
    _train(tmp_path / "parado", tok, cfg)
    for f in ("adapter_model.safetensors", ponteiro.FILE):
        a = load_file(str(adir / f))
        b = load_file(str(tmp_path / "parado" / "adapter" / f))
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)

    # leitura pelo backend (cabeça recarregada) == caminho de treino no mesmo modelo
    from peft import PeftModel
    model = PeftModel.from_pretrained(_tiny(), str(adir)).eval()
    head = ponteiro.load_head(adir, "cpu")
    be = HFBackend(model_obj=model.get_base_model(), tokenizer_obj=tok, device="cpu")
    enc = lora.Encoder(lambda t: tok(t, add_special_tokens=False)["input_ids"], load_template(TEMPLATE),
                       n_checks=2, offsets=_offsets(tok))
    for c in (case(1, 3), case(5, 4)):
        e = enc(c, "reversed", soft=False)
        with torch.no_grad():
            z = lora.option_scores(model, [e], 0, pointer=head)[0]
        got = be.pointer_readout_batch([e.ids], [e.opt_pos], pointer=head)[0][0]
        assert max(abs(a - b) for a, b in zip(torch.log_softmax(z, 0).tolist(), got)) < 1e-5


@needs_tok
def test_letters_path_unchanged_by_pointer_fields(tok, tmp_path):
    """O E3 (letras) não grava cabeça e mantém um único grupo no otimizador."""
    cfg = lora.TrainConfig(batch_examples=6, token_budget=900, max_micro=4, r=4, alpha=8, dropout=0.0, seed=1)
    _train(tmp_path / "x", tok, cfg)
    assert not (tmp_path / "x" / "adapter" / ponteiro.FILE).exists()


def _rows(model, flips):
    """Linhas sintéticas: 2 famílias x 4 grupos x 1 caso, 2 ordens; `flips` = casos que mudam de escolha."""
    rows = []
    for f in ("fa", "fb"):
        for g in range(4):
            cid = f"{f}{g}"
            for o in ("canonical", "reversed"):
                pred = "b" if (cid in flips and o == "reversed") else "a"
                rows.append({"case_id": cid, "order": o, "family": f, "group_id": f"{f}:{g}", "gold": "a",
                             "pred": pred, "presented": ["a", "b"], "prompt_sha1": f"{cid}{o}"})
    return rows


def test_flip_bootstrap_and_h3_rule():
    a = _rows("A", {"fa0", "fa1", "fb0", "fb1"})
    b = _rows("B", set())
    r = metrics.family_mean_flip_bootstrap(a, b, n_boot=200, require_families=["fa", "fb"])
    assert r["delta_media_familias"] == -0.5 and r["taxa_base"] == {"fa": 0.5, "fb": 0.5}
    with pytest.raises(ValueError):
        metrics.family_mean_flip_bootstrap(a, b[:-1])
    v = metrics.h3_verdict([0.003, 0.017], [-0.3, -0.1])
    assert v["qualidade"] == "melhora" and v["direcao"] == "B melhor" and v["equivalencia"].startswith("equivalentes")
    assert v["estabilidade"] == "melhora"
    assert metrics.h3_verdict([-0.015, 0.012], [-0.3, -0.1])["qualidade"] == "empate"
    assert metrics.h3_verdict([-0.03, 0.01], [-0.3, -0.1]) == {**metrics.h3_verdict([-0.03, 0.01], [-0.3, -0.1]),
                                                               "qualidade": "inconclusivo", "estabilidade": "sem melhora demonstrada"}
    assert metrics.h3_verdict([-0.05, -0.01], [0.01, 0.2])["estabilidade"] == "piora"


def test_require_readout_refuses_letters_in_pointer_arm():
    pointer = [{"readout": "ponteiro"}] * 3
    metrics.require_readout(pointer, "ponteiro", "B")
    metrics.require_readout([{}, {"readout": "letras"}], "letras", "A1")  # E3: campo ausente = letras
    with pytest.raises(ValueError, match="B"):
        metrics.require_readout(pointer + [{"readout": "letras"}], "ponteiro", "B")
    with pytest.raises(ValueError):
        metrics.require_readout([{}], "ponteiro", "B")
    with pytest.raises(ValueError):
        metrics.require_readout([], "ponteiro", "B")


@needs_tok
def test_diag_ponteiro_runs_end_to_end(tok, tmp_path, monkeypatch):
    """Smoke do F0a com um GPT-2 minúsculo no lugar do Lumi (o script real roda no cluster)."""
    import argparse
    import json

    from lumi_decision import diag_ponteiro
    from lumi_decision.backends import hf

    real = hf.HFBackend

    def fake(model=None, revision=None, dtype=None, trust_remote_code=None, adapter=None, **kw):
        be = real(model_obj=_tiny().eval(), tokenizer_obj=tok, device="cpu")
        if adapter == "e4":
            be.pointer = ponteiro.new_head(16, 8, seed=1).eval()
        return be

    monkeypatch.setattr(hf, "HFBackend", fake)
    a = argparse.Namespace(model="x", revision="y", e3="e3", e4="e4", source="felhen", family="factck_veracidade",
                           alvo="impreciso", contra="falso", max_orders=6, out=str(tmp_path))
    rep = diag_ponteiro.run(a)
    assert rep["meta"]["n_casos"] == 102 and len(rep["meta"]["ordens"]) == 6
    assert set(rep["resumo"]) == {"A0_congelado", "A1_E3", "E4_pelas_letras", "B_E4_ponteiro"}
    m = rep["cruzamento"]["matriz_q_linha_k_coluna"]
    assert len(m) == 6 and rep["cruzamento"]["ambos_mudam"]["perm012"] == 0.0
    json.dumps(rep)
    assert (tmp_path / "qk.npz").exists()


@needs_tok
def test_residual_starts_at_letters_has_gradient_and_reloads(tok, tmp_path):
    """F2.3a: cabeça residual sobre um adaptador treinado começa idêntica às letras, recebe gradiente em W_k e
    o backend reproduz a leitura do treino depois da recarga."""
    import json

    from lumi_decision.backends.hf import HFBackend

    base = lora.TrainConfig(batch_examples=6, token_budget=900, max_micro=4, r=4, alpha=8, dropout=0.0, lr=5e-3,
                            seed=3)
    _train(tmp_path / "e3", tok, base)
    a3 = tmp_path / "e3" / "adapter"
    cfg = lora.TrainConfig(**{**base.__dict__, "readout": "residual", "head_dim": 8, "init_adapter": str(a3),
                              "epoch_offset": 1})
    res = _train(tmp_path / "res", tok, cfg)
    assert not res["interrompido"]
    meta = json.loads((tmp_path / "res" / "treino_meta.json").read_text(encoding="utf-8"))
    assert meta["residual_inicial_max_dif"] == 0.0
    adir = tmp_path / "res" / "adapter"
    head = ponteiro.load_head(adir, "cpu")
    assert head.residual and float(head.k.weight.abs().max()) > 0  # a cabeça saiu do zero

    # gradiente em W_k já no passo 0 (W_q só depois, porque k começa em zero)
    h0 = ponteiro.new_head(16, 8, seed=1, residual=True)
    z = h0(torch.randn(16), torch.randn(3, 16))
    assert torch.equal(z, torch.zeros(3))
    torch.log_softmax(z + torch.tensor([0.1, 0.2, 0.3]), 0)[0].backward()
    assert float(h0.k.weight.grad.abs().max()) > 0 and float(h0.q.weight.grad.abs().max()) == 0

    from peft import PeftModel
    model = PeftModel.from_pretrained(_tiny(), str(adir)).eval()
    be = HFBackend(model_obj=model.get_base_model(), tokenizer_obj=tok, device="cpu")
    enc = lora.Encoder(lambda t: tok(t, add_special_tokens=False)["input_ids"], load_template(TEMPLATE),
                       n_checks=2, offsets=_offsets(tok))
    e = enc(case(5, 4), "reversed", soft=False)
    with torch.no_grad():
        z = lora.option_scores(model, [e], 0, pointer=head)[0]
    got = be.pointer_readout_batch([e.ids], [e.opt_pos], pointer=head, batch_cands=[e.cands])[0][0]
    assert max(abs(x - y) for x, y in zip(torch.log_softmax(z, 0).tolist(), got)) < 1e-5


@needs_tok
def test_init_adapter_continues_from_trained_lora(tok, tmp_path):
    """O braço de controle (letras) começa exatamente do LoRA do E3: com lr 0, o adaptador final é o inicial."""
    from safetensors.torch import load_file

    base = lora.TrainConfig(batch_examples=6, token_budget=900, max_micro=4, r=4, alpha=8, dropout=0.0, lr=5e-3,
                            seed=3)
    _train(tmp_path / "e3", tok, base)
    a3 = tmp_path / "e3" / "adapter"
    cfg = lora.TrainConfig(**{**base.__dict__, "lr": 0.0, "init_adapter": str(a3), "epoch_offset": 1})
    _train(tmp_path / "ctrl", tok, cfg)
    a = load_file(str(a3 / "adapter_model.safetensors"))
    b = load_file(str(tmp_path / "ctrl" / "adapter" / "adapter_model.safetensors"))
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
