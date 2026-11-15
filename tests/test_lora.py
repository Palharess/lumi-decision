"""Núcleo do LoRA (dados, empacotamento, readout diferenciável) num GPT-2 minúsculo em CPU.

Não testa o Nemotron-H (isso é o E2 no cluster)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from lumi_decision import lora  # noqa: E402
from lumi_decision.data import ARTIFACTS_DIR  # noqa: E402
from lumi_decision.prompts import load_template  # noqa: E402
from lumi_decision.schema import Case  # noqa: E402

TOKENIZER = ARTIFACTS_DIR / "lumi" / "tokenizer.json"
pytestmark = pytest.mark.skipif(not TOKENIZER.exists(), reason="artefatos do Lumi não baixados")


def case(i, n=3, gold_dist=None):
    return Case(id=f"c{i}", group_id=f"g{i}", source="t", family="f", split="train", held_out=False,
                qtype="choice", state="estado " * (i + 1), question="qual?",
                options=[(f"o{k}", f"opção {k}") for k in range(n)], gold="o1", gold_dist=gold_dist)


@pytest.fixture(scope="module")
def setup():
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER), eos_token="<|im_end|>", pad_token="<unk>")
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=131072, n_embd=32, n_layer=2, n_head=2, n_positions=2048, eos_token_id=11)
    model = GPT2LMHeadModel(cfg).float()
    for p in model.parameters():
        p.requires_grad_(False)
    model = lora.apply_lora(model, r=4, alpha=8, dropout=0.0, targets=["c_attn"])
    enc = lora.Encoder(lambda t: tok(t, add_special_tokens=False)["input_ids"],
                       load_template(ARTIFACTS_DIR / "lumi" / "chat_template.jinja"), n_checks=2)
    return tok, model, enc


def test_target_vector_soft_and_hard():
    c = case(0, gold_dist={"o0": 0.2, "o1": 0.6, "o2": 0.2})
    opts = c.options[::-1]
    assert lora.target_vector(c, opts, soft=True) == pytest.approx([0.2, 0.6, 0.2])
    assert lora.target_vector(c, opts, soft=False) == [0.0, 1.0, 0.0]
    assert lora.target_vector(case(1), c.options, soft=True) == [0.0, 1.0, 0.0]


def test_plan_covers_each_example_once_within_budget(setup):
    _, _, enc = setup
    exs = [enc(case(i % 9), "canonical", soft=True) for i in range(50)]
    plan = lora.plan_microbatches(exs, token_budget=600, max_micro=8, seed=1, pool=16)
    flat = sorted(i for b in plan for i in b)
    assert flat == list(range(50))
    for b in plan:
        assert len(b) <= 8
        assert len(b) == 1 or len(b) * max(len(exs[i].ids) for i in b) <= 600
    assert plan == lora.plan_microbatches(exs, token_budget=600, max_micro=8, seed=1, pool=16)


def test_encoder_gold_position_follows_order(setup):
    _, _, enc = setup
    c = case(2)
    assert enc(c, "canonical", soft=False).gold_pos == 1
    e = enc(c, "reversed", soft=False)
    assert e.gold_pos == 1 and e.target == [0.0, 1.0, 0.0]
    assert len(set(e.cands)) == 3


def test_train_readout_matches_backend_and_gets_gradients(setup):
    from lumi_decision.backends.hf import HFBackend

    tok, model, enc = setup
    exs = [enc(case(i), "canonical", soft=True) for i in range(3)]
    model.eval()
    with torch.no_grad():
        lps = lora.letter_logprobs(model, exs, pad_id=0)
    be = HFBackend(model_obj=model.get_base_model(), tokenizer_obj=tok, device="cpu", eos_ids=[11])
    for e, lp in zip(exs, lps):
        ref, _ = be.readout(e.ids, e.cands)
        assert max(abs(a - b) for a, b in zip(lp.tolist(), ref)) < 1e-4
    model.train()
    loss, hits, mass = lora.batch_loss(model, exs, pad_id=0)
    loss.backward()
    b_grads = [p.grad for n, p in model.named_parameters() if "lora_B" in n]
    assert b_grads and all(g is not None and float(g.abs().max()) > 0 for g in b_grads)
    assert 0 <= hits <= 3 and 0 < mass <= 1
    model.zero_grad(set_to_none=True)


def test_lora_modules_lists_targets(setup):
    _, model, _ = setup
    mods = lora.lora_modules(model)
    assert mods and set(mods.values()) == {"c_attn"}


def test_training_set_caps_and_excludes_held_out():
    tr = lora.build_training_set(("felhen",), cap_per_family=50, seed=0)
    from collections import Counter
    cnt = Counter(c.family for c in tr)
    assert max(cnt.values()) <= 50
    assert not ({"juristcu_area", "factck_veracidade"} & set(cnt))
    assert all(c.split == "train" and not c.held_out for c in tr)
    assert [c.id for c in tr] == [c.id for c in lora.build_training_set(("felhen",), cap_per_family=50, seed=0)]


def test_adapter_save_reload_through_backend(setup, tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel

    from lumi_decision.backends.hf import HFBackend

    tok, model, enc = setup
    with torch.no_grad():  # sai do zero para o adaptador ter efeito
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.add_(0.05)
    model.eval()
    model.save_pretrained(str(tmp_path / "ad"))
    e = enc(case(4), "canonical", soft=True)
    with torch.no_grad():
        ref = lora.letter_logprobs(model, [e], pad_id=0)[0].tolist()
    torch.manual_seed(0)
    fresh = GPT2LMHeadModel(GPT2Config(vocab_size=131072, n_embd=32, n_layer=2, n_head=2, n_positions=2048,
                                       eos_token_id=11)).float()
    be = HFBackend(model_obj=fresh, tokenizer_obj=tok, device="cpu", eos_ids=[11], adapter=str(tmp_path / "ad"))
    got, _ = be.readout(e.ids, e.cands)
    assert max(abs(a - b) for a, b in zip(got, ref)) < 1e-5
    assert be.describe()["adapter_sha256"]


def test_expected_counts_match_lumi_pattern():
    import json
    from types import SimpleNamespace

    from lumi_decision.e2_lora import expected_counts
    cfg = json.loads((ARTIFACTS_DIR / "lumi" / "config.json").read_text(encoding="utf-8"))
    c = expected_counts(SimpleNamespace(**cfg))
    assert c["in_proj"] == 21 and c["up_proj"] == 17 and c["q_proj"] == 4


def test_group_steps_uses_real_plan():
    plan = [[0, 1, 2], [3], [4, 5, 6, 7], [8], [9, 10]]
    assert lora.group_steps(plan, 4) == [[0, 1], [2], [3, 4]]
    assert sum(len(plan[m]) for g in lora.group_steps(plan, 4) for m in g) == 11


def test_loss_is_normalized_over_options(setup):
    _, model, enc = setup
    e = enc(case(3), "canonical", soft=False)
    model.eval()
    with torch.no_grad():
        loss, _, mass = lora.batch_loss(model, [e], pad_id=0)
        lp = lora.letter_logprobs(model, [e], pad_id=0)[0]
    expected = -(lp - torch.logsumexp(lp, 0))[e.gold_pos]
    assert float(loss) == pytest.approx(float(expected), rel=1e-5)
    assert mass < 0.5  # modelo aleatório põe pouca massa nas letras; a perda não depende disso


def _tiny(seed=0):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(vocab_size=131072, n_embd=16, n_layer=1, n_head=2, n_positions=2048,
                                      eos_token_id=11)).float()


def _train(tmp, setup, cfg, targets=("c_attn",), **kw):
    tok, _, _ = setup
    cases = [case(i % 7, n=2 + i % 3, gold_dist={"o0": .3, "o1": .7} if i % 4 == 0 else None) for i in range(24)]
    for i, c in enumerate(cases):
        c.id = f"c{i}"
    return lora.train(cfg, tmp, model=_tiny(), tok=tok, device="cpu", cases=cases,
                      template=load_template(ARTIFACTS_DIR / "lumi" / "chat_template.jinja"), targets=list(targets), **kw)


def test_interrupt_and_resume_equals_uninterrupted(setup, tmp_path):
    from safetensors.torch import load_file

    cfg = lora.TrainConfig(batch_examples=6, token_budget=900, max_micro=4, r=4, alpha=8, dropout=0.1,
                           lr=5e-3, epochs=2, save_every=100, seed=3)
    full = _train(tmp_path / "direto", setup, cfg)
    assert not full["interrompido"] and full["passo"] == full["total"] >= 6
    part = _train(tmp_path / "parado", setup, cfg, _stop_after=3)
    assert part["interrompido"] and part["passo"] == 3
    assert (tmp_path / "parado" / "checkpoint" / "estado.json").exists()
    assert not (tmp_path / "parado" / "adapter").exists()
    rest = _train(tmp_path / "parado", setup, cfg)
    assert not rest["interrompido"] and rest["passo"] == full["passo"]
    a = load_file(str(tmp_path / "direto" / "adapter" / "adapter_model.safetensors"))
    b = load_file(str(tmp_path / "parado" / "adapter" / "adapter_model.safetensors"))
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_stop_file_interrupts_and_is_consumed(setup, tmp_path):
    cfg = lora.TrainConfig(batch_examples=6, token_budget=900, max_micro=4, r=4, alpha=8, dropout=0.0, seed=2)
    out = tmp_path / "arq"
    out.mkdir()
    (out / "PARAR").touch()
    part = _train(out, setup, cfg)
    assert part["interrompido"] and part["passo"] == 1
    assert not (out / "PARAR").exists() and (out / "checkpoint" / "estado.json").exists()
    rest = _train(out, setup, cfg)
    assert not rest["interrompido"] and rest["passo"] == rest["total"]


def test_resume_refuses_other_config(setup, tmp_path):
    cfg = lora.TrainConfig(batch_examples=6, token_budget=900, max_micro=4, r=4, alpha=8, dropout=0.0, seed=1)
    _train(tmp_path / "x", setup, cfg, _stop_after=1)
    with pytest.raises(SystemExit, match="outra configuração"):
        _train(tmp_path / "x", setup, lora.TrainConfig(**{**cfg.__dict__, "lr": 9e-4}))


def test_data_seed_freezes_selection_independently_of_seed():
    """F1: com data_seed fixa, trocar a semente do treino não troca os exemplos."""
    a = lora.build_training_set(seed=0)
    cfg = lora.TrainConfig(seed=1, data_seed=0)
    b = lora.build_training_set(cfg.sources, cfg.cap_per_family, cfg.seed if cfg.data_seed < 0 else cfg.data_seed)
    assert [c.id for c in a] == [c.id for c in b]
    assert lora.TrainConfig().data_seed == -1


def test_frozen_targets_keep_init_and_base_output(setup, tmp_path):
    """F2.2: o alvo congelado nasce como no braço completo (mesmo RNG) e não treina; B = 0 deixa o módulo igual ao base."""
    import json

    from safetensors.torch import load_file

    T = ("c_attn", "c_proj")
    kw = dict(batch_examples=6, token_budget=900, max_micro=4, r=4, alpha=8, dropout=0.1, seed=4)
    ld = lambda d: load_file(str(d / "adapter" / "adapter_model.safetensors"))  # noqa: E731
    # lr 0: o adaptador final é a inicialização; congelar não pode mudar nenhum peso inicial
    _train(tmp_path / "c0", setup, lora.TrainConfig(lr=0.0, **kw), targets=T)
    _train(tmp_path / "f0", setup, lora.TrainConfig(lr=0.0, frozen_targets=["c_proj"], **kw), targets=T)
    a, b = ld(tmp_path / "c0"), ld(tmp_path / "f0")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    # lr > 0: c_proj fica na inicialização (B = 0), c_attn treina
    _train(tmp_path / "f1", setup, lora.TrainConfig(lr=5e-3, frozen_targets=["c_proj"], **kw), targets=T)
    c = ld(tmp_path / "f1")
    proj = [k for k in c if ".c_proj." in k]
    attn = [k for k in c if ".c_attn." in k and "lora_B" in k]
    assert proj and all(torch.equal(c[k], a[k]) for k in proj)
    assert all(float(c[k].abs().max()) == 0.0 for k in proj if "lora_B" in k)
    assert any(not torch.equal(c[k], a[k]) for k in attn)
    meta = json.loads((tmp_path / "f1" / "treino_meta.json").read_text(encoding="utf-8"))
    assert meta["modulos_congelados"] == {"c_proj": 2} and meta["congelados_max_abs_B"] == 0.0
    with pytest.raises(RuntimeError, match="nenhum módulo"):
        _train(tmp_path / "x", setup, lora.TrainConfig(frozen_targets=["in_proj"], **kw), targets=T)


@pytest.mark.skipif(not (ARTIFACTS_DIR.parent / "data" / "f3" / "raw" / "massive" / "pt-PT.jsonl").exists(),
                    reason="brutos do F3 não baixados")
def test_source_caps_pilot_composition():
    """F3: o braço diverso tem o mesmo nº de exemplos únicos do F2.1, e o Felhen dele é um prefixo do F2.1."""
    from collections import Counter

    ctrl = lora.training_set_for(lora.TrainConfig(cap_per_family=1300, data_seed=0))
    div = lora.training_set_for(lora.TrainConfig(sources=["felhen", "telepatia", "traces", "f3"], cap_per_family=1300,
                                                 data_seed=0, source_caps=["felhen=650", "f3=1625"]))
    assert len(ctrl) == len(div) == 10516
    src = Counter(c.source for c in div)
    assert src["felhen"] == 3250 and src["f3"] == 3250 and src["telepatia"] == 3740 and src["traces"] == 276
    assert {c.family for c in div if c.source == "f3"} == {"massive_intencao", "goemo_emocao"}
    assert {c.id for c in div if c.source == "felhen"} <= {c.id for c in ctrl if c.source == "felhen"}
    assert lora.parse_source_caps(["felhen=650", "f3=1625"]) == {"felhen": 650, "f3": 1625}
