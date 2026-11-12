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
