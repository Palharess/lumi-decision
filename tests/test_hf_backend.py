"""Lógica do backend HF num modelo minúsculo aleatório (CPU), com o tokenizer real do Lumi.

Não testa o Nemotron-H (isso é o E0 no cluster); testa readout, padding e geração do nosso código.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from lumi_decision.data import ARTIFACTS_DIR  # noqa: E402

TOKENIZER = ARTIFACTS_DIR / "lumi" / "tokenizer.json"
pytestmark = pytest.mark.skipif(not TOKENIZER.exists(), reason="artefatos do Lumi não baixados")


@pytest.fixture(scope="module")
def backend():
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from lumi_decision.backends.hf import HFBackend

    tok = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER), eos_token="<|im_end|>", pad_token="<unk>")
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=131072, n_embd=32, n_layer=2, n_head=2, n_positions=2048,
                     eos_token_id=11, bos_token_id=1)
    model = GPT2LMHeadModel(cfg).float()
    return HFBackend(model_obj=model, tokenizer_obj=tok, device="cpu", eos_ids=[11])


def test_readout_equals_full_forward(backend):
    ids = backend.tokenize("Contexto: teste.\nResponda apenas com a letra.")
    cands = backend.tokenize("ABCD")[:4] if len(backend.tokenize("ABCD")) >= 4 else [1065, 1066, 1067, 1068]
    lp, _ = backend.readout(ids, cands)
    full = backend.full_forward_last_logprobs(ids)[cands].tolist()
    assert max(abs(a - b) for a, b in zip(lp, full)) < 1e-4


def test_right_padding_matches_single(backend):
    prompts = [backend.tokenize(t) for t in ("um", "um texto um pouco maior", "outro exemplo de prompt aqui")]
    cands = [[1065, 1066, 1067]] * 3
    batched, _ = backend.readout_batch(prompts, cands)
    for p, b in zip(prompts, batched):
        single, _ = backend.readout(p, [1065, 1066, 1067])
        assert max(abs(x - y) for x, y in zip(b, single)) < 1e-4


def test_generate_matches_no_cache(backend):
    ids = backend.tokenize("Pergunta qualquer para gerar")
    new, _, info = backend.generate(ids, max_new_tokens=4)
    assert backend.greedy_no_cache(ids, n_tokens=len(new)) == new
    assert info["stop_type"] in ("eos", "limit")
