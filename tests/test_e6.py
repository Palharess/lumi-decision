"""E6 (analises/e6_protocolo.md): formatos de decisão com prefixo, âncoras KL e adaptadores intermediários, num
GPT-2 minúsculo em CPU com o tokenizador do Lumi."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from lumi_decision import lora, prompts  # noqa: E402
from lumi_decision.data import ARTIFACTS_DIR  # noqa: E402
from lumi_decision.prompts import load_template  # noqa: E402
from lumi_decision.schema import Case  # noqa: E402

TOKENIZER = ARTIFACTS_DIR / "lumi" / "tokenizer.json"
pytestmark = pytest.mark.skipif(not TOKENIZER.exists(), reason="artefatos do Lumi não baixados")
TEMPLATE = ARTIFACTS_DIR / "lumi" / "chat_template.jinja"
# textos das instruções do Lumi-Benchmark (lumi_bench/prompts.py, v1): nenhum formato de treino pode copiá-los
TEXTOS_DO_BENCHMARK = ("Pode raciocinar antes", "Responda somente no formato 'Resposta: X', sem explicacao",
                       "Responda a questao de multipla escolha abaixo")


def case(i, n=3, gold_dist=None):
    return Case(id=f"c{i}", group_id=f"g{i}", source="t", family="f", split="train", held_out=False,
                qtype="choice", state="estado " * (i + 1), question="qual?",
                options=[(f"o{k}", f"opção {k}") for k in range(n)], gold="o1", gold_dist=gold_dist)


@pytest.fixture(scope="module")
def tok():
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER), eos_token="<|im_end|>", pad_token="<unk>")


def _enc(tok, n_checks=2):
    return lora.Encoder(lambda t: tok(t, add_special_tokens=False)["input_ids"], load_template(TEMPLATE),
                        n_checks=n_checks)


def _modelo(seed=0, sem_dropout=True):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    kw = dict(resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0) if sem_dropout else {}
    m = GPT2LMHeadModel(GPT2Config(vocab_size=131072, n_embd=32, n_layer=2, n_head=2, n_positions=2048,
                                   eos_token_id=11, **kw)).float()
    for p in m.parameters():
        p.requires_grad_(False)
    return lora.apply_lora(m, r=4, alpha=8, dropout=0.0, targets=["c_attn"])


def test_v1_identico_e_formatos_sem_texto_do_benchmark():
    tpl = load_template(TEMPLATE)
    c = case(2)
    assert prompts.user_message_formato(c, c.options, "v1") == prompts.user_message(c, c.options)
    assert prompts.render_formato(tpl, c, c.options, "v1") == (prompts.render(tpl, c, c.options), "")
    for nome, f in {**prompts.FORMATOS, "cot": prompts.FORMATO_COT}.items():
        msg = prompts.user_message_formato(c, c.options, f if nome == "cot" else nome)
        assert not any(t in msg for t in TEXTOS_DO_BENCHMARK), nome
        assert c.state in msg and "opção 2" in msg
    assert {f["prefixo"] for k, f in prompts.FORMATOS.items() if k != "v1"} == {"Resposta:"}


def test_leitura_depois_do_prefixo_e_cache_por_formato(tok):
    enc = _enc(tok, n_checks=1)
    c = case(3)
    v1 = enc(c, "canonical", soft=False)
    assert v1.formato == "v1" and v1.pos_inicio == len(v1.ids) - 1
    tpl = load_template(TEMPLATE)
    for fmt in ("p1", "p2", "p3"):
        e = enc(c, "canonical", soft=False, formato=fmt)
        prompt, prefixo = prompts.render_formato(tpl, c, c.options, fmt)
        base = tok(prompt, add_special_tokens=False)["input_ids"]
        assert e.pos_inicio == len(base) - 1 < len(e.ids) - 1 and e.ids[:len(base)] == base
        for k, letra in enumerate("ABC"):  # a letra lida é exatamente o próximo token da geração
            full = tok(prompt + prefixo + " " + letra, add_special_tokens=False)["input_ids"]
            assert full == e.ids + [e.cands[k]]
        assert e.cands != v1.cands  # " A" depois do prefixo não é o mesmo token que "A" no início da resposta
        # depois das checagens, o cache de (formato, n) serve e dá o mesmo resultado
        assert enc(c, "canonical", soft=False, formato=fmt).cands == e.cands
    assert enc(case(4), "canonical", soft=False).cands == v1.cands
    cot = enc.cot_ids(c, "canonical")
    assert cot[-1] == v1.ids[-1] and cot != v1.ids  # termina no mesmo início de resposta, com outra instrução


def test_kl_topk_com_resto_e_bernoulli():
    g = torch.Generator().manual_seed(0)
    lp0 = torch.log_softmax(torch.randn(500, generator=g), 0)
    lp = torch.log_softmax(torch.randn(500, generator=g), 0)
    exata = float((lp0.exp() * (lp0 - lp)).sum())
    assert float(lora._kl_topk(lp0, lp, 500)) == pytest.approx(exata, rel=1e-5)
    aprox = float(lora._kl_topk(lp0, lp, 64))
    assert 0 <= aprox <= exata + 1e-6  # agrupar o resto num balde nunca aumenta a KL
    assert float(lora._kl_topk(lp0, lp0, 64)) == pytest.approx(0.0, abs=1e-6)
    assert float(lora._kl_bern(torch.tensor(0.3), torch.tensor(0.3))) == pytest.approx(0.0, abs=1e-7)
    assert float(lora._kl_bern(torch.tensor(0.9), torch.tensor(0.5))) > 0
    fora = lora._fora(lp0, [0, 1, 2])
    assert torch.isinf(fora[:3]).all() and float(fora[3:].exp().sum()) == pytest.approx(1.0, rel=1e-5)


def test_ancora_zero_com_adaptador_zerado_e_gradiente_so_no_lora(tok):
    enc = _enc(tok)
    model = _modelo()
    model.train()
    exs = []
    for i, fmt in enumerate(["v1", "p1", "p2", "p3"]):
        e = enc(case(i, n=2 + i % 3), "canonical", soft=True, formato=fmt)
        e.cot_ids = enc.cot_ids(case(i, n=2 + i % 3), "canonical")
        exs.append(e)
    dec, anc, hits, mass = lora.batch_loss_ancorado(model, exs, pad_id=0)
    ref_dec, ref_hits, ref_mass = lora.batch_loss(model, exs, pad_id=0)
    assert float(dec) == pytest.approx(float(ref_dec), rel=1e-5) and hits == ref_hits
    assert mass == pytest.approx(ref_mass, rel=1e-5)
    # B do LoRA nasce zerado: o modelo com adaptador é o Lumi (aqui, o GPT-2) e toda KL é zero
    assert all(abs(float(v)) < 1e-5 for v in anc.values())
    assert abs(float(lora.batch_cot(model, exs, pad_id=0))) < 1e-5
    # com o LoRA perturbado, as âncoras ficam positivas e o gradiente cai só nos parâmetros do LoRA
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.5)
    for p in model.parameters():
        p.grad = None
    dec, anc, _, _ = lora.batch_loss_ancorado(model, exs, pad_id=0)
    total = anc["mass"] + anc["out"] + anc["pos"] + lora.batch_cot(model, exs, pad_id=0)
    assert all(float(v) > 0 for v in anc.values())
    total.backward()
    com_grad = {n for n, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().max()) > 0}
    assert com_grad and all("lora_" in n for n in com_grad)


def _treinar(tmp, tok, cfg):
    cases = [case(i % 7, n=2 + i % 3, gold_dist={"o0": .3, "o1": .7} if i % 4 == 0 else None) for i in range(24)]
    for i, c in enumerate(cases):
        c.id = f"c{i}"
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(vocab_size=131072, n_embd=16, n_layer=1, n_head=2, n_positions=2048,
                                   eos_token_id=11, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)).float()
    return lora.train(cfg, tmp, model=m, tok=tok, device="cpu", cases=cases, template=load_template(TEMPLATE),
                      targets=["c_attn"])


def test_treino_e6_mantem_plano_do_v1_e_guarda_intermediarios(tok, tmp_path):
    kw = dict(batch_examples=6, token_budget=900, max_micro=4, r=4, alpha=8, dropout=0.0, lr=5e-3, seed=3)
    r0 = _treinar(tmp_path / "r0", tok, lora.TrainConfig(**kw))
    r2 = _treinar(tmp_path / "r2", tok, lora.TrainConfig(formatos=["v1", "p1", "p2", "p3"], ancora_lambda=1.0,
                                                          ancora_cot=True, snapshots=["50"], **kw))
    assert r0["passo"] == r2["passo"] == r0["total"] == r2["total"]
    m0 = json.loads((tmp_path / "r0" / "treino_meta.json").read_text(encoding="utf-8"))
    m2 = json.loads((tmp_path / "r2" / "treino_meta.json").read_text(encoding="utf-8"))
    assert m0["dados_sha256"] == m2["dados_sha256"]  # mesmos casos e mesmo plano de lotes do R0
    assert "e6" not in m0 and sum(m2["e6"]["exposicoes_por_formato"].values()) == 24
    assert set(m2["e6"]["exposicoes_por_formato"]) <= {"v1", "p1", "p2", "p3"} and m2["e6"]["tokens_cot"] > 0
    assert (tmp_path / "r2" / "adapter_p50" / "adapter_model.safetensors").exists()
    assert (tmp_path / "r2" / "adapter" / "adapter_model.safetensors").exists()
    log = [json.loads(x) for x in (tmp_path / "r2" / "treino.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {"ancora_mass", "ancora_out", "ancora_pos", "ancora_cot"} <= set(log[-1])
    with pytest.raises(ValueError, match="ancora_cot"):
        _treinar(tmp_path / "x", tok, lora.TrainConfig(ancora_cot=True, **kw))
    with pytest.raises(ValueError, match="formato desconhecido"):
        _treinar(tmp_path / "y", tok, lora.TrainConfig(formatos=["p9"], **kw))
