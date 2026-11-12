"""E2 — sanidade do treino LoRA no Lumi nativo (rodar no cluster, 1 GPU). Libera e dimensiona o E3.

Verificações OBRIGATÓRIAS (todas precisam passar; senão sai com código 1):
  a. alvos: LoRA em in_proj (Mamba), up/down_proj (MLP), q/k/v/o_proj (atenção), nas quantidades do
     padrão híbrido do config; nenhum em out_proj do Mamba;
  b. passo 0: perda e gradientes finitos; TODO lora_B com gradiente != 0 (lora_A começa com gradiente
     zero porque o PEFT inicia B = 0 — registrado, não cobrado);
  c. após alguns passos: TODO lora_A e lora_B com gradiente finito e != 0, e todos mudaram desde o início;
  d. sobreajuste: em 64 exemplos fixos a perda cai a < 25% da inicial e a acurácia de treino chega a >= 90%;
  e. coerência: readout de treino (modo eval, sem gradiente) == readout do HFBackend no mesmo modelo
     (|Δlogprob| <= 1e-3, argmax igual);
  f. recarga: adaptador salvo e recarregado num modelo base novo dá o mesmo readout (|Δlogprob| <= 1e-3,
     argmax igual).

DIAGNÓSTICOS: modo treino x eval (kernel fundido x caminho de inferência), adaptador fundido em BF16
(|Δ| e argmax), memória de pico e tokens/s por orçamento de tokens do micro-lote (para o E3).

    lumi-e2 --out runs/e2_<job>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from .data import ARTIFACTS_DIR, SOURCES
from .lora import (Encoder, LORA_TARGETS, apply_lora, batch_loss, build_training_set, letter_logprobs,
                   lora_modules)
from .prompts import load_template

REQUIRED = ("a_alvos", "b_passo0", "c_gradientes_apos_passos", "d_sobreajuste", "e_coerencia_readout", "f_recarga")


def verdict(rep: dict) -> bool:
    return all(rep.get(k, {}).get("ok") is True for k in REQUIRED)


def expected_counts(config) -> dict:
    pat = config.hybrid_override_pattern
    n_m, n_mlp, n_att = pat.count("M"), pat.count("-"), pat.count("*")
    return {"in_proj": n_m, "up_proj": n_mlp, "down_proj": n_mlp,
            "q_proj": n_att, "k_proj": n_att, "v_proj": n_att, "o_proj": n_att}


def run(args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .backends.hf import HFBackend
    from .evaluate import git_commit

    torch.manual_seed(0)
    rep = {"meta": {"git": git_commit(), "args": vars(args), "inicio": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}}
    out = Path(args.out)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def load_base():
        m = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16).to("cuda")
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    model = apply_lora(load_base(), r=args.r, alpha=args.alpha, dropout=0.0)
    import peft
    import transformers
    rep["meta"].update(peft=peft.__version__, transformers=transformers.__version__)

    # a. alvos
    mods = lora_modules(model)
    cnt = dict(Counter(mods.values()))
    exp = expected_counts(model.get_base_model().config)
    rep["a_alvos"] = {"contagem": cnt, "esperado": exp, "alvos": LORA_TARGETS,
                      "nomes_exemplo": sorted(mods)[:6],
                      "ok": cnt == exp and "out_proj" not in cnt}

    # dados: 64 exemplos de treino de famílias variadas, ordem sorteada
    cases = build_training_set(cap_per_family=4000, seed=0)
    by = {}
    for c in cases:
        by.setdefault((c.source, c.family), []).append(c)
    pick = []
    while len(pick) < args.n_overfit:
        for k in sorted(by):
            if by[k] and len(pick) < args.n_overfit:
                pick.append(by[k].pop(0))
    enc = Encoder(lambda t: tok(t, add_special_tokens=False)["input_ids"],
                  load_template(ARTIFACTS_DIR / "lumi" / "chat_template.jinja"))
    # alvo duro no sobreajuste: com alvo suave a perda tem piso na entropia da referência
    exs = [enc(c, "shuffle7", soft=False) for c in pick]
    rep["meta"]["familias_sobreajuste"] = dict(Counter(e.family for e in exs))

    lora_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    init = {n: p.detach().clone() for n, p in lora_params.items()}
    head32 = model.get_base_model().get_output_embeddings().weight.detach().float()

    def grads_report():
        bad, zero_a, zero_b = [], [], []
        for n, p in lora_params.items():
            g = p.grad
            if g is None or not torch.isfinite(g).all():
                bad.append(n)
            elif float(g.abs().max()) == 0.0:
                (zero_a if "lora_A" in n else zero_b).append(n)
        return bad, zero_a, zero_b

    # b. passo 0
    model.train()
    mb = exs[: args.micro]
    loss, _, _ = batch_loss(model, mb, pad_id, head32)
    (loss / len(mb)).backward()
    bad, zero_a, zero_b = grads_report()
    n_a = sum("lora_A" in n for n in lora_params)
    rep["b_passo0"] = {"perda": float(loss) / len(mb), "grad_nao_finito": bad[:5], "lora_B_grad_zero": zero_b[:5],
                       "n_lora_B_grad_zero": len(zero_b), "n_lora_A_grad_zero": len(zero_a), "n_lora_A": n_a,
                       "ok": bool(torch.isfinite(loss)) and not bad and not zero_b}
    model.zero_grad(set_to_none=True)

    # c + d. sobreajuste nos 64 exemplos
    opt = torch.optim.AdamW(lora_params.values(), lr=args.lr)
    hist = []
    t0 = time.time()
    for step in range(args.steps):
        tot, hits = 0.0, 0
        for s in range(0, len(exs), args.micro):
            b = exs[s:s + args.micro]
            l, h, _ = batch_loss(model, b, pad_id, head32)
            (l / len(exs)).backward()
            tot, hits = tot + float(l), hits + h
        if step == 2:  # c: gradientes depois de alguns passos (B já saiu de zero)
            bad, zero_a, zero_b = grads_report()
            rep["c_gradientes_apos_passos"] = {"passo": step, "grad_nao_finito": bad[:5],
                                               "n_lora_A_grad_zero": len(zero_a), "n_lora_B_grad_zero": len(zero_b),
                                               "exemplos_zero": (zero_a + zero_b)[:5]}
        torch.nn.utils.clip_grad_norm_(lora_params.values(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        hist.append({"passo": step, "perda": tot / len(exs), "acc": hits / len(exs)})
        if step % 10 == 0:
            print(json.dumps(hist[-1]), flush=True)
    unchanged = [n for n, p in lora_params.items() if torch.equal(p.detach(), init[n])]
    c = rep["c_gradientes_apos_passos"]
    c["parametros_que_nao_mudaram"] = unchanged[:5]
    c["n_nao_mudaram"] = len(unchanged)
    c["ok"] = not c["grad_nao_finito"] and c["n_lora_A_grad_zero"] == 0 and c["n_lora_B_grad_zero"] == 0 and not unchanged

    model.eval()
    with torch.no_grad():
        tot, hits = 0.0, 0
        for s in range(0, len(exs), args.micro):
            l, h, _ = batch_loss(model, exs[s:s + args.micro], pad_id, head32)
            tot, hits = tot + float(l), hits + h
    final = {"perda": tot / len(exs), "acc": hits / len(exs)}
    rep["d_sobreajuste"] = {"inicial": hist[0], "final_eval": final, "segundos": time.time() - t0,
                            "curva": hist[:: max(1, len(hist) // 10)],
                            "ok": final["perda"] < 0.25 * hist[0]["perda"] and final["acc"] >= 0.9}

    # e. readout de treino (eval) x HFBackend sobre o MESMO modelo com LoRA
    probe = exs[: args.n_probe]
    be = HFBackend(model_obj=model.get_base_model(), tokenizer_obj=tok, device="cuda")
    with torch.no_grad():
        train_lp = [x.tolist() for e in probe for x in letter_logprobs(model, [e], pad_id, head32)]
    be_lp = [be.readout(e.ids, e.cands)[0] for e in probe]

    def cmp(xs, ys):
        d = max(max(abs(a - b) for a, b in zip(x, y)) for x, y in zip(xs, ys))
        am = sum(max(range(len(x)), key=x.__getitem__) == max(range(len(y)), key=y.__getitem__) for x, y in zip(xs, ys))
        return d, am

    d, am = cmp(train_lp, be_lp)
    rep["e_coerencia_readout"] = {"max_abs_dlogprob": d, "argmax_igual": f"{am}/{len(probe)}",
                                  "ok": d <= args.tol and am == len(probe)}
    # diagnóstico: modo treino (kernel fundido) x eval, sem dropout (dropout=0 neste modelo)
    model.train()
    with torch.no_grad():
        tr_lp = [x.tolist() for e in probe for x in letter_logprobs(model, [e], pad_id, head32)]
    model.eval()
    d2, am2 = cmp(tr_lp, train_lp)
    rep["diag_modo_treino_vs_eval"] = {"max_abs_dlogprob": d2, "argmax_igual": f"{am2}/{len(probe)}"}

    # f. recarga
    adir = out / "adapter_e2"
    model.save_pretrained(str(adir))
    del be
    reloaded = HFBackend(model=args.model, revision=args.revision, dtype="bfloat16", trust_remote_code=False,
                         adapter=str(adir))
    re_lp = [reloaded.readout(e.ids, e.cands)[0] for e in probe]
    d3, am3 = cmp(re_lp, be_lp)
    rep["f_recarga"] = {"max_abs_dlogprob": d3, "argmax_igual": f"{am3}/{len(probe)}",
                        "ok": d3 <= args.tol and am3 == len(probe)}

    # diagnóstico: adaptador fundido nos pesos BF16
    try:
        from peft import PeftModel
        merged = PeftModel.from_pretrained(load_base(), str(adir)).merge_and_unload().eval()
        mb_ = HFBackend(model_obj=merged, tokenizer_obj=tok, device="cuda")
        mg_lp = [mb_.readout(e.ids, e.cands)[0] for e in probe]
        d4, am4 = cmp(mg_lp, be_lp)
        rep["diag_fundido_bf16"] = {"max_abs_dlogprob": d4, "argmax_igual": f"{am4}/{len(probe)}"}
        del merged, mb_
    except Exception as e:  # noqa: BLE001
        rep["diag_fundido_bf16"] = {"erro": f"{type(e).__name__}: {e}"}
    del reloaded
    torch.cuda.empty_cache()

    # diagnóstico: memória e velocidade por orçamento de tokens (exemplos mais longos do treino)
    longest = sorted(cases, key=lambda c: len(c.state), reverse=True)[:64]
    lex = [enc(c, "shuffle7", soft=True) for c in longest]
    model.train()
    mem = {}
    for budget in args.budgets:
        L = max(len(e.ids) for e in lex)
        n = max(1, budget // L)
        b = lex[:n]
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t1 = time.time()
            for _ in range(3):
                l, _, _ = batch_loss(model, b, pad_id, head32)
                l.backward()
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = (time.time() - t1) / 3
            mem[str(budget)] = {"exemplos": n, "maior_comprimento": L, "pico_gb": torch.cuda.max_memory_allocated() / 1e9,
                                "s_por_microlote": dt, "tokens_s": sum(len(e.ids) for e in b) / dt}
        except torch.cuda.OutOfMemoryError:
            mem[str(budget)] = {"exemplos": n, "maior_comprimento": L, "erro": "OOM"}
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    rep["diag_memoria"] = mem
    rep["aprovado"] = verdict(rep)
    return rep


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="lumi-e2")
    p.add_argument("--model", default=SOURCES["lumi"]["repo"])
    p.add_argument("--revision", default=SOURCES["lumi"]["revision"])
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--n-overfit", type=int, default=64)
    p.add_argument("--micro", type=int, default=8)
    p.add_argument("--n-probe", type=int, default=16)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--budgets", type=lambda s: [int(x) for x in s.split(",")], default=[4096, 8192, 16384, 32768])
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rep = run(a)
    (Path(a.out) / "e2.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    print(f"E2 {'APROVADO' if rep['aprovado'] else 'REPROVADO'}", flush=True)
    sys.exit(0 if rep["aprovado"] else 1)


if __name__ == "__main__":
    main()
