"""E4 — sanidade da pointer head no Lumi nativo (rodar no cluster, 1 GPU), ANTES do treino completo.

Verificações OBRIGATÓRIAS (todas precisam passar; senão sai com código 1):
  a. posições: em 200 casos de cada fonte (treino, dev e calib), nas ordens canônica, invertida e sorteada,
     o token escolhido contém o último caractere de cada opção e só avança sobre espaço/quebra de linha; a
     tokenização com offsets é idêntica à do readout;
  b. passo 0: perda e gradientes finitos; cabeça (q e k) e todo lora_B com gradiente != 0;
  c. após alguns passos: todo lora_A, lora_B e a cabeça com gradiente finito e != 0, e todos mudaram;
  d. sobreajuste: em 64 exemplos fixos (alvo duro) a perda cai a < 25% da inicial e a acurácia chega a >= 90%;
  e. coerência: leitura do caminho de treino (modo eval) == `HFBackend.pointer_readout_batch` no mesmo modelo
     (|Δlogprob| <= 1e-3, argmax igual);
  f. recarga: adaptador + cabeça salvos e recarregados num modelo base novo dão a mesma leitura;
  g. kernel fundido: no forward de treino da pointer head, o kernel fundido do Mamba roda em todas as camadas;
     em modo eval, em nenhuma;
  h. RNG: criar a cabeça não muda o estado dos geradores da CPU nem da CUDA (LoRA e dropout iguais ao E3).

DIAGNÓSTICOS: fração de opções cujo token também cobre a quebra de linha; modo treino x eval; memória e
tokens/s com 8192 tokens por micro-lote.

    python -m lumi_decision.e4_ponteiro --out runs/e4_sanidade_<job>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from .data import ARTIFACTS_DIR, LOADERS, SOURCES, load_config
from .e2_lora import expected_counts
from .lora import Encoder, apply_lora, batch_loss, build_training_set, option_scores
from .prompts import load_template, option_char_ends, option_token_positions, order_options, render

REQUIRED = ("a_posicoes", "b_passo0", "c_gradientes_apos_passos", "d_sobreajuste", "e_coerencia_readout",
            "f_recarga", "g_kernel_fundido", "h_rng")


def verdict(rep: dict) -> bool:
    return all(rep.get(k, {}).get("ok") is True for k in REQUIRED)


def check_positions(tok, template, n_per_source: int = 200, seed: int = 0) -> dict:
    """Confere as posições das opções em casos reais das três fontes (todos os splits ativos)."""
    cfg = load_config()
    rng = random.Random(seed)
    stats = Counter()
    erros = []
    for src in LOADERS:
        cases = [c for c in LOADERS[src](cfg) if c.split in ("train", "dev", "calib")]
        for c in rng.sample(cases, min(n_per_source, len(cases))):
            for order in ("canonical", "reversed", "shuffle0"):
                opts = order_options(c, order)
                prompt = render(template, c, opts)
                e = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
                stats["prompts"] += 1
                try:
                    if list(e["input_ids"]) != list(tok(prompt, add_special_tokens=False)["input_ids"]):
                        raise ValueError("offsets mudam a tokenização")
                    ends = option_char_ends(prompt, c, opts)
                    pos = option_token_positions(e["offset_mapping"], prompt, ends)
                    for ch, t in zip(ends, pos):
                        s, en = e["offset_mapping"][t]
                        stats["opcoes"] += 1
                        stats["token_cobre_quebra"] += "\n" in prompt[ch + 1:en]
                except ValueError as err:
                    stats["falhas"] += 1
                    if len(erros) < 5:
                        erros.append(f"{c.id} {order}: {err}")
    return {**stats, "fracao_token_cobre_quebra": stats["token_cobre_quebra"] / max(1, stats["opcoes"]),
            "exemplos_de_erro": erros, "ok": stats["falhas"] == 0 and stats["opcoes"] > 0}


def run(args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .backends.hf import HFBackend
    from .evaluate import git_commit
    from .ponteiro import new_head, save_head

    torch.manual_seed(0)
    rep = {"meta": {"git": git_commit(), "args": vars(args), "inicio": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}}
    out = Path(args.out)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    template = load_template(ARTIFACTS_DIR / "lumi" / "chat_template.jinja")

    rep["a_posicoes"] = check_positions(tok, template, args.n_pos)
    print(json.dumps({"a_posicoes": rep["a_posicoes"]}, ensure_ascii=False), flush=True)

    def load_base():
        m = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16).to("cuda")
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    model = apply_lora(load_base(), r=args.r, alpha=args.alpha, dropout=0.0)
    d = model.get_base_model().get_output_embeddings().weight.shape[1]
    before, before_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    head = new_head(d, args.head_dim, seed=1).to("cuda")
    cpu_ok = bool(torch.equal(before, torch.get_rng_state()))
    after_cuda = torch.cuda.get_rng_state_all()
    cuda_ok = len(before_cuda) == len(after_cuda) and all(torch.equal(x, y) for x, y in zip(before_cuda, after_cuda))
    rep["h_rng"] = {"cpu_intacto": cpu_ok, "cuda_intacto": cuda_ok, "ok": cpu_ok and cuda_ok}

    def _offsets(t):
        e = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        return e["input_ids"], e["offset_mapping"]

    cases = build_training_set(cap_per_family=4000, seed=0)
    by = {}
    for c in cases:
        by.setdefault((c.source, c.family), []).append(c)
    pick = []
    while len(pick) < args.n_overfit:
        for k in sorted(by):
            if by[k] and len(pick) < args.n_overfit:
                pick.append(by[k].pop(0))
    enc = Encoder(lambda t: tok(t, add_special_tokens=False)["input_ids"], template, offsets=_offsets)
    exs = [enc(c, "shuffle7", soft=False) for c in pick]

    lora_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    head_params = {f"cabeca.{n}": p for n, p in head.named_parameters()}
    allp = {**lora_params, **head_params}
    init = {n: p.detach().clone() for n, p in allp.items()}

    def grads_report():
        bad, zero_a, zero_b, zero_h = [], [], [], []
        for n, p in allp.items():
            g = p.grad
            if g is None or not torch.isfinite(g).all():
                bad.append(n)
            elif float(g.abs().max()) == 0.0:
                (zero_h if n.startswith("cabeca.") else zero_a if "lora_A" in n else zero_b).append(n)
        return bad, zero_a, zero_b, zero_h

    # g. kernel fundido no forward de treino da pointer head
    try:
        import transformers.models.nemotron_h.modeling_nemotron_h as mn
        orig = mn.mamba2_split_conv1d_scan_combined
        calls = {"n": 0, "ok": 0}

        def counted(*a, **k):
            r = orig(*a, **k)
            calls["n"] += 1
            calls["ok"] += r is not None
            return r

        mn.mamba2_split_conv1d_scan_combined = counted
        g = {}
        for modo in ("treino", "eval"):
            calls.update(n=0, ok=0)
            model.train(modo == "treino")
            with torch.no_grad():
                option_scores(model, exs[:1], pad_id, pointer=head)
            g[modo] = dict(calls)
        mn.mamba2_split_conv1d_scan_combined = orig
        n_m = expected_counts(model.get_base_model().config)["in_proj"]
        rep["g_kernel_fundido"] = {"camadas_mamba": n_m, **g, "ok": g["treino"]["ok"] == n_m and g["eval"]["n"] == 0}
    except Exception as e:  # noqa: BLE001
        rep["g_kernel_fundido"] = {"erro": f"{type(e).__name__}: {e}", "ok": False}

    # b. passo 0
    model.train()
    mb = exs[: args.micro]
    loss, _, _ = batch_loss(model, mb, pad_id, pointer=head)
    (loss / len(mb)).backward()
    bad, zero_a, zero_b, zero_h = grads_report()
    rep["b_passo0"] = {"perda": float(loss) / len(mb), "grad_nao_finito": bad[:5], "lora_B_grad_zero": len(zero_b),
                       "lora_A_grad_zero": len(zero_a), "cabeca_grad_zero": zero_h,
                       "ok": bool(torch.isfinite(loss)) and not bad and not zero_b and not zero_h}
    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)

    # c + d. sobreajuste (mesma lr na cabeça e no LoRA, como no registro do E4)
    opt = torch.optim.AdamW(allp.values(), lr=args.lr)
    hist = []
    t0 = time.time()
    for step in range(args.steps):
        tot, hits = 0.0, 0
        for s in range(0, len(exs), args.micro):
            l, h, _ = batch_loss(model, exs[s:s + args.micro], pad_id, pointer=head)
            (l / len(exs)).backward()
            tot, hits = tot + float(l), hits + h
        if step == 2:
            bad, zero_a, zero_b, zero_h = grads_report()
            rep["c_gradientes_apos_passos"] = {"passo": step, "grad_nao_finito": bad[:5], "n_lora_A_grad_zero": len(zero_a),
                                               "n_lora_B_grad_zero": len(zero_b), "cabeca_grad_zero": zero_h}
        torch.nn.utils.clip_grad_norm_(list(allp.values()), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        hist.append({"passo": step, "perda": tot / len(exs), "acc": hits / len(exs)})
        if step % 10 == 0:
            print(json.dumps(hist[-1]), flush=True)
    unchanged = [n for n, p in allp.items() if torch.equal(p.detach(), init[n])]
    c = rep["c_gradientes_apos_passos"]
    c.update(n_nao_mudaram=len(unchanged), parametros_que_nao_mudaram=unchanged[:5])
    c["ok"] = (not c["grad_nao_finito"] and c["n_lora_A_grad_zero"] == 0 and c["n_lora_B_grad_zero"] == 0
               and not c["cabeca_grad_zero"] and not unchanged)

    model.eval()
    head.eval()
    with torch.no_grad():
        tot, hits = 0.0, 0
        for s in range(0, len(exs), args.micro):
            l, h, _ = batch_loss(model, exs[s:s + args.micro], pad_id, pointer=head)
            tot, hits = tot + float(l), hits + h
    final = {"perda": tot / len(exs), "acc": hits / len(exs)}
    rep["d_sobreajuste"] = {"inicial": hist[0], "final_eval": final, "segundos": time.time() - t0,
                            "curva": hist[:: max(1, len(hist) // 10)],
                            "ok": final["perda"] < 0.25 * hist[0]["perda"] and final["acc"] >= 0.9}

    # e. caminho de treino (eval) x HFBackend no mesmo modelo
    probe = exs[: args.n_probe]
    be = HFBackend(model_obj=model.get_base_model(), tokenizer_obj=tok, device="cuda")
    with torch.no_grad():
        train_lp = [torch.log_softmax(z, 0).tolist() for e in probe for z in option_scores(model, [e], pad_id, pointer=head)]
    be_lp = [be.pointer_readout_batch([e.ids], [e.opt_pos], pointer=head)[0][0] for e in probe]

    def cmp(xs, ys):
        dd = max(max(abs(a - b) for a, b in zip(x, y)) for x, y in zip(xs, ys))
        am = sum(max(range(len(x)), key=x.__getitem__) == max(range(len(y)), key=y.__getitem__) for x, y in zip(xs, ys))
        return dd, am

    dd, am = cmp(train_lp, be_lp)
    rep["e_coerencia_readout"] = {"max_abs_dlogprob": dd, "argmax_igual": f"{am}/{len(probe)}",
                                  "ok": dd <= args.tol and am == len(probe)}
    model.train()
    with torch.no_grad():
        tr_lp = [torch.log_softmax(z, 0).tolist() for e in probe for z in option_scores(model, [e], pad_id, pointer=head)]
    model.eval()
    d2, am2 = cmp(tr_lp, train_lp)
    rep["diag_modo_treino_vs_eval"] = {"max_abs_dlogprob": d2, "argmax_igual": f"{am2}/{len(probe)}"}

    # f. recarga conjunta (adaptador + cabeça)
    adir = out / "adapter_e4"
    model.save_pretrained(str(adir))
    save_head(head, adir)
    del be
    reloaded = HFBackend(model=args.model, revision=args.revision, dtype="bfloat16", trust_remote_code=False,
                         adapter=str(adir))
    re_lp = [reloaded.pointer_readout_batch([e.ids], [e.opt_pos])[0][0] for e in probe]
    d3, am3 = cmp(re_lp, be_lp)
    rep["f_recarga"] = {"max_abs_dlogprob": d3, "argmax_igual": f"{am3}/{len(probe)}",
                        "cabeca_carregada": reloaded.pointer is not None,
                        "ok": reloaded.pointer is not None and d3 <= args.tol and am3 == len(probe)}
    del reloaded
    torch.cuda.empty_cache()

    # diagnóstico: memória com 8192 tokens por micro-lote nos exemplos mais longos
    longest = sorted(cases, key=lambda c: len(c.state), reverse=True)[:16]
    lex = [enc(c, "shuffle7", soft=True) for c in longest]
    L = max(len(e.ids) for e in lex)
    b = lex[: max(1, 8192 // L)]
    model.train()
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(3):
            l, _, _ = batch_loss(model, b, pad_id, pointer=head)
            l.backward()
            model.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.time() - t1) / 3
        rep["diag_memoria_8192"] = {"exemplos": len(b), "maior_comprimento": L,
                                    "pico_gb": torch.cuda.max_memory_allocated() / 1e9,
                                    "tokens_s": sum(len(e.ids) for e in b) / dt}
    except torch.cuda.OutOfMemoryError:
        rep["diag_memoria_8192"] = {"erro": "OOM"}
    rep["aprovado"] = verdict(rep)
    return rep


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="lumi-e4")
    p.add_argument("--model", default=SOURCES["lumi"]["repo"])
    p.add_argument("--revision", default=SOURCES["lumi"]["revision"])
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--n-overfit", type=int, default=64)
    p.add_argument("--n-pos", type=int, default=200)
    p.add_argument("--micro", type=int, default=8)
    p.add_argument("--n-probe", type=int, default=16)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rep = run(a)
    (Path(a.out) / "e4.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    print(f"E4 SANIDADE {'APROVADA' if rep['aprovado'] else 'REPROVADA'}", flush=True)
    sys.exit(0 if rep["aprovado"] else 1)


if __name__ == "__main__":
    main()
