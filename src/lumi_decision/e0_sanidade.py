"""E0 — sanidade do checkpoint oficial no backend HF (rodar no cluster, 1 GPU).

Verificações OBRIGATÓRIAS (todas precisam passar; senão o processo sai com código 1):
  0. pesos: nenhum peso ausente, inesperado ou com formato errado ao carregar;
  1. template: nosso render == tokenizer.apply_chat_template;
  2. readout: o estado projetado == entrada do LM head no forward completo, e argmax igual
     (a projeção do readout é FP32; a do modelo pode ser BF16 — essa diferença fica no diagnóstico);
  3. lote com padding à direita == lote de 1 (decisão igual e diferença dentro do ruído numérico do lote);
  4. geração: cache == sem cache nos primeiros tokens, e TODAS as gerações param num token de EOS.

DIAGNÓSTICOS (registrados, não reprovam): 1º token gerado == argmax do readout (a geração livre pode
sair das letras), massa dos candidatos, concordância com a execução de referência (GGUF local).

    lumi-e0 run --source referencias/piloto40/casos.json --ref-run referencias/piloto40 --out runs/e0_remote
    lumi-e0 comparar runs/e0_remote runs/e0_native       # compara as duas implementações BF16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .data import ARTIFACTS_DIR, SOURCES, load_cases, load_config
from .evaluate import git_commit, load_rows, predict_one
from .prompts import candidate_tokens, load_template, order_options, render, user_message

REQUIRED = ("0_pesos_carregados", "1_template_igual_ao_tokenizer", "2_readout_vs_forward",
            "3_padding_direita_vs_lote_1", "4_geracao")


def verdict(rep: dict) -> bool:
    return all(rep.get(k, {}).get("ok") is True for k in REQUIRED)


def run(args) -> dict:
    import torch

    from .backends.hf import HFBackend

    be = HFBackend(model=args.model, revision=args.revision, dtype=args.dtype,
                   trust_remote_code=not args.native)
    tpl = load_template(Path(args.template))
    cases = load_cases(args.source, load_config())
    if args.split != "all":
        cases = [c for c in cases if c.split == args.split] or cases
    cases = cases[: args.n]
    orders = args.orders.split(",")
    rep = {"meta": {"git": git_commit(), "backend": be.describe(), "args": vars(args),
                    "inicio": time.strftime("%Y-%m-%dT%H:%M:%S")}}

    # 0. pesos
    info = be.loading_info or {}
    bad = {k: v for k, v in info.items() if k in ("missing_keys", "unexpected_keys", "mismatched_keys") and v}
    rep["0_pesos_carregados"] = {"ok": not bad, "problemas": bad, "error_msgs": info.get("error_msgs", [])}

    # 1. template
    mism = []
    for c in cases:
        opts = order_options(c, orders[0])
        ours = render(tpl, c, opts)
        theirs = be.tok.apply_chat_template([{"role": "user", "content": user_message(c, opts)}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        if ours != theirs:
            mism.append(c.id)
    rep["1_template_igual_ao_tokenizer"] = {"ok": not mism, "divergentes": mism[:10]}

    # 2. readout vs forward completo.
    # Obrigatório: o estado que o readout projeta é o MESMO que o modelo entrega ao LM head na última posição
    # (mesma posição, norma final incluída) e o argmax coincide. A diferença de log-prob entre a nossa projeção
    # FP32 e a do modelo (BF16 na implementação nativa: logits de módulo ~20 têm passo de 0,125) é diagnóstico.
    prompts, diffs, hdiffs, argmax_ok, masses = [], [], [], 0, []
    for c in cases:
        opts = order_options(c, orders[0])
        ids, cands = candidate_tokens(be.tokenize, render(tpl, c, opts), len(opts))
        prompts.append((ids, cands))
        lp, _ = be.readout(ids, cands)
        x_head, full_all = be.full_forward_head_input(ids)
        full = full_all[cands].tolist()
        h_ours = be.last_hidden([ids])[0]
        hdiffs.append(float((h_ours.float() - x_head.float()).abs().max() / x_head.float().abs().max().clamp_min(1e-6)))
        diffs.append(max(abs(a - b) for a, b in zip(lp, full)))
        masses.append(sum(2.718281828459045 ** x for x in lp))
        argmax_ok += int(max(range(len(lp)), key=lp.__getitem__) == max(range(len(full)), key=full.__getitem__))
    rep["2_readout_vs_forward"] = {
        "estado_rel_max_dif": max(hdiffs), "argmax_igual": f"{argmax_ok}/{len(cases)}",
        "diag_max_abs_dlogprob_fp32_vs_modelo": max(diffs),
        "ok": argmax_ok == len(cases) and max(hdiffs) < args.tol_estado,
    }

    # 3. padding à direita vs lote de 1.
    # Com padding à direita e modelo causal, o padding não pode alterar as posições reais; o que sobra é ruído
    # numérico de BF16 (kernels escolhem outra ordem de soma conforme o tamanho do lote e o comprimento).
    # Piso de ruído: o MESMO prompt repetido num lote do mesmo tamanho, sem padding, contra o lote de 1.
    # Obrigatório: decisão (argmax entre candidatos) igual em todos os casos e diferença no máximo
    # 2x o piso de ruído + 0,02. Registramos também a diferença nas probabilidades renormalizadas.
    def renorm(lps):
        m = max(lps)
        w = [2.718281828459045 ** (x - m) for x in lps]
        return [v / sum(w) for v in w]

    pad_diffs, pad_dp, pad_argmax, noise, leak, leak_dp = [], [], 0, [], [], []
    filler = [t for p in prompts for t in p[0]]  # tokens reais para preencher os sufixos do controle
    for i in range(0, len(prompts), args.batch):
        chunk = prompts[i:i + args.batch]
        batched, _ = be.readout_batch([p[0] for p in chunk], [p[1] for p in chunk])
        T = max(len(p[0]) for p in chunk)
        suf = be.readout_unmasked_at([p[0] + filler[: T - len(p[0])] for p in chunk], [len(p[0]) for p in chunk],
                                     [p[1] for p in chunk])
        for (ids, cands), lb, lr in zip(chunk, batched, suf):
            ls, _ = be.readout(ids, cands)
            pad_diffs.append(max(abs(a - b) for a, b in zip(lb, ls)))
            pad_dp.append(max(abs(a - b) for a, b in zip(renorm(lb), renorm(ls))))
            pad_argmax += int(max(range(len(lb)), key=lb.__getitem__) == max(range(len(ls)), key=ls.__getitem__))
            leak.append(max(abs(a - b) for a, b in zip(lb, lr)))
            leak_dp.append(max(abs(a - b) for a, b in zip(renorm(lb), renorm(lr))))
    for ids, cands in prompts[: args.n_ruido]:
        copies, _ = be.readout_batch([ids] * args.batch, [cands] * args.batch)
        ls, _ = be.readout(ids, cands)
        noise.append(max(abs(a - b) for row in copies for a, b in zip(row, ls)))
    limite = 2 * max(noise) + 0.02
    rep["3_padding_direita_vs_lote_1"] = {
        "max_abs_dlogprob": max(pad_diffs), "piso_de_ruido_mesmo_lote_sem_padding": max(noise),
        "limite": limite, "argmax_igual": f"{pad_argmax}/{len(prompts)}",
        "max_abs_dprob_renormalizada": max(pad_dp),
        "controle_padding_vs_sufixo_real_mesmo_formato": {"max_abs_dlogprob": max(leak), "max_abs_dprob_renormalizada": max(leak_dp)},
        "ok": pad_argmax == len(prompts) and max(pad_diffs) <= limite,
    }

    # 4. geração (um erro aqui não impede gravar as predições do readout, usadas na comparação remote x nativa)
    stops, first_ok, cache_ok, last_tokens, texts = [], 0, 0, [], []
    try:
        for ids, cands in prompts[: args.n_gen]:
            new, text, ginfo = be.generate(ids, 16)
            stops.append(ginfo["stop_type"])
            last_tokens.append(new[-1] if new else None)
            texts.append(text)
            lp, _ = be.readout(ids, cands)
            first_ok += int(bool(new) and new[0] == cands[max(range(len(lp)), key=lp.__getitem__)])
            nc = be.greedy_no_cache(ids, n_tokens=min(4, len(new)))
            cache_ok += int(nc == new[: len(nc)])
        gen_err = None
    except Exception as e:  # noqa: BLE001 — registrado e reprovado
        gen_err = f"{type(e).__name__}: {e}"
        print(f">> geração falhou: {gen_err}", file=sys.stderr, flush=True)
    n_gen = len(stops)
    rep["4_geracao"] = {
        "ok": gen_err is None and n_gen > 0 and cache_ok == n_gen and stops.count("eos") == n_gen,
        "erro": gen_err,
        "cache_igual_sem_cache": f"{cache_ok}/{n_gen}", "parou_em_eos": f"{stops.count('eos')}/{n_gen}",
        "eos_aceitos": be.eos_ids, "ultimos_tokens": sorted({str(t) for t in last_tokens}),
        "exemplos_texto": texts[:5],
    }
    rep["diagnostico"] = {
        "primeiro_token_igual_readout": f"{first_ok}/{n_gen}",
        "massa_dos_candidatos_min": min(masses), "massa_dos_candidatos_mediana": sorted(masses)[len(masses) // 2],
    }

    # predições completas (para comparar implementações e com a referência)
    rows = [predict_one(be, tpl, c, o, generate=False) for c in cases for o in orders]
    out = Path(args.out)
    (out / "predicoes.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                         encoding="utf-8")
    if args.ref_run:
        rep["diagnostico"]["vs_referencia"] = compare_rows(load_rows(Path(args.ref_run)), rows)
    rep["memoria_pico_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    rep["aprovado"] = verdict(rep)
    return rep


def compare_rows(a: list[dict], b: list[dict]) -> dict:
    ra = {(r["case_id"], r["order"]): r for r in a}
    keys = [(r["case_id"], r["order"]) for r in b if (r["case_id"], r["order"]) in ra]
    rb = {(r["case_id"], r["order"]): r for r in b}
    dps = sorted(max(abs(ra[k]["probs"][x] - rb[k]["probs"][x]) for x in rb[k]["probs"]) for k in keys)
    return {"n": len(keys),
            "decisao_igual": f"{sum(ra[k]['pred'] == rb[k]['pred'] for k in keys)}/{len(keys)}",
            "tokens_do_prompt_iguais": f"{sum(ra[k].get('prompt_sha1') == rb[k].get('prompt_sha1') for k in keys)}/{len(keys)}",
            "max_abs_dp": dps[-1] if dps else None, "mediana_abs_dp": dps[len(dps) // 2] if dps else None}


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="lumi-e0")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default=SOURCES["lumi"]["repo"])
    r.add_argument("--revision", default=SOURCES["lumi"]["revision"])
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--native", action="store_true", help="implementação nativa do transformers (sem remote code)")
    r.add_argument("--template", default=str(ARTIFACTS_DIR / "lumi" / "chat_template.jinja"))
    r.add_argument("--source", required=True)
    r.add_argument("--split", default="dev")
    r.add_argument("--orders", default="canonical,reversed")
    r.add_argument("--n", type=int, default=40)
    r.add_argument("--n-gen", type=int, default=20)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--tol-estado", type=float, default=1e-3,
                   help="diferença relativa máxima entre o estado do readout e a entrada do LM head no forward")
    r.add_argument("--n-ruido", type=int, default=10, help="prompts usados para medir o piso de ruído do lote")
    r.add_argument("--ref-run", default=None)
    r.add_argument("--out", required=True)
    c = sub.add_parser("comparar")
    c.add_argument("a")
    c.add_argument("b")
    a = p.parse_args(argv)
    if a.cmd == "comparar":
        print(json.dumps(compare_rows(load_rows(Path(a.a)), load_rows(Path(a.b))), indent=2, ensure_ascii=False))
        return
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rep = run(a)
    (Path(a.out) / "e0.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    print(f"E0 {'APROVADO' if rep['aprovado'] else 'REPROVADO'}", flush=True)
    sys.exit(0 if rep["aprovado"] else 1)


if __name__ == "__main__":
    main()
