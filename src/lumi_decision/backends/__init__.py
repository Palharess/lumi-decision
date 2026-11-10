"""Backends de inferência. Todos expõem a mesma interface:

    tokenize(text) -> list[int]
    readout(prompt_ids, cand_ids) -> (logprobs_dos_candidatos_no_vocabulário, info)
    generate(prompt_ids, max_new_tokens) -> (token_ids, texto, info)
    describe() -> dict  (metadados da execução)

`readout` devolve log-probabilidades no vocabulário inteiro (não renormalizadas);
a renormalização sobre os candidatos é feita no avaliador.
"""


def make_backend(name: str, **kw):
    if name == "llamacpp":
        from .llamacpp import LlamaCppBackend
        return LlamaCppBackend(**kw)
    if name == "hf":
        from .hf import HFBackend
        return HFBackend(**kw)
    raise ValueError(f"backend desconhecido: {name}")
