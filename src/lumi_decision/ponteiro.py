"""E4 — pointer head (H3), no molde do kev (github.com/jaredpalmer/kev, kev/model.py `PointerHead`).

Pontuação de cada opção: s_i = (W_q h_q) · (W_k h_i) / √dp, com h_q = estado final na última posição do prompt
(onde o A1 lê as letras) e h_i = estado final no token que contém o último caractere da opção i. Os estados são
os da saída do backbone (depois da norma final: a mesma entrada do LM head, conferida no E0). A cabeça roda em
FP32. Numa passada única as opções interagem (o Mamba carrega estado de token em token): a sensibilidade à
ordem é medida, não assumida.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

FILE, CONFIG = "ponteiro.safetensors", "ponteiro.json"


class PointerHead(nn.Module):
    def __init__(self, d: int, dp: int = 256):
        super().__init__()
        self.d, self.dp = d, dp
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)  # com viés, como no kev
        self.scale = 1 / math.sqrt(dp)

    def forward(self, h_q: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:  # [d], [K, d] -> [K]
        return (self.k(h_opts) @ self.q(h_q)) * self.scale


def new_head(d: int, dp: int, seed: int) -> PointerHead:
    """Inicializa sem mexer no RNG do treino: a inicialização do LoRA e o dropout ficam idênticos aos do E3.
    Os pesos nascem na CPU; só o gerador da CPU é semeado (e restaurado). `torch.manual_seed` NÃO serve: ele
    também resemeia a CUDA, que `fork_rng(devices=[])` não restaura (revisão 7)."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        return PointerHead(d, dp).float()


def pointer_logits(h: torch.Tensor, lens: torch.Tensor, batch_pos: list[list[int]], head: PointerHead):
    """h: [B, T, d] estados finais (padding à direita); devolve lista de logits FP32 [n_i]."""
    out = []
    for i, pos in enumerate(batch_pos):
        hq = h[i, int(lens[i]) - 1].float()
        ho = h[i, torch.tensor(pos, device=h.device)].float()
        out.append(head(hq, ho))
    return out


def save_head(head: PointerHead, folder: Path) -> None:
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, str(folder / FILE))
    (folder / CONFIG).write_text(json.dumps({"d": head.d, "dp": head.dp}))


def load_head(folder: Path, device) -> PointerHead | None:
    from safetensors.torch import load_file

    folder = Path(folder)
    if not (folder / FILE).exists():
        return None
    cfg = json.loads((folder / CONFIG).read_text())
    head = PointerHead(cfg["d"], cfg["dp"]).float()
    head.load_state_dict(load_file(str(folder / FILE)))
    return head.to(device).eval()


def head_sha256(folder: Path) -> str | None:
    p = Path(folder) / FILE
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
