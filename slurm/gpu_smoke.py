"""Teste rápido de GPU dentro do container: kernel Triton do mamba-ssm (o mesmo caminho que o E0 usa)."""

import os

import torch
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

b, seqlen, h, p, g, n = 1, 64, 2, 16, 1, 16
x = torch.randn(b, seqlen, h, p, device="cuda", dtype=torch.bfloat16)
dt = torch.rand(b, seqlen, h, device="cuda", dtype=torch.float32)
A = -torch.rand(h, device="cuda", dtype=torch.float32)
B = torch.randn(b, seqlen, g, n, device="cuda", dtype=torch.bfloat16)
C = torch.randn(b, seqlen, g, n, device="cuda", dtype=torch.bfloat16)
y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=32)
torch.cuda.synchronize()
assert torch.isfinite(y.float()).all(), "saída não finita"
print("gpu smoke ok:", torch.cuda.get_device_name(0), "| TRITON_LIBCUDA_PATH =", os.environ.get("TRITON_LIBCUDA_PATH"),
      "| y", tuple(y.shape))
