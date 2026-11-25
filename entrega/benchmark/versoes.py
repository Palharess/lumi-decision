"""Versões do ambiente em que o pacote foi conferido (vai para ambiente.txt; não vai na entrega)."""
import sys

import causal_conv1d
import mamba_ssm
import peft
import torch
import transformers

print("python", sys.version.split()[0])
for m in (torch, transformers, peft, mamba_ssm, causal_conv1d):
    print(m.__name__, m.__version__)
print("cuda", torch.version.cuda)
