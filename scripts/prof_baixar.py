"""Baixa um modelo do HF numa revisão fixada (reservadas com professor, slurm/prof_preparar.sbatch).
Uso: python scripts/prof_baixar.py <repo> <revisao>"""

import sys

from huggingface_hub import snapshot_download

repo, rev = sys.argv[1], sys.argv[2]
p = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model",
                                                          "tokenizer*"])
print(">> baixado:", p)
