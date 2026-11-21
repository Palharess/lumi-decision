"""Imprime as versões da imagem do vLLM (slurm/prof_preparar.sbatch)."""
import torch
import vllm

print(">> vllm", vllm.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda)
