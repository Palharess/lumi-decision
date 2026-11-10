"""Backend local via `llama-server` (llama.cpp) com GGUF. Uso: desenvolvimento e diagnóstico.

Números oficiais vêm do backend HF com o checkpoint BF16 no cluster.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

FULL_VOCAB = 131072


class LlamaCppBackend:
    def __init__(self, model: str | None = None, server_bin: str | None = None, url: str | None = None,
                 n_gpu_layers: int = 99, ctx: int = 8192, threads: int = 6, log_path: str | None = None,
                 top_n: int = 64):
        self.model = model or os.environ.get("LUMI_GGUF")
        self.server_bin = server_bin or os.environ.get("LUMI_LLAMA_SERVER")
        self.top_n = top_n
        self.proc = None
        if url:
            self.url = url.rstrip("/")
        else:
            if not (self.model and self.server_bin):
                raise SystemExit("informe --model e --server-bin (ou LUMI_GGUF / LUMI_LLAMA_SERVER) ou --url")
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            self.url = f"http://127.0.0.1:{port}"
            args = [self.server_bin, "-m", self.model, "--host", "127.0.0.1", "--port", str(port),
                    "-ngl", str(n_gpu_layers), "-c", str(ctx), "-t", str(threads), "-np", "1", "--no-webui"]
            log = open(log_path or os.devnull, "w", encoding="utf-8")
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            self.proc = subprocess.Popen(args, cwd=Path(self.server_bin).parent, stdout=log,
                                         stderr=subprocess.STDOUT, creationflags=flags)
            self._wait_ready()
        self.props = self._get("/props")

    # ------------------------------------------------------------ HTTP
    def _post(self, path, payload):
        req = Request(self.url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=300) as r:
            return json.load(r)

    def _get(self, path):
        with urlopen(self.url + path, timeout=30) as r:
            return json.load(r)

    def _wait_ready(self, timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server saiu com código {self.proc.returncode}")
            try:
                if self._get("/health").get("status") == "ok":
                    return
            except (OSError, URLError):
                pass
            time.sleep(1)
        raise TimeoutError("llama-server não ficou pronto")

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=30)

    # ------------------------------------------------------------ interface
    def tokenize(self, text: str) -> list[int]:
        return self._post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]

    def _completion(self, ids, **kw):
        t0 = time.perf_counter()
        r = self._post("/completion", {"prompt": ids, "cache_prompt": False, "return_tokens": True,
                                        "repeat_penalty": 1, "seed": 0, **kw})
        if r.get("truncated"):
            raise ValueError("contexto truncado")
        return r, (time.perf_counter() - t0) * 1000

    def readout(self, prompt_ids, cand_ids):
        """Log-probs brutas (sem amostragem) do próximo token para cada candidato."""
        info = {}
        for n_probs in (self.top_n, FULL_VOCAB):
            r, ms = self._completion(prompt_ids, n_predict=1, n_probs=n_probs, temperature=-1,
                                     post_sampling_probs=False)
            top = r["completion_probabilities"][0]["top_logprobs"]
            lp = {e["id"]: e["logprob"] for e in top}
            if all(c in lp for c in cand_ids):
                info = {"latency_ms": ms, "n_probs": n_probs, "timings": r.get("timings")}
                return [lp[c] for c in cand_ids], info
        raise ValueError("candidatos ausentes mesmo com o vocabulário inteiro")

    def generate(self, prompt_ids, max_new_tokens=16):
        r, ms = self._completion(prompt_ids, n_predict=max_new_tokens, temperature=0)
        return r["tokens"], r["content"], {"latency_ms": ms, "stop_type": r.get("stop_type"),
                                            "timings": r.get("timings")}

    def describe(self) -> dict:
        return {"backend": "llamacpp", "model": self.model, "server_bin": self.server_bin,
                "build": self.props.get("build_info"), "model_path_server": self.props.get("model_path")}
