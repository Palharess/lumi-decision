"""E3c: famílias vistas (Felhen e Telepatia, dev, 2 ordens), média com peso igual, e acertos por fonte.

    python scripts/analise_e3c_vistas.py <pasta> 0 1
"""
import json, sys
from pathlib import Path
from lumi_decision import metrics
sys.stdout.reconfigure(encoding="utf-8")
R = Path(sys.argv[1])
def load(n, src):
    return [r for r in (json.loads(l) for l in (R / f"{n}_{src}_dev" / "predicoes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()) if not r["held_out"]]
for s in sys.argv[2:]:
    P, Q, C = f"e3c_proximo_s{s}", f"e3c_resto_s{s}", f"e3c_ctrl_s{s}"
    for src in ("felhen", "telepatia"):
        rows = {n: load(n, src) for n in (P, Q, C)}
        acc = {n: f"{sum(r['pred']==r['gold'] for r in v)}/{len(v)}" for n, v in rows.items()}
        out = {}
        for rot, a, b in (("próximo−resto", Q, P), ("próximo−controle", C, P), ("resto−controle", C, Q)):
            m = metrics.family_mean_bootstrap(rows[a], rows[b])
            out[rot] = f"{100*m['delta_media_familias']:+.1f} [{100*m['ic95'][0]:+.1f}; {100*m['ic95'][1]:+.1f}]"
        print(f"s{s} {src}: {acc} {out}")
