"""Download fixado por revisão, conversão para `Case` e atribuição determinística de splits.

Uso:
    lumi-dados baixar            # datasets + chat template/tokenizer do Lumi em revisões fixas
    lumi-dados resumo            # contagens por fonte/split/família
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from . import REPO_ROOT
from .schema import Case

DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
SPLITS_CONFIG = REPO_ROOT / "configs" / "splits.toml"

FELHEN_TASKS = ("camara_tema", "factck_veracidade", "faquad_resposta", "juristcu_area",
                "olid_alvo", "olid_ofensivo", "scielo_area")

# Revisões fixadas: mudar qualquer uma delas exige registrar no README e nos resultados.
SOURCES = {
    "felhen": {
        "repo": "felhen-ai/ptbr-typed-decisions-bench", "repo_type": "dataset",
        "revision": "f4028c3712f2eae307bfd9e605746000ce1f1fa3",
        "files": [f"data/{t}/{s}.jsonl" for t in FELHEN_TASKS for s in ("train", "dev", "test")],
    },
    "telepatia": {
        "repo": "telepatia-ai/typed-decisions-pt-es", "repo_type": "dataset",
        "revision": "8c85a4d2017c12599857ad083f87536fa6424596",
        "files": ["data/train.pt.jsonl", "data/test.pt.jsonl"],
    },
    "traces": {
        "repo": "annajuliaasf/tool-calling-traces-ptbr", "repo_type": "dataset",
        "revision": "021b543b8d01a5efa39011fbfda029a90696fdff",
        "files": ["treino.jsonl", "validacao.jsonl", "teste.jsonl", "tools_catalogo.jsonl"],
    },
    "lumi": {
        "repo": "ceia-br/lumi-4b", "repo_type": "model",
        "revision": "27b30f40b930fd05cee5e1c1f8790f70fa5b816d",
        "files": ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "config.json",
                  "generation_config.json"],
    },
}


def source_dir(name: str) -> Path:
    base = ARTIFACTS_DIR if SOURCES[name]["repo_type"] == "model" else DATA_DIR
    return base / name


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(names: list[str] | None = None) -> list[dict]:
    from huggingface_hub import hf_hub_download

    records = []
    for name in names or list(SOURCES):
        spec = SOURCES[name]
        dest = source_dir(name)
        for filename in spec["files"]:
            path = Path(hf_hub_download(spec["repo"], filename, repo_type=spec["repo_type"],
                                        revision=spec["revision"], local_dir=dest))
            records.append({"source": name, "repo": spec["repo"], "revision": spec["revision"],
                            "file": filename, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
            print(f"{name}: {filename}", flush=True)
    manifest = DATA_DIR / "manifesto.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    keep = [r for r in old if (r["source"], r["file"]) not in {(x["source"], x["file"]) for x in records}]
    manifest.write_text(json.dumps(keep + records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


def load_config(path: Path = SPLITS_CONFIG) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def bucket(key: str, salt: str) -> float:
    """Número em [0, 1) estável para uma chave; base de todas as divisões aleatórias."""
    return int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:12], 16) / 16**12


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- agrupamento por texto

DISCARDED = "descartado"  # linha de treino removida por repetir texto de dev/teste ou de família reservada


def text_group(prefix: str, text: str) -> str:
    """Grupo = texto normalizado. O mesmo texto em tarefas diferentes (ex.: as duas OLID) cai no
    mesmo grupo, e grupos nunca podem cruzar splits."""
    norm = " ".join(text.casefold().split())
    return f"{prefix}:txt:{hashlib.sha1(norm.encode()).hexdigest()[:16]}"


def purge_train(cases: list[Case]) -> None:
    """Treino/calibração não podem repetir texto de dev/teste oficiais nem de famílias reservadas.
    Essas linhas viram `descartado` (o teste oficial é preservado). Dev repetido no teste também."""
    protected = {c.group_id for c in cases if c.split in ("dev", "test") or c.held_out}
    test_groups = {c.group_id for c in cases if c.split == "test"}
    for c in cases:
        if c.split in ("train", "calib") and not c.held_out and c.group_id in protected:
            c.meta["descarte"] = "texto em dev/teste/reservada"
            c.split = DISCARDED
        elif c.split == "dev" and c.group_id in test_groups:
            c.meta["descarte"] = "texto também no teste"
            c.split = DISCARDED


# ---------------------------------------------------------------- Felhen

NOUL_OPTIONS = [("True", "Sim"), ("False", "Não")]


def felhen_cases(cfg: dict) -> list[Case]:
    salt, conf = cfg["salt"], cfg["felhen"]
    held = set(conf["held_out_tasks"])
    cases = []
    calib_frac = {}
    for task in FELHEN_TASKS:
        for split in ("train", "dev", "test"):
            rows = read_jsonl(source_dir("felhen") / "data" / task / f"{split}.jsonl")
            if split == "train":
                calib_frac[task] = min(1.0, conf["calib_per_task"] / max(1, len(rows)))
            for i, row in enumerate(rows):
                if row["type"] == "noul":
                    options, gold = NOUL_OPTIONS, str(bool(row["gold"]))
                else:
                    options, gold = list(row["criteria"].items()), str(row["gold"])
                cases.append(Case(
                    id=f"felhen:{task}:{split}:{i}", group_id=text_group("felhen", row["text"]),
                    source="felhen", family=task, split=split, held_out=task in held, qtype=row["type"],
                    state=row["text"], question=row["instructions"], options=options, gold=gold,
                    meta={"official_split": split},
                ))
    purge_train(cases)
    # Calibração decidida por GRUPO: o limiar do grupo é o maior entre as tarefas vistas em que ele
    # aparece, para que o mesmo texto nunca fique em calib numa tarefa e em treino em outra.
    thr = defaultdict(float)
    for c in cases:
        if c.split == "train" and not c.held_out:
            thr[c.group_id] = max(thr[c.group_id], calib_frac[c.family])
    for c in cases:
        if c.split == "train" and not c.held_out and bucket(c.group_id, salt) < thr[c.group_id]:
            c.split = "calib"
    return cases


# ---------------------------------------------------------------- Typed Decisions PT

def _telepatia_options(q: dict) -> list[tuple[str, str]]:
    if q["type"] == "score":
        return [(str(i), d) for i, d in enumerate(q["criteria"])]
    if q["type"] == "noul":
        # Em noul a instrução é uma afirmação; critérios costumam vir vazios.
        crit = q.get("criteria") or {}
        return [("true", crit.get("true") or "Sim, a afirmação é verdadeira."),
                ("false", crit.get("false") or "Não, a afirmação é falsa.")]
    return [(str(k), d) for k, d in q["criteria"].items()]


def telepatia_cases(cfg: dict) -> list[Case]:
    salt, conf = cfg["salt"], cfg["telepatia"]
    held = set(conf["held_out_workflows"])
    cases = []
    for official in ("train", "test"):
        for row in read_jsonl(source_dir("telepatia") / "data" / f"{official}.pt.jsonl"):
            wf = row["workflow"]
            if official == "test":
                split = "test"
            elif wf in held:
                split = "dev"  # família reservada: usada só para medir transferência
            else:
                b = bucket(row["id"], salt)
                split = "dev" if b < conf["dev_frac"] else (
                    "calib" if b < conf["dev_frac"] + conf["calib_frac"] else "train")
            state = json.dumps(row["state"], ensure_ascii=False, indent=2)
            for key, q in row["questions"].items():
                g = row["gold"][key]
                probs = {str(k): float(v) for k, v in g["probabilities"].items()}
                total = sum(probs.values())
                cases.append(Case(
                    id=f"telepatia:{row['id']}:{key}", group_id=f"telepatia:{row['id']}",
                    source="telepatia", family=wf, split=split, held_out=wf in held,
                    qtype=q["type"], state=state, question=q["instructions"],
                    options=_telepatia_options(q), gold=str(g["label"]),
                    gold_dist={k: v / total for k, v in probs.items()},
                    meta={"official_split": official, "question_key": key,
                          "gold_peak": max(probs.values()) / total},
                ))
    return cases


# ---------------------------------------------------------------- Tool Calling Traces PT-BR

NO_TOOL = ("no_tool", "Responder diretamente, sem chamar nenhuma ferramenta.")
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def _trigrams(text: str) -> Counter:
    t = f"  {' '.join(text.casefold().split())}  "
    return Counter(t[i:i + 3] for i in range(len(t) - 2))


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def traces_cases(cfg: dict) -> list[Case]:
    salt, conf = cfg["salt"], cfg["traces"]
    base = source_dir("traces")
    tools = {t["nome"]: t for t in read_jsonl(base / "tools_catalogo.jsonl")}
    held_tipos = set(conf["held_out_tipos"])
    held_tools = {n for n, t in tools.items() if t["tipo"] in held_tipos}
    grams = {n: _trigrams(f"{n} {t['descricao']}") for n, t in tools.items()}
    cases = []
    for official, split in (("treino", "train"), ("validacao", "dev"), ("teste", "test")):
        for i, row in enumerate(read_jsonl(base / f"{official}.jsonl")):
            msgs = row["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            first = next(m["content"] for m in msgs if m["role"] == "assistant")
            m = _TOOL_CALL.search(first)
            gold = json.loads(m[1])["nome_tool"] if m else NO_TOOL[0]
            if gold != NO_TOOL[0] and gold not in tools:
                raise ValueError(f"ferramenta desconhecida {gold} em {official}:{i}")
            held = gold in held_tools
            # Treino nunca mostra ferramentas reservadas, nem como distratoras. Em dev/teste as distratoras
            # vêm do catálogo inteiro: assim "a ferramenta que nunca apareceu no treino" não é sinônimo de
            # gabarito (ver `traces_shortcut_stats`). Mesmo assim o Traces é só diagnóstico para ferramentas
            # inéditas; a conclusão vem de catálogos reservados montados sem olhar o gabarito.
            pool = [n for n in tools if n != gold and (split != "train" or n not in held_tools)]
            q = _trigrams(user)
            rivals = sorted(pool, key=lambda n: (-_cosine(q, grams[n]), n))
            n_rivals = conf["menu_size"] - 1 - (gold != NO_TOOL[0])
            names = ([gold] if gold != NO_TOOL[0] else []) + rivals[:n_rivals]
            options = [(n, f"{n}: {tools[n]['descricao']}") for n in sorted(names)] + [NO_TOOL]
            cid = f"traces:{official}:{i}"
            cases.append(Case(
                id=cid, group_id=text_group("traces", user), source="traces",
                family=tools[gold]["tipo"] if gold in tools else "no_tool",
                split=split, held_out=held, qtype="choice", state=user,
                question="Qual ação o assistente deve tomar para atender ao pedido?",
                options=options, gold=gold,
                meta={"official_split": official,
                      "ferramentas_reservadas_no_menu": sorted(n for n in names if n in held_tools)},
            ))
    purge_train(cases)
    # Calibração: fração dos estados de treino vistos, sorteada por grupo (texto).
    for c in cases:
        if c.split == "train" and not c.held_out and bucket(c.group_id, salt) < conf.get("calib_frac", 0.0):
            c.split = "calib"
    return cases


def traces_shortcut_stats(cases: list[Case]) -> dict:
    """Quanto o atalho "escolha a ferramenta reservada do menu" acertaria, por split de avaliação."""
    out = {}
    for sp in ("dev", "test"):
        held = [c for c in cases if c.split == sp and c.held_out]
        seen = [c for c in cases if c.split == sp and not c.held_out]
        out[sp] = {
            "reservados": len(held),
            "gabarito_e_a_unica_reservada_no_menu": sum(c.meta["ferramentas_reservadas_no_menu"] == [c.gold] for c in held),
            "vistos": len(seen),
            "vistos_com_reservada_no_menu": sum(bool(c.meta["ferramentas_reservadas_no_menu"]) for c in seen),
        }
    return out


# ---------------------------------------------------------------- arquivo de casos (ex.: piloto)

def file_cases(path: Path) -> list[Case]:
    """Lê casos no formato do piloto (lista JSON com id/group/task/text/question/options/gold)
    ou no formato `Case.to_dict()` (JSON ou JSONL)."""
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
    cases = []
    for r in rows:
        if "source" in r:
            cases.append(Case.from_dict(r))
            continue
        cases.append(Case(
            id=r["id"], group_id=r["id"], source="arquivo", family=r.get("task", r.get("group", "")),
            split="dev", held_out=False, qtype="choice", state=r["text"], question=r["question"],
            options=[tuple(o) for o in r["options"]], gold=str(r["gold"]), meta={"group": r.get("group")},
        ))
    return cases


LOADERS = {"felhen": felhen_cases, "telepatia": telepatia_cases, "traces": traces_cases}


def load_cases(source: str, cfg: dict | None = None) -> list[Case]:
    if source in LOADERS:
        return LOADERS[source](cfg or load_config())
    return file_cases(Path(source))


def check_integrity(cases: list[Case]) -> None:
    """Falha se um grupo (texto) aparece em mais de um split ativo, se um texto de família
    reservada aparece no treino/calibração de uma família vista, ou se o gabarito não está no menu.
    O treino oficial das famílias reservadas fica no arquivo, mas `training_cases` o exclui."""
    splits = defaultdict(set)
    held_groups = {c.group_id for c in cases if c.held_out}
    for c in cases:
        if c.split == DISCARDED:
            continue
        if not (c.held_out and c.split == "train"):
            splits[c.group_id].add(c.split)
        if c.held_out and c.split == "calib":
            raise ValueError(f"caso reservado em calibração: {c.id}")
        if not c.held_out and c.split in ("train", "calib") and c.group_id in held_groups:
            raise ValueError(f"texto de família reservada no treino/calibração: {c.id}")
        if c.gold not in {o[0] for o in c.options}:
            raise ValueError(f"gabarito fora das opções: {c.id}")
    crossing = [g for g, s in splits.items() if len(s) > 1]
    if crossing:
        raise ValueError(f"{len(crossing)} grupos cruzam splits, ex.: {crossing[:3]}")


def training_cases(cases: list[Case]) -> list[Case]:
    """Único ponto de entrada para dados de treino: exclui famílias reservadas."""
    return [c for c in cases if c.split == "train" and not c.held_out]


def summary(sources: list[str]) -> dict:
    cfg = load_config()
    out = {}
    for s in sources:
        cases = load_cases(s, cfg)
        check_integrity(cases)
        table = Counter((c.split, c.family, c.held_out) for c in cases)
        out[s] = {f"{sp}|{fam}|{'reservada' if h else 'vista'}": n for (sp, fam, h), n in sorted(table.items())}
        out[s]["_max_opcoes"] = max(len(c.options) for c in cases)
    return out


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="lumi-dados")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baixar")
    b.add_argument("fontes", nargs="*", default=None)
    r = sub.add_parser("resumo")
    r.add_argument("fontes", nargs="*", default=list(LOADERS))
    a = p.parse_args(argv)
    if a.cmd == "baixar":
        download(a.fontes or None)
    else:
        print(json.dumps(summary(a.fontes), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
