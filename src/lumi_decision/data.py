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
    # Eikos (CC BY 4.0): `main` depois do PR #2 (770d6bdf), que tirou os documentos do FinEntity.
    # Só `core/train` entra, e só no E3b; nunca é avaliado. Ver `eikos_cases`.
    "eikos": {
        "repo": "caiovicentino1/eikos-decisions", "repo_type": "dataset",
        "revision": "cebb4a26687cb14742fd07c0ebe17387ee51d369",
        "files": ["data/core/train.jsonl"],
    },
    # DecisionBank (CC BY-SA 4.0 na parte usada): só a `j11_generator`, só no braço DecisionBank do E5. Ver
    # `decisionbank.py`; os sha256 das partes ficam lá.
    "decisionbank": {
        "repo": "jgeuter/DecisionBank", "repo_type": "dataset",
        "revision": "fbeee67758966fe78bfb008b28fc771924c0bc71",
        "files": [f"data/DecisionBank-625k/train-{i:05d}-of-00012.jsonl" for i in range(12)],
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


# ---------------------------------------------------------------- Eikos (só treino, E3b)

EIKOS_SHA256 = "2e5546946d28a7c9c479cd335752f2ca9ca430c94748fcec3574043c7e04d0b6"  # data/core/train.jsonl
EIKOS_FILTER_VERSION = "eikos-filtro-v1"
# Exclusões por SEMELHANÇA DE TAREFA com as famílias reservadas (mesmo com outro texto e outro nome):
# sem isso, "transferência para família inédita" deixaria de valer. Regex sobre instrução + estado + tópico.
EIKOS_EXCLUDE = {
    "invoice_processing": re.compile(
        r"invoice|fatura|nota fiscal|notas fiscais|purchase order|ordem de compra|pedido de compra|\bPO-?\d"
        r"|accounts payable|contas a pagar|duplicate (payment|charge|bill)|pagamento duplicado|three.?way match",
        re.I),
    "factck_veracidade": re.compile(
        r"fact.?check|checagem|misinformation|disinformation|desinforma|fake news|not[íi]cia falsa"
        r"|veracidade|veracity", re.I),
    "juristcu_area": re.compile(
        r"\bTCU\b|tribunal de contas|ac[óo]rd[ãa]o|jurisprud|court of audit", re.I),
}
EIKOS_EXCLUDE_SOURCES = ("prog_fin:t_three_way",)  # conferência fatura × pedido × recebimento


def _eikos_reason(row: dict) -> str | None:
    if row.get("source") in EIKOS_EXCLUDE_SOURCES:
        return "tarefa: invoice_processing"
    text = f"{row.get('topic', '')}\n{row['instructions']}\n{row['state']}"
    for fam, rx in EIKOS_EXCLUDE.items():
        if rx.search(text):
            return f"tarefa: {fam}"
    if not 2 <= len(row["options"]) <= 26:
        return "nº de opções"
    return None


def eikos_cases(cfg: dict, langs: set[str] | None = None, path: Path | None = None) -> list[Case]:
    """`core/train` do Eikos como casos de treino. Exclusões viram `descartado` com o motivo em
    `meta["descarte"]` (a auditoria conta por motivo). A sobreposição de TEXTO com dev/teste/reservadas
    das outras fontes é tirada depois, em `drop_eval_overlap`, porque precisa delas carregadas.
    `path` só para testes; o arquivo padrão precisa ter o sha256 da revisão fixada."""
    if path is None:
        path = source_dir("eikos") / "data" / "core" / "train.jsonl"
        if sha256_file(path) != EIKOS_SHA256:
            raise ValueError(f"{path} não é a revisão fixada do Eikos (sha256 diferente)")
    cases = []
    for row in read_jsonl(path):
        labels = [o["label"] for o in row["options"]]
        probs = [float(p) for p in row["target_probs"]]
        z = sum(probs)
        reason = _eikos_reason(row)
        if reason is None and langs is not None and row["lang"] not in langs:
            reason = "idioma"
        meta = {"official_split": "train", "lang": row["lang"], "topic": row["topic"],
                "origem": row["source"], "filtro": EIKOS_FILTER_VERSION}
        if reason:
            meta["descarte"] = reason
        cases.append(Case(
            id=f"eikos:{row['id']}", group_id=text_group("eikos", row["state"]), source="eikos",
            family=row["family"], split=DISCARDED if reason else "train", held_out=False,
            qtype=row["question_type"], state=row["state"], question=row["instructions"],
            options=[(o["label"], o.get("description") or o["label"]) for o in row["options"]],
            gold=str(row["expected"]), gold_dist={k: p / z for k, p in zip(labels, probs)} if z > 0 else None,
            meta=meta,
        ))
    return cases


# E3c (analises/plano_fase2.md, "Protocolo E3c"): o Eikos separado por tipo de decisão. Lista fixada antes de
# rodar, com os nomes de família do próprio Eikos: aplicar uma regra ou política escrita a um caso, ou raciocinar
# sobre datas e prazos (os tipos de `regra_beneficio`, `restricoes_oferta` e `ordem_eventos`).
EIKOS_PROXIMO = frozenset({"rules_book", "rules_permit", "rules_route", "rules_tier", "rules_score",
                           "policy", "long_policy", "temporal_numeric"})
EIKOS_RESTO_SAL = "lumi-decision-e3c-resto-v1"


def eikos_tipo_cases(cfg: dict, grupo: str, path: Path | None = None) -> list[Case]:
    """Eikos filtrado (como no E3b) e restrito a um grupo de famílias:
    - "proximo": todas as linhas mantidas das famílias de `EIKOS_PROXIMO`;
    - "resto": uma amostra das outras famílias com o MESMO nº de linhas do "proximo", sorteada por hash
      (sal fixo) entre as linhas e estratificada por idioma (o mesmo nº por idioma do "proximo", quando há).
    As linhas fora do grupo viram `descartado` com o motivo em `meta["descarte"]`."""
    if grupo not in ("proximo", "resto"):
        raise ValueError(f"grupo do Eikos desconhecido: {grupo}")
    cases = eikos_cases(cfg, path=path)
    kept = [c for c in cases if c.split == "train"]
    proximo = [c for c in kept if c.family in EIKOS_PROXIMO]
    if grupo == "proximo":
        keep = {c.id for c in proximo}
    else:
        resto = sorted((c for c in kept if c.family not in EIKOS_PROXIMO),
                       key=lambda c: hashlib.sha256(f"{EIKOS_RESTO_SAL}:{c.id}".encode()).hexdigest())
        if len(resto) < len(proximo):
            raise ValueError(f"o resto tem {len(resto)} linhas, menos que as {len(proximo)} do grupo próximo")
        # Estratificado por idioma: cada idioma com o nº de linhas que tem no "proximo" (ou todas, se faltar);
        # a falta é completada na ordem do sorteio. Assim tipo de decisão e idioma não andam juntos.
        keep, alvo = set(), Counter(c.meta["lang"] for c in proximo)
        for lang, n in sorted(alvo.items()):
            keep |= {c.id for c in [c for c in resto if c.meta["lang"] == lang][:n]}
        keep |= {c.id for c in [c for c in resto if c.id not in keep][:len(proximo) - len(keep)]}
    for c in kept:
        if c.id not in keep:
            c.split = DISCARDED
            c.meta["descarte"] = "tipo: fora do grupo" if (c.family in EIKOS_PROXIMO) != (grupo == "proximo") \
                else "amostra do resto"
    return cases


_WORD = re.compile(r"\w+", re.U)


def _shingles(text: str, n: int = 13) -> set[int]:
    w = _WORD.findall(text.casefold())
    return {hash(" ".join(w[i:i + n])) for i in range(max(0, len(w) - n + 1))}


def drop_eval_overlap(train: list[Case], others: list[Case], n: int = 13) -> int:
    """Tira do treino casos cujo estado repete (texto normalizado igual, ou qualquer sequência de `n`
    palavras) um texto de dev/teste/calib ou de família reservada das outras fontes. Devolve quantos."""
    prot = [c for c in others if c.split in ("dev", "test", "calib") or c.held_out]
    exact = {" ".join(c.state.casefold().split()) for c in prot}
    grams = set().union(*(_shingles(c.state, n) for c in prot)) if prot else set()
    k = 0
    for c in train:
        if c.split != "train":
            continue
        if " ".join(c.state.casefold().split()) in exact or not grams.isdisjoint(_shingles(c.state, n)):
            c.split, c.meta["descarte"] = DISCARDED, "texto em avaliação"
            k += 1
    return k


def decisionbank_audit(cfg: dict | None = None) -> dict:
    """Contagens das exclusões do DecisionBank (`j11_generator`) e da sobreposição com todas as avaliações, para
    registro antes do E5. A sobreposição usa as mesmas fontes protegidas do treino (`PROTEGE_TAMBEM`)."""
    from .decisionbank import DB_FILTER_VERSION, REVISAO
    cfg = cfg or load_config()
    cases = load_cases("decisionbank", cfg)
    others = [c for s in list(LOADERS) + list(PROTEGE_TAMBEM["decisionbank"]) for c in load_cases(s, cfg)]
    n_overlap = drop_eval_overlap(cases, others)
    kept = [c for c in cases if c.split == "train"]
    return {
        "revisao": REVISAO, "filtro": DB_FILTER_VERSION, "linhas_j11_generator": len(cases), "mantidas": len(kept),
        "por_motivo": dict(Counter(c.meta.get("descarte") for c in cases if c.split != "train").most_common()),
        "sobreposicao_texto": n_overlap,
        "grupos_repetidos": len(kept) - len({c.group_id for c in kept}),
        "mantidas_por_familia": dict(Counter(c.family for c in kept).most_common()),
        "mantidas_por_tipo": dict(Counter(c.qtype for c in kept)),
        "mantidas_por_n_opcoes": dict(sorted(Counter(len(c.options) for c in kept).items())),
        "descartadas_por_familia": dict(Counter(c.family for c in cases if c.split != "train").most_common()),
    }


def eikos_audit(cfg: dict | None = None) -> dict:
    """Contagens das exclusões do Eikos, para registro antes do E3b."""
    cfg = cfg or load_config()
    cases = eikos_cases(cfg)
    others = [c for s in ("felhen", "telepatia", "traces") for c in LOADERS[s](cfg)]
    n_overlap = drop_eval_overlap(cases, others)
    kept = [c for c in cases if c.split == "train"]
    return {
        "revisao": SOURCES["eikos"]["revision"], "filtro": EIKOS_FILTER_VERSION, "linhas": len(cases),
        "mantidas": len(kept), "por_motivo": dict(Counter(c.meta.get("descarte") for c in cases if c.split != "train")),
        "sobreposicao_texto": n_overlap,
        "mantidas_por_idioma": dict(Counter(c.meta["lang"] for c in kept)),
        "mantidas_por_familia": dict(Counter(c.family for c in kept).most_common()),
        "mantidas_por_tipo": dict(Counter(c.qtype for c in kept)),
        "descartadas_por_familia_eikos": dict(Counter(c.family for c in cases if c.split != "train").most_common()),
    }


# ---------------------------------------------------------------- provas (VestBench-Jev), só avaliação

VESTBENCH = {"repo": "danielpedrozo/vestbench-jev", "revision": "64b3a109ff713c9e38018358b9fb465338d2c91d"}  # tag v1.3
VESTBENCH_PARQUET_SHA256 = {"enem": "d4e0f263241795190ca45cf13b4ea57791eac8fd2f6e12eb2f3d19280f7ad91e"}
VESTBENCH_JSONL_SHA256 = {"enem": "84c6f33c4af354750b3781c1becfc5a09fd53b60dc4943407f4373a302c51a82"}  # saída de convert_vestbench
VESTBENCH_SALT = "vestbench-metades-v1"  # metade "explorar" x metade "confirmar" (intocada até as receitas congelarem)
# Alternativa que cita outras por posição ("A e C.", "todas as anteriores") muda de sentido ao inverter a ordem.
_POSICIONAL = re.compile(
    r"(todas|nenhuma)\s+(as|das)\s+(alternativas|anteriores|op[çc][õo]es|acima)"
    r"|^\s*\(?[a-e]\)?\s*(e|,)\s*\(?[a-e]\)?\s*\.?\s*$|\b(acima|anteriores)\s*\.?\s*$", re.I)


def convert_vestbench(exam: str) -> Path:
    """Parquet do VestBench-Jev (revisão fixada) -> JSONL só com as questões válidas. Roda no PC (pyarrow);
    o cluster lê o JSONL. Confere o sha256 do parquet baixado."""
    import pyarrow.parquet as pq

    src = DATA_DIR / "vestbench" / "raw" / f"{exam}.parquet"
    if sha256_file(src) != VESTBENCH_PARQUET_SHA256[exam]:
        raise ValueError(f"{src} não é o parquet da revisão fixada")
    out = DATA_DIR / "vestbench" / f"{exam}.jsonl"
    rows = [r for r in pq.read_table(src).to_pylist() if not r["excluded"]]
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "exam": r["exam"], "edition": r["exam_edition"], "year": r["year"],
                                "state": r["state"], "instructions": r["instructions"],
                                "options": [[o["label"], o["text"]] for o in r["options"]], "answer": r["answer"],
                                "has_latex": r["has_latex"]}, ensure_ascii=False) + "\n")
    return out


def vestbench_cases(cfg: dict, exam: str = "enem") -> list[Case]:
    """Questões válidas de uma prova como casos de AVALIAÇÃO (held_out; nunca treinam). Split `explorar` ou
    `confirmar` por hash do id; alternativas com referência posicional viram `descartado` (motivo registrado)."""
    path = DATA_DIR / "vestbench" / f"{exam}.jsonl"
    if sha256_file(path) != VESTBENCH_JSONL_SHA256[exam]:
        raise ValueError(f"{path} não corresponde à conversão fixada (sha256)")
    cases = []
    for r in read_jsonl(path):
        pos = any(_POSICIONAL.search(txt) for _, txt in r["options"])
        split = "explorar" if bucket(r["id"], VESTBENCH_SALT) < 0.5 else "confirmar"
        meta = {"edicao": r["edition"], "ano": r["year"], "latex": r["has_latex"], "fonte": VESTBENCH["revision"]}
        if pos:
            meta["descarte"] = "alternativa com referência posicional"
        cases.append(Case(
            id=f"vestbench:{r['id']}", group_id=f"vestbench:{r['id']}", source=f"vestbench_{exam}", family=exam,
            split=DISCARDED if pos else split, held_out=True, qtype="choice", state=r["state"],
            question=r["instructions"], options=[tuple(o) for o in r["options"]], gold=str(r["answer"]), meta=meta,
        ))
    return cases


def exportar_enem_bench(saida: Path) -> dict:
    """ENEM do VestBench no formato do Lumi-Benchmark, para o guarda-corpo do E6 em `cot` (analises/e6_protocolo.md).
    Uma linha por questão, na ordem canônica das alternativas, com `split` = `explorar` (seleção) ou `confirmar`
    (confirmação final). As descartadas (alternativa com referência posicional) ficam fora, como na avaliação."""
    rows = []
    for c in vestbench_cases(load_config(), "enem"):
        if c.split not in ("explorar", "confirmar"):
            continue
        letras = [chr(65 + i) for i in range(len(c.options))]
        rows.append({"id": c.id, "dataset": "enem_vestbench", "sector": "enem", "task": "mcq",
                     "question": f"{c.state}\n\n{c.question}", "choices": {l: t for l, (_, t) in zip(letras, c.options)},
                     "answer": letras[[o[0] for o in c.options].index(c.gold)], "group_id": c.id, "split": c.split})
    _write_jsonl(saida, rows)
    return {"explorar": sum(r["split"] == "explorar" for r in rows),
            "confirmar": sum(r["split"] == "confirmar" for r in rows), "saida": str(saida),
            "sha256": sha256_file(saida)}


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


def _f3_cases(cfg: dict) -> list[Case]:
    from .f3 import f3_cases
    return f3_cases(cfg)


def _res2_cases(cfg: dict) -> list[Case]:
    from .res2 import res2_cases
    return res2_cases(cfg)


LOADERS = {"felhen": felhen_cases, "telepatia": telepatia_cases, "traces": traces_cases}
# F3 (piloto de diversidade): fora de LOADERS para não mudar as auditorias e os dados dos treinos já feitos.
# Reservadas v2 (analises/reservadas_v2.md): só avaliação, todas as famílias reservadas.
# Reservadas com professor (protocolo de 16/11): arquivo fixado pelo sha256. Antes do registro, só o split
# `dificuldade` abre (o filtro de dificuldade com o congelado faz parte do protocolo); dev e teste ficam fechados.
def _prof_cases(cfg: dict) -> list[Case]:
    from .prof import prof_cases
    if cfg["prof"].get("sha256"):
        return prof_cases(cfg)
    print(">> [prof] sem registro: só o split 'dificuldade' é carregado")
    return [c for c in prof_cases(cfg, antes_do_registro=True) if c.split == "dificuldade"]


NEW_SOURCES = {"f3": _f3_cases, "res2": _res2_cases, "prof": _prof_cases}
# Fontes só de treino (não entram em `lumi-dados resumo` nem na avaliação).
TRAIN_ONLY = {"eikos": eikos_cases, "eikos_pt": lambda cfg: eikos_cases(cfg, langs={"Brazilian Portuguese"}),
              "eikos_proximo": lambda cfg: eikos_tipo_cases(cfg, "proximo"),
              "eikos_resto": lambda cfg: eikos_tipo_cases(cfg, "resto"),
              "decisionbank": lambda cfg: _decisionbank_cases(cfg)}
# Fontes de treino que, além das avaliações de LOADERS, precisam ficar sem texto em comum com estas outras
# (E5: as reservadas v2 e com professor, o F3 e o ENEM). As fontes antigas não mudam (os treinos já feitos).
PROTEGE_TAMBEM = {"decisionbank": ("f3", "res2", "prof", "vestbench_enem"),
                  # E6 (27/11): o Eikos também. A auditoria do outro chat deu zero sobreposição nos 16.191 exemplos
                  # do E3b, então os dados não mudam; o R0 confere pelo `dados_sha256` igual ao do E3b.
                  "eikos": ("f3", "res2", "prof", "vestbench_enem")}


def _decisionbank_cases(cfg: dict) -> list[Case]:
    from .decisionbank import decisionbank_cases
    return decisionbank_cases(cfg)
# Fontes só de avaliação, fora do fine-tuning (provas). Não entram em LOADERS (nem nas auditorias de treino).
EVAL_ONLY = {"vestbench_enem": lambda cfg: vestbench_cases(cfg, "enem")}


def load_cases(source: str, cfg: dict | None = None) -> list[Case]:
    if source in LOADERS:
        return LOADERS[source](cfg or load_config())
    if source in TRAIN_ONLY:
        return TRAIN_ONLY[source](cfg or load_config())
    if source in EVAL_ONLY:
        return EVAL_ONLY[source](cfg or load_config())
    if source in NEW_SOURCES:
        return NEW_SOURCES[source](cfg or load_config())
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


def _sint_cli(a) -> None:
    from . import sint
    if a.cmd == "sint-pedidos":
        rows = sint.pedidos_escritor(a.inicio, a.fim, a.familias)
        _write_jsonl(REPO_ROOT / a.saida, rows)
        print(json.dumps({"pedidos": len(rows), "saida": a.saida}, ensure_ascii=False))
    elif a.cmd == "sint-candidatos":
        resp = [r for p in a.respostas for r in read_jsonl(REPO_ROOT / p)]
        casos, motivos, pedidos = sint.candidatos(resp)
        _write_jsonl(REPO_ROOT / a.saida, [c.to_dict() for c in casos])
        _write_jsonl(REPO_ROOT / a.pedidos_conferente, pedidos)
        rel = defaultdict(dict)
        for (fam, m), n in sorted(motivos.items()):
            rel[fam][m] = n
        print(json.dumps({"filtro_de_forma": rel, "casos": len(casos), "pedidos_conferente": len(pedidos)},
                         indent=2, ensure_ascii=False))
    elif a.cmd == "sint-montar":
        casos = [Case.from_dict(r) for p in a.candidatos for r in read_jsonl(REPO_ROOT / p)]
        conf = [r for p in a.conferencia for r in read_jsonl(REPO_ROOT / p)]
        final, rel = sint.montar(casos, conf)
        sha = sint.gravar(final, REPO_ROOT / a.saida)
        print(json.dumps({"conferencia": rel, "auditoria": sint.auditoria(final), "casos": len(final),
                          "arquivo": a.saida, "sha256": sha}, indent=2, ensure_ascii=False))
    elif a.cmd == "sint-controles":
        casos = [Case.from_dict(r) for p in a.casos for r in read_jsonl(REPO_ROOT / p)]
        if a.familias:
            casos = [c for c in casos if c.family in a.familias]
        out = {}
        for cond, cs in sint.controles(casos).items():
            p = REPO_ROOT / a.saida_dir / f"{cond}.jsonl"
            sint.gravar(cs, p)
            out[cond] = dict(Counter(c.family for c in cs))
        print(json.dumps(out, ensure_ascii=False))


def _prof_cli(a) -> None:
    from . import prof
    cfg = load_config()
    if a.cmd == "prof-pedidos":
        rows = prof.pedidos_escritor(cfg, a.inicio, a.fim, a.familias)
        _write_jsonl(REPO_ROOT / a.saida, rows)
        print(json.dumps({"pedidos": len(rows), "saida": a.saida}, ensure_ascii=False))
    elif a.cmd == "prof-candidatos":
        resp = [r for p in a.respostas for r in read_jsonl(REPO_ROOT / p)]
        casos, motivos, pedidos = prof.candidatos(cfg, resp)
        _write_jsonl(REPO_ROOT / a.saida, [c.to_dict() for c in casos])
        _write_jsonl(REPO_ROOT / a.pedidos_conferente, pedidos)
        rel = defaultdict(dict)
        for (fam, m), n in sorted(motivos.items()):
            rel[fam][m] = n
        print(json.dumps({"filtro_de_forma": rel, "casos": len(casos), "pedidos_conferente": len(pedidos)},
                         indent=2, ensure_ascii=False))
    elif a.cmd == "prof-montar":
        casos = [Case.from_dict(r) for p in a.candidatos for r in read_jsonl(REPO_ROOT / p)]
        conf = [r for p in a.conferencia for r in read_jsonl(REPO_ROOT / p)]
        final, rel = prof.montar(cfg, casos, conf)
        sha = prof.gravar_casos(final, REPO_ROOT / a.saida)
        completo = all(r["faltam"] == 0 for r in rel.values())
        print(json.dumps({"conferencia": rel, "casos": len(final), "arquivo": a.saida, "sha256": sha,
                          "completo": completo}, indent=2, ensure_ascii=False))
        if not completo:
            print(">> ATENÇÃO: faltam grupos; o arquivo não pode ser registrado (rode outra rodada de geração)")
    elif a.cmd == "prof-corrigir-revisao":
        gab = json.loads((REPO_ROOT / a.planilha).with_name(Path(a.planilha).stem + "_gabarito.json")
                         .read_text(encoding="utf-8"))
        rel = prof.corrigir_revisao((REPO_ROOT / a.planilha).read_text(encoding="utf-8"), gab)
        (REPO_ROOT / a.saida).write_text(json.dumps(rel, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({f: {k: (len(v) if k == "grupos_errados" else v) for k, v in r.items()}
                          for f, r in rel.items()}, indent=2, ensure_ascii=False))
        print(f">> correção em {a.saida}; depois da adjudicação, salve o resultado final em "
              "data/prof/revisao_adjudicada.json")
    elif a.cmd == "prof-registrar":
        casos = prof.prof_cases(cfg, REPO_ROOT / a.arquivo, antes_do_registro=True)
        adj = json.loads((REPO_ROOT / a.adjudicacao).read_text(encoding="utf-8"))
        probs = prof.checar_registro(cfg, casos, adj, prof.prof_audit(cfg, casos), prof.prof_sobreposicao(cfg, casos))
        if probs:
            print(">> NÃO registrar:\n  - " + "\n  - ".join(probs))
            raise SystemExit(1)
        print(f'>> pode registrar: em configs/splits.toml [prof], sha256 = "{prof.sha256_arquivo(REPO_ROOT / a.arquivo)}"')
    elif a.cmd == "auditar-prof":
        casos = prof.prof_cases(cfg, REPO_ROOT / a.arquivo if a.arquivo else None, antes_do_registro=True)
        rep = prof.prof_audit(cfg, casos)
        if not a.sem_sobreposicao:
            rep["sobreposicao_com_treino_e_calib"] = prof.prof_sobreposicao(cfg, casos)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    elif a.cmd == "prof-revisao":
        casos = prof.prof_cases(cfg, REPO_ROOT / a.arquivo, antes_do_registro=True)
        if a.familias:  # emenda 4: a revisão pode sair por partes (postura e correferência antes das v2)
            cfg = {**cfg, "prof": {**cfg["prof"], "familias": a.familias}}
        md, gab = prof.revisao_humana(cfg, casos)
        saida = REPO_ROOT / a.saida
        saida.parent.mkdir(parents=True, exist_ok=True)
        saida.write_text(md, encoding="utf-8")
        saida.with_name(saida.stem + "_gabarito.json").write_text(json.dumps(gab, indent=1), encoding="utf-8")
        print(json.dumps({"itens": len(gab), "saida": a.saida}, ensure_ascii=False))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="lumi-dados")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baixar")
    b.add_argument("fontes", nargs="*", default=None)
    r = sub.add_parser("resumo")
    r.add_argument("fontes", nargs="*", default=list(LOADERS))
    sub.add_parser("auditar-eikos")
    sub.add_parser("auditar-decisionbank")
    sub.add_parser("baixar-f3")
    sub.add_parser("auditar-f3")
    sub.add_parser("auditar-res2")
    pp = sub.add_parser("prof-pedidos")
    pp.add_argument("--inicio", type=int, default=0)
    pp.add_argument("--fim", type=int, default=1000)
    pp.add_argument("--saida", default="data/prof/pedidos_escritor.jsonl")
    pp.add_argument("--familias", nargs="+", default=None, help="só estas famílias (padrão: todas)")
    pc = sub.add_parser("prof-candidatos")
    pc.add_argument("--respostas", nargs="+", default=["data/prof/respostas_escritor.jsonl"])
    pc.add_argument("--saida", default="data/prof/candidatos.jsonl")
    pc.add_argument("--pedidos-conferente", default="data/prof/pedidos_conferente.jsonl")
    pm = sub.add_parser("prof-montar")
    pm.add_argument("--candidatos", nargs="+", default=["data/prof/candidatos.jsonl"])
    pm.add_argument("--conferencia", nargs="+", default=["data/prof/respostas_conferente.jsonl"])
    pm.add_argument("--saida", default="data/prof/prof_v1.jsonl")
    ap = sub.add_parser("auditar-prof")
    ap.add_argument("--arquivo", default=None, help="sem isto, o arquivo registrado no splits.toml")
    ap.add_argument("--sem-sobreposicao", action="store_true")
    pr = sub.add_parser("prof-revisao")
    pr.add_argument("--arquivo", default="data/prof/prof_v1.jsonl")
    pr.add_argument("--saida", default="data/prof/revisao_humana.md")
    pr.add_argument("--familias", nargs="+", default=None, help="só estas famílias (padrão: todas)")
    pq = sub.add_parser("prof-corrigir-revisao")
    pq.add_argument("--planilha", default="data/prof/revisao_humana.md")
    pq.add_argument("--saida", default="data/prof/revisao_correcao.json")
    pg = sub.add_parser("prof-registrar", help="confere o protocolo e imprime o sha256 para o splits.toml")
    pg.add_argument("--arquivo", default="data/prof/prof_v1.jsonl")
    pg.add_argument("--adjudicacao", default="data/prof/revisao_adjudicada.json")
    # E5 (analises/e5_dados.md): sintético em PT-BR para treino, com os papéis trocados (o Gemma escreve).
    sp = sub.add_parser("sint-pedidos")
    sp.add_argument("--inicio", type=int, default=0)
    sp.add_argument("--fim", type=int, default=100)
    sp.add_argument("--saida", default="data/sint/pedidos_escritor.jsonl")
    sp.add_argument("--familias", nargs="+", default=None)
    sc = sub.add_parser("sint-candidatos")
    sc.add_argument("--respostas", nargs="+", default=["data/sint/respostas_escritor.jsonl"])
    sc.add_argument("--saida", default="data/sint/candidatos.jsonl")
    sc.add_argument("--pedidos-conferente", default="data/sint/pedidos_conferente.jsonl")
    sm = sub.add_parser("sint-montar")
    sm.add_argument("--candidatos", nargs="+", default=["data/sint/candidatos.jsonl"])
    sm.add_argument("--conferencia", nargs="+", default=["data/sint/respostas_conferente.jsonl"])
    sm.add_argument("--saida", default="data/sint/sint_pt.jsonl")
    sk = sub.add_parser("sint-controles")  # variantes para o Lumi congelado: completo, sem_texto, so_versao_2
    sk.add_argument("--casos", nargs="+", required=True)
    sk.add_argument("--familias", nargs="+")
    sk.add_argument("--saida-dir", required=True)
    ee = sub.add_parser("exportar-enem-bench")  # E6: ENEM do VestBench no formato do Lumi-Benchmark (guarda-corpo)
    ee.add_argument("--saida", default="data/vestbench/enem_bench.jsonl")
    cv =sub.add_parser("converter-vestbench")
    cv.add_argument("prova", default="enem", nargs="?")
    a = p.parse_args(argv)
    if a.cmd == "baixar":
        download(a.fontes or None)
    elif a.cmd == "exportar-enem-bench":
        print(json.dumps(exportar_enem_bench(REPO_ROOT / a.saida), ensure_ascii=False))
    elif a.cmd == "converter-vestbench":
        print(convert_vestbench(a.prova))
    elif a.cmd == "baixar-f3":
        from .f3 import baixar_f3
        print(json.dumps(baixar_f3(), indent=2))
    elif a.cmd == "auditar-f3":
        from .f3 import f3_audit
        print(json.dumps(f3_audit(load_config()), indent=2, ensure_ascii=False))
    elif a.cmd == "auditar-res2":
        from .res2 import res2_audit, res2_sobreposicao
        cfg = load_config()
        rep = res2_audit(cfg)
        rep["sobreposicao_com_treino_e_calib"] = res2_sobreposicao(cfg)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    elif a.cmd.startswith("prof-") or a.cmd == "auditar-prof":
        _prof_cli(a)
    elif a.cmd.startswith("sint-"):
        _sint_cli(a)
    elif a.cmd == "auditar-decisionbank":
        print(json.dumps(decisionbank_audit(), indent=2, ensure_ascii=False))
    elif a.cmd == "auditar-eikos":
        print(json.dumps(eikos_audit(), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(summary(a.fontes), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
