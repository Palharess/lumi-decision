"""F3: famílias novas de decisão para o piloto de diversidade (analises/f3_fontes.md, plano_fase2.md).

Três fontes com licença permissiva, baixadas de URLs fixadas e conferidas por sha256:
- PorSimplesSent (CC BY 4.0, PT-BR nativo): `pss_mais_simples`, qual de duas frases é mais fácil de ler.
- MASSIVE 1.1, locale pt-PT (CC BY 4.0, localizado do inglês): `massive_intencao`, a intenção de um pedido a
  um assistente de voz, entre as intenções do mesmo cenário (2 a 9 opções).
- GoEmotions traduzido para PT (Apache-2.0, tradução automática): `goemo_emocao`, a emoção predominante em 7
  classes (Ekman + neutro), com rótulo suave tirado dos votos dos anotadores.

Os dados brutos ficam em `data/f3/raw` (fora do git). A conversão é Python puro (csv/json) e roda em qualquer
lugar; nada aqui executa script de terceiros.
"""

from __future__ import annotations

import csv
import json
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from .data import DATA_DIR, DISCARDED, bucket, drop_eval_overlap, sha256_file, text_group
from .schema import Case

RAW = DATA_DIR / "f3" / "raw"

_PSS = "https://raw.githubusercontent.com/sidleal/porsimplessent/c0b7bb6ccda6b40ebd7f5524b08a5699b2266ffe/pss"
_GOEMO = ("https://huggingface.co/datasets/antoniomenezes/go_emotions_ptbr/resolve/"
          "635f1f6b24ac85800a7c0b94c644fd888a5c8e4e")
F3_FILES = {  # arquivo local -> (url fixada, sha256)
    "pss/pss2_align_length_ori_nat.tsv": (f"{_PSS}/pss2_align_length_ori_nat.tsv",
                                          "883ae60658aef8bfb961b558f7c744aa977beddc9861bf6aa066b0982d71cace"),
    "pss/pss2_align_length_nat_str.tsv": (f"{_PSS}/pss2_align_length_nat_str.tsv",
                                          "6a1408f79d594dbe2d76a1b61153ecde0485327c36568c521b477f5ac22e3d47"),
    "pss/pss2_align_length_ori_str.tsv": (f"{_PSS}/pss2_align_length_ori_str.tsv",
                                          "46d2e1a8f26d6d5ac1cd8b8abd40908e03849383d8cc3b94e15fc372988663e9"),
    "goemo/goemotions_1_pt.csv": (f"{_GOEMO}/goemotions_1_pt.csv",
                                  "0bb9f86b6f88eeda65700783f46e085ddd22df742d19e11a9c843ed379ee0ebc"),
    "goemo/goemotions_2_pt.csv": (f"{_GOEMO}/goemotions_2_pt.csv",
                                  "1a552e1576af5b0e60ebf861af367aed4d8179fb10a27eb4a8cc318f1cc6f4bb"),
    "goemo/goemotions_3_pt.csv": (f"{_GOEMO}/goemotions_3_pt.csv",
                                  "2e19d234b7f8e98f44ef37240d488c13f8948abdbe2644880fcfd139f8fd50e8"),
    "massive/amazon-massive-dataset-1.1.tar.gz": (
        "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
        "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"),
}
MASSIVE_PT_SHA256 = "af89ce740b98e8fe73a0fb3ba69a296837910f037e954b87409ef41d467d3da0"  # 1.1/data/pt-PT.jsonl

F3_FAMILIES = ("pss_mais_simples", "massive_intencao", "goemo_emocao")


def baixar_f3() -> list[dict]:
    """Baixa os brutos das URLs fixadas, confere o sha256 e extrai só o pt-PT do MASSIVE."""
    out = []
    for rel, (url, sha) in F3_FILES.items():
        dest = RAW / rel
        if not dest.exists() or sha256_file(dest) != sha:
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, dest)
        got = sha256_file(dest)
        if got != sha:
            raise ValueError(f"{rel}: sha256 {got} != {sha}")
        out.append({"arquivo": rel, "sha256": got, "bytes": dest.stat().st_size})
    tgz = RAW / "massive" / "amazon-massive-dataset-1.1.tar.gz"
    pt = RAW / "massive" / "pt-PT.jsonl"
    if not pt.exists():
        with tarfile.open(tgz) as tf:
            member = tf.getmember("1.1/data/pt-PT.jsonl")
            pt.write_bytes(tf.extractfile(member).read())
    if sha256_file(pt) != MASSIVE_PT_SHA256:
        raise ValueError("pt-PT.jsonl do MASSIVE não confere")
    return out


def _check(rel: str) -> Path:
    p = RAW / rel
    if sha256_file(p) != F3_FILES[rel][1]:
        raise ValueError(f"{p} não é o arquivo fixado; rode `lumi-dados baixar-f3`")
    return p


# ------------------------------------------------------------------ PorSimplesSent

PSS_QUESTION = ("Qual das duas frases é mais simples, isto é, mais fácil de entender para um leitor com pouca "
                "escolaridade?")
PSS_OPTIONS = [("1", "A frase 1 é mais simples."), ("2", "A frase 2 é mais simples.")]


def pss_rows() -> list[dict]:
    """Pares (original, simplificada) em que a frase mudou. O grupo é o texto de origem (production_id)."""
    rows = []
    for name in ("ori_nat", "nat_str", "ori_str"):
        path = _check(f"pss/pss2_align_length_{name}.tsv")
        with path.open(encoding="utf-8", newline="") as f:
            for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
                a, b = (r["sentence_text_from"] or "").strip(), (r["sentence_text_to"] or "").strip()
                if r["changed"] != "S" or not a or not b or a == b:
                    continue
                rows.append({"key": f"{name}:{i}", "prod": r["production_id"], "nivel": r["level"],
                             "complexa": a, "simples": b})
    return rows


# ------------------------------------------------------------------ MASSIVE (pt-PT)

MASSIVE_QUESTION = "Qual é a intenção deste pedido feito a um assistente de voz?"
# Cenários fora: `transport` (pedidos de viagem: mesma tarefa da família reservada `viagens` do Traces),
# `general` (a classe `quirky` não tem definição) e `cooking` (na prática uma intenção só).
MASSIVE_EXCLUDED_SCENARIOS = ("transport", "general", "cooking")
MASSIVE_INTENTS = {
    "alarm_query": "Consultar os alarmes que já estão definidos.",
    "alarm_remove": "Apagar ou cancelar um alarme.",
    "alarm_set": "Criar um alarme novo.",
    "audio_volume_down": "Diminuir o volume.",
    "audio_volume_mute": "Silenciar o som.",
    "audio_volume_other": "Outro ajuste de volume, sem dizer se é para aumentar ou diminuir.",
    "audio_volume_up": "Aumentar o volume.",
    "calendar_query": "Consultar compromissos ou eventos da agenda.",
    "calendar_remove": "Apagar ou cancelar um compromisso da agenda.",
    "calendar_set": "Marcar um compromisso ou lembrete na agenda.",
    "datetime_convert": "Converter horário ou data entre fusos ou formatos.",
    "datetime_query": "Perguntar a hora ou a data.",
    "email_addcontact": "Adicionar um contato à lista de contatos.",
    "email_query": "Consultar e-mails recebidos.",
    "email_querycontact": "Consultar dados de um contato.",
    "email_sendemail": "Escrever ou enviar um e-mail.",
    "iot_cleaning": "Ligar o aspirador ou robô de limpeza.",
    "iot_coffee": "Fazer café na cafeteira inteligente.",
    "iot_hue_lightchange": "Mudar a cor das luzes.",
    "iot_hue_lightdim": "Diminuir a intensidade das luzes.",
    "iot_hue_lightoff": "Apagar as luzes.",
    "iot_hue_lighton": "Acender as luzes.",
    "iot_hue_lightup": "Aumentar a intensidade das luzes.",
    "iot_wemo_off": "Desligar uma tomada inteligente.",
    "iot_wemo_on": "Ligar uma tomada inteligente.",
    "lists_createoradd": "Criar uma lista ou acrescentar um item a ela.",
    "lists_query": "Consultar uma lista ou o que há nela.",
    "lists_remove": "Apagar uma lista ou tirar um item dela.",
    "music_dislikeness": "Dizer que não gosta da música ou do artista.",
    "music_likeness": "Dizer que gosta de uma música, artista ou gênero.",
    "music_query": "Perguntar sobre a música que está tocando (nome, artista, banda).",
    "music_settings": "Mudar o modo de reprodução (repetir, aleatório).",
    "news_query": "Pedir notícias.",
    "play_audiobook": "Tocar ou continuar um audiolivro.",
    "play_game": "Jogar um jogo.",
    "play_music": "Tocar uma música, artista ou playlist.",
    "play_podcasts": "Tocar ou navegar em um podcast.",
    "play_radio": "Tocar uma estação de rádio.",
    "qa_currency": "Perguntar sobre moedas ou câmbio.",
    "qa_definition": "Pedir a definição ou o significado de algo.",
    "qa_factoid": "Fazer uma pergunta factual de conhecimento geral.",
    "qa_maths": "Fazer uma conta ou pergunta de matemática.",
    "qa_stock": "Perguntar sobre ações ou a bolsa de valores.",
    "recommendation_events": "Pedir sugestões de eventos.",
    "recommendation_locations": "Pedir sugestões de lugares (restaurantes, lojas).",
    "recommendation_movies": "Pedir sugestões de filmes.",
    "social_post": "Publicar ou reclamar em uma rede social.",
    "social_query": "Consultar novidades em redes sociais.",
    "takeaway_order": "Pedir comida para entrega ou para levar.",
    "takeaway_query": "Consultar opções ou o andamento de um pedido de comida.",
    "weather_query": "Perguntar sobre o tempo ou a previsão.",
}


def massive_rows() -> list[dict]:
    path = RAW / "massive" / "pt-PT.jsonl"
    if sha256_file(path) != MASSIVE_PT_SHA256:
        raise ValueError(f"{path} não confere; rode `lumi-dados baixar-f3`")
    rows = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if r["scenario"] in MASSIVE_EXCLUDED_SCENARIOS:
            continue
        rows.append({"key": r["id"], "cenario": r["scenario"], "intencao": r["intent"], "texto": r["utt"].strip(),
                     "particao": r["partition"]})
    return rows


# ------------------------------------------------------------------ GoEmotions (PT)

GOEMO_QUESTION = "Qual emoção predomina neste comentário?"
EKMAN = {
    "raiva": ("anger", "annoyance", "disapproval"),
    "nojo": ("disgust",),
    "medo": ("fear", "nervousness"),
    "alegria": ("joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride",
                "admiration", "desire", "caring"),
    "tristeza": ("sadness", "disappointment", "embarrassment", "grief", "remorse"),
    "surpresa": ("surprise", "realization", "confusion", "curiosity"),
    "neutro": ("neutral",),
}
GOEMO_OPTIONS = [
    ("raiva", "Raiva, irritação ou desaprovação."),
    ("nojo", "Nojo ou repulsa."),
    ("medo", "Medo ou nervosismo."),
    ("alegria", "Alegria, gratidão, carinho, admiração ou aprovação."),
    ("tristeza", "Tristeza, decepção, vergonha ou remorso."),
    ("surpresa", "Surpresa, confusão ou curiosidade."),
    ("neutro", "Nenhuma emoção clara (neutro)."),
]
_TO_EKMAN = {fine: ek for ek, fines in EKMAN.items() for fine in fines}


def goemo_rows(min_raters: int = 3) -> list[dict]:
    """Um texto por linha, com a distribuição dos votos (cada anotador divide 1 voto entre as suas classes).
    Fica só o texto com pelo menos `min_raters` anotadores e uma classe que ganha com folga (>= 2 votos inteiros
    e estritamente acima da segunda)."""
    votes, text = defaultdict(list), {}
    for i in (1, 2, 3):
        path = _check(f"goemo/goemotions_{i}_pt.csv")
        with path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["example_very_unclear"] == "True":
                    continue
                labs = {_TO_EKMAN[k] for k in _TO_EKMAN if r.get(k) == "1"}
                if labs:
                    votes[r["id"]].append(labs)
                    text[r["id"]] = (r["texto"] or "").strip()
    rows = []
    for k, vs in votes.items():
        if len(vs) < min_raters or not text[k]:
            continue
        dist, whole = Counter(), Counter()
        for labs in vs:
            for lab in labs:
                dist[lab] += 1 / len(labs)
                whole[lab] += 1
        (top, n1), *rest = dist.most_common()
        if whole[top] < 2 or (rest and rest[0][1] >= n1):
            continue
        z = sum(dist.values())
        rows.append({"key": k, "texto": text[k], "gold": top, "dist": {e: v / z for e, v in dist.items()}})
    return rows


# ------------------------------------------------------------------ casos

def _split_seen(group: str, conf: dict) -> str:
    u = bucket(group, conf["salt"])
    return "dev" if u < conf["dev_frac"] else "calib" if u < conf["dev_frac"] + conf["calib_frac"] else "train"


def _split_held(group: str, conf: dict) -> str:
    """Família reservada: metade dev (exploração), metade teste (lida uma vez, com a receita congelada)."""
    return "dev" if bucket(group, conf["salt"]) < 0.5 else "test"


def f3_cases(cfg: dict, protect: list[Case] | None = None) -> list[Case]:
    """Casos das três famílias. `protect`: casos de avaliação das outras fontes (padrão: Felhen, Telepatia,
    Traces e ENEM); treino que repete texto deles vira `descartado`."""
    conf = cfg["f3"]
    held = set(conf["held_out_families"])
    split = lambda fam, g: _split_held(g, conf) if fam in held else _split_seen(g, conf)  # noqa: E731
    cases = []
    fam = "pss_mais_simples"
    for r in pss_rows():
        g = f"f3:pss:{r['prod']}"
        first_simple = bucket(f"pss-ordem:{r['key']}", conf["salt"]) < 0.5
        f1, f2 = (r["simples"], r["complexa"]) if first_simple else (r["complexa"], r["simples"])
        cases.append(Case(
            id=f"f3:{fam}:{r['key']}", group_id=g, source="f3", family=fam, split=split(fam, g), held_out=fam in held,
            qtype="choice", state=f"Frase 1: {f1}\nFrase 2: {f2}", question=PSS_QUESTION, options=list(PSS_OPTIONS),
            gold="1" if first_simple else "2", meta={"nivel": r["nivel"], "texto_origem": r["prod"]}))
    fam = "massive_intencao"
    by_scen = defaultdict(set)
    rows = massive_rows()
    for r in rows:
        by_scen[r["cenario"]].add(r["intencao"])
    for r in rows:
        if len(by_scen[r["cenario"]]) < 2:  # `news` e `weather` têm uma intenção só: não há decisão
            continue
        g = text_group("f3", r["texto"])
        menu = sorted(by_scen[r["cenario"]])
        cases.append(Case(
            id=f"f3:{fam}:{r['key']}", group_id=g, source="f3", family=fam, split=split(fam, g), held_out=fam in held,
            qtype="choice", state=r["texto"], question=MASSIVE_QUESTION,
            options=[(i, MASSIVE_INTENTS[i]) for i in menu], gold=r["intencao"],
            meta={"cenario": r["cenario"], "particao_original": r["particao"]}))
    fam = "goemo_emocao"
    for r in goemo_rows():
        g = text_group("f3", r["texto"])
        cases.append(Case(
            id=f"f3:{fam}:{r['key']}", group_id=g, source="f3", family=fam, split=split(fam, g), held_out=fam in held,
            qtype="choice", state=r["texto"], question=GOEMO_QUESTION, options=list(GOEMO_OPTIONS), gold=r["gold"],
            gold_dist=r["dist"]))
    # o mesmo texto em dois cenários/famílias: o grupo manda; um grupo em mais de um split vira descarte
    seen = defaultdict(set)
    for c in cases:
        seen[c.group_id].add(c.split)
    for c in cases:
        if len(seen[c.group_id]) > 1:
            c.split, c.meta["descarte"] = DISCARDED, "grupo em mais de um split"
    if protect is None:
        from .data import LOADERS, vestbench_cases
        protect = [c for s in ("felhen", "telepatia", "traces") for c in LOADERS[s](cfg)] + vestbench_cases(cfg)
    drop_eval_overlap(cases, protect)
    return cases


def f3_audit(cfg: dict) -> dict:
    """Contagens para registro antes do piloto: por família e split, descartes, rótulos e o cruzamento dos
    textos das famílias reservadas novas com o TREINO das fontes atuais (que não pode existir)."""
    from .data import LOADERS, _shingles, vestbench_cases
    others = {s: LOADERS[s](cfg) for s in ("felhen", "telepatia", "traces")}
    protect = [c for cs in others.values() for c in cs] + vestbench_cases(cfg)
    cases = f3_cases(cfg, protect)
    held_text = [c for c in cases if c.held_out and c.split != DISCARDED]
    grams = set().union(*(_shingles(c.state) for c in held_text)) if held_text else set()
    exact = {" ".join(c.state.casefold().split()) for c in held_text}
    cruz = {s: sum(1 for c in cs if c.split in ("train", "calib") and not c.held_out and (
        " ".join(c.state.casefold().split()) in exact or not grams.isdisjoint(_shingles(c.state))))
        for s, cs in others.items()}
    return {
        "reservadas": sorted(cfg["f3"]["held_out_families"]),
        "por_familia_split": {f"{f}|{s}": n for (f, s), n in sorted(Counter((c.family, c.split) for c in cases).items())},
        "descartes": {f"{f}|{m}": n for (f, m), n in Counter((c.family, c.meta.get("descarte"))
                                                              for c in cases if c.split == DISCARDED).items()},
        "gabarito_por_familia": {f: dict(Counter(c.gold for c in cases if c.family == f and c.split != DISCARDED)
                                         .most_common()) for f in F3_FAMILIES},
        "opcoes_por_familia": {f: dict(sorted(Counter(len(c.options) for c in cases if c.family == f).items()))
                               for f in F3_FAMILIES},
        "treino_atual_com_texto_das_reservadas_novas": cruz,
    }
