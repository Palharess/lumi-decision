"""Formato comum de um caso de decisão, independente da fonte."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Case:
    id: str
    group_id: str  # itens que precisam ficar sempre no mesmo split (ex.: perguntas do mesmo estado)
    source: str  # felhen | telepatia | traces | arquivo
    family: str  # unidade de generalização: tarefa, workflow ou tipo de ferramenta
    split: str  # train | dev | calib | test
    held_out: bool  # família reservada: nunca entra em treino, seleção ou calibração
    qtype: str  # choice | noul | score
    state: str  # contexto mostrado ao modelo
    question: str
    options: list[tuple[str, str]]  # (id, descrição) na ordem canônica
    gold: str  # id da opção correta
    gold_dist: dict[str, float] | None = None  # distribuição de referência, quando existe
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["options"] = [list(o) for o in self.options]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Case":
        d = dict(d)
        d["options"] = [tuple(o) for o in d["options"]]
        return cls(**d)
