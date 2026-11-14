# F3: levantamento de fontes para o piloto de diversidade (iniciado em 14/11/2026)

Protocolo do piloto em `plano_fase2.md` (seção F3). Esta nota lista candidatas; nada aqui entrou no treino.

## O que a mistura já cobre (contado no dev)

| Fonte | Famílias vistas | Tipos de decisão |
|---|---|---|
| Felhen | `camara_tema` (24 opções), `scielo_area` (8), `olid_ofensivo` (2), `olid_alvo` (3), `faquad_resposta` (2) | tema/área, ofensa, resposta correta |
| Telepatia | `customer_service`, `security_incidents`, `agent_trace_observability` | escolha (4–5), sim/não, nota (4–5) |
| Traces | 9 domínios de agente | escolha da ferramenta (5) |

**Reservadas atuais:** `factck_veracidade` (veracidade, 3), `juristcu_area` (área jurídica, 10),
`invoice_processing` (escolha, sim/não e nota sobre faturas); ENEM (VestBench-Jev) só na avaliação.

**Faltam:** inferência (acarreta / contradiz), similaridade graduada, sentimento e nota de avaliação, postura
(a favor / contra), nível de complexidade do texto, intenção. Quase tudo que temos é "a que categoria isto
pertence".

## Candidatas (PT-BR)

| Dataset | Tipo de decisão novo | Formato provável | Licença (a conferir no card fixado) | Risco de sobreposição |
|---|---|---|---|---|
| ASSIN 2 | inferência (RTE) e similaridade 1–5 (STS) | `noul` 2 e `score` 5 | não confirmada | baixo |
| PorSimplesSent | nível de simplificação | `choice` 3 | não confirmada | baixo |
| B2W-Reviews01 | nota da avaliação 1–5; recomendaria? | `score` 5 e `noul` 2 | CC BY 4.0 no espelho `ruanchaves/b2w-reviews01` (conferir a original) | baixo; domínio de e-commerce, perto do Traces visto, longe de `invoice` |
| TweetSentBR | sentimento | `choice` 3 | não encontrada | baixo |
| UstanceBR | postura sobre um alvo | `choice` 2–3 | não confirmada | baixo |
| UlyssesSD | postura sobre projetos de lei | `choice` | não confirmada | **médio:** domínio legislativo, igual ao `camara_tema` visto; auditar contra `juristcu` |
| ToLD-Br | categorias de toxicidade | `choice`/`noul` | CC BY-SA 4.0 | tipo já coberto pelo `olid_*`: pouca diversidade nova |
| HateBR | discurso de ódio | `noul` | **só pesquisa acadêmica, uso comercial proibido** | tipo já coberto; licença incompatível com um modelo publicado: fora |

**Excluídas desde já:**
- **FaQuAD-NLI:** os mesmos textos do `faquad_resposta` que já está no Felhen.
- **Fake.br, FactNews e afins:** a mesma tarefa de `factck_veracidade` (reservada). A auditoria é por tarefa, como
  no filtro do Eikos, não só por texto.
- **Classificação jurídica por área (STF, TCU):** a mesma tarefa de `juristcu_area`.
- **Provas (BLUEX, OAB e afins):** sobrepõem a avaliação nas provas (VestBench-Jev).

## Próximos passos da auditoria

1. Para cada candidata: card no Hugging Face, licença, revisão fixada e sha256, como no Eikos e no VestBench.
2. Converter para `Case` (estado, pergunta, opções com descrição), com rótulos suaves quando houver mais de um
   anotador (ASSIN 2 tem a média das notas de similaridade).
3. `drop_eval_overlap` contra todas as fontes de avaliação (texto exato e shingles de 13 palavras), mais a
   checagem por tarefa acima.
4. **Escolher e registrar as famílias reservadas novas antes do treino** (por exemplo, uma de inferência e uma de
   postura). Elas não entram no braço diverso.
5. Montar o braço diverso com o mesmo número de atualizações e de tokens do controle (mistura do F2.1), trocando
   parte do orçamento do Felhen. Conferir as contagens com `scripts/contar_orcamento.py` antes de submeter.

Fontes da busca: [Napolab](https://huggingface.co/datasets/ruanchaves/napolab/blob/main/README.md),
[B2W-Reviews01](https://huggingface.co/datasets/ruanchaves/b2w-reviews01/blob/main/README.md),
[ToLD-Br](https://github.com/JAugusto97/ToLD-Br), [UstanceBR](https://arxiv.org/abs/2312.06374).
