# E5: dados novos de decisão (rascunho de 21/11/2026, ainda não registrado)

Este documento é o plano, não o registro. O critério do E5 é fixado aqui antes de qualquer treino, mas só
depois de registradas as reservadas com professor (`analises/reservadas_v2.md`, protocolo de 16/11).

## Por que dados

- A forma de leitura e a capacidade não mudaram o resultado: o pointer piorou (E4, F1), o residual empatou
  (F2.3a) e o `in_proj` empatou (F2.2).
- Somar o Eikos ajudou nas 4 reservadas programáticas, +3,42 e +3,41 (E3b e E3b-r).
- No E3c, as duas metades do Eikos ajudaram contra o controle. A diferença próximo − resto (+0,8 e +1,3) é
  inconclusiva pela regra.
- Leitura defensável: a composição dos dados importa, com efeitos diferentes por tarefa. Não é "mais dados
  ajuda sempre".
  - O Eikos desloca o limiar de `duplicate` na invoice.
  - O piloto F3 (classificação traduzida, Felhen cortado) piorou o factck.

## Ordem

1. Registrar as reservadas com professor com `lumi-dados prof-registrar`, que confere:
   - 650 grupos completos por família;
   - auditoria de atalhos dentro dos limites;
   - sobreposição zero com treino e calib;
   - revisão humana adjudicada com no máximo 3 de 30 grupos errados.

   Antes do registro, só o split `dificuldade` abre.
2. Montar o corpus do E5 com o filtro de sobreposição contra as reservadas já congeladas.
3. Registrar o critério do E5 (abaixo) e só então treinar.

## Fontes candidatas

| fonte | idioma | licença | uso proposto |
|---|---|---|---|
| sintético PT (pipeline das reservadas) | PT-BR nativo | gerado por nós (Qwen3.8 e Gemma 4 são Apache-2.0) | principal |
| DecisionBank (jgeuter, v2 de 19/11) | 22 línguas, inclui PT | mista, por linha (`source_license`) | filtrado por origem e por tipo |
| Open-Jev release-v2 (só workflow e suporte) | EN | CC0 no conteúdo gerado | braço opcional |

### Sintético PT

- **Famílias diferentes das reservadas.** Ficam de fora:
  - os tipos reservados: postura, inferência, implicatura e correferência;
  - as 4 famílias do res2: restrições, regra de benefício, custo de plano e ordem de eventos.
- **Para não repetir a semelhança do mesmo escritor:**
  - inverter os papéis (Gemma-4-31B escreve, Qwen3.8-27B confere às cegas);
  - usar domínios e temas disjuntos dos `TEMAS` das reservadas;
  - usar modelos de texto (gêneros) diferentes dos das reservadas.
- **Controles:** os mesmos do protocolo de 16/11, ou seja, gabarito pela especificação antes do texto, filtro
  de forma, conferente em 2 ordens e auditoria de atalhos. Sem revisão humana completa, só uma amostra de
  controle de qualidade.

### DecisionBank

- [Card](https://huggingface.co/datasets/jgeuter/DecisionBank) e [artigo](https://arxiv.org/html/2610.17205).
- **Tamanho:** 625 mil linhas no total. O subconjunto `in_jeff_1_1` tem 261.561.
  - Partes: decider (163.732), votos humanos (9.168), escrito por LLM (60.219) e programático (28.442).
- **Filtragem:** cada linha tem `source_dataset`, `source_license`, `part`, `family` e `target_origin`, então dá
  para filtrar por origem e por tipo.
- **Cuidados antes de usar:**
  - **Tirar a parte escrita por LLM.** Ela foi gerada pelo Qwen3.8-27B, o mesmo escritor das nossas
    reservadas com professor. Também desconfiar dos rótulos `teacher`, que são a média de 2 amostras do
    Qwen3.8-27B com raciocínio.
  - **Tirar os tipos reservados.** O `mnli` (NLI) colide com `inferencia_pt`. Também postura, implicatura e
    correferência, se aparecerem entre as 98 famílias.
  - **Licenças:** lista explícita, comparada por igualdade e não por prefixo: `apache-2.0`, `mit`, `cc0-1.0`,
    `cc-by-3.0`, `cc-by-4.0`, `cc-by-sa-3.0` e `cc-by-sa-4.0`. Um padrão `cc-by-*` também pegaria `cc-by-nc-*` e
    `cc-by-nd-*` (revisão de 23/11). Ficam fora `unknown`, `other`, qualquer variante NC ou ND e as `mixed` que
    incluem termos de pesquisa.
  - **Deduplicar** contra Eikos, Open-Jev, Felhen, Telepatia, Traces, factck, juristcu, invoice, res2, prof e
    ENEM.
  - **Contar as linhas em português** e verificar se são nativas ou traduzidas. O card não diz.
- **O que o card da revisão `fbeee677` (26/10) diz, lido em 25/11:**
  - **Português.** Só a parte `j11_written` tem idioma marcado: 1.091 a 1.343 linhas por idioma fora do inglês.
    Como ela é a parte escrita pelo Qwen3.8, que já está excluída, o que sobra do DecisionBank é praticamente
    todo em inglês.
  - **Alvos.** Na parte `decider` (300 mil linhas), o alvo é 50% o rótulo do decider e 50% o Qwen3.8-27B. O campo
    `label` é o mais provável dessa mistura. Usar o rótulo original exige voltar à fonte pelo `source_ref`, ou
    ficar só com as linhas em que os dois concordam.
  - **A parte limpa é a `j11_generator`.** São 67.910 questões programáticas, rotuladas por código, com licença CC
    BY-SA 4.0. Antes de usar, conferir se alguma família colide com as reservadas do `res2`: "composite rules" e
    "counting traps" podem se parecer com `regra_beneficio` e `restricoes_oferta`.
  - **Consequência para o E5.** O DecisionBank responde a uma pergunta diferente: dado de decisão em inglês ajuda a
    decidir em português? A fonte principal de dado em PT continua sendo o sintético.
  - O inventário completo, com as contagens por parte, família, idioma, licença e origem do alvo, sai de
    `slurm/decisionbank_inventario.sbatch` em `analises/decisionbank_inventario.json`.
- **Inventário (job 38155, 25/11, revisão `fbeee677`): 624.517 linhas, 98 famílias.**
  - **Partes:** decider 300.042, `j11_written` 138.782, `j11_generator` 67.910, `human_soft` 67.646, `jeff10_llm`
    50.137. O card de 19/11 dava 625 mil no total; as contagens por parte diferem das citadas acima, que vinham
    do subconjunto `in_jeff_1_1`.
  - **Idioma:** só a `j11_written` tem idioma marcado; PT são 1.337 linhas, todas escritas pelo Qwen3.8 e, portanto,
    excluídas. Confirma que o braço DecisionBank é em inglês.
  - **Origem do alvo:** só a `j11_generator` (67.910) tem alvo "computed by code". Todo o resto mistura 50% Qwen3.8-27B
    (ou é 100% Qwen3.8, na `j11_written`).
  - **`j11_generator`, 6 famílias:** `j11_jaggedness` 19.741 (aritmética de datas e contagens), `j11_calibration_gen`
    17.998 e `j11_calibration` 4.253 (probabilidades a partir de frequências), `j11_composite_gen` 11.998 (regras
    compostas de elegibilidade), `j11_negation_pair_gen` 7.950 (qual afirmação o registro sustenta) e `j11_usecase`
    5.970.
  - **Colisões com as reservadas:**
    - `j11_composite_gen` colide com `regra_beneficio` e `restricoes_oferta` do res2 (regras com severidade aplicadas a
      um pedido). Fica fora.
    - `j11_jaggedness` pode colidir com `ordem_eventos` nas questões de datas. Precisa olhar os subtipos antes.
    - Nenhuma família de postura, implicatura ou correferência. O `mnli` (decider) colide com `inferencia_pt`, mas já
      sai por estar na parte decider.
  - **Tipos de questão:** `choice` 318.777, `noul` 171.257, `score` 134.483. Nosso treino usa `choice`; `noul` e `score`
    têm alvo de probabilidade e exigiriam outra perda.
  - **Decider com licença da lista:** 140.653 linhas em 36 famílias, mas com alvo 50% Qwen3.8. Só entraria
    voltando ao rótulo original pelo `source_ref`.
  - **Consequência:** o braço DecisionBank, se rodar, usa a `j11_generator` em `choice`, sem `j11_composite_gen` e
    com `j11_jaggedness` conferida contra `ordem_eventos`. Fica abaixo do sintético PT na prioridade.

## Desenho proposto do E5

- **Mesmo ponto de partida por semente:** o adaptador do E3b na semente 0 e na semente 1.
- **Braços:**
  - novo: treino adicional com dados novos mais replay da mistura do E3b;
  - controle: o mesmo treino adicional só com replay.

  Os dois braços têm o mesmo número de passos. Ficam registrados passos, exposições aos exemplos originais e
  tokens de conteúdo.
- **Pergunta principal:** novo − controle na média das 4 famílias com professor (dev, todas as ordens).
  - "Ajuda" se o IC > 0 nas 2 sementes; "atrapalha" se o IC < 0 nas 2; caso contrário, inconclusivo.
- **Leituras secundárias:**
  - as 4 famílias do res2;
  - as 3 reservadas originais pela regra do M1 (factck em 6 ordens);
  - as vistas, para conferir que não há regressão;
  - `duplicate` da invoice em acurácia balanceada.
- **Teste de idioma (opcional, se sobrar GPU):** os mesmos exemplos em EN e em PT, com o mesmo número de
  passos.

## Protocolo do E5, braço DecisionBank (registrado em 26/11/2026, antes de treinar)

**Pergunta:** somar dado de decisão novo, rotulado por código e em inglês, à mistura do E3b melhora a decisão em
português nas famílias que o treino nunca viu?

**Dados novos:** a parte `j11_generator` do DecisionBank (`src/lumi_decision/decisionbank.py`, filtro v1).
- **Ficam fora:**
  - a `j11_composite_gen` inteira;
  - na `j11_jaggedness`, prazos, devolução, cupons e soma de itens;
  - na `j11_negation_pair_gen`, tudo menos as afirmações sobre registros (reserva, envio, funcionário) e a
    consistência com a folha de referência.
- **Por quê:** a lista saiu dos modelos de pergunta contados no cluster em 26/11, e cada exclusão colide com uma
  família reservada:
  - `regra_beneficio` e `restricoes_oferta`: regras compostas e aplicação de política;
  - `ordem_eventos`: prazos e datas de devolução;
  - `custo_plano`: cupons;
  - `invoice`: soma de itens contra o total.
- **Sobreposição de texto:** sai do treino tudo o que repete texto das avaliações de Felhen, Telepatia e Traces,
  das reservadas v2, das reservadas com professor, do F3 e do ENEM (`PROTEGE_TAMBEM`, mesmo critério do Eikos).
- **Teto:** 4.000 por família, como no E3b. O alvo é a distribuição do próprio dado (`target_distribution`, perda
  suave).
- **Contagens:** saem de `slurm/e5_preparar.sbatch` (`analises/decisionbank_auditoria.json`) e entram aqui antes do
  treino.

**Braços** (sementes 0 e 1; cada semente parte do adaptador do E3b da mesma semente, `--init-adapter`):
- **Novo (`e5_db_s<s>`):** `--sources felhen,telepatia,traces,eikos,decisionbank`, 1 época, `--epoch-offset 1`.
  O E3b vira replay, com ordens e planos novos.
- **Controle (`e5_ctrl_s<s>`):** `--sources felhen,telepatia,traces,eikos --epochs 2 --epoch-offset 1 --max-steps N`.
  N é o nº de atualizações do braço novo na mesma semente, contado antes do treino (`analises/e5_orcamento.jsonl`).
- **Igual nos dois:** lr 1e-4, LoRA r16, lotes de 32 exemplos e leitura pelas letras.
  - A s1 usa `--data-seed 0`, como o E3b s1.
  - O otimizador e o cronograma recomeçam nos dois braços.
  - A comparação recusa rodar se os dois não tiverem o mesmo nº real de atualizações.
- **Registrar:** atualizações, exemplos vistos e tokens de conteúdo de cada braço.

**Critério:**
- **Principal:** novo − controle na média com peso igual das 4 reservadas com professor (dev, todas as
  rotações, IC com famílias fixas reamostrando grupos, como na M3).
  - "Ajuda" se o IC ficar inteiro acima de zero nas 2 sementes; "atrapalha" se ficar inteiro abaixo nas 2.
  - Caso contrário, inconclusivo.
- **Secundários:**
  - as 4 reservadas v2;
  - as 3 reservadas originais pela regra do M1 (`factck` em 6 ordens);
  - cada família;
  - as famílias vistas (dev de Felhen, Telepatia e Traces), para conferir que não há regressão;
  - `duplicate` da invoice em acurácia balanceada.
- **Diagnóstico:** novo − E3b e controle − E3b. Isso separa o efeito do dado do efeito de treinar mais.
- **Leitura:**
  - o dado é em inglês e o teste é em português, então um resultado nulo não diz que dado de decisão não ajuda;
  - o sintético PT, mais difícil, entra depois como outro braço contra o mesmo controle.

## A decidir antes do registro

- Treino adicional a partir do E3b ou mistura do zero com atualizações casadas, como no E3b. O primeiro é mais
  barato e isola o efeito do dado novo.
- Proporção entre dado novo e replay, e tamanho do corpus. Depende da taxa de aprovação medida no lote pequeno
  das reservadas.

## Piloto sintético em PT-BR (decidido em 25/11, depois da revisão do outro chat)

O E5 começa pelo sintético em PT-BR. O DecisionBank fica para um braço separado, depois do inventário.

- **Famílias do piloto:**
  - **`extracao_campo`:** numa ficha ou registro não financeiro (equipamento, patrimônio, protocolo, matrícula,
    lote), qual valor é o campo pedido. A ficha tem valores-distratores do mesmo tipo: outros códigos, outras
    datas, outros nomes. Paralelo no treino: `extraction` do Eikos. O experimento mede o valor de exemplos novos
    em PT-BR para essa habilidade.
  - **`campo_alterado`:** duas versões do mesmo registro; qual campo mudou. O registro estruturado e a mudança
    são sorteados por código, então o gabarito vem da diferença entre as versões. O escritor só escreve as duas
    versões em texto corrido, e o filtro de forma confere que cada valor aparece onde deve.
- **Ficam fora do piloto:**
  - prioridade, porque `urgency` já está no treino da Telepatia e na `invoice` reservada;
  - encaminhamento e melhor resposta, que têm paralelos no Eikos e ficam para depois.
- **Controles do lote pequeno, antes de escalar:**
  - dá para acertar sem a ficha (só pergunta e opções)?
  - dá para acertar com só uma das versões do registro?
  - os mesmos atalhos de forma do protocolo de 16/11.
- **Desenho:** como acima. Ponto de partida no E3b por semente; sintético PT mais replay contra só replay, com os
  mesmos passos e exposição a tokens comparável, registrada.
- **Papéis:** o Gemma-4-31B escreve e o Qwen3.8-27B confere às cegas. Domínios e gêneros de texto ficam disjuntos
  dos `TEMAS` das reservadas.

### Lote `_teste1` (job 38158, 26/11): índices 0–24, 72 casos

- Filtro de forma 24/24 e conferente 24/24 grupos nas duas famílias (96/96 e 48/48 respostas).
- Posição do gabarito: a mais comum é 14/48 (extração) e 8/24 (campo alterado). Classe mais comum: 6/24.
- **Leitura dos textos:**
  - `extracao_campo` está boa: documentos naturais e no gênero pedido. Os dois itens vêm sempre em blocos separados.
  - `campo_alterado` v1 tinha campos iguais em todos os domínios ("bairro de entrega" na biblioteca, "65 unidades"
    num pedido de exame). As versões eram quase listas com o rótulo do campo, e o gênero não aparecia.
- **Mudança para `campo_alterado` v2** (`SINT_VERSION` por família; a extração continua v1):
  - cada domínio tem o seu registro e de 6 a 7 campos próprios, um por tipo de valor, para que o texto dispense o
    rótulo sem criar ambiguidade;
  - cada versão tem de 3 a 5 frases em texto corrido;
  - o filtro rejeita o nome do campo usado como rótulo ("campo: valor") e códigos a mais.
- **Controles com o Lumi congelado** (`slurm/sint_dificuldade.sbatch`), nas 4 rotações:
  - completo;
  - sem o texto;
  - só com a versão 2 (no `campo_alterado`).

  Nas duas últimas o esperado é o acaso.

### Lote `_teste2` (jobs 38174 e 38175, 26/11): índices 0–100, 298 casos

- **Filtro de forma:** extração 100/100. Campo alterado 98/100: 1 caso com o nome do dado como rótulo, 1 com valor
  ausente ou repetido.
- **Conferente:** 100/100 e 98/98 grupos nas 2 ordens.
- **Posição do gabarito:** a mais comum é 57/200 e 26/98. A classe mais comum do campo alterado é 7/98.
- **Lumi congelado** (4 rotações, `analises/sint_controles_teste2.json`):

  | condição | `extracao_campo` | `campo_alterado` |
  |---|---:|---:|
  | completo | 800/800 | 352/392 (81/98 grupos 4 de 4) |
  | sem o texto | 210/800 (acaso 200) | 89/392 (acaso 98) |
  | só a versão 2 | — | 120/392 (acaso 98) |

- **Controles:** sem o texto dá o acaso nas duas famílias.
  - Só com a versão 2 fica 22 acertos acima do acaso, cerca de 1,3 erro-padrão contando grupos. Não é atalho claro.
  - O campo que muda está distribuído por igual na ordem real do texto da versão 2 (22, 12, 17, 21 e 26 nas posições
    1 a 5).
  - O modelo prefere o campo citado primeiro, o que não favorece o gabarito.
- **Textos do campo alterado v2:** bem melhores, em texto corrido e com os campos do domínio. Às vezes o escritor
  muda o sentido do campo; num caso, "data de entrega do carro" virou "aberta em".
- **Leitura:**
  - a extração v1 é fácil demais para o Lumi (800/800) e quase não dá sinal de treino;
  - o campo alterado v2 fica no limite de cima do critério das reservadas (89,8%, com 90% de teto).

  As duas precisam ficar mais difíceis antes de escalar.
