# F3: fontes para o piloto de diversidade (14/11/2026)

Protocolo do piloto em `plano_fase2.md` (seção F3). Código em `src/lumi_decision/f3.py`
(`lumi-dados baixar-f3`, `lumi-dados auditar-f3`). Nada daqui entrou em treino ainda.

## O que a mistura já cobre (contado no dev)

| Fonte | Famílias vistas | Tipos de decisão |
|---|---|---|
| Felhen | `camara_tema` (24 opções), `scielo_area` (8), `olid_ofensivo` (2), `olid_alvo` (3), `faquad_resposta` (2) | tema/área, ofensa, resposta correta |
| Telepatia | `customer_service`, `security_incidents`, `agent_trace_observability` | escolha (4–5), sim/não, nota (4–5) |
| Traces | 9 domínios de agente | escolha da ferramenta (5) |

**Reservadas atuais:** `factck_veracidade` (veracidade, 3), `juristcu_area` (área jurídica, 10),
`invoice_processing` (faturas); ENEM só na avaliação.

**Faltam:** comparação entre dois textos, emoção, intenção, inferência, similaridade graduada, postura. Quase tudo
que temos é "a que categoria isto pertence".

## Licenças conferidas na origem (não no espelho)

| Dataset | Tipo de decisão | Licença na origem | Situação |
|---|---|---|---|
| PorSimplesSent | qual frase é mais simples | **CC BY 4.0** (README do repositório) | **entra** (PT-BR nativo) |
| MASSIVE 1.1, pt-PT | intenção de um pedido | **CC BY 4.0** (arquivo LICENSE) | **entra** (português europeu, localizado do inglês) |
| GoEmotions PT | emoção | **Apache-2.0** (card) | **entra** (tradução automática do inglês) |
| ToLD-Br | toxicidade | CC BY-SA 4.0 | fora: o tipo já existe (`olid_*`) |
| B2W-Reviews01 | nota de avaliação | **CC BY-NC-SA 4.0** na origem; o espelho no HF diz CC BY 4.0, errado | fora: não comercial |
| HateBR | discurso de ódio | só pesquisa acadêmica | fora |
| ASSIN 2 | inferência, similaridade | não declarada (HF e TFDS: "unknown") | fora até alguém confirmar com os autores |
| ReLi-SA, TweetSentBR, RePro, BPSAD | sentimento, opinião | não declarada, ou NC-SA (RePro) | fora |

**Excluídas por tarefa (iguais às reservadas ou à avaliação):** FaQuAD-NLI (mesmos textos do `faquad_resposta`),
Fake.br e afins (`factck`), classificação jurídica por área (`juristcu`), provas como BLUEX e OAB (ENEM).

**Achado:** dataset de decisão em PT-BR **nativo e com licença permissiva** é raro. Das três fontes que entram,
só o PorSimplesSent é PT-BR nativo; as outras duas vêm do inglês. Isso pesa a favor de gerar famílias sintéticas
em PT-BR com um professor (como no *Large Decision Models*) como o passo seguinte ao piloto.

## As três famílias novas

| Família | Fonte | Pergunta | Opções | Casos | Uso no piloto |
|---|---|---|---|---|---|
| `pss_mais_simples` | PorSimplesSent | qual das duas frases é mais fácil de entender | 2 | dev 2.063, teste 2.873 | **reservada nova** |
| `massive_intencao` | MASSIVE pt-PT | qual a intenção do pedido ao assistente | 2–9 (as do mesmo cenário) | treino 10.319, calib 1.273, dev 1.271 | vista |
| `goemo_emocao` | GoEmotions PT | qual emoção predomina | 7, com rótulo suave | treino 41.220, calib 5.066, dev 5.319 | vista |

**Decisões de construção:**
- **PorSimplesSent:** pares em que a frase mudou (original → natural, natural → forte, original → forte).
  - Grupo = texto de origem, então as frases de um mesmo texto ficam todas no mesmo split.
  - A frase simples cai na posição 1 ou 2 por sorteio fixo: 2.439 contra 2.497.
- **MASSIVE:** o menu são as intenções do mesmo cenário. Cenários fora:
  - `transport`: pedidos de viagem, a mesma tarefa da família reservada `viagens` do Traces;
  - `general`: a classe `quirky` não tem definição;
  - `cooking`, `news` e `weather`: uma intenção só, então não há decisão.
- **GoEmotions:** as 27 emoções viram as 6 de Ekman mais neutro.
  - Ficam os textos com pelo menos 3 anotadores e uma classe que ganha com folga.
  - O rótulo suave é a divisão dos votos.
  - A classe "alegria" domina: 21.368 de 51.605.

## Auditoria (registrada antes de qualquer treino)

- **Sobreposição de texto** com dev, teste, calib e reservadas de Felhen, Telepatia e Traces, e com o ENEM (texto
  exato ou 13 palavras seguidas): 1 texto do GoEmotions descartado; nada no MASSIVE nem no PorSimplesSent.
- **O treino atual nunca viu texto da reservada nova:** 0 linhas de treino ou calib do Felhen, do Telepatia e do
  Traces repetem texto do PorSimplesSent.
- **`f3` fica fora de `LOADERS`:** os dados e as auditorias dos treinos já feitos (E3, E3b, F2) não mudam.
- **Atalho na reservada nova:** escolher a frase mais curta acerta **3.936/4.936** (80%).
  - Original → forte: 992/1.084.
  - Natural → forte: 1.197/1.485.
  - Original → natural: 1.747/2.367.
  - Por isso a `pss_mais_simples` também será lida nos casos em que a simples **não** é a mais curta
    (1.000 pares), registrado agora como leitura secundária.
- **Linha de base de classe majoritária (dev):**
  - `massive_intencao`: 587/1.271, com a intenção mais comum do cenário;
  - `goemo_emocao`: 2.230/5.319, sempre "alegria".

## Desenho do piloto (proposta; detalhar no protocolo antes de rodar)

- **Controle:** a mistura do F2.1, com Felhen 1.300 por família, Telepatia e Traces inteiros e 508
  atualizações.
- **Braço diverso:** troca metade dos exemplos do Felhen (3.250) por `massive_intencao` e `goemo_emocao` (1.625
  cada). As 508 atualizações e o nº de exemplos vistos ficam iguais.
- **Tokens:** os textos novos são curtos (pedidos e comentários), então o braço diverso vê **menos** tokens que
  o controle. Não dá para casar atualizações e tokens ao mesmo tempo. Casamos as atualizações e reportamos os
  tokens com `scripts/contar_orcamento.py`. O viés é contra a diversidade, ou seja, conservador.
- **Precisa de código:** teto por fonte (hoje o teto é um só para todas as famílias).
- **Leituras:**
  - principal: as 3 reservadas atuais, IC nas duas sementes;
  - secundárias: `pss_mais_simples` (toda e só os pares em que a simples não é a mais curta), as vistas novas e
    as antigas.

Fontes: [PorSimplesSent](https://github.com/sidleal/porsimplessent),
[MASSIVE](https://github.com/alexa/massive), [GoEmotions PT](https://huggingface.co/datasets/antoniomenezes/go_emotions_ptbr),
[B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01), [ToLD-Br](https://github.com/JAugusto97/ToLD-Br),
[ASSIN 2 no HF](https://huggingface.co/datasets/nilc-nlp/assin2), [Napolab](https://huggingface.co/datasets/ruanchaves/napolab).
