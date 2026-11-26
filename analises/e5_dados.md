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
