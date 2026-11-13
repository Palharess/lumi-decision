# Resultados E0–E3 (11–13/11/2026)

O que testamos, o que deu e o que isso quer dizer. Números do dev; o teste não foi tocado.

## A ideia em uma frase

Damos ao Lumi-4B um contexto, uma pergunta e opções rotuladas A, B, C…; no ponto em que ele escreveria a
resposta, lemos a probabilidade de cada letra. Isso é a **decisão**: uma distribuição sobre as opções, numa
única passada, sem gerar texto.

## Como medir sem se enganar

- **Média das duas ordens.** Cada caso é avaliado com as opções na ordem original e invertida, e a acurácia é a
  média das duas. Só a ordem original engana: o Lumi congelado acerta 54,7% das reservadas do Felhen na ordem
  original e 45,5% na invertida.
- **Famílias vistas × reservadas.** Vistas: tipos de tarefa que entram no treino (com outros exemplos).
  Reservadas: tipos de tarefa que o modelo nunca vê no treino (`factck_veracidade`, `juristcu_area`,
  `invoice_processing`). Só as reservadas dizem se ele *generaliza*.
- **IC 95% por bootstrap pareado por grupo.** Comparamos os dois modelos nos mesmos casos e reamostramos grupos
  de casos (o mesmo texto aparece em várias perguntas). Ganho só conta se o intervalo inteiro ficar acima de zero.
- **Abstenção.** O modelo pode não responder quando a confiança é baixa. O limiar é fixado no split `calib` para
  no máximo 10% de erro nas respostas aceitas e depois aplicado, sem ajuste, ao dev.

## E0 — o modelo está sendo lido certo? (job 36805) ✅

Antes de medir qualquer coisa, conferimos que a leitura é fiel ao modelo:

- os pesos carregam sem faltas e o template de conversa é o oficial;
- o estado que usamos é exatamente a entrada da última camada (LM head);
- o conteúdo do preenchimento (padding) de um lote não vaza para os outros exemplos.

Usamos a implementação nativa do `transformers`. A versão `remote code` do modelo quebra ao gerar texto com
transformers 5.x.

## E1 — o Lumi congelado já decide? (H1) (jobs 36807/36808)

| Grupo | Lumi | Nemotron base | Lumi − Nemotron (IC 95%) |
|---|---|---|---|
| Felhen vistas | 0,723 | 0,718 | empate |
| Felhen reservadas | 0,501 | 0,524 | −2,2 [−5,3; +1,0] |
| Telepatia vistas | 0,520 | 0,552 | −3,2 [−6,0; −0,2] |
| Telepatia reservadas | 0,557 | 0,389 | **+16,9 [+15,1; +18,6]** |
| Traces vistas | 0,851 | 0,824 | — |

**Leitura:** congelado, o Lumi decide bem acima do acaso e no nível do Nemotron base, com a exceção grande de
`invoice_processing`, onde fica bem à frente. Três problemas apareceram:

1. **Viés de ordem.** A decisão muda quando a ordem das opções muda em 18–32% dos casos.
2. **Abstenção não transfere.** Nas vistas do Felhen o limiar funciona: o modelo responde 16% dos casos e erra
   4,5% deles, contra 28% de erro se aceitasse a mesma fração ao acaso. Nas reservadas o limiar não segura o erro.
3. **Um baseline simples ganha.** Nas vistas do Felhen, um TF-IDF supervisionado acerta mais que o Lumi
   (0,818 contra 0,716 na ordem original).

## E2 — o treino LoRA está montado certo? (job 36812) ✅

O E2 confere o treino antes de gastar GPU com ele:

- **Alvos:** o LoRA entra nas 71 camadas esperadas e não entra no `out_proj` do Mamba, que não recebe gradiente
  no kernel rápido.
- **Gradientes:** todos são finitos e não nulos depois do primeiro passo.
- **Capacidade de aprender:** o modelo decora 64 exemplos (acurácia de 70% para 100%).
- **Coerência:** a leitura usada no treino é idêntica à da avaliação, e o adaptador salvo e recarregado dá o
  mesmo resultado.
- **Bug encontrado:** o `transformers` liga o cache por padrão, e com o cache ligado o Mamba abandona o kernel
  rápido de treino (0 de 21 camadas). Com `use_cache=False`, passam 21 de 21.
- **Memória:** com 8192 tokens por micro-lote o pico é de 35 GB. Com 16384 vai a 72,8 GB de 80 sem ficar mais
  rápido, por isso o padrão é 8192.

Testamos também a parada e a retomada: o Slurm avisa perto do limite de tempo, o treino salva e sai, e o mesmo
comando retoma do checkpoint (jobs 36849 → 36851). Nisso apareceram dois bugs, ambos corrigidos:

- o Apptainer não repassa o sinal do Slurm ao Python, e agora o pedido de parada vai por um arquivo `PARAR`;
- o checkpoint era carregado na GPU, e o estado do gerador aleatório precisa ficar na CPU.

## E3 — LoRA melhora a decisão em tarefas novas? (H2) (job 36852)

**Receita.** LoRA r=16, uma época sobre 20.350 exemplos de Felhen, Typed Decisions PT e Traces, só famílias
vistas, no máximo 4 mil por família. A ordem das opções é sorteada por exemplo e o alvo é suave quando há
distribuição de rótulos. São 508 passos e 37 minutos numa H100, incluindo a avaliação.

**Critério da H2 (fixado antes de treinar):** média, com peso igual, do ganho nas três famílias reservadas.

| Família reservada | LoRA − congelado |
|---|---|
| factck_veracidade | +2,9 |
| juristcu_area | +0,2 |
| invoice_processing | +4,8 [+3,0; +6,8] |
| **H2** | **+2,7 [−0,3; +5,9] → inconclusivo** |

| Famílias vistas | Congelado | LoRA | Ganho (IC 95%) |
|---|---|---|---|
| Felhen | 0,723 | 0,869 | +14,7 [+12,8; +16,6] |
| Telepatia | 0,520 | 0,723 | +20,2 [+15,7; +25,0] |
| Traces | 0,851 | 0,878 | +2,7 [−2,7; +8,1] |

Nas vistas do Felhen, na ordem original, o LoRA acerta 0,874 e passa o TF-IDF (0,818): +5,7 [+3,8; +7,6]
nos mesmos 1.498 casos.

**`invoice_processing` por pergunta (ordem e casos pareados, 600 linhas cada):**

| Pergunta | Acurácia antes → depois | Acurácia balanceada antes → depois |
|---|---|---|
| urgency | 0,442 → 0,607 | 0,425 → 0,514 |
| matches_order | 0,583 → 0,712 | 0,586 → 0,713 |
| disposition | 0,385 → 0,448 | 0,254 → 0,284 |
| discrepancy_severity | 0,475 → 0,515 | 0,326 → 0,338 |
| duplicate | 0,902 → 0,747 | 0,497 → 0,628 |

A queda em `duplicate` é efeito da classe majoritária. Só 56 de 600 linhas são duplicatas. O modelo congelado
respondia "não" em 597 delas, o que dá 90% de acurácia sem detectar nada. O LoRA marca 150 como duplicata: passa
a detectar, com mais falsos positivos. Em acurácia balanceada, as cinco perguntas melhoram. O critério da H2
continua sendo a acurácia, como fixado; a balanceada entra como diagnóstico.

**Viés de ordem (% de casos em que a decisão muda com a ordem):**

| Grupo | Antes | Depois |
|---|---|---|
| Felhen vistas | 18% | 3% |
| Felhen reservadas | 32% | 12% |
| Telepatia reservadas | 26% | 28% |

**Abstenção (limiar do `calib`, meta de 10% de erro):**

| Grupo | Responde | Erra entre as aceitas | Erro ao acaso |
|---|---|---|---|
| Felhen vistas | 90% (era 16%) | 8,6% | 13% |
| Telepatia vistas | 36% | 10,9% | 28% |
| Felhen reservadas | 38% | 28,5% | 49% |
| Telepatia reservadas | 38% | 35,0% | 39% |

## O que concluímos

1. **O LoRA aprende muito bem o que vê.** Ganha de 15 a 20 pontos nas famílias vistas, passa o TF-IDF e torna a
   abstenção útil: no Felhen o modelo responde 90% dos casos dentro da meta de erro.
2. **O viés de ordem do Felhen quase some**, inclusive nas famílias reservadas. No Telepatia reservadas, não muda.
3. **A transferência para tarefas novas é pequena e incerta.** O ganho é de +2,7 pontos, mas o IC encosta no
   zero, então pela regra fixada a H2 é **inconclusiva**. A abstenção continua não transferindo.
4. **Cuidado com o próximo passo.** Testar variações no mesmo dev até a H2 "passar" anula o critério. Toda
   variante nova (E3b, E4) é um teste novo, com critério escrito antes de rodar.

## Próximos experimentos

- **E3b (ablação de dados).** Mesma receita, acrescentando o corpus público Eikos (decisões com distribuições de
  um professor). Pergunta se a diversidade de tarefas melhora a transferência para as reservadas. Fatos já
  conferidos e cuidados:
  - **Revisão:** a revisão sem FinEntity (commit `cebb4a26` na `main`, vindo do PR `770d6bdf`) tem 19.239
    linhas em `core/train`.
  - **Idioma:** 85% das linhas são em inglês (16.398) e 2.841 em PT-BR. Isso muda o idioma do treino, além da
    variedade.
  - **Sobreposição com faturas:** o Eikos tem decisões sobre faturas (590 linhas citam *invoice* na instrução;
    411 estão em tópicos de contas a pagar e faturas; há casos de pagamento duplicado e de pedido de compra). Elas
    precisam sair do treino, senão `invoice_processing` deixa de ser uma família inédita.
  - **Sem sobreposição evidente:** com `juristcu_area` não há menção a TCU. A família `fact` do Eikos checa fatos
    *dentro do texto*, não a veracidade de notícias como `factck_veracidade`.
  - **Controle de duração:** o E3 consumiu 8,11 milhões de tokens (8,74 milhões com padding) em 508
    atualizações. O cronograma de lr é derivado do total de atualizações, então somar o Eikos numa época quase
    dobra o treino. Por isso o E3b precisa de um braço de controle só com os dados originais, no mesmo número
    de atualizações e com o mesmo cronograma.
  - **Critério:** comparação principal E3b − controle nas três reservadas; E3b − congelado como secundária.
    Tudo escrito antes de rodar.
- **E4 (H3).** Uma pointer head no lugar da leitura pelas letras, contra o LoRA do E3 com os mesmos dados.

Dados brutos: `runs/e1_*`, `runs/e3_lora_r16*` no cluster; análise do E1 em `analises/e1_resumo.txt`.
