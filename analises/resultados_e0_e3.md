# Lumi Decision: resultados dos experimentos E0–E4 (11–13/11/2026)

Tudo medido no dev, com uma semente de treino. O teste oficial não foi usado para nada.

## 1. A ideia

Damos ao Lumi-4B um contexto, uma pergunta e opções rotuladas A, B, C… No ponto exato em que ele escreveria a
resposta, lemos a probabilidade de cada letra. Essa é a **decisão**: uma distribuição sobre as opções, numa
passada só, sem gerar texto.

## 2. Como medimos

- **Duas ordens.** Cada caso é lido com as opções na ordem original e na invertida. A métrica principal é a
  média das acurácias nas duas ordens. As opções foram invertidas, não embaralhadas.
- **Famílias vistas × reservadas.** As vistas são tipos de tarefa que entram no treino, com outros exemplos.
  As reservadas são tipos de tarefa que o treino nunca mostra: `factck_veracidade` (checagem de fatos),
  `juristcu_area` (área de acórdãos do TCU) e `invoice_processing` (faturas). Só as reservadas dizem se o modelo
  **generaliza**.
- **Intervalo de confiança (IC 95%).** Os dois modelos são comparados nos mesmos casos, reamostrando grupos
  inteiros de casos que compartilham o mesmo texto. Uma diferença só conta se o intervalo inteiro ficar de um
  lado do zero.
- **Critério antes do resultado.** Cada hipótese tem a regra de decisão escrita no README antes de rodar.

## 3. Resumo

| Exp. | Pergunta | Resultado | O que aprendemos |
|---|---|---|---|
| E0 | Estamos lendo o modelo do jeito certo? | ✅ Sim, com a implementação nativa do `transformers` | A leitura pelas letras é fiel ao modelo |
| E1 (H1) | O Lumi congelado já decide bem? | Em parte | Decide bem acima do acaso, mas muda de resposta com a ordem das opções em 18–32% dos casos |
| E2 | O treino LoRA está montado certo? | ✅ Sim, depois de corrigir um bug de cache | Sem `use_cache=False`, o Mamba treina pelo caminho lento |
| E3 (H2) | LoRA melhora em tarefas novas? | **Inconclusivo**: +2,7 [−0,3; +5,9] | Ganha muito no que vê (+15 a +20 pontos) e quase elimina o viés de ordem do Felhen; transfere pouco |
| E3b | Um corpus externo (Eikos) ajuda a generalizar? | **Inconclusivo**: −0,8 [−3,1; +1,3] contra o controle | Mais dados variados, com o mesmo orçamento, não melhoraram a transferência |
| E4 (H3) | Uma pointer head lê melhor que as letras? | **Piora**: −4,4 [−7,2; −1,6] | Uma cabeça nova, sem conhecimento prévio, generaliza pior que o LM head pré-treinado |

**Escolha:** Lumi-4B + LoRA lido pelas letras (receita do E3), com temperatura e limiar de abstenção ajustados
no `calib`.

## 4. Viés de ordem: quanto cada modelo muda de ideia ao inverter as opções

O mesmo caso, com as mesmas opções em outra ordem, deveria receber a mesma resposta. A tabela mostra em quantos
casos a escolha **mudou** ao inverter as opções (`scripts/analise_ordem.py`, saída em
`analises/ordem_resumo.json`):

| Modelo | Felhen vistas | Felhen reservadas | Telepatia vistas | Telepatia reservadas | Reservadas (média das 3 famílias) |
|---|---|---|---|---|---|
| Nemotron base congelado | 17% | 23% | 21% | 30% | 27% |
| **Lumi congelado (E1)** | **18%** | **32%** | **22%** | **26%** | **30%** |
| **LoRA + letras (E3)** | **3%** | **12%** | **14%** | **28%** | **17%** |
| Controle do E3b (treino mais longo) | 2% | 14% | 10% | 23% | 17% |
| LoRA + Eikos (E3b) | 4% | 13% | 11% | 14% | 14% |
| LoRA + pointer head (E4) | 2% | 20% | 10% | 9% | 22% |

**Do Lumi congelado para o E3**, diferença na taxa de mudança (IC 95%, média por família):

| Grupo | E3 − congelado |
|---|---|
| Felhen vistas | −14,8 [−16,8; −12,8] |
| Felhen reservadas | −19,8 [−25,5; −14,1] |
| Telepatia vistas | −7,4 [−12,4; −2,1] |
| Telepatia reservadas (`invoice_processing`) | +1,9 [−0,4; +4,3] |
| **Reservadas, três famílias** | **−12,6 [−16,6; −8,7]** |

E a fração de casos **acertados nas duas ordens** (certo e estável ao mesmo tempo):

| Modelo | Felhen vistas | Felhen reservadas | Telepatia vistas | Telepatia reservadas |
|---|---|---|---|---|
| Lumi congelado (E1) | 65% | 40% | 43% | 46% |
| LoRA + letras (E3) | 86% | 47% | 65% | 49% |
| LoRA + pointer head (E4) | 87% | 43% | 70% | 58% |

**Como ler:**
- **O congelado não erra muito mais numa ordem do que na outra; ele erra casos diferentes.** A acurácia média
  quase não muda ao inverter (Felhen vistas 0,72 contra 0,73), mas um quarto a um terço das respostas troca.
  Só 40–65% dos casos saem certos nas duas ordens. A exceção é Felhen reservadas, onde a acurácia cai de 0,55
  para 0,46 ao inverter. O Nemotron base tem o mesmo padrão, então isso é próprio de ler as letras de um LLM
  congelado, não um defeito do Lumi.
- **O LoRA mitiga o viés onde ele aprende.** No treino a ordem das opções é sorteada a cada exemplo, então a
  posição deixa de ser pista e o modelo passa a decidir pelo conteúdo (é uma hipótese, mas consistente com os
  números). No Felhen o viés cai de 18% para 3% e transfere para as duas famílias reservadas do Felhen (factck
  31% → 11%, juristcu 32% → 13%).
- **Em `invoice_processing` o E3 não mudou nada** (26% → 28%). O E3b (14%) e o E4 (9%) reduzem o viés ali, mas
  sem ganho de acurácia que se sustente.
- **Estabilidade não é qualidade.** Um modelo pode ficar estável por sempre dar a mesma resposta errada: o E3b
  quase nunca muda de ideia em `duplicate` e erra 83%. E o E4 piora a estabilidade justamente em factck (11% →
  45%), onde também piora a acurácia. Por isso os critérios olham acurácia primeiro.

## 5. Os experimentos, um a um

### E0: a leitura é fiel ao modelo? ✅ (job 36805)

Conferimos que os pesos carregam sem faltas, que o template de conversa é o oficial, que o estado lido é
exatamente a entrada do LM head e que o preenchimento de um lote não vaza para os outros exemplos. A versão
`remote code` do modelo quebra ao gerar texto com transformers 5.x, então usamos a implementação nativa.

### E1: o Lumi congelado já decide? (H1) (jobs 36807/36808)

| Grupo | Lumi | Nemotron base | Lumi − Nemotron (IC 95%) |
|---|---|---|---|
| Felhen vistas | 0,723 | 0,718 | empate |
| Felhen reservadas | 0,501 | 0,524 | −2,2 [−5,3; +1,0] |
| Telepatia vistas | 0,520 | 0,552 | −3,2 [−6,0; −0,2] |
| Telepatia reservadas | 0,557 | 0,389 | **+16,9 [+15,1; +18,6]** |

**Resultado:** o Lumi congelado decide bem acima do acaso e no nível do Nemotron base, com uma vantagem grande em
faturas. **Problemas:**
- o viés de ordem (seção 4);
- a abstenção funciona nas famílias vistas (responde 16% dos casos do Felhen com 4,5% de erro), mas não
  transfere para as reservadas;
- um TF-IDF supervisionado acerta mais nas vistas do Felhen (0,818 contra 0,716).

**Lição:** o modelo já tem a habilidade; falta consistência.

### E2: o treino está montado certo? ✅ (job 36812)

**O que conferimos:**
- o LoRA entra nas 71 camadas certas;
- os gradientes chegam a todas elas;
- o modelo decora 64 exemplos (70% → 100%);
- a leitura no treino é idêntica à da avaliação;
- o adaptador recarrega igual.

**Bug encontrado:** o `transformers` liga o cache por padrão, e com o cache o Mamba deixa de usar o kernel
rápido de treino (0 de 21 camadas). O treino passou a usar `use_cache=False`. A parada perto do limite de tempo
e a retomada também foram testadas, e corrigimos mais dois bugs nelas: o sinal do Slurm não chegava ao Python, e
o estado do gerador aleatório precisa ficar na CPU.

**Lição:** validar o caminho de treino antes de gastar GPU evitou um E3 inteiro treinado pelo caminho errado.

### E3: LoRA melhora em tarefas novas? (H2) (job 36852)

**Receita:** LoRA r=16, uma época sobre 20.350 exemplos (Felhen, Typed Decisions PT e Traces, só famílias
vistas), ordem das opções sorteada por exemplo e alvo suave quando há distribuição de rótulos. São 508
atualizações e 37 minutos numa H100.

| | Lumi congelado | LoRA | Ganho (IC 95%) |
|---|---|---|---|
| Felhen vistas | 0,723 | 0,869 | +14,7 [+12,8; +16,6] |
| Telepatia vistas | 0,520 | 0,723 | +20,2 [+15,7; +25,0] |
| **H2: reservadas (média das 3 famílias)** | | | **+2,7 [−0,3; +5,9] → inconclusivo** |

Por família reservada: factck +2,9; juristcu +0,2; invoice +4,8 [+3,0; +6,8]. Nas vistas do Felhen, na ordem
original, o LoRA passa o TF-IDF (0,874 contra 0,818; +5,7 [+3,8; +7,6]). A abstenção fica útil nas vistas: no
Felhen o modelo responde 90% dos casos com 8,6% de erro, mas ela continua sem transferir para as reservadas.

**Por quê:** o LoRA aprende o formato e as famílias que vê. O ganho nas famílias novas é pequeno: o que ele
aprende é em boa parte específico das tarefas do treino.

**Lição:** o LoRA é claramente útil para as tarefas do cliente que entram no treino; para tarefas inéditas, o
ganho existe na média mas não foi demonstrado.

### E3b: um corpus externo ajuda a generalizar? (jobs 36857/36858)

**Desenho:**
- **Braço E3b:** o E3 com mais 16.191 decisões do corpus público Eikos, depois de tirar tudo que se parecia com
  as famílias reservadas (2.886 itens de fatura, por exemplo).
- **Controle:** só os dados do E3, com o mesmo número de atualizações (891) e o mesmo cronograma.

| Comparação nas reservadas | Δ (IC 95%) |
|---|---|
| **E3b − controle (principal)** | **−0,8 [−3,1; +1,3] → inconclusivo** |
| E3b − Lumi congelado | +1,3 [−1,8; +4,4] |

A queda do E3b em invoice (−4,3) veio de uma pergunta só, `duplicate`, em que ele passou a dizer "sim" em 557 de
600 casos (detalhes no apêndice).

**Por quê:** o Eikos é 87% em inglês e tem tarefas muito diferentes das nossas; trocar parte da repetição dos
dados originais por ele não trouxe transferência.

**Lição:** mais variedade de dados não é atalho para generalizar. O controle com o mesmo orçamento foi
essencial: sem ele, um ganho poderia vir só de treinar mais tempo.

### E4: uma pointer head lê melhor que as letras? (H3) (jobs 36884–36886)

**Desenho:** a mesma receita, os mesmos dados e as mesmas 508 atualizações do E3; só muda a leitura. Uma cabeça
nova, no molde do kev (projeções 3136 → 256), compara o estado do fim de cada opção com o da última posição.

| Critério nas reservadas | E4 − E3 (IC 95%) | factck | juristcu | invoice | Regra |
|---|---|---|---|---|---|
| **Qualidade (principal)** | **−4,4 [−7,2; −1,6]** | −17,6 | +3,8 | +0,7 | **piora** |
| Estabilidade | +4,7 [+0,6; +8,7] (mais mudanças) | | | | **piora** |

Nas vistas o E4 empata (Felhen +0,4) ou ganha pouco (Telepatia +2,6 [+0,5; +4,6]). Em factck, na ordem invertida,
ele escolhe "impreciso" em 66 de 102 casos, quando o gabarito tem 24.

**Por quê (hipótese):** a cabeça nova começa do zero e aprende a ler os estados das opções nas famílias vistas.
Em factck as opções se definem umas pelas outras, e numa passada causal o estado de cada opção carrega as
anteriores. Ao trocar as vizinhas, a nota de "impreciso" sobe em 99 de 102 casos, mesmo com a opção na mesma
posição. As letras usam o LM head pré-treinado, que já sabe ler a lista inteira.

**Lição:** com o mesmo orçamento de treino, trocar a leitura pelas letras por uma cabeça nova custa
generalização. A leitura pelas letras aproveita o que o modelo já sabe.

## 6. O que aprendemos

1. **O Lumi já sabe decidir; o problema é consistência.** Congelado, ele muda de resposta com a ordem das
   opções em até um terço dos casos.
2. **O LoRA com ordem sorteada no treino resolve boa parte disso** e dá +15 a +20 pontos nas tarefas vistas. No
   Felhen o viés de ordem cai para 3% e a queda transfere para as famílias reservadas do Felhen.
3. **Generalizar para tarefas inéditas é o ponto fraco.** Nenhuma das três receitas (E3, E3b, E4) mostrou ganho
   demonstrado nas famílias reservadas.
4. **Mais dados (Eikos) e outra cabeça (pointer head) não ajudaram**, e a pointer head piorou.
5. **Estabilidade e qualidade precisam ser medidas juntas.** Um modelo estável pode estar estavelmente errado.
6. **Validação e critério antes de rodar evitam erros caros.** O E2 achou o bug de cache antes do treino, e o
   critério fixado impediu "salvar" o E3b tirando `duplicate` da conta.

## 7. Limites e próximos passos

- **Uma semente de treino.** Os intervalos medem a variação entre casos, não entre treinos. Replicar o E3 com
  mais sementes diria quanto dos resultados, principalmente em `duplicate` e factck, é oscilação.
- **Só duas ordens.** Avaliar mais permutações daria uma medida mais completa do viés.
- **Teste oficial:** mede a receita escolhida uma única vez, depois de congelada.

## Apêndice: detalhes

**`invoice_processing` por pergunta** (acurácia / acurácia balanceada, 600 linhas cada):

| Pergunta | Congelado | E3 | Controle | E3b | E4 |
|---|---|---|---|---|---|
| urgency | 0,44 / 0,43 | 0,61 / 0,51 | 0,62 / 0,52 | 0,66 / 0,57 | 0,62 / 0,48 |
| matches_order | 0,58 / 0,59 | 0,71 / 0,71 | 0,70 / 0,70 | 0,81 / 0,81 | 0,60 / 0,61 |
| disposition | 0,39 / 0,25 | 0,45 / 0,28 | 0,44 / 0,29 | 0,47 / 0,30 | 0,48 / 0,31 |
| discrepancy_severity | 0,48 / 0,33 | 0,52 / 0,34 | 0,52 / 0,35 | 0,59 / 0,40 | 0,50 / 0,33 |
| duplicate | 0,90 / 0,50 | 0,75 / 0,63 | 0,61 / 0,61 | 0,17 / 0,54 | 0,86 / 0,50 |
| *"sim" em `duplicate` (56 corretos)* | 3 | 150 | 245 | 557 | 32 |

`duplicate` é a pergunta mais instável. Só 56 das 600 linhas são duplicatas, então dizer sempre "não" já dá 90% de
acurácia sem detectar nada (é o que o congelado faz). A capacidade de separar duplicatas (AUROC) é 0,82 no E3,
0,74 no controle, 0,73 no E3b e 0,66 no E4. Entre o controle e o E3b a AUROC quase não muda, mas as
probabilidades se deslocam para "sim" e cruzam o 0,5.

**Abstenção** (limiar fixado no `calib` para 10% de erro, aplicado ao dev):

| Grupo | Congelado: responde / erra | E3: responde / erra | Erro sem abster (E3) |
|---|---|---|---|
| Felhen vistas | 16% / 4,5% | 90% / 8,6% | 13% |
| Telepatia vistas | 13% / 11% | 36% / 10,9% | 28% |
| Felhen reservadas | 7% / 23% | 38% / 28,5% | 49% |
| Telepatia reservadas | 17% / 31% | 38% / 35,0% | 39% |

**Eikos (E3b):**
- **Revisão:** `cebb4a26`, sem os documentos do FinEntity; 19.239 linhas, das quais ficaram 16.191.
- **Exclusões:** por semelhança com as reservadas, saíram 2.886 itens de fatura, 157 de checagem de fatos e 5 do
  TCU. Nenhum texto se sobrepõe à avaliação.
- **O que o controle controla:** os dois braços têm 891 atualizações e o mesmo cronograma. O E3b vê 13% mais
  tokens e 43% menos repetições dos exemplos originais.

**Dados e código:**
- predições em `runs/` (no cluster e localmente, fora do git);
- critérios em `lumi-eval h2` e `lumi-eval h3`;
- viés de ordem em `scripts/analise_ordem.py`;
- análise do E1 em `analises/e1_resumo.txt`;
- auditoria do Eikos em `analises/eikos_auditoria.json`.
