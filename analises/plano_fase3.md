# Plano revisado depois do Lumi-Benchmark (proposta v2, 27/11/2026, ainda não registrada)

Este documento revisa o plano à luz do Lumi-Benchmark (`analises/lumi_benchmark.md`) e do diagnóstico em andamento
(`analises/bench_diag.md`). Ele propõe; nada aqui é critério registrado. Cada etapa com treino ganha o seu protocolo
antes de rodar.

**O que mudou da v1 para a v2** (revisão do outro chat, 27/11, conferida aqui):
- O E5 antigo fica **suspenso**, não cancelado.
- O ENEM `explorar` tem 763 questões em 2 ordens, não 1.526 itens. O IC reamostra questões. A metade `confirmar`
  fica para a confirmação final.
- O test do Lumi-Benchmark já orientou a mudança de receita. Ele passa a ser referência de regressão, e a seleção usa
  só o dev.
- A margem de preservação é uma decisão do projeto e precisa ser fixada antes de escolher qualquer checkpoint.
- O `in_proj` fica no primeiro comparativo.
- **A âncora principal passa a ser a KL do LLM-as-Jev** (§3.4 e §4.4.1, arXiv 2610.02076), com a decisão lida
  depois de um prefixo. A âncora por autodestilação vira o braço de reserva. Motivo na seção 3.
- A leitura com prefixo ("Resposta: X") precisa de conferência de tokenização por formato. O `Encoder` atual guarda
  os ids das letras só pelo nº de opções.
- Os treinos curtos do diagnóstico não formam uma curva de dose. O job com raciocínio ligado também dobra o limite
  de geração.
- **DecisionBank:**
  - a composição por tipo fica registrada (`choice`, `noul` e `score`, como no Eikos do E3b);
  - o carregador foi corrigido (v3);
  - a pergunta do E5 fica explícita: substituição sob orçamento fixo de atualizações.
- A auditoria do Eikos contra as avaliações deu zero, o que fecha a lacuna apontada na v1.

## Resumo

- **Recomeçar do Lumi congelado.**
  - O E3b fica como referência de decisão e o Lumi-4B como referência de preservação.
  - O E5 antigo, que parte do E3b, fica suspenso.
- **O problema principal é o formato do treino somado à falta de âncora,** mais do que os dados em si. Os 36.541
  exemplos ensinam a decidir no 1º token da resposta, num só modelo de prompt, e nenhum termo segura o resto do
  comportamento.
- **A correção com mais respaldo é a do LLM-as-Jev,** que usa o mesmo porte (4B, raciocínio desligado) e o mesmo
  LoRA (r16, α32):
  - ler a decisão depois de um prefixo;
  - ancorar por KL, contra o modelo congelado, o início da resposta e a posição logo depois da decisão.

  Sem âncoras, o 4B deles também "continua gerando texto redundante depois do identificador" em 69 de 231 consultas.
  Com λ = 1, eles tiveram o melhor acerto e a menor ECE.
- **Dois eixos em todo checkpoint:** decisão (reservadas, formato v1) e preservação (Lumi-Benchmark dev e ENEM
  `explorar` em `cot`, contra o Lumi-4B). A regra de preservação é registrada antes de escolher qualquer checkpoint.
- **O DecisionBank entra depois,** num experimento que mede quanto ele acrescenta à receita escolhida.

## 1. O que deu errado, com a evidência

### P1. Vazamento de formato (medido)

| test do Lumi-Benchmark | Lumi-4B | E3b s1 | diferença |
|---|---:|---:|---|
| A, `cot`, acertos | 319/547 | 261/547 | −10,6 [−15,7; −5,7] (bootstrap pareado por grupo) |
| A, `cot`, MATH-PT | 71/140 | 38/140 | a queda persiste nas questões em que os dois deram resposta válida |
| A, `cot`, tokens gerados em média | 277 | 20 | |
| A, `cot`, começa pela letra da opção | 4/547 | 110/547 | |
| A, `direct`, acertos | 261/547 | 256/547 | inconclusivo; no dev, 138 → 116 de 253 |
| B, chamada exata | 103/140 | 113/140 | o E3b melhora |

- O E3b responde direto mesmo quando o prompt deixa raciocinar. Às vezes copia a alternativa ("B) texto…
  Resposta: B"), que é o formato do treino.
- **O treino não ensinou a responder com letra.** Com o prompt do treino, o Lumi já põe de 0,97 a 0,99 da massa nas
  letras desde o 1º passo. O que mudou foi o comportamento fora desse prompt.
- **A causa exata ainda precisa ser isolada.** A perda compara só as letras candidatas na 1ª posição e deixa o resto
  da distribuição sem supervisão.
- **Pela definição do projeto,** o Decision responde direto quando se pede uma decisão. O defeito é fazer isso
  também quando a instrução deixa raciocinar.

### P2. Ponto cego da avaliação

- Todas as métricas de seleção usavam o formato do treino:
  - as reservadas, as famílias vistas e o ENEM do F0, todos lidos pela letra no 1º token;
  - o avaliador oficial do projeto (o Lumi-Benchmark) não entrava.

### P3. Objetivo sem âncora e sem dose controlada

- **Sem termo de preservação:** a perda olha só a 1ª posição da resposta, renormalizada entre as letras.
- **LoRA largo:** r16 em atenção, MLP e `in_proj`, com lr 1e-4.
- **Só o adaptador final é guardado.**

### P4. Dados homogêneos em formato e tipo

- Um só prompt (v1): opções "A." e "Responda apenas com a letra".
- Quase só classificação de um passo, e o Eikos tem 85% em inglês.
- Faltam dados de decisão em PT difíceis para o Lumi.

### P5. O ganho de decisão é real, mas pequeno

- **No geral:** o E3b fica +3,4 e +3,3 acima do controle nas 4 reservadas com professor, nas 2 sementes.
- **Por família:** a inferência ganha +6,5 nas duas e a correferência praticamente não se move. Isso justifica
  manter o Eikos no primeiro comparativo.

### Sobreposição do Eikos

- A conferência das reservadas v2 não cobria o Eikos.
- O outro chat auditou os 16.191 exemplos do Eikos usados no treino e não achou nenhuma sobreposição (texto
  normalizado ou 13 palavras seguidas) com as avaliações verificadas.
- Na receita nova, a proteção do `PROTEGE_TAMBEM` passa a valer também para o Eikos. Como a auditoria deu zero, os
  dados do E3b não mudam, e a reprodução do R0 confere isso pelo `dados_sha256`.

## 2. Continuar do E3b ou recomeçar do Lumi?

**Recomeçar do Lumi congelado; o E5 antigo fica suspenso.**

| critério | continuar do E3b | recomeçar do Lumi |
|---|---|---|
| Vazamento | herdado; a correção primeiro precisa desfazê-lo | a correção age desde o 1º passo |
| Atribuição | "dado novo" se mistura com "desfazer o E3b" | mesmo desenho do E3b: novo × controle, mesmo ponto de partida |
| Custo | uns 45 min por treino | uns 45 min por treino |
| Ganho já obtido | mantém os +3,4 | a receita do E3b é fixa, então o R0 reproduz os +3,4 |
| Entrega | adaptador sobre adaptador | uma receita reproduzível a partir do Lumi |

## 3. Princípios da receita nova

1. **Dois eixos em todo checkpoint.** Decisão: as reservadas no formato v1, comparáveis com tudo o que já foi
   medido. Preservação: o conjunto fixo da Etapa 1.
2. **Decisão lida depois de um prefixo, e não no 1º token da resposta.**
   - Nos formatos novos, a resposta começa preenchida com "Resposta:" e a letra é lida na posição seguinte. É assim
     que o LLM-as-Jev lê ("Best answer: [") e que o próprio benchmark tira as probabilidades.
   - O 1º token da resposta deixa de ser o lugar onde se decide. Assim ele pode ser ancorado no Lumi.
   - O formato v1 (letra no 1º token) continua na mistura, para a avaliação das reservadas seguir comparável.
3. **Âncora por KL contra o Lumi congelado** (LLM-as-Jev, §3.4). Os termos, adaptados ao nosso caso:
   - `L_mass`: KL de Bernoulli da massa nas letras contra o resto do vocabulário, na posição da decisão;
   - `L_out`: KL da distribuição renormalizada fora das letras, na posição da decisão (top-64 mais um balde de
     resto);
   - `L_pos`: KL da distribuição completa no início da resposta e na posição logo depois da letra (top-64 mais o
     resto).
   - O início da resposta é ancorado também em prompts `cot` montados a partir dos casos de treino, sem perda de
     decisão. Isso mira exatamente o defeito medido, em que o 1º token num prompt `cot` vira letra.
   - λ = 1, o padrão do artigo e o melhor no 4B deles. A referência sai do mesmo modelo com o adaptador desligado,
     ou de um cache calculado antes do treino, como no artigo.
4. **Autodestilação como reserva** (braço R3, condicional): treinar nas respostas do próprio Lumi com raciocínio,
   filtradas pelas que acertam. Entra se a KL não bastar para preservar o `cot`.
5. **O `in_proj` fica nos alvos.** No F2.2, tirá-lo custou −2,3 numa semente e empatou na outra.
6. **Dose controlada:** adaptadores em 25, 50, 75 e 100% do treino. A escolha do ponto segue uma regra registrada.
7. **O rigor de sempre:**
   - critério registrado antes;
   - controle casado;
   - 2 sementes;
   - exposições e tokens registrados por objetivo;
   - o teste oficial lido uma vez.

## 4. Plano de ação

### Etapa 0. Diagnóstico com checkpoints existentes (jobs 38292–38294, na fila)

**Ressalvas:**
- **Os treinos de 40, 200 e 508 passos tiveram cronogramas próprios** (aquecimento e cosseno completos). Eles
  respondem "quanto treino completo é preciso para o vazamento aparecer", não a dose dentro de um treino. A curva de
  dose limpa vem dos adaptadores intermediários do E6.
- **O job com raciocínio ligado usa 4.096 tokens**, contra 2.048 no desligado. A comparação Lumi × E3b dentro dele é
  limpa; a comparação entre os modos mistura duas mudanças.
- **Conferir que saíram todas as rodadas previstas.** Antes, o job podia terminar com sucesso mesmo com uma
  avaliação interna falhando. Desde 27/11 a análise lista as rodadas que faltam, as que têm menos itens que as
  outras, e sai com erro. Os jobs na fila usam a análise nova; uma fusão que falhe aparece no log.

| se o diagnóstico mostrar | então |
|---|---|
| o vazamento já aparece no treino de 40 passos | é o formato, não a quantidade de treino: âncora e prefixo são obrigatórios |
| o vazamento cresce de 40 para 891 passos | menos treino ajuda, mas a âncora continua sendo a correção principal |
| E3 e controle (sem Eikos) também vazam | a causa é o treino pela letra, não o Eikos; o Eikos fica |
| só os braços com Eikos vazam | rever o Eikos antes de qualquer outra coisa |
| o F2.2 (sem `in_proj`) preserva o `cot` | testar a retirada como braço próprio, depois do E6; o `in_proj` fica no primeiro comparativo |
| o E4 (pointer head) também vaza | o efeito está na representação da posição de resposta: a âncora é obrigatória |
| com o raciocínio ligado, o E3b raciocina como o Lumi | o vazamento é do modo desligado; medir os dois modos daqui em diante |

### Etapa 1. Guarda-corpo e regra de preservação (sem treino; 1 dia)

1. **Conjunto de preservação (dev, usado na seleção):**
   - Lumi-Benchmark dev: A `direct` (253), A `cot` (253) e B (60), servido no vLLM com o LoRA fundido, na
     configuração do job 38177 (2.048 tokens, temperatura 0).
   - ENEM `explorar` em `cot`: 763 questões, uma ordem só, no formato do benchmark ("Pode raciocinar antes…
     Resposta: X"). O IC reamostra questões.
2. **Confirmação final (fora da seleção):**
   - ENEM `confirmar` em `cot`;
   - o teste das reservadas;
   - o Lumi-Benchmark test, como referência de regressão, porque ele já orientou esta mudança.
3. **Referências medidas no mesmo conjunto:** Lumi-4B, E3b s0 e E3b s1.
4. **Regra de preservação, a registrar antes do E6.** A margem é a perda aceitável para o projeto.
   - **Proposta:** em cada medida (A `direct`, A `cot`, ENEM `cot`), a estimativa pontual não fica mais de **2
     pontos** abaixo do Lumi-4B e o IC 95% pareado não fica inteiro abaixo de zero. Na B, nenhuma piora pontual.
   - O E3b reprova (A `cot` −13 no dev). Um modelo igual ao Lumi passa na maior parte das vezes.
   - Uma não inferioridade pura com IC (limite inferior acima de −2) reprovaria até um modelo igual ao Lumi,
     porque o erro-padrão pareado fica entre 1,8 e 3 pontos.
5. **Opcional:** o Lumi-4B gerando com raciocínio nas reservadas dev, ordem canônica, como teto de referência para
   a decisão.

### Etapa 2. Infraestrutura da receita nova (código; 1 a 2 dias)

1. **Formatos de decisão:**
   - v1 (letra no 1º token), mais formatos com prefixo "Resposta:";
   - rótulos de opção "A.", "A)" e "(A)";
   - variações da instrução.
   - Conferência de tokenização **por formato e nº de opções**: o cache de ids das letras passa a ser indexado por
     (formato, n), com as checagens iniciais em cada chave.
2. **Âncoras KL** (`L_mass`, `L_out`, `L_pos`), referência com o adaptador desligado.
   - Testes num modelo minúsculo: KL zero quando o adaptador é zero, gradiente só no LoRA, top-64 mais resto
     somando 1.
3. **Contextos de âncora `cot`:** os casos de treino renderizados com a instrução `cot`, só com `L_pos` no início da
   resposta.
4. **Adaptadores intermediários** em 25, 50, 75 e 100%.
5. **Registro por objetivo:** exposições e tokens de decisão e de âncora, em cada formato.
6. **Pipeline num job:** treino → reservadas → fusão → guarda-corpo, com os resultados nos dois eixos.
7. **Proteção completa:** `PROTEGE_TAMBEM` também para o Eikos.
8. **Conferência de reprodução:** o R0 tem de bater o `dados_sha256` do E3b original, e as reservadas, dentro do
   ruído do bf16.

### Etapa 3. E6: três receitas a partir do Lumi (2 sementes)

| braço | o que muda | pergunta |
|---|---|---|
| R0 | o E3b reproduzido no pipeline novo (E3 + Eikos, formato v1, 1 época, com adaptadores intermediários) | referência e conferência de reprodução |
| R1 | os mesmos exemplos e atualizações, em formatos variados (v1 incluso), leitura depois do prefixo nos formatos novos | a variedade de formato e o prefixo sozinhos reduzem o vazamento? |
| R2 | R1 + âncoras KL (λ = 1), nas mesmas atualizações e com os mesmos exemplos de decisão | a âncora preserva o Lumi sem perder a decisão? |
| R3 (condicional) | R2 + autodestilação | só se R2 reprovar na preservação |

- **Critério (a registrar):**
  - preservação de cada braço contra o Lumi-4B pela regra da Etapa 1;
  - decisão contra o R0 nas 4 reservadas com professor, por não inferioridade com margem a registrar (proposta:
    1,5 ponto, perto do IC das comparações da M3).
  - A receita adotada é a que passa na preservação com a melhor decisão.
- **Exposição casada:** as âncoras entram no mesmo lote de cada atualização. Com isso, R1 e R2 têm as mesmas
  atualizações e os mesmos exemplos de decisão do R0; o que muda são os tokens, que ficam registrados.

### Etapa 4. E5 refeito: DecisionBank sobre a receita escolhida

- **Pergunta (explícita):** substituição sob orçamento fixo de atualizações, como no E3b.
  - Com as mesmas atualizações, o braço novo troca a 2ª exposição de parte dos exemplos antigos por 18.086 exemplos
    novos.
  - Os tokens ficam registrados; no E5 antigo, o braço DecisionBank tinha de 15 a 16% menos tokens.
  - Adição de dados (mais atualizações) seria outra pergunta, não testada aqui.
- **Composição (a registrar):**
  - a `j11_generator` filtrada (v3), com o teto de 4.000 por família;
  - os três tipos de pergunta, `choice`, `noul` (verdadeiro/falso) e `score` (escala ordinal). Todos têm opções e
    rótulo, e alguns têm distribuição. O Eikos do E3b também tinha os três: 6.345, 8.601 e 1.245;
  - a proporção real por tipo e por família, com o teto, sai do `scripts/contar_orcamento.py`
    (`decisionbank_por_familia_e_tipo`) antes do registro.
- **Carregador v3:** rejeita distribuição com valor negativo ou não finito. Um rótulo fora das opções vira descarte,
  sem `IndexError`. A auditoria refeita mostra se alguma linha real mudou.
- **Critério:**
  - o do E5 (novo − controle nas 4 com professor, IC > 0 nas 2 sementes);
  - mais a regra de preservação.

### Etapa 5. Dados de decisão em PT mais difíceis (sintético v3)

- **Alvo:** famílias em que o Lumi congelado fica entre o acaso + 5 e 80% no modo direto, fora dos tipos reservados.
  Exemplos:
  - registro com correções no meio do texto;
  - contagem com condições;
  - comparação de valores e prazos.
- **Cada caso entra nos dois modos:** decisão com o gabarito do código e contexto de âncora `cot`.

### Etapa 6. Seleção final e entrega

- Receita e ponto do treino pela regra registrada.
- Leitura única da confirmação final da Etapa 1.
- Adaptador no HF e pacote de entrega atualizados.

## 5. Lista de conferência para qualquer fonte de dados

- **Licença explícita,** comparada por igualdade.
- **Sem sobreposição com nenhuma avaliação:** `PROTEGE_TAMBEM` em todas as fontes externas.
- **Sem colisão com os tipos reservados.**
- **Dificuldade:** o Lumi congelado não pode acertar quase tudo.
- **Validade do alvo:** rótulo dentro das opções, distribuição sem negativos, soma positiva.
- **Registrar:** o tipo de pergunta, o formato e a proporção de PT.
- **Orçamento:** atualizações, exemplos e tokens, por fonte e por objetivo.

## 6. Riscos e pontos em aberto

- **A âncora pode custar decisão.** No 4B do LLM-as-Jev não custou, mas a nossa medida é outra (reservadas em PT).
  O R2 mede essa troca.
- **A KL ancora contextos de decisão e `cot` dos nossos dados**, não conversa geral. O LLM-as-Jev viu a âncora
  generalizar para a conversa, mas isso precisa ser medido no nosso caso.
- **Prefixo em produção:** o Decision passa a ser usado com a resposta preenchida com "Resposta:" (ou a instrução
  equivalente). Documentar para o time.
- **Modo de raciocínio:** treinamos e servimos com ele desligado; a Etapa 0 diz se é preciso medir os dois.
- **Saída de emergência:** servir o Decision como adaptador carregado só nas chamadas de decisão. Isso resolve o uso,
  mas não o que o time de benchmark mede.

## 7. Decisões (27/11/2026)

1. **Margem de preservação:** 2 pontos, e o IC não pode ficar inteiro abaixo de zero. Na B, nenhuma piora pontual.
   Aceita pelo Gabriel.
2. **Margem de não inferioridade da decisão contra o R0:** 1,5 ponto. Aceita pelo Gabriel.
3. **O R3 (autodestilação) só roda se R1 e R2 reprovarem na preservação.**

O protocolo do E6 com essas margens está registrado em `analises/e6_protocolo.md`. Infraestrutura pronta:
- formatos e leitura depois do prefixo;
- âncoras KL;
- adaptadores intermediários;
- guarda-corpo (`slurm/guarda_corpo.sbatch`, `scripts/guarda_corpo.py`).
