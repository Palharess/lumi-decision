# Plano revisado depois do Lumi-Benchmark (proposta de 27/11/2026, ainda não registrada)

Este documento revisa o plano à luz do Lumi-Benchmark (`analises/lumi_benchmark.md`) e do diagnóstico em andamento
(`analises/bench_diag.md`). Ele propõe; nada aqui é critério registrado. Cada etapa com treino ganha o seu protocolo
antes de rodar, como sempre.

## Resumo

- **Recomeçar do Lumi congelado, não continuar do E3b.**
  - O adaptador do E3b carrega o problema (responder direto mesmo quando o prompt deixa raciocinar).
  - Um treino custa uns 45 minutos, então refazer do zero sai barato.
  - O E3b vira a referência de decisão a bater; o Lumi-4B, a referência de preservação.
- **O problema principal não são os dados em si, e sim o formato único do treino somado a nenhuma âncora no
  comportamento original.**
  - Os 36.541 exemplos ensinam a mesma coisa: decidir no 1º token, num só modelo de prompt.
  - Sem um termo que preserve o resto, o LoRA levou esse hábito para prompts que pedem outra coisa.
- **"Mais dados" do mesmo formato tende a piorar o vazamento.** O dado novo precisa vir com variedade de formato e
  com exemplos que mantenham o modo de raciocinar do Lumi.
- **Toda receita passa a ser julgada em dois eixos:**
  - decisão: as reservadas, no nosso formato;
  - preservação: o Lumi-Benchmark e o ENEM com raciocínio, fora do nosso formato.

  Um ganho de decisão que piora a preservação não conta como avanço.
- **Os 4 treinos do E5 a partir do E3b (suspensos) devem ser cancelados.** A pergunta do DecisionBank passa para a
  Etapa 4, a partir do Lumi.

## 1. O que deu errado, com a evidência

### P1. Vazamento de formato (medido)

| test do Lumi-Benchmark | Lumi-4B | E3b s1 |
|---|---:|---:|
| A, `cot`, acertos | 319/547 | 261/547 |
| A, `cot`, MATH-PT | 71/140 | 38/140 |
| A, `cot`, tokens gerados em média | 277 | 20 |
| A, `cot`, começa pela letra da opção | 4/547 | 110/547 |
| A, `direct`, acertos | 261/547 | 256/547 |
| B, chamada exata | 103/140 | 113/140 |

- O E3b responde direto mesmo quando o prompt diz "pode raciocinar antes". Às vezes copia a alternativa ("B) texto…
  Resposta: B"), que é o formato do treino.
- **O treino não ensinou a responder com letra.** Com o prompt do treino, o Lumi já põe de 0,97 a 0,99 da massa nas
  letras desde o 1º passo (`massa_letras` nos `treino.jsonl`). O que mudou foi o comportamento fora desse prompt.
- **Pela definição do projeto,** o Lumi Decision "retorna a opção diretamente, sem explicação intermediária".
  Responder direto quando se pede uma decisão está certo. O defeito é fazer isso também quando a instrução deixa
  raciocinar, e o time de benchmark compara as duas variantes.
- **O conhecimento no modo direto não está garantido.**
  - No test, `direct` empata (261 × 256), mas o OAB cai 3,7 pontos.
  - No dev, `direct` cai de 138 para 116 de 253.
  - Com 253 itens, isso pode ser ruído, mas não dá para afirmar que não houve perda.

### P2. Ponto cego da avaliação

- Todas as nossas métricas de seleção usam o mesmo formato do treino: as reservadas, as famílias vistas e o ENEM do
  F0, este lido pela letra no 1º token.
- Escolhemos o E3b justamente pelo que ele mais aprendeu: decidir naquele formato. Nada do que medíamos via o
  vazamento.
- O avaliador oficial do projeto é o Lumi-Benchmark do time. A nossa seleção não olhava para ele.

### P3. Objetivo sem âncora e sem dose controlada

- **A perda olha só a 1ª posição da resposta,** renormalizada entre as letras. Nenhum termo segura o resto da
  distribuição do modelo.
- **O LoRA é largo:** r16 em atenção, MLP e `in_proj` do Mamba, com lr 1e-4.
- **O treino só guarda o adaptador final.** Não há pontos intermediários para escolher a dose por um guarda-corpo.

### P4. Dados homogêneos em formato e tipo

- Um só modelo de prompt (`PROMPT_VERSION` v1): opções "A." e "Responda apenas com a letra". O benchmark usa "A)"
  e "Resposta: X".
- Quase tudo é classificação de um passo, e o Eikos tem 85% em inglês.
- Faltam dados de decisão em PT nativo que sejam difíceis para o Lumi:
  - o sintético do piloto ficou fácil demais (extração 800/800);
  - o DecisionBank limpo é todo em inglês.

### P5. O ganho de decisão é real, mas pequeno

- O E3b ganha 3,4 pontos do controle nas 4 reservadas com professor, nas duas sementes.
- O ganho se concentra em famílias parecidas com o treino:
  - a inferência ganha de 22 a 28 pontos com o E3 (é a mais próxima do `factck`);
  - a `ordem_eventos` melhora com qualquer metade do Eikos.
- A correferência não se move com nenhum treino.
- A H2 original (3 reservadas, 2 ordens) segue inconclusiva.

### Lacuna encontrada nesta revisão

- A conferência de sobreposição das reservadas v2 (`res2_sobreposicao`) cobre Felhen, Telepatia, Traces e F3, mas
  **não o Eikos**.
- A das reservadas com professor (`prof_sobreposicao`) cobre o Eikos.
- O risco é baixo (as v2 são programáticas e o Eikos é 85% inglês), mas precisa de uma auditoria antes de a receita
  nova usar o Eikos de novo.

## 2. Continuar do E3b ou recomeçar do Lumi?

**Recomendação: recomeçar do Lumi congelado.**

| critério | continuar do E3b | recomeçar do Lumi |
|---|---|---|
| Vazamento | herdado; qualquer correção primeiro precisa desfazê-lo | a correção (âncora, formatos) age desde o 1º passo |
| Atribuição | "dado novo" se mistura com "desfazer o E3b" | mesmo desenho do E3b: novo × controle, mesmo ponto de partida |
| Custo | uns 45 min por treino | uns 45 min por treino (o E3b inteiro tem 891 atualizações) |
| Ganho já obtido | mantém os +3,4 | a receita do E3b é fixa (dados e sementes), então os +3,4 se refazem |
| Entrega | adaptador em cima de adaptador, receita difícil de reproduzir | uma receita reproduzível a partir do Lumi |

- **O argumento a favor de continuar** era o custo e o isolamento do efeito do dado novo (`e5_dados.md`). Com
  treinos de 45 minutos, o custo deixa de pesar. O isolamento vem igual de um controle casado a partir do Lumi.
- **O E3b continua útil como referência:**
  - de decisão, a bater nas reservadas;
  - de vazamento, no benchmark e no diagnóstico.

## 3. Princípios da receita nova

1. **Dois eixos em todo checkpoint.**
   - **Decisão:** as reservadas, no formato v1 de sempre, para manter a comparação com tudo o que já foi medido.
   - **Preservação:** o Lumi-Benchmark dev (`direct`, `cot` e trilha B) e o ENEM em `cot`, contra o Lumi-4B.
2. **Ensinar a decisão como comportamento condicionado à instrução.** O modelo responde direto quando se pede só a
   letra e raciocina quando a instrução deixa. Para isso, o treino precisa ter os dois casos, em formatos
   variados.
3. **Ancorar no Lumi.**
   - Parte de cada lote é o comportamento do próprio Lumi-4B (autodestilação), para o LoRA aprender a decisão sem
     mudar o resto.
   - Ideia próxima da *self-distillation fine-tuning* e do *rejection sampling fine-tuning*: treinar nas respostas
     do próprio modelo, filtradas pelas que acertam.
4. **Controlar a dose.** Guardar o adaptador em 25, 50, 75 e 100% do treino. A escolha do ponto segue uma regra
   registrada antes.
5. **Manter o rigor que já temos:**
   - critério registrado antes;
   - controle casado em atualizações;
   - 2 sementes;
   - o teste oficial lido uma vez só, no fim.

## 4. Plano de ação

### Etapa 0. Diagnóstico com checkpoints existentes (em andamento: jobs 38292–38294)

| se o diagnóstico mostrar | então |
|---|---|
| a piora no `cot` cresce com as atualizações (40 → 200 → 508 → 891) | menos passos ajudam, mas a âncora continua necessária |
| a piora já aparece com 40 atualizações | é o formato, não a dose: âncora e variedade de formato são obrigatórias |
| E3 e controle (sem Eikos) também pioram | a causa é o treino pela letra, não o Eikos; o Eikos pode ficar |
| só os braços com Eikos pioram | rever o Eikos (inglês, tipos) antes de qualquer outra coisa |
| o F2.2 (sem `in_proj`) preserva o `cot` | tirar o `in_proj` dos alvos (no F2.2 ele ficou neutro na decisão) |
| o E4 (pointer head) também piora | o efeito está na representação da posição de resposta, não na cabeça de LM: âncora obrigatória |
| com o raciocínio ligado, o E3b raciocina normalmente | o vazamento é do modo sem raciocínio; documentar o modo de uso e medir os dois |

### Etapa 1. Guarda-corpo e linhas de base (sem treino; 1 dia)

1. **Conjunto de preservação fixo:**
   - Lumi-Benchmark dev: A `direct` (253), A `cot` (253) e B (60). Serve no vLLM com o LoRA fundido, como no job
     38177.
   - ENEM em `cot`: os 1.526 itens da metade `explorar` do VestBench, no formato do benchmark ("Pode raciocinar
     antes… Resposta: X"). Com 253 itens, o dev do benchmark não basta: o erro-padrão pareado é de uns 3 pontos.
     Com 1.526, cai para perto de 1,3.
   - Opcional: o mesmo `cot` com o raciocínio ligado, se a Etapa 0 mostrar diferença.
2. **Referências nos mesmos conjuntos:** Lumi-4B, E3b s0 e E3b s1.
3. **Regra de preservação**, registrada antes da Etapa 3:
   - não inferior ao Lumi-4B com margem fixa (proposta: 2 pontos no ENEM `cot`, IC pareado por item);
   - a margem final sai do ruído medido nesta etapa.
4. **Referência de decisão com raciocínio (opcional):** o Lumi-4B gerando com raciocínio nas reservadas dev (ordem
   canônica).
   - Se ele já superar o E3b lendo a letra, o valor do Decision é velocidade e calibração, não acerto.
   - Isso muda a história que contamos ao time.
5. **Auditoria do Eikos contra as reservadas v2, as com professor, o F3 e o ENEM** (a lacuna da seção 1).
   - Se der zero, nada muda.
   - Se não der, as linhas saem da receita nova e o achado fica registrado.

### Etapa 2. Infraestrutura da receita nova (código; 1 a 2 dias)

1. **Formatos variados nos exemplos de decisão.**
   - Varia: o rótulo das opções ("A.", "A)", "(A)"), a instrução e o prefixo da resposta (letra pura ou
     "Resposta: X").
   - O v1 continua entre os formatos, e a avaliação das reservadas fica no v1.
2. **Âncoras por autodestilação.**
   - Prompts montados a partir dos casos de **treino** em formato `cot` (nunca itens do benchmark).
   - O Lumi-4B gera no vLLM e fica só a geração cuja resposta final é o gabarito.
   - A perda de LM vale nos tokens gerados.
   - Os mesmos dados servem à frente do Lumi Reasoning: alinhar com ela.
3. **Opcional: KL contra o Lumi** nas primeiras posições da resposta dos prompts de âncora. A distribuição do base
   sai do mesmo modelo com o adaptador desligado (`disable_adapter`), sem uma segunda cópia na GPU.
4. **Adaptadores intermediários** em 25, 50, 75 e 100% do treino.
5. **Pipeline num job só:** treino → avaliação das reservadas → fusão → guarda-corpo no vLLM. Os resultados saem nos
   dois eixos.

### Etapa 3. Receita nova a partir do Lumi (2 sementes; 1 dia de cluster)

| braço | dados | pergunta |
|---|---|---|
| R0 = E3b (já existe) | E3 + Eikos, formato v1 | referência |
| R1 | os mesmos dados, formatos variados | a variedade de formato sozinha reduz o vazamento? |
| R2 | R1 + âncoras (proporção a registrar, por exemplo 25% dos exemplos) | a âncora elimina o vazamento sem perder a decisão? |

- **Critério (a registrar):**
  - preservação de R2 contra o Lumi-4B pela regra da Etapa 1;
  - decisão de R2 não inferior ao R0 nas 4 reservadas com professor (margem a registrar).
- **Variações conforme a Etapa 0:** sem `in_proj`, r menor e menos passos.
- **Curva de dose:** os adaptadores intermediários dão a curva de decisão e de preservação dentro de cada treino.

### Etapa 4. Dados novos sobre a receita nova (o E5 refeito a partir do Lumi)

- **Novo:** a receita vencedora mais o DecisionBank `j11_generator` (18.086 exemplos, filtro v2, já auditado). As
  âncoras incluem os prompts do DecisionBank em `cot`.
- **Controle:** a receita vencedora com o mesmo nº de atualizações.
- **Critério:**
  - o do E5 já escrito (novo − controle nas 4 com professor, IC > 0 nas 2 sementes);
  - mais a regra de preservação.
- **Depois, como braços próprios:** o sintético PT da Etapa 5 e o Open-Jev, opcional.

### Etapa 5. Dados de decisão em PT mais difíceis (sintético v3)

- **Famílias em que o Lumi congelado fica entre o acaso + 5 e 80% no modo direto.** O piloto mostrou que a extração
  simples é resolvida (800/800). Exemplos:
  - registro com correções no meio do texto;
  - contagem com condições;
  - comparação de valores e prazos.
  Todas fora dos tipos reservados.
- **Cada caso entra nos dois modos:**
  - decisão direta, com o gabarito do código;
  - âncora com raciocínio, por amostragem de rejeição com o Lumi.

### Etapa 6. Seleção final e entrega

- Escolher a receita e o ponto do treino pela regra registrada (decisão sujeita à preservação).
- Ler o teste oficial uma vez só: reservadas e Lumi-Benchmark test.
- Atualizar o adaptador no HF e o pacote de entrega para o time de benchmark.

## 5. O que "mais dados" deve significar

Lista de conferência para qualquer fonte nova (vale também para as antigas na receita nova):
- **Licença explícita,** comparada por igualdade.
- **Sem sobreposição com nenhuma avaliação.** A proteção do `PROTEGE_TAMBEM` (LOADERS, F3, res2, prof e ENEM)
  passa a valer para todas as fontes externas, Eikos incluso.
- **Sem colisão com os tipos reservados.**
- **Dificuldade:** o Lumi congelado não pode acertar quase tudo.
- **Formato variado e uma versão com raciocínio** (âncora) quando fizer sentido.
- **Proporção de PT registrada.**
- **Orçamento registrado:** atualizações, exemplos e tokens.

## 6. Riscos e pontos em aberto

- **A âncora pode reduzir o ganho de decisão.** A Etapa 3 mede essa troca, e a proporção de âncoras pode precisar
  de 2 valores.
- **Treinar nas respostas do próprio Lumi** só funciona onde ele acerta. Onde ele erra, a âncora some e o vazamento
  pode continuar.
- **O dev do benchmark é pequeno.** O ENEM em `cot` dá a precisão, e o dev entra como conferência no formato do time.
- **Modo de raciocínio:** decidir com o time se o Decision é usado sempre com o raciocínio desligado. Hoje
  treinamos assim.
- **Saída de emergência, se a troca não fechar:** servir o Decision como adaptador carregado só nas chamadas de
  decisão (multi-LoRA no vLLM), sem fundir no modelo de uso geral. Isso resolve o uso, mas não o que o time de
  benchmark mede, porque eles rodam o benchmark no Decision.
