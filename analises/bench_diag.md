# Por que o E3b piora o raciocínio no Lumi-Benchmark (diagnóstico, 27/11/2026)

Ponto de partida: `analises/lumi_benchmark.md`. No test da trilha A com o prompt `cot`, o Lumi-4B acerta 319/547 e o
E3b s1 acerta 261/547. Na trilha B, o E3b melhora: 113 × 103/140.

## O que já está medido (rodadas do job 38177, `scripts/analise_bench_diag.py`)

| rodada | acertos | tokens médio | diretas (≤ 8 tokens) | começa pela letra da opção | MATH-PT |
|---|---:|---:|---:|---:|---:|
| Lumi-4B, cot, dev | 158/253 | 271,8 | 157/253 | 4/253 | 32/60 |
| E3b s1, cot, dev | 125/253 | 29,6 | 199/253 | 49/253 | 16/60 |
| Lumi-4B, cot, test | 319/547 | 276,9 | 330/547 | 4/547 | 71/140 |
| E3b s1, cot, test | 261/547 | 20,0 | 425/547 | 110/547 | 38/140 |
| Lumi-4B, direct, test | 261/547 | 5,1 | 544/547 | 3/547 | 31/140 |
| E3b s1, direct, test | 256/547 | 2,4 | 535/547 | 22/547 | 33/140 |

- **Bootstrap pareado por grupo** (revisão do outro chat, 27/11):
  - no test `cot`, a diferença E3b − Lumi é −10,6 pontos [−15,7; −5,7];
  - em MATH-PT, a queda persiste nas questões em que os dois deram resposta válida;
  - no `direct`, o test é inconclusivo.
- Mesmo quando o prompt pede raciocínio, o E3b começa pela letra da opção ("A) texto da opção… Resposta: A") em
  110/547 itens no test. O Lumi-4B faz isso em 4/547. Esse é o formato do treino: a letra é o primeiro token da
  resposta.
- **Massa das letras no treino** (`treino.jsonl`, `massa_letras`):
  - com o prompt do treino ("Responda apenas com a letra"), o Lumi já põe de 0,97 a 0,99 da massa nas letras
    desde o 1º passo, e isso quase não muda até o fim (0,986 a 0,998);
  - logo, o treino não ensinou a responder com letra. Ele generalizou "começar pela resposta" para prompts que
    pedem outra coisa;
  - a perda renormalizada entre as letras não mede esse efeito, então só uma avaliação fora do formato do treino
    (o benchmark) o enxerga.

## Diagnóstico com checkpoints que já existem (job `slurm/benchmark_diag.sbatch`, dev, `cot`)

Sem treino novo. Cada comparação isola uma causa:

| pergunta | checkpoints |
|---|---|
| Dose: a perda cresce com o nº de atualizações? | `e3_curto` (40), `e3_curto3` (200), `e3_lora_r16` (508), `e3b_ctrl_s0` (891) |
| É o Eikos ou o treino pela letra em geral? | `e3_lora_r16_s1` e `e3b_ctrl_s1` (sem Eikos) × `e3b_eikos_s1` |
| O `in_proj` do Mamba carrega o efeito? | `f22_sem_inproj_s1` × `e3_lora_r16_s1` |
| A perda pela cabeça de LM (letras) é a causa? | `e4_ponteiro_s1` (o LoRA aprende pela pointer head, sem a cabeça de LM) |
| Dados mais variados ajudam? | `f3p_diverso_s1` |
| Com o raciocínio ligado (padrão do Lumi), a perda continua? | Lumi-4B e `e3b_eikos_s1`, `THINKING=1`, 4096 tokens |

- **Leitura:**
  - a métrica principal é acertos no dev cot, MATH-PT separado;
  - os mecanismos são tokens médio, diretas e "começa pela letra";
  - o dev tem 253 itens, então diferenças de poucos acertos são ruído.
- **Ressalvas (revisão de 27/11):**
  - **Os treinos de 40, 200 e 508 passos tiveram cronogramas próprios** (aquecimento e cosseno completos). A
    comparação diz quanto treino completo basta para o vazamento aparecer, não a dose dentro de um treino. A curva
    de dose limpa vem dos adaptadores intermediários do E6 (`analises/plano_fase3.md`).
  - **O job com raciocínio ligado usa 4.096 tokens,** contra 2.048 no desligado. Lumi × E3b dentro dele é uma
    comparação limpa; a comparação entre os modos mistura o raciocínio com o limite.
  - **O job podia terminar com sucesso mesmo com uma avaliação interna falhando.** Desde 27/11, a análise lista as
    rodadas sem resultado e as que têm menos itens que as outras, e sai com erro. O `sbatch` também acumula as
    falhas (fusão, vLLM, avaliação) e termina com erro. Os jobs já na fila usam a análise nova, mas o corpo antigo
    do `sbatch`: uma fusão que falhe neles só aparece no log, então é preciso conferir as rodadas uma a uma.
- O `fundir.py` passou a aceitar alvos congelados (F2.2): exige `lora_B` zerado neles e não os conta entre os
  tensores que mudam. Para os outros adaptadores, nada muda.

## Resultado do diagnóstico (jobs 38292 e 38293, 27/11/2026)

Lumi-Benchmark dev, `cot`, raciocínio desligado, 2.048 tokens (253 questões). Todas as rodadas previstas saíram
(`_faltam` e `_incompletas` vazios). O Lumi-4B reproduziu exatamente os 158/253 do job 38177.

| checkpoint | atualizações | acertos | tokens médio | diretas | começa pela opção | MATH-PT |
|---|---:|---:|---:|---:|---:|---:|
| Lumi-4B | — | 158/253 | 271,8 | 157/253 | 4/253 | 32/60 |
| `e3_curto` (E3, s0) | 40 | 151/253 | 286,9 | 157/253 | 6/253 | 33/60 |
| `e3_curto3` (E3, s0) | 200 | 142/253 | 225,9 | 191/253 | 2/253 | 31/60 |
| `e3_lora_r16` (E3, s0) | 508 | 147/253 | 79,0 | 228/253 | 0/253 | 26/60 |
| `e3b_ctrl_s0` (E3, 2 épocas) | 891 | 137/253 | 35,2 | 238/253 | 1/253 | 23/60 |
| `e3b_eikos_s0` (E3 + Eikos) | 891 | 125/253 | 11,4 | 210/253 | 42/253 | 15/60 |
| `e3_lora_r16_s1` (E3, s1) | 502 | 133/253 | 17,2 | 238/253 | 11/253 | 19/60 |
| `e3b_ctrl_s1` | 900 | 127/253 | 63,2 | 216/253 | 15/253 | 25/60 |
| `e3b_eikos_s1` | 900 | 125/253 | 29,6 | 199/253 | 49/253 | 16/60 |
| `f22_sem_inproj_s1` (E3 sem `in_proj`) | 502 | 140/253 | 188,6 | 191/253 | 11/253 | 28/60 |
| `e4_ponteiro_s1` (pointer head) | 502 | 110/253 | 14,6 | 215/253 | 29/253 | 22/60 |
| `f3p_diverso_s1` (piloto F3) | 508 | 138/253 | 244,1 | 155/253 | 9/253 | 27/60 |

Leitura, na ordem da tabela da Etapa 0 (`analises/plano_fase3.md`):
- **Quantidade de treino.** Com 40 atualizações, o comportamento é o do Lumi (157 respostas diretas, 287 tokens).
  Com 200, já são 191 diretas. Com 508 e 891, os tokens caem para 79 e 35.
  - O vazamento cresce com a quantidade de treino. Não é o formato aparecendo logo nos primeiros passos.
  - Lembrete: são treinos curtos completos, cada um com o próprio cronograma, não pontos de um mesmo treino.
- **Eikos.** Sem Eikos também vaza: o E3 s1 cai para 17 tokens e 238 diretas, e o controle s0 para 35 tokens.
  - O Eikos acrescenta o padrão "letra + texto da alternativa" (42 e 49 de 253, contra 1 e 15 sem ele) e um pouco
    mais de perda.
  - A causa é o treino pela letra, não o Eikos. Pela tabela, o Eikos fica.
- **`in_proj`.** Com a mesma semente, os mesmos dados e as mesmas 502 atualizações, congelar o `in_proj` sobe de
  17 para 189 tokens e de 133 para 140 acertos.
  - É o maior efeito de preservação entre os checkpoints que já existem, mas não chega ao Lumi (191 diretas contra
    157) e foi medido em uma semente só.
  - Pela tabela, a retirada vira braço próprio depois do E6, e o `in_proj` fica no primeiro comparativo.
- **Pointer head.** Também vaza, e é o pior em acertos (110/253). O LoRA mudou a representação na posição da resposta
  mesmo sem a perda pelas letras da cabeça de LM. Pela tabela, a âncora é obrigatória.
- **Piloto F3.** Os dados mais variados mantêm o comportamento (155 diretas, 244 tokens), mas os acertos caem 20
  (BLUEX 45 contra 54, DrBode 37 contra 45). A receita é outra (o teto do F2.1), então isso fica como observação.
- **Semente.** No E3, a s0 e a s1 diferem muito (79 contra 17 tokens; 147 contra 133 acertos). Isso justifica o
  critério por semente.
- **Raciocínio ligado** (job 38294): ainda rodando. Pela tabela, o resultado só muda o que é medido, não a receita.

## Guarda-corpo da Etapa 1 (job 38297, 27/11/2026; `analises/guarda_corpo/dev_38297.json`)

| medida (dev) | Lumi-4B | E3b s0 | E3b s1 |
|---|---:|---:|---:|
| A `direct` | 138/253 | 121/253 (−6,7 [−12,2; −1,6]) | 116/253 (−8,7 [−14,2; −3,2]) |
| A `cot` | 158/253 | 125/253 (−13,0 [−19,4; −6,7]) | 126/253 (−12,6 [−19,0; −6,3]) |
| ENEM `cot` (763 questões `explorar`) | 583/763 | 533/763 (−6,5 [−9,7; −3,5]) | 515/763 (−8,9 [−12,1; −5,6]) |
| B, chamada exata | 49/60 | 50/60 (+1,7) | 51/60 (+3,3) |
| veredito pela regra registrada | — | reprova | reprova |

- **No ENEM `cot` o Lumi raciocina (157 tokens em média) e o E3b não (13 e 15).** Na leitura pela letra do F0, o
  mesmo ENEM não mostrava perda (E3 −0,7 [−2,5; +1,2]). O guarda-corpo fora do formato do treino enxerga o que a
  leitura pela letra escondia.
- **Determinismo:** o E3b s1 no `cot` deu 126/253 aqui e 125/253 no diagnóstico, com o mesmo modelo fundido e
  temperatura 0. É uma diferença de 1 questão entre duas rodadas do vLLM.

## O que isso muda nos próximos treinos (a confirmar com o diagnóstico)

- **Guarda-corpo obrigatório:** todo checkpoint candidato passa pelo dev cot do Lumi-Benchmark contra o Lumi-4B
  (acertos e tokens médio), além das reservadas.
  - Um treino que melhora a decisão e piora esse dev não conta como avanço.
- **Candidatos a correção**, a escolher pelo diagnóstico:
  - **Autodestilação:** gerar com o próprio Lumi-4B respostas com raciocínio para prompts variados e para os de
    decisão em formato cot, e treinar nelas (perda de LM na sequência gerada) junto com a perda das letras.
  - **KL contra o base:** nas primeiras posições da resposta, em prompts gerais e de cot.
  - **Letra depois do raciocínio:** treinar a perda na letra que vem depois de um raciocínio gerado pelo base, e
    não na primeira posição.
  - **Adaptador mais estreito ou menos passos:** se a dose ou o `in_proj` explicarem a perda.
- **O E5 parte do E3b:**
  - os braços do E5 herdam o problema;
  - os 4 treinos ficam suspensos (`scontrol hold`);
  - a pergunta do DecisionBank passa para a Etapa 4 de `analises/plano_fase3.md`, a partir do Lumi.
- **O plano revisado** (`analises/plano_fase3.md`, v2) troca a âncora principal pela KL do LLM-as-Jev
  (2610.02076, §3.4): a decisão é lida depois de um prefixo, e o início da resposta e a posição depois da decisão
  ficam ancorados no Lumi congelado.
