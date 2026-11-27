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
- O `fundir.py` passou a aceitar alvos congelados (F2.2): exige `lora_B` zerado neles e não os conta entre os
  tensores que mudam. Para os outros adaptadores, nada muda.

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
  - os 4 treinos ficaram suspensos (`scontrol hold`) até o resultado do diagnóstico;
  - se forem rodar, entram no mesmo guarda-corpo.
