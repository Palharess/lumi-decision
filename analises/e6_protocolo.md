# E6: receita nova a partir do Lumi (protocolo registrado em 27/11/2026, antes de treinar)

Contexto: `analises/plano_fase3.md` (v2) e `analises/bench_diag.md`. O E3b decide melhor nas reservadas, mas piora
o Lumi-Benchmark porque passou a responder direto mesmo quando o prompt deixa raciocinar. O E6 testa, a partir do
Lumi congelado, se a variedade de formato com leitura depois de um prefixo e as âncoras KL do LLM-as-Jev
(arXiv 2610.02076, §3.4 e §4.4.1) preservam o Lumi sem perder a decisão.

**Só é submetido depois do diagnóstico** (jobs 38292–38294). Se ele mostrar que só os braços com Eikos vazam, os
dados mudam antes, por emenda registrada aqui.

## Braços (sementes 0 e 1)

Todos partem do Lumi-4B (`27b30f4`). A receita base é a do E3b:
- fontes `felhen,telepatia,traces,eikos`, teto de 4.000 por família, 1 época;
- lr 1e-4, LoRA r16/α32 nos alvos do E3, `in_proj` incluso;
- lotes de 32 exemplos e 8.192 tokens por micro-lote.

A semente 1 usa `--data-seed 0`, como o E3b s1.

| braço | argumentos além da receita | pergunta |
|---|---|---|
| R0 `e6_r0_s<s>` | `--snapshots 25,50,75` | referência; reproduz o E3b |
| R1 `e6_r1_s<s>` | R0 + `--formatos v1,p1,p2,p3` | a variedade de formato, com leitura depois do prefixo, reduz o vazamento? |
| R2 `e6_r2_s<s>` | R1 + `--ancora-lambda 1.0 --ancora-cot 1` | as âncoras KL preservam o Lumi sem perder a decisão? |
| R3 (condicional) | R2 + autodestilação (protocolo próprio) | só se R1 e R2 reprovarem na preservação |

- **Exposição casada.** R1 e R2 usam exatamente o plano de lotes do v1, o mesmo do R0. São os mesmos exemplos, a
  mesma ordem e as mesmas atualizações; só o texto de cada exemplo muda (o formato) e, no R2, entram as âncoras na
  mesma atualização.
  - O `treino_meta.json` registra, em `e6`, as exposições por formato, os tokens de decisão, os tokens dos
    contextos `cot` e o sha256 dos formatos sorteados.
- **Conferência de reprodução.**
  - O `dados_sha256` do R0 (e também do R1 e do R2) tem de ser igual ao do `runs/e3b_eikos_s<s>/treino_meta.json`.
  - A média do R0 nas 4 com professor tem de ficar a até 1 ponto da do E3b na mesma semente.
  - Se uma das duas falhar, o E6 para e investigamos.

## Formatos de decisão (`src/lumi_decision/prompts.py`, `FORMATOS`)

| formato | cabeçalhos | rótulo das opções | instrução | resposta |
|---|---|---|---|---|
| `v1` | Contexto / Pergunta / Opções | `A. ` | Responda apenas com a letra da opção escolhida. | letra no 1º token |
| `p1` | Contexto / Pergunta / Opções | `A) ` | Responda no formato 'Resposta: <letra>'. | "Resposta:" preenchido; lê " A" |
| `p2` | Texto / Questão / Alternativas | `(A) ` | Indique a alternativa correta no formato 'Resposta: <letra>', sem justificar. | idem |
| `p3` | Situação / O que decidir / Escolhas | `A - ` | Escolha uma das opções e responda só com 'Resposta: <letra>'. | idem |

- **Sorteio do formato:** uniforme entre os 4, por hash de (caso, semente da época).
- **Nenhum formato copia o texto do Lumi-Benchmark** (teste em `tests/test_e6.py`).
- **Leitura depois do prefixo.** A letra lida é o próximo token da geração depois de "Resposta:", conferido nas 20
  primeiras ocorrências de cada (formato, nº de opções) e indexado por esse par.
- **A avaliação das reservadas continua no `v1`**, comparável com tudo o que já foi medido.

## Âncoras KL (R2; `src/lumi_decision/lora.py`, `batch_loss_ancorado` e `batch_cot`)

- **Referência:** o próprio modelo com o adaptador desligado (`disable_adapter`), sem gradiente e no mesmo modo de
  treino. O Lumi não tem dropout fora do LoRA. Sem treino, toda KL é zero (teste).
- **Por exemplo de decisão**, numa passada sobre o prompt mais a letra do gabarito:
  - `L_mass`: KL de Bernoulli da massa nas letras contra o resto, na posição da decisão;
  - `L_out`: KL da distribuição fora das letras, renormalizada, na posição da decisão;
  - `L_pos`: KL da distribuição inteira logo depois da letra e, nos formatos com prefixo, no início da resposta
    (pesos 0,5 e 0,5). No `v1` o início da resposta é a própria decisão, coberta por `L_mass` e `L_out`.
- **Contexto `cot`** (`--ancora-cot 1`): o mesmo caso com a instrução "Se ajudar, pense passo a passo antes de
  responder. Termine com uma linha 'Resposta: <letra>'.". Ancora a KL da distribuição inteira no 1º token da
  resposta, sem perda de decisão. Vai numa segunda passada da mesma atualização.
- **Detalhes:**
  - todas as KL são KL(Lumi ‖ modelo), com o top-64 da referência mais um balde de resto;
  - λ = 1, o padrão do artigo e o melhor no 4B deles;
  - a perda da atualização é (decisão + λ·âncoras) / nº de exemplos de decisão.

## Avaliação

- **Pontos avaliados:** os adaptadores de 25, 50, 75 e 100% de cada treino.
- **Decisão (primária):** dev das 4 reservadas com professor, todas as rotações, por ponto (`slurm/m3_prof.sbatch
  <run>:adapter_pNN`).
  - A métrica é a média com peso igual das 4 famílias.
  - O IC é de famílias fixas, reamostrando grupos, como na M3.
- **Preservação:** `slurm/guarda_corpo.sbatch`, parte `dev`, por ponto, sempre contra o Lumi-4B nas mesmas
  questões. As medidas:
  - Lumi-Benchmark dev, commit `4d517d8`: A `direct` (253 questões) com a trilha B (60) e A `cot` (253);
  - ENEM do VestBench em `cot`, metade `explorar`: 763 questões, uma ordem
    (`data/vestbench/enem_bench.jsonl`, sha256 `79e75795dd5f928462883f1ddd6882a20ac2bc4b186480710338376a10f96ec8`).
  - Configuração do job 38177: vLLM 0.33, bf16, raciocínio desligado, 2.048 tokens, temperatura 0, seed 42.
- **Secundárias, no adaptador final e no ponto escolhido:**
  - as 4 reservadas v2;
  - as 3 originais pela regra do M1;
  - as vistas;
  - `duplicate` da invoice em acurácia balanceada.

## Critério (margens aceitas pelo Gabriel em 27/11/2026)

1. **Preservação** (`scripts/guarda_corpo.py`), em cada ponto e cada semente:
   - em A `direct`, A `cot` e ENEM `cot`, a diferença contra o Lumi-4B fica em no mínimo −2 pontos e o IC 95%
     pareado não fica inteiro abaixo de zero (bootstrap reamostrando questões, 10.000 réplicas, semente 0);
   - na B (chamada exata), a diferença fica em no mínimo 0.
   - Passa quem passa nas quatro medidas.
2. **Decisão:** R1 − R0 e R2 − R0 na média das 4 com professor, no adaptador final, por semente.
   - É "não inferior" se o limite inferior do IC 95% ficar em no mínimo −1,5 ponto.
3. **Receita adotada:** a que passa na preservação (adaptador final) e é não inferior na decisão, nas 2 sementes.
   - Se mais de uma passar, fica a de maior média de decisão nas 2 sementes.
   - Se nenhuma passar na preservação, entra o R3, com protocolo próprio.
4. **Ponto do treino para a entrega:** dentro da receita adotada, o ponto (25, 50, 75 ou 100%) que passa na
   preservação nas 2 sementes com a maior média de decisão.
   - A confirmação final (Lumi-Benchmark test, ENEM `confirmar`, teste das reservadas) é lida uma vez, só nesse
     ponto.
5. **O R0 e o E3b reprovam na preservação por construção.** No dev, o E3b s1 fica em −8,7 [−14,2; −3,2] em
   A `direct` e −13,0 [−19,4; −6,7] em A `cot`. A medida no Lumi-4B e nos E3b s0/s1 sai antes do E6 (Etapa 1).

## Comandos (por semente `s`; na s1 acrescentar `--seed 1 --data-seed 0`)

```
sbatch slurm/e3_lora_h2.sbatch e6_r0_s0 --sources felhen,telepatia,traces,eikos --snapshots 25,50,75
sbatch slurm/e3_lora_h2.sbatch e6_r1_s0 --sources felhen,telepatia,traces,eikos --snapshots 25,50,75 --formatos v1,p1,p2,p3
sbatch slurm/e3_lora_h2.sbatch e6_r2_s0 --sources felhen,telepatia,traces,eikos --snapshots 25,50,75 --formatos v1,p1,p2,p3 --ancora-lambda 1.0 --ancora-cot 1
sbatch --dependency=afterok:<treino> slurm/m3_prof.sbatch e6_rX_s0 e6_rX_s0:adapter_p25 e6_rX_s0:adapter_p50 e6_rX_s0:adapter_p75
sbatch --dependency=afterok:<treino> slurm/guarda_corpo.sbatch e6_rX_s0 e6_rX_s0:adapter_p25 e6_rX_s0:adapter_p50 e6_rX_s0:adapter_p75
```
