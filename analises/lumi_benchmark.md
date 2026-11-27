# Lumi-Benchmark: Lumi-4B e Lumi Decision E3b (semente 1)

Rodadas feitas pelo time do lumi-decision a pedido do time de benchmark, em 26/11/2026.

## O que foi rodado

| modelo | o que é |
|---|---|
| `lumi-4b` | `ceia-research/lumi-4b`, revisão `27b30f40b930fd05cee5e1c1f8790f70fa5b816d`, sem alteração |
| `lumi-decision-e3b-s1` | o mesmo Lumi-4B com o adaptador LoRA `lumi-decision/lumi-decision-e3b-s1` fundido nos pesos |

Cada modelo foi rodado nas duas variantes (`direct` e `cot`) e nos dois splits (`dev` e `test`), o que dá 8 rodadas.

## Configuração, igual para os dois modelos

- **Repositório:** commit `4d517d8` do Lumi-Benchmark. O `manifest.json` mostra `git_commit: None` porque o runner rodou dentro de um container sem git.
- **Dados:** os sha256 dos arquivos de `data/benchmark/` batem com a tabela do README. Os hashes da tabela são dos arquivos com fim de linha CRLF.
- **Servidor:** vLLM 0.33, bf16, numa H100. O contexto máximo foi de 16384 tokens, com `--max-logprobs 20`.
- **Raciocínio desligado.** O chat template do Lumi liga o raciocínio (`<think>`) por padrão. Usamos o mesmo template com `enable_thinking=False` como padrão, porque o adaptador foi treinado sem raciocínio e o runner não passa argumentos ao template. Os números deste pacote valem para o Lumi **sem raciocínio**. Com o raciocínio ligado, o resultado do Lumi-4B pode ser outro.
- **Geração:** como no guia: `temperature` 0, `seed` 42, `max_new_tokens` 2048, adaptador `default`, prompts v1 sem alteração.

## Resumo no test (acertos/total)

| trilha | métrica | lumi-4b | lumi-decision-e3b-s1 |
|---|---|---|---|
| A, `direct` | acurácia | 261/547 (47,7%) | 256/547 (46,8%) |
| A, `cot` | acurácia | 319/547 (58,3%) | 261/547 (47,7%) |
| B | chamada exata | 103/140 (73,6%) | 113/140 (80,7%) |
| B | sem chamada esperada, acerto | 15/21 | 19/21 |

**Leitura:**
- **Trilha B:** o adaptador melhora. Na comparação item a item, o E3b acerta 15 chamadas que o Lumi-4B erra e erra 5 que o Lumi-4B acerta.
- **Trilha A, `cot`:** o adaptador piora, porque o E3b quase não raciocina antes de responder (20 tokens em média, contra 277 do Lumi-4B). A maior queda é em MATH-PT, de 50,7% para 27,1%.
- **Trilha A, `direct`:** os dois modelos ficam praticamente empatados.
- **Validade:** no `cot`, a validade do Lumi-4B em MATH-PT é 73,6%, porque parte das respostas passa de 2048 tokens.

## Dev (acertos/total)

| trilha | métrica | lumi-4b | lumi-decision-e3b-s1 |
|---|---|---|---|
| A, `direct` | acurácia | 138/253 | 116/253 |
| A, `cot` | acurácia | 158/253 | 125/253 |
| B | chamada exata | 49/60 | 51/60 |

## Conferências feitas antes das rodadas

- **Job 38177, fusão:**
  - o LoRA foi carregado em 71 módulos;
  - o modelo fundido decide igual ao LoRA sem fundir em 20/20 prompts do dev das reservadas, com KL máximo de 0,006;
  - o fundido difere da base nos 20/20 prompts;
  - só os 71 tensores com LoRA mudaram.
- **Job 38176, pacote de entrega:** o `exemplo.py`, na implementação nativa, reproduz as predições do dev em 40/40 casos.

As rodadas estão em `/raid/$USER/Lumi-Benchmark/runs/*sem-thinking*`. O job que as gerou é o `slurm/lumi_benchmark.sbatch`.
