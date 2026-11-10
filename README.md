# lumi-decision

Experimentos para adaptar o **Lumi-4B** (CEIA/UFG, `ceia-br/lumi-4b`) como *decision model* tipado em PT-BR:
dado um contexto, uma pergunta e um conjunto de opções, o modelo devolve uma distribuição sobre as opções
numa única passada, sem gerar texto (linha Jev / System One).

## Hipóteses

| | Hipótese | Comparação |
|---|---|---|
| **H1** | O Lumi congelado já decide com qualidade suficiente, e sua distribuição permite reduzir o erro entre as decisões aceitas quando ele pode se abster. | Readout congelado (A0) contra acaso e maioria; risco × cobertura contra aceitar a mesma fração ao acaso. Geração curta como referência operacional. |
| **H2** | LoRA melhora a decisão também em famílias inéditas. | A1 (LoRA, mesmo readout) contra A0, separando famílias vistas e reservadas. TF-IDF supervisionado como referência nas vistas. |
| **H3** | Uma pointer head melhora qualidade ou estabilidade em relação ao LM head com LoRA. | B (pointer head + LoRA) contra A1 com mesmos dados, passos e alvos. Margem de equivalência ±2 pontos. |

## Mecanismo de readout

O prompt termina em `<|im_start|>assistant\n<think></think>` (thinking desligado), exatamente onde a geração
do Lumi escreve a letra. As opções recebem letras `A`..`Z`; cada uma precisa ser um único token nessa posição
(checado caso a caso em `prompts.candidate_tokens`). A decisão é o softmax das log-probs dessas letras.
No piloto, nessa posição, readout e geração gulosa escolheram a mesma opção em 80 de 80 casos.

## Dados e splits (`configs/splits.toml`)

| Fonte | Papel | Reservado para generalização |
|---|---|---|
| Felhen PT-BR Typed Decisions Bench | 7 tarefas brasileiras, avaliação principal | tarefas `juristcu_area`, `factck_veracidade` |
| Typed Decisions PT (telepatia) | workflows com distribuição de referência; as 5 perguntas de um estado ficam no mesmo split | workflow `invoice_processing` |
| Tool Calling Traces PT-BR | escolher ferramenta ou responder direto (menu de 5 com distratoras) | tipos `viagens`, `saude` |

- Revisões fixadas em `src/lumi_decision/data.py`. `lumi-dados baixar` grava `data/manifesto.json` com SHA-256.
- Splits: `train`, `dev`, `calib` (temperatura e limiares), `test` (oficial, não usado para nenhuma escolha).
- Famílias reservadas nunca entram em treino (`data.training_cases`) nem em calibração. O dev delas mede transferência.
- `lumi-dados resumo` mostra as contagens e falha se um grupo cruzar splits.

## Onde roda o quê

| Trabalho | Onde |
|---|---|
| Dados, prompts, avaliador, métricas, testes, iteração de template com GGUF | PC local (`--backend llamacpp`, Vulkan) |
| Números oficiais (checkpoint BF16), E0, E1, treinos (E2–E4), Nemotron base, Saracura/Eikos | Cluster H100 via Slurm (`--backend hf`) |

## Uso

```bash
uv venv --python 3.12 && uv pip install -e ".[dev,baselines]"
lumi-dados baixar
pytest -q

# Local (GGUF via llama.cpp)
lumi-eval run --backend llamacpp --model lumi-4b-Q6_K_XL.gguf --server-bin llama-server.exe \
  --source felhen --split dev --orders canonical,reversed --out runs/h1_felhen_gguf
lumi-eval baselines --source felhen --split dev --out runs/baselines_felhen_dev
lumi-eval comparar runs/A runs/B          # IC 95% pareado, bootstrap por grupo

# Cluster DGX (partição h100n2), sempre da raiz do repo em /raid/$USER/lumi-decision
mkdir -p logs
sbatch slurm/00_preparar_ambiente.sbatch  # imagem Apptainer, venv, kernels, dados, checkpoints, testes
sbatch slurm/e0_sanidade.sbatch
sbatch slurm/e1_h1_dev.sbatch
squeue -u $USER
```

### Regras do cluster aplicadas nos scripts

- Tudo (repo, venv, caches do HF/pip/torch/triton, imagens `.sif`, checkpoints) fica em `/raid/$USER`, nunca na home (`slurm/config.sh`).
- Nada usa GPU fora do Slurm: preparação do ambiente, compilação do `mamba-ssm`, E0 e E1 são `sbatch`.
- Execução dentro de container Apptainer (`apptainer exec --nv`), imagem NGC PyTorch por padrão
  (`LUMI_NGC_IMAGE` em `slurm/config.sh`; confirmar a tag).
- Jobs de avaliação são reentrantes (pulam casos já gravados); treinos terão checkpoint em `/raid/$USER/checkpoints`
  e tratamento de `--signal=B:SIGUSR1@300`.

Cada execução grava `predicoes.jsonl` (uma linha por caso × ordem), `meta.json` (backend, modelo, versão do
prompt, commit) e `relatorio.{json,md}` (acurácia, acurácia balanceada por família, NLL, Brier, ECE,
risco × cobertura, sensibilidade à ordem, latência).

## Experimentos

| | Objetivo | Estado |
|---|---|---|
| Piloto | 40 casos de dev, GGUF Q4 local | feito; reproduzido pelo repo em `referencias/piloto40` |
| E0 | Sanidade do BF16 oficial: template, readout × forward, padding, parada, cache, concordância com o piloto | `lumi-e0` pronto; falta rodar no cluster |
| E1 | H1 no dev completo das três fontes, duas ordens; Nemotron base como controle | `slurm/e1_h1_dev.sbatch` pronto; depende do E0 |
| E2 | Gradientes do LoRA por tipo de módulo, salvar e recarregar | a fazer |
| E3 | H2: A1 contra A0 | a fazer |
| E4 | H3: pointer head contra A1 | a fazer |

## Achados técnicos já verificados

- O Jinja padrão deixa um `\n` antes de `<|im_start|>` neste template; o transformers (trim/lstrip_blocks)
  não. Seguimos o transformers. No piloto, essa diferença mudou 4 de 80 decisões, todas em quase-empates.
- No código remoto do Lumi, o bloco não repassa `attention_mask` aos mixers: usar padding à direita.
- No treino com kernel fundido, `out_proj` do Mamba recebe o peso direto, então LoRA nesse módulo não treina.
- `transformers>=5` tem `nemotron_h` nativo; o E0 compara remote code e implementação nativa.
- Na geração com GGUF, todas as respostas terminaram no token 11 (`<|im_end|>`), mas o config do HF diz EOS 2.
  O E0 verifica isso no BF16.
- Para o controle com o Nemotron base, confirmar que o chat template dele é igual ao do Lumi antes de reaproveitar o prompt.
