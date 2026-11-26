# Lumi Decision: E3b, semente 1

Este pacote traz o **Lumi-4B** com um adaptador LoRA treinado pelo projeto lumi-decision (CEIA/UFG). O adaptador ensina o modelo a **escolher uma opção entre várias**. Ele é o melhor checkpoint que temos nas famílias de decisão reservadas, ou seja, que nunca entraram no treino.

| pasta ou arquivo | o que é |
|---|---|
| `base/` | Lumi-4B original (`ceia-br/lumi-4b`, revisão `27b30f40b930fd05cee5e1c1f8790f70fa5b816d`), sem alterações |
| `adapter/` | adaptador LoRA (PEFT): r=16, alpha=32, alvos `in_proj` (Mamba), `q/k/v/o_proj` e `up/down_proj`, 68 MB |
| `exemplo.py` | carrega base + adaptador e decide; aceita um JSONL de casos |
| `treino.json` | configuração do treino e número de exemplos por família de treino |
| `ambiente.txt` | versões em que o pacote foi conferido |
| `SHA256SUMS` | hashes de todos os arquivos |

## Como usar

O modelo precisa de GPU CUDA, `transformers`, `peft`, `mamba-ssm` e `causal-conv1d`, porque o Lumi é híbrido Mamba-2 + atenção. As versões conferidas estão em `ambiente.txt`.

> **Use a implementação nativa do transformers (`trust_remote_code=False`).** O adaptador foi treinado nela, com os módulos chamados `model.layers.*`. Com `trust_remote_code=True` os módulos viram `backbone.layers.*`, o PEFT não encontra as chaves do adaptador e carrega o LoRA zerado, sem erro (só um aviso de "missing adapter keys"). O resultado é o Lumi-4B original. O `exemplo.py` confere isso e para se o adaptador não entrar.

Para rodar o exemplo embutido:

```bash
python exemplo.py
```

Para rodar os seus casos:

```bash
python exemplo.py casos.jsonl > saida.jsonl
```

Cada linha de `casos.jsonl` tem este formato:

```json
{"contexto": "...", "pergunta": "...", "opcoes": ["...", "..."]}
```

Em Python:

```python
from exemplo import carregar, decidir
tok, model = carregar()
decidir(tok, model, contexto, pergunta, ["opção 1", "opção 2", "opção 3"])
# {"escolha": 2, "letra": "C", "probs": [...], "gerado": "C"}
```

## Formato com que o modelo foi treinado e avaliado

O adaptador só foi treinado neste formato. Se o prompt mudar, o desempenho pode cair.

- **Mensagem do usuário** (chat template do Lumi, com `enable_thinking=False`):

  ```
  Contexto:
  <contexto>

  Pergunta: <pergunta>

  Opções:
  A. <opção 1>
  B. <opção 2>
  ...

  Responda apenas com a letra da opção escolhida.
  ```

- **Leitura da resposta.** Usamos as probabilidades das letras candidatas no primeiro token da resposta, que vem logo depois de `<|im_start|>assistant\n<think></think>`. Escolhemos a letra mais provável. Isso é o que `decidir` devolve em `escolha` e `probs`.
- **Geração gulosa.** `max_new_tokens=4` devolve a mesma letra em quase todos os casos (campo `gerado`).
- **Ordem das opções.** Nós avaliamos com as opções em várias ordens (rotações) e tiramos a média, porque a decisão muda com a ordem em parte dos casos. Se o benchmark de vocês usa uma ordem só, vale rodar também com as opções embaralhadas.

## Treino

O treino foi uma época de LoRA sobre 36.541 exemplos de escolha entre opções, com 900 atualizações, lr 1e-4 e semente 1:

- **Felhen** (~16 mil): classificação em PT-BR.
- **Telepatia** (~3,7 mil): fluxos de trabalho de agentes e atendimento.
- **Traces** (~300): escolha de ferramenta.
- **Eikos** (~16 mil): corpus de decisão, 85% em inglês, com regras, políticas e raciocínio temporal.

A lista completa de famílias está em `treino.json`, em `por_familia`. Se o benchmark de vocês tiver dados dessas fontes, avisem, porque isso seria contaminação.

## Resultados internos

Todos os números são do dev do projeto, nas famílias reservadas, nunca vistas no treino. A tabela dá a acurácia média por família, com todas as ordens das opções:

| modelo | 4 famílias PT-BR geradas por professor | 7 reservadas programáticas / de datasets | 11 reservadas |
|---|---|---|---|
| Lumi-4B sem adaptador | 60,8% | 58,5% | 59,3% |
| LoRA só com Felhen/Telepatia/Traces | 68,9% | 61,5% | 64,2% |
| **este pacote** (+ Eikos) | **72,6%** | **63,9%** | **67,1%** |

Nas famílias vistas no treino, o Eikos não piorou nada.

## Limitações conhecidas

- **Só foi medido em escolha entre opções lida pela letra.** Não medimos geração livre, conhecimento geral nem raciocínio longo com este adaptador. Uma variante anterior, sem Eikos, não piorou no ENEM; esta não foi testada lá.
- **Tende a responder "sim"** em perguntas sim/não sobre se uma fatura é duplicada: disse "sim" em 557 de 600, e só 56 eram duplicadas.
- **Não melhora em correferência** (pares mínimos do tipo Winograd): fica em ~61%, contra 58% sem o adaptador.
- **Não pede informação faltante.** Quando o documento necessário para decidir não está no contexto, quase nunca escolhe a opção "falta informação".
