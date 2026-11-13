# Fase 2: entender os resultados antes de mudar a receita (13/11/2026)

Plano revisado depois do E4, de duas revisões externas (8 e 9) e da literatura de set–nov 2026. Os números
de "o que sabemos" foram conferidos no código e nos dados; o que é hipótese está marcado como tal.

## Onde estamos em relação à literatura

| Escolha nossa | O que a literatura recente diz | Leitura |
|---|---|---|
| LoRA r=16 no backbone, LM head congelado, leitura pelas letras (E3) | *Releasing the Inner Jev* (2610.17205): o ganho vem de adaptar o backbone; congelar ou treinar as linhas das letras dá resultado parecido; ranks de 16 a 128 empatam. *Our Own Jev* (2610.29052): treinar as linhas das letras não ajudou. | Receita alinhada; r=16 fica como controle. |
| Mistura de treino: 20.350 exemplos, 3 fontes | *Small Heads, Strong Decisions* (2611.15592): 15 fontes, até 10 mil linhas cada, 8.594 passos. *Large Decision Models* (2610.15501): variedade deliberada de regras, domínios e significados dos rótulos melhora esquemas novos. | **Nossa mistura é estreita.** O Felhen tem 80% das linhas (16.334) com só 5 textos de instrução; o treino todo tem 83 instruções distintas; todas as linhas com 3 opções vêm do `olid_alvo`; nenhuma tem 10 opções (o `juristcu` tem). |
| Pointer head nova: bilinear 3136 → 256 sobre o fim de cada opção, lr igual à do LoRA (E4) | *Small Heads*: num estudo controlado, cinco tipos de leitura empatam nas tarefas vistas, e a vantagem da leitura pelas letras nas tarefas novas vem **toda** do conhecimento prévio do LM head sobre as letras. Ela só aparece quando a tarefa nova também rotula as opções por letras, e some se as letras forem trocadas na avaliação. Lá a lr de cada cabeça foi escolhida numa busca, pela NLL calibrada na validação das fontes de treino. | O E4 é compatível com isso: as nossas reservadas usam letras, e a cabeça nova perdeu esse conhecimento prévio. A lr igual à do LoRA controlou a comparação, mas não foi otimizada para a cabeça. |
| Leitura da pointer head no último token de cada opção, sem delimitadores | O kev usa `</opt>` e `<decide>`. *Small Heads*: a cabeça de cosseno lê o último token da pergunta e da opção. | Manter o prompt do E3 foi um bom controle; delimitadores próprios ficam como variante. |
| Uma passada, opções interagindo | *Candidate-Independent Block-Causal Attention* (2610.01601): isolar candidatos reduz a sensibilidade à ordem. No Lumi, uma máscara não isola, porque o Mamba carrega estado. *PACT* (2609.35865): um termo de consistência entre permutações reduz o viés de posição, comparado contra um controle com o mesmo otimizador e cronograma. | Isolamento real exige uma passada por opção. A consistência entre ordens é a alternativa barata. |
| LoRA em `in_proj` do Mamba, sem `out_proj` | O kernel fundido lê `out_proj.weight` direto e pula o LoRA. *MambaPEFT* (2411.03855): a melhor projeção depende da tarefa e do volume de dados. | Correto para o caminho fundido. O `in_proj` concentra 6,94 M dos 17,05 M parâmetros LoRA (41%); o efeito dele nunca foi medido. |

## O LoRA e a pointer head "não generalizaram"? Em parte

- **O LoRA transfere pouco, mas transfere algo.**
  - Nas reservadas, a estimativa é +2,7 [−0,3; +5,9]; `invoice_processing` sozinha tem +4,8 [+3,0; +6,8].
  - O viés de ordem do Felhen reservado cai de 32% para 12%.
  - O que chama atenção é a **distância** entre +15 a +20 nas famílias vistas e +3 nas novas.
- **A pointer head transfere pior que as letras**, sobretudo em `factck`.
- **Estudar essa distância é legítimo**, desde que como pergunta (o que limita a transferência?) e não como premissa de falha. Hipóteses, cada uma com um teste:
  1. **Pouca variedade de esquemas no treino:** o modelo aprende muitos textos, mas poucos tipos de decisão. Teste: mistura balanceada por esquema, com o mesmo orçamento de atualizações (F2).
  2. **Pouca margem de ganho:** o Lumi congelado já resolve parte das reservadas, e `factck` e `juristcu` exigem conhecimento que 508 atualizações não ensinam. Diagnóstico: ganho por pergunta e por número de opções contra a acurácia do congelado (F0).
  3. **Referências ambíguas:** nas famílias do Telepatia, a distribuição de referência dá em média só 0,61 a 0,77 à opção do gabarito (`urgency` 0,61; `duplicate` 0,93). Isso não é um teto (o gabarito é o argmax da distribuição em 98–100% dos casos), mas indica perguntas onde anotadores divergem e onde a acurácia oscila mais.
  4. **Para a pointer head: a perda do conhecimento prévio das letras** (*Small Heads*), além das opções que interagem numa passada causal. Testes: o diagnóstico de `factck` (F0) e a cabeça residual ou iniciada a partir das linhas das letras (F2).
  5. **Ruído entre treinos:** com uma semente não sabemos o tamanho da oscilação. Teste: réplicas (F1).

## F0: diagnósticos sem treino

**a. `factck` em 6 permutações** (Lumi congelado, E3, E4)
- Guardar as notas antes do softmax.
- No E4, cruzar no mesmo caso o estado final de uma ordem com os estados das opções de outra, alinhados pelo rótulo. Primeiro fixar um e trocar o outro; depois o inverso. A medida é a diferença de nota entre "impreciso" e "falso".
- Ler o **adaptador do E4 pelas letras**, sem a cabeça. Isso separa uma falha da cabeça de uma mudança nas representações que o LoRA aprendeu junto com ela.
- Nas outras reservadas: todas as ordens nas binárias e uma amostra balanceada de permutações em `juristcu`, que tem 10 opções.

**b. Provas em português: piloto com o ENEM**
- **Fonte:** VestBench-Jev (`danielpedrozo/vestbench-jev`, revisão fixada). Usar só as questões válidas (`excluded == False`) e, à parte, as que mencionam outras alternativas ("todas as anteriores"), que não podem ser invertidas.
- **Divisão:** cada prova em metade de exploração e metade intocada, por hash do id.
- **Modelos:** Lumi congelado (A0), E3 (A1), E4 (B), com o mesmo prompt e duas ordens. Depois estender ao E3b, ao controle e ao Nemotron, e às outras provas (BLUEX, FUVEST, OAB, POSCOMP), sempre por prova e com média entre provas.
- **Linguagem:** as provas ficam "fora do nosso fine-tuning, após auditoria", não "inéditas", porque a exposição no pré-treino é desconhecida.
- **Ressalvas:** a licença da coleção de origem não é declarada e o OAB veio de um site de terceiros; uso só para avaliação interna.

**c. Anatomia da distância entre famílias vistas e novas**
- Ganho do E3 sobre o congelado por família, pergunta, tipo (`noul`/`choice`/`score`) e número de opções.
- Esses ganhos cruzados com o que existe no treino: tipo de pergunta, número de opções e texto da instrução.

## F1: réplicas pareadas

- E3 e E4 com sementes 1 e 2 (4 treinos).
- A **lista de exemplos fica fixa** (hoje, trocar a semente de 0 para 1 troca 3.169 dos 20.350 exemplos): separar `data_seed` de `seed`.
- Dentro de cada réplica, os dois braços usam o mesmo plano e as mesmas ordens.
- Reportar a diferença por semente e por família.

## F2: variantes (cada uma registrada antes de rodar, julgada nas reservadas e na metade de exploração das provas, contra o ruído medido no F1)

1. **Mistura balanceada por esquema, mesmo orçamento:** teto por família menor (o Felhen deixa de dominar), com o mesmo número de atualizações do E3.
2. **E3 sem LoRA no `in_proj` do Mamba** (só atenção e MLP). Mede quanto os 41% de parâmetros no Mamba contribuem.
3. **Pointer head que preserva as letras:**
   - (a) residual: letras + cabeça iniciada em zero, ainda recebendo gradiente;
   - (b) leitura de letras aprendida, iniciada das linhas do LM head (Decision 3.0 / Inner Jev).
4. **Busca curta da lr da cabeça**, com a do LoRA fixa, escolhida pela NLL calibrada no `calib` das famílias **vistas** (como no *Small Heads*), nunca nas reservadas.
5. **Consistência entre ordens** (como no PACT): duas ordens com a perda normal, contra as mesmas duas ordens com a perda normal mais KL alinhado pelo significado da opção. Isso separa o efeito da penalidade do efeito de ver mais ordens.
6. **Âncora no Lumi congelado,** só se as provas mostrarem uma regressão que queremos preservar.

## Ajustes depois do primeiro resultado do F0a (revisão 10, conferida)

**`factck` nas 6 ordens (102 casos):**

| Leitura | Acurácia média | Decisão muda entre ordens | Escolhe a posição A |
|---|---|---|---|
| Lumi congelado | 44,1% | 80% | 45% |
| E3 pelas letras | 55,1% | 40% | 40% |
| Adaptador do E4 pelas letras | 43,6% | 100% | **75%** |
| E4 pela pointer head | 40,5% | 59% | 38% |

- **No cálculo da cabeça, o efeito está nos estados das opções.** Da ordem original para a invertida, a margem
  de "impreciso" sobre "falso" sobe 0,38 em logit. Trocando só os estados das opções, sobe 0,38; trocando só o
  estado final, 0,05. A origem disso no treino segue em aberto.
- **O adaptador do E4 lido pelas letras não mede bem as representações.** Ele escolhe a letra A em 75% dos
  casos: o LoRA do E4 nunca foi treinado para responder por letras, e essa leitura fica quebrada. Os −11,4 pontos
  contra o E3 misturam essa leitura quebrada com uma possível piora das representações. Separar as duas pede uma
  sonda linear treinada nos estados (das famílias vistas) e avaliada em `factck`, nos dois adaptadores.
- **Ordens repetidas:** nas perguntas binárias, `canonical,reversed,rot1,rot2,rot3` dá só duas ordens únicas.
  A análise de `invoice` deve pesar cada ordem única uma vez. Em `juristcu`, as 5 ordens são uma amostra
  exploratória; as 10 rotações fariam cada opção passar por cada posição.

**Mudanças no plano:**
- **F2.1 testa balanceamento, não diversidade.** Reduzir o peso do Felhen redistribui os esquemas que já
  existem. Diversidade nova (regras e famílias novas) é o F3. Um resultado nulo no F2.1 não fecha a pergunta da
  diversidade.
- **Prioridade depois do F0:**
  - **réplicas pareadas** do E3 e do E4 (F1);
  - **pointer residual iniciada em zero sobre o E3 já treinado.** O controle é o próprio E3 com o mesmo treino
    adicional, só pelas letras. Assim a cabeça herda o que o E3 aprendeu, e o treino a mais é igual nos dois
    braços.
- **Ablação do `in_proj` do Mamba** fica como teste próprio do LoRA (F2.2), com os outros alvos e o kernel
  fixos. Nada até aqui atribui a queda do E4 ao Mamba: ela apareceu na combinação de cabeça e treino.

## Protocolo F2.3a: pointer residual sobre o E3 (registrado em 13/11/2026, antes de rodar)

**Pergunta:** somar uma pointer head à leitura pelas letras, partindo do E3 já treinado, melhora a decisão nas
reservadas e evita o colapso em `factck` que a pointer head pura teve em 2 de 3 sementes?

**Braços (sementes 0 e 1; cada semente parte do E3 da mesma semente):**
- **Residual (`f23_residual_s<s>`):** o LoRA continua do adaptador do E3 (`--init-adapter`). A nota de cada
  opção é log-prob da letra + nota da pointer head.
  - A cabeça começa com W_k = 0, então a leitura inicial é exatamente a do E3. O treino confere isso e recusa
    rodar se houver diferença.
  - W_k recebe gradiente desde o 1º passo.
  - lr 1e-4 no LoRA e na cabeça.
- **Controle (`f23_controle_s<s>`):** o mesmo treino adicional, só pelas letras.
- **Igual nos dois braços:** os mesmos 20.350 exemplos, 508 atualizações e o mesmo cronograma. As ordens e os
  planos são os da 2ª época do E3 (`--epoch-offset 1`), então nenhum braço repete as ordens já vistas.

**Critério** (`lumi-eval h3 --readout-novo residual`, residual − controle, mesmo bootstrap da H3), por semente:
- **Principal:** qualidade nas 3 reservadas, peso igual, 2 ordens. Direção e equivalência ±2 registradas
  separadamente.
- **Secundário:** estabilidade (casos que mudam com a ordem).
- **Leitura entre sementes:** o residual "ajuda" só se o IC de qualidade ficar inteiro acima de zero nas duas
  sementes, e "atrapalha" só se ficar inteiro abaixo nas duas. Divergência ou cruzamento de zero é inconclusivo.
- **Diagnósticos:** acertos em `factck` por braço (o colapso some?); vistas; ENEM só depois de congelar.

## F3: dados mais variados e treino maior (depois de F0–F2)

**Hipótese:** o limite principal é a mistura de treino. Ela tem muitos textos, mas poucos tipos de decisão.
Mais variedade de esquemas e, depois, mais escala devem aumentar a transferência para famílias novas.

**Ressalva:** o E3b já acrescentou um corpus variado (Eikos, 27 famílias) e não ajudou (−0,8 [−3,1; +1,3]
contra o controle). Mas o Eikos é 87% inglês e focado em finanças e regras, e uma semente só não separa
efeito de ruído. Então F3 só começa depois de:
- a variante F2.1 (mistura balanceada, mesmo orçamento) mostrar algum sinal;
- o F1 dizer quanto os resultados oscilam.

Sem isso, mais dados é uma aposta cara sem leitura clara.

**Busca de datasets: critérios**
- **Idioma:** PT-BR primeiro. O Eikos sugere que dados em inglês ajudam pouco nas nossas reservadas.
- **Tipos de decisão:** variedade é o objetivo. Cobrir `noul`, `choice` e `score`; números de opções variados
  (3, 10 e outros que hoje faltam); significados de rótulo diferentes (entailment, severidade, intenção,
  postura, rota); instruções diferentes.
- **Licença** compatível com treino, revisão fixada e sha256, como já fazemos.
- **Auditoria de sobreposição** contra as três famílias reservadas (por tarefa, não só por texto, como no
  filtro do Eikos), contra as provas do VestBench-Jev e contra o teste oficial.
- **Rótulos:** distribuições suaves quando existirem. Medir a ambiguidade, que hoje varia de 0,61 a 0,93 de
  pico.

**Fontes a investigar**
- Datasets públicos de classificação e inferência em PT-BR, auditando se não repetem as reservadas: por
  exemplo, um dataset de notícias falsas se parece com `factck`, e um de classificação jurídica com
  `juristcu`.
- A parte em PT-BR do Eikos (2.158 linhas, fonte `eikos_pt` já implementada).
- Tarefas sintéticas geradas por um professor, no molde do *Large Decision Models*: tarefas novas com
  rótulos tirados de várias rodadas de raciocínio do professor e filtradas por consistência.

**Desenho, para não confundir variedade com volume**
- Variedade e volume em 2×2: mesma variedade com mais dados; mais variedade com o mesmo orçamento; e os dois
  juntos, cada braço com controle no mesmo número de atualizações.
- Reservar famílias inteiras dos dados novos como reservadas novas. Hoje são só três; a conclusão sobre
  transferência precisa de mais.
- **Escala:** o E3 levou 37 min por 508 atualizações. Um treino do porte do *Small Heads* (cerca de 8.600
  passos) leva perto de 10 h numa H100. Cabe no limite de 12 h, e já temos retomada testada.
- O *Small Heads* viu um 3º epoch custar de 0,7 a 2 pontos nas fontes novas: mais passos sobre os mesmos dados
  não é escala útil.

**Para fechar conclusões:** a metade intocada das provas e o teste oficial, lidos uma única vez com as receitas congeladas.

As hipóteses ficam como estão: H1 (Lumi congelado), H2 (LoRA contra congelado) e H3 (pointer head contra letras). As provas ampliam a avaliação das três.
