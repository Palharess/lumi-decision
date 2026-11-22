# Reservadas v2: desenho de novas famílias reservadas (15/11/2026, proposta)

**Por quê:** a pergunta do projeto é se o treino transfere para **famílias novas**. Hoje há 3 reservadas atuais
(`factck`, `juristcu`, `invoice`), mais a `pss_mais_simples`, que é lida pelo tamanho da frase. Com tão poucas
famílias, o veredito depende de qual família reage, e não da receita. O `factck` decidiu os vereditos do F1, do
F2.1, do F2.2 e do piloto F3. Nada aqui foi gerado nem avaliado ainda.

## Quantas famílias e quantos casos (medido nos dados do M1)

**Dentro de uma família.** O ruído vem da diferença pareada por grupo (caso ou fatura) entre dois braços
treinados. Nas comparações do F2.1, F2.2 e F3, o desvio padrão dessa diferença fica entre 0,14 e 0,28 em
`factck` (6 ordens) e `juristcu` (2 ordens). Em `invoice` fica entre 0,08 e 0,11, porque cada fatura junta 5
perguntas. Com desvio padrão de 0,25:

| Grupos por família (dev) | Meia-largura do IC numa família |
|---|---|
| 102 (`factck` hoje) | ±4,9 pontos |
| 300 | ±2,8 |
| 500 | ±2,2 |
| 1.000 | ±1,5 |

**Entre famílias.** As famílias não reagem igual:
- piloto F3: −8,0, +1,5 e +3,7 (s0); −11,3, +1,5 e +2,2 (s1);
- E3 − congelado: +10,9, +0,2 e +4,8.

O desvio padrão entre famílias fica entre 5 e 8 pontos. O bootstrap registrado trata as 3 famílias como fixas:
ele diz quanto o braço ganha **nestas três**, não numa família nova qualquer. Tratando a família como unidade
amostral (efeito aleatório), a meia-largura do IC da média fica perto de t × 6/√K:

| Reservadas (K) | IC da média sobre famílias novas |
|---|---|
| 3 | ±15 pontos |
| 8 | ±5 |
| 11 (3 atuais + 8 novas) | ±4 |
| 15 | ±3,3 |

**Conclusão:** o gargalo é o número de famílias, não o número de casos. A proposta é **8 famílias novas**, cada
uma com **pelo menos 300 grupos no dev e 300 no teste**. Com as 3 atuais, são 11 reservadas.

## Critérios para uma família entrar

1. **Tipo de decisão ausente das famílias vistas e das outras reservadas.** Hoje as vistas cobrem:
   - categoria de um texto (tema, área, ofensa, intenção, emoção);
   - resposta correta a uma pergunta sobre um texto;
   - escolha, sim/não e nota em atendimento, segurança e observabilidade;
   - escolha da ferramenta do agente.
   As reservadas atuais cobrem veracidade, área jurídica e faturas.
2. **Domínio separado:** não usar atendimento ao cliente (Telepatia), saúde nem viagens (reservadas do Traces),
   nem notícias falsas ou classificação jurídica.
3. **PT-BR.**
4. **Gabarito verificável:**
   - por construção (regra executável),
   - ou fixado antes do texto (o professor escreve o texto a partir do gabarito) e conferido por um segundo
     modelo e por amostra humana.
   Gabarito só "na opinião do professor" mede concordância com o professor, e isso não entra.
5. **Sem atalhos conhecidos, auditados antes de registrar:**
   - posição: todas as ordens previstas abaixo;
   - tamanho da opção correta;
   - sobreposição de palavras entre o estado e a opção correta;
   - classe majoritária;
   - só a hipótese (nas famílias de inferência).
   - **Atalhos de forma** (não leem o problema: posição, tamanho, sobreposição de palavras, classe majoritária,
     um atributo só): nenhum pode passar de min(60%, acaso + 20 pontos).
   - **Atalhos de leitura parcial** (ignoram uma parte do problema: uma regra, um termo do custo, as datas
     relativas): nenhum pode passar de 80%, ou seja, cada parte decide pelo menos 1/5 dos casos.
   - Refinado em 15/11, antes de qualquer avaliação. A regra única de 60% não servia para as leituras
     parciais: numa família de regras com 4 classes equilibradas, ignorar a regra que define uma classe acerta
     as outras três por construção (cerca de 75%).
6. **Dificuldade, medida só com o Lumi congelado,** numa fatia de calibração de 50 grupos que depois é
   descartada:
   - fica a família com acerto entre o acaso + 5 e 90%;
   - **não se olha nenhum adaptador treinado** antes de registrar, senão a escolha das famílias passa a
     favorecer um braço.
7. **Independência do treino:**
   - nenhuma família vista futura sai do mesmo gerador ou do mesmo molde de uma reservada;
   - auditoria de sobreposição (texto exato e 13 palavras seguidas) contra todo o treino e calib, como no F3.
8. **Ordens:** todas as rotações das opções, para cada opção passar por cada posição uma vez (n ordens para n
   opções). O `factck` segue nas 6 permutações (regra da M1).

## As 8 famílias propostas

**Programáticas (regra executável, molde escrito por nós em PT-BR, sem licença de terceiros):**

| Família | Decisão | Opções | Gabarito |
|---|---|---|---|
| `restricoes_oferta` | qual das ofertas (aluguel, curso, equipamento) cumpre todas as condições do pedido | 4 | exatamente uma cumpre; cada distratora viola uma condição diferente |
| `regra_beneficio` | pedido de bolsa, licença ou auxílio contra um regulamento curto gerado por caso: conceder / negar / pedir documento / encaminhar | 4 | motor de regras; os limiares mudam a cada caso |
| `custo_plano` | qual plano (energia, internet, frete) sai mais barato para um perfil de uso, com taxas e faixas | 3–4 | cálculo; sem empate; o mais barato nem sempre tem o menor preço unitário |
| `ordem_eventos` | num relato curto, qual evento aconteceu primeiro, ou por último | 3–4 | linha do tempo gerada antes do texto; a ordem do relato não é a cronológica |

**Geradas por um professor com o gabarito fixado antes (texto natural):**

| Família | Decisão | Opções | Controle extra |
|---|---|---|---|
| `postura_alvo` | um comentário é a favor, contra ou neutro sobre um alvo (política pública, produto, regra de condomínio) | 3 | o alvo é dado na pergunta; o mesmo texto aparece com alvos diferentes |
| `inferencia_pt` | dada uma premissa, a hipótese decorre, contradiz ou é neutra | 3 | teste só com a hipótese; um trio de hipóteses por premissa no mesmo grupo |
| `pedido_agente` | um pedido a um assistente: executar, pedir esclarecimento (falta um dado) ou recusar (fora do escopo declarado) | 3 | o escopo e a ferramenta disponível vêm no estado; um pedido de cada tipo por cenário |
| `resposta_suficiente` | um trecho responde à pergunta: sim, em parte ou não | 3 | os trechos "em parte" e "não" são editados do "sim", para não diferirem em tema |

- Tipos novos entre as 8: satisfazer restrições, aplicar uma regra, comparar custos, ordem temporal, postura,
  inferência, decidir entre agir e perguntar, e suficiência de evidência. Nenhum aparece nas vistas.
- `resposta_suficiente` fica perto do `faquad_resposta` visto. Ela entra como teste de transferência próxima e é
  lida à parte.
- **A `pss_mais_simples` continua,** com leitura principal nos 406 pares estritamente mais longos.

## Geração e registro

1. **Código:** uma fonte nova, `res2` (fora de `LOADERS`, como o `f3`), com sal próprio no `splits.toml`.
   - A semente do gerador e a versão do molde ficam fixadas.
   - Cada família gera 650 grupos: 50 de calibração de dificuldade, 300 de dev e 300 de teste.
2. **Professor (só nas 4 famílias geradas):**
   - modelo aberto rodando no cluster, com revisão fixada;
   - o gabarito é sorteado antes, e o professor escreve o texto para ele;
   - um segundo modelo, de outra família, classifica às cegas, e os grupos em que ele discorda saem;
   - amostra humana de 30 grupos por família, com a taxa de erro registrada.
3. **Auditorias:** atalhos, dificuldade (só o congelado) e sobreposição, num `auditar-res2` como no F3.
4. **Registro:** as famílias que passam entram no `splits.toml` com data e commit, **antes** de qualquer treino
   ou avaliação de adaptador nelas.
5. **Leitura retroativa:** depois do registro, todos os checkpoints existentes são avaliados no dev das
   reservadas v2, como foi feito na M1. É uma leitura secundária e não muda vereditos.

## O que muda no critério dos próximos testes (proposta; registrar junto com as famílias)

- **Principal:** a média com peso igual sobre as 11 reservadas, com IC em dois estágios (reamostra famílias, e
  grupos dentro de cada família). Esse IC responde se o braço ajuda numa família nova qualquer.
- **Ao lado:** o critério antigo (3 reservadas, famílias fixas) e o nº de famílias em que o braço ganha e perde.
- **Ajuda / atrapalha:** IC inteiro acima ou abaixo de zero nas duas sementes, como antes.

## Decisões em aberto

- **Qual professor:** é o que mais pesa no custo e no prazo.
- **A ordem do trabalho:** as 4 programáticas não dependem de professor e podem sair primeiro.
- **O critério novo:** se ele passa a ser o principal assim que as reservadas v2 forem registradas, ou só depois
  de uma rodada de leitura retroativa.

## Programáticas: implementação e auditoria (15/11/2026)

- **Código:** `src/lumi_decision/res2.py`, fonte `res2` (fora de `LOADERS`), seção `[res2]` do `splits.toml`
  (sal próprio). Testes em `tests/test_res2.py`; auditoria com `lumi-dados auditar-res2`.
- **Tamanho:** 650 grupos por família: 50 na fatia `dificuldade`, e o resto dividido em dev e teste por
  sorteio (de 274 a 326 em cada).
- **Gabarito:** os 2.600 conferem por um verificador independente. No `ordem_eventos`, por exemplo, as datas
  são reconstruídas só a partir das frases do relato.
- **Posição do correto** na ordem canônica: equilibrada (de 147 a 184 por posição nas famílias de 4 opções;
  de 201 a 229 no `custo_plano`).
- **Sobreposição** com o treino e o calib de Felhen, Telepatia, Traces e F3: 0 casos.

**Atalhos (acertos em 650):**

| Família | Acaso | Pior atalho de forma (limite) | Pior leitura parcial (limite 80%) |
|---|---|---|---|
| `restricoes_oferta` | 25% | mais palavras do pedido: 212 (33%; limite 45%) | ignora a condição de categoria: 450 (69%) |
| `regra_beneficio` | 25% | classe majoritária: 168 (26%; limite 45%) | ignora os documentos: 495 (76%) |
| `custo_plano` | 33% | menor mensalidade: 246 (38%; limite 53%) | ignora a mensalidade: 392 (60%) |
| `ordem_eventos` | 25% | classe majoritária: 177 (27%; limite 45%) | só as datas explícitas: 303 (47%) |

Na primeira versão, "maior franquia" acertava 406/650 no `custo_plano` e "sem precedência" 454/650 na
`regra_beneficio`. A geração foi corrigida para controlar a maior franquia e criar mais conflitos de
precedência. Tudo isso foi feito antes de qualquer modelo ver os casos.

## Protocolo do filtro de dificuldade (registrado em 15/11/2026, antes de rodar)

- **Modelo:** só o Lumi congelado (mesma revisão e leitura pelas letras). Nenhum adaptador vê a fatia nem o dev
  antes do registro.
- **Casos:** a fatia `dificuldade` (50 grupos por família), em todas as rotações (`canonical,rot1,rot2,rot3`;
  `canonical,rot1,rot2` no `custo_plano`). Script: `slurm/res2_dificuldade.sbatch`.
- **Regra:** a família fica se o acerto (acertos/respostas) estiver entre o acaso + 5 pontos e 90%.
  - Abaixo disso, mesmo um modelo bom pode não se distinguir do acaso.
  - Acima, sobra pouco espaço para medir ganho.
- **Ajuste:** a família que falhar pode ser ajustada **uma vez** (por exemplo, menos ou mais condições). O
  ajuste é registrado e conferido numa fatia nova, com outro sal. Se falhar de novo, sai.
- **Também reportado** (não decide nada): acerto por rotação e por classe, e casos em que a resposta muda com
  a rotação.
- **Depois do filtro:** as famílias aprovadas são registradas aqui com data e commit. A fatia `dificuldade` é
  descartada. Segue a leitura retroativa dos checkpoints existentes no dev.

## Resultado do filtro de dificuldade (job 37122, Lumi congelado, fatia `dificuldade`)

| Família | Acertos | Faixa exigida | Veredito | Muda com a rotação | Observação |
|---|---|---|---|---|---|
| `custo_plano` | 76/150 (50,7%) | 38,3%–90% | **fica** | 36/50 | escolhe pouco a posição B (28/150) |
| `ordem_eventos` | 115/200 (57,5%) | 30%–90% | **fica** | 23/50 | |
| `regra_beneficio` | 155/200 (77,5%) | 30%–90% | **fica** | 9/50 | nunca escolhe "pedir documento" (0/36); as outras classes, 155/164 |
| `restricoes_oferta` | 192/200 (96%) | 30%–90% | **falha** (fácil demais) | 6/50 | |

**Ajuste 1 da `restricoes_oferta` (o único permitido):**
- 4 ou 5 condições por pedido, em vez de 3 ou 4;
- metade das condições numéricas ditas de forma indireta, para o modelo derivar o limite: orçamento anual,
  10 parcelas, dois dias de festa, pessoas que querem o próprio quarto, convidados mais a equipe, semanas de
  20 horas;
- metade das condições de categoria negativas ("não quero morar no bairro X").

O ajuste vale só para essa família, e as outras três ficam idênticas (conferido pela impressão sha256 dos
casos). Ela ganha sal próprio (`[res2.sal_por_familia]`), então a fatia de dificuldade é nova.

A auditoria refeita passa: o pior atalho de forma, "mais barata", acerta 218/650; o pior de leitura parcial,
"ignora a condição de preço", 464/650. Sobreposição com o treino: 0. O que é refeito: o filtro, na fatia nova
(`RES2_SUFIXO=_ajuste1`).

**Filtro da `restricoes_oferta` ajustada (job 37123, fatia nova):** 176/200 (88%), dentro da faixa: **fica**.
O acerto por rotação vai de 43/50 a 47/50, e a resposta muda com a rotação em 13 dos 50 grupos. Fica perto do
teto: um braço treinado só pode ganhar até cerca de 12 pontos nela.

## Registro (15/11/2026)

**As 4 famílias programáticas estão registradas como reservadas:** `restricoes_oferta` (ajuste 1),
`regra_beneficio`, `custo_plano` e `ordem_eventos`. Fonte `res2`, `splits.toml [res2]`, gerador
`RES2_VERSION = v1`. A fatia `dificuldade` está descartada: nenhum resultado dela entra em comparação.

| Família | Dev | Teste | Opções | Ordens |
|---|---|---|---|---|
| `restricoes_oferta` | 313 | 287 | 4 | 4 rotações |
| `regra_beneficio` | 308 | 292 | 4 | 4 rotações |
| `custo_plano` | 324 | 276 | 3 | 3 rotações |
| `ordem_eventos` | 321 | 279 | 4 | 4 rotações |

## Protocolo M2: leitura retroativa no dev das reservadas v2 (registrado em 15/11/2026, antes de rodar)

**O que é:** só avaliação. Os mesmos 20 checkpoints da M1, com a mesma leitura registrada de cada um, no `dev`
das 4 famílias, em todas as rotações. Script: `slurm/m2_res2.sbatch`.

**Leituras:**
- **Por checkpoint e família:** acertos/respostas; casos que mudam com a rotação; na `regra_beneficio`, também
  o acerto por classe.
- **Por comparação registrada** (os mesmos pares da M1):
  - diferença por família, com IC pareado reamostrando grupos;
  - média com peso igual sobre as 4 famílias novas;
  - média com peso igual sobre as 7 reservadas: as 3 atuais pela regra da M1 (`factck` em 6 ordens) e as 4
    novas. O IC sai do bootstrap de famílias fixas (o registrado) e do de dois estágios (reamostra famílias,
    depois grupos).
  - Também o nº de famílias em que o braço ganha e perde.

**O que decide:** nenhum veredito registrado muda; a M2 é secundária. Ela serve para ver se as conclusões
das 3 reservadas se repetem em 4 famílias independentes, sobretudo a H2 (E3 − congelado) e o piloto F3.

## Protocolo das 4 famílias com professor (registrado em 16/11/2026, antes de gerar)

**Por quê agora:** no E3c (`analises/e3c_eikos_tipo.md`), o Eikos ajudou nas 4 programáticas, mas duas delas
(`regra_beneficio`, `ordem_eventos`) têm tipos que o Eikos também tem. Para medir transferência **longe** do
treino, faltam reservadas de tipos que nem as vistas nem o Eikos cobrem.

**As famílias (escolha do Gabriel em 16/11).** Comparamos o desenho de 15/11 com as 27 famílias do Eikos:
- `pedido_agente` ficava perto de `tool_selection`, `routing` e `ambiguous` do Eikos e do Traces visto;
- `resposta_suficiente` ficava perto do `faquad` visto e das classes `unclear` e `cannot_determine` do Eikos.

As duas foram trocadas. Ficam:

| Família | Decisão | Opções | Grupo (mesmo split) | Controle por construção |
|---|---|---|---|---|
| `postura_alvo` | posição do autor de um comentário sobre um alvo | 3: a favor, contra, sem posição | 1 comentário, 2 alvos com posturas diferentes | ignorar o alvo acerta no máximo metade |
| `inferencia_pt` | relação entre um texto e uma hipótese | 3: decorre, contradiz, não dá para concluir | 1 texto, 3 hipóteses (uma por classe) | negação sorteada por hipótese, independente da classe |
| `resposta_indireta` | o que uma resposta indireta quer dizer a uma pergunta de sim/não | 3: sim, não, sim com condição | 1 pergunta, 3 respostas (uma por classe) | nenhuma resposta usa "sim", "não" e afins |
| `correferencia` | a quem um pronome se refere | 2: as duas pessoas | par de textos que diferem em até 3 palavras e trocam o referente | ignorar a palavra trocada acerta no máximo metade |

Domínios: condomínio, bairro e prefeitura, escola e universidade, trabalho, clube e esporte amador, cultura e
eventos, cidade e transporte, casa e família. Fora: atendimento ao cliente, saúde, viagens, notícias falsas,
direito, faturas e finanças.

**Modelos (revisões em `slurm/prof_config.sh`):**
- **Escritor:** Qwen3.8-27B (`37561816`), sem raciocínio, temperatura 0,8, top-p 0,95, semente por pedido. (Emenda de 22/11: com raciocínio, 0,6.)
- **Conferente:** Gemma-4-31B-it (`842da379`), de outra família, temperatura 0.
- Os dois rodam em vLLM v0.33.0 (imagem fixada), uma H100.

**Gabarito fixado antes do texto.** Cada grupo nasce de uma especificação sorteada por código (sal
`[prof]` do `splits.toml`): domínio, pessoas, alvos, a classe de cada item e as restrições de forma. O
escritor recebe a especificação e escreve o texto para aquele gabarito. Ele nunca escolhe a classe.

**Filtros, nesta ordem (nenhum olha um adaptador treinado):**
1. **Forma, por código:**
   - JSON válido e completo;
   - tamanhos dentro da faixa;
   - palavras proibidas ausentes:
     - em `postura_alvo`: "a favor", "contra", "neutro", "concordo", "discordo";
     - em `resposta_indireta`: "sim", "não" e afins;
     - em `inferencia_pt`: "talvez", "provavelmente", "possivelmente", "pode ser";
   - negação presente só onde a especificação manda (`inferencia_pt`);
   - em `correferencia`: os dois textos diferem em 1 a 3 palavras, as duas pessoas aparecem nos dois, e a
     pergunta não cita nenhum nome.
2. **Conferência às cegas:**
   - o Gemma recebe cada item no mesmo prompt que o Lumi vê, em duas ordens (canônica e `rot1`);
   - pode raciocinar em até 3 frases e termina com a letra;
   - o grupo fica só se o Gemma acerta todos os itens nas duas ordens.
3. **Seleção:** entre os grupos que passam, os 650 primeiros por sorteio (sal) ficam.
   - Divisão: 50 de dificuldade, e o resto meio a meio entre dev e teste, como no `res2`.
   - Se passarem menos de 650, gera-se uma rodada extra com índices novos (de 1.000 em diante) e a mesma
     regra. Se mesmo assim faltar, a família sai.

**Revisão humana (Gabriel), antes do registro:**
- 30 grupos por família, sorteados do dev e do teste.
- **Às cegas:** a planilha mostra o texto e as opções, sem o gabarito.
- Nas discordâncias, o Gabriel vê o gabarito e decide se ele é defensável.
- Conta como erro o grupo com algum item de gabarito indefensável.
- A família fica com até 3 grupos errados em 30; com 4 ou mais, sai.

**Auditoria de atalhos** (`lumi-dados auditar-prof`), com os limites de 15/11:
- **Atalhos de forma, limite min(60%, acaso + 20):**
  - posição do correto na ordem canônica;
  - opção mais longa e mais curta;
  - sobreposição de palavras;
  - classe majoritária;
  - um classificador TF-IDF + regressão logística (validação cruzada em 5 partes, por grupo), treinado só na
    parte do item que não basta para decidir:
    - em `inferencia_pt`: só a hipótese, mais a sobreposição com o texto;
    - em `resposta_indireta`: só a resposta;
    - em `postura_alvo`: só o comentário, sem o alvo;
  - em `correferencia`: a primeira pessoa citada e a pessoa citada mais perto do pronome.
- **Leitura parcial, limite 80%:**
  - em `postura_alvo`: ignorar o alvo (a mesma resposta para os 2 alvos do grupo; teto de 50% por
    construção);
  - em `correferencia`: ignorar a palavra trocada (teto de 50% por construção).
- **Sobreposição:** zero casos com texto igual ou 13 palavras seguidas no treino e calib de Felhen, Telepatia,
  Traces, F3 e Eikos.

**Dificuldade** (só o Lumi congelado, fatia `dificuldade`, todas as rotações): a mesma regra do `res2`. A
família fica com acerto entre o acaso + 5 e 90%. Uma família que falhar na auditoria ou na dificuldade pode
ser ajustada **uma vez**, com o ajuste registrado e um sal novo; se falhar de novo, sai.

**Registro:** as famílias aprovadas entram no `splits.toml [prof]` com o sha256 do arquivo final de casos
(`data/prof/prof_v1.jsonl`, fora do git, como os outros dados).

**Leitura M3 (só avaliação, depois do registro).** Os textos do professor não são reprodutíveis bit a bit, por
isso o arquivo é fixado pelo sha256. Checkpoints:
- congelado;
- E3 s0–s2;
- E3b e controle, s0 e s1;
- os 6 braços do E3c.

Leituras:
- **Pergunta principal: o ganho do Eikos chega a tipos que ele não tem?**
  - Comparação: E3b − controle na média com peso igual das 4 famílias com professor (dev, todas as rotações,
    IC com famílias fixas, reamostrando grupos).
  - "Transfere longe": IC acima de zero nas duas sementes. "Atrapalha longe": IC abaixo de zero nas duas.
    Qualquer outro caso: inconclusivo.
- **Secundárias:**
  - H2 (E3 − congelado) nas 4 com professor;
  - as 11 reservadas, com famílias fixas e em dois estágios;
  - cada família;
  - próximo − controle e resto − controle do E3c nas 4 com professor.
- **Previsão, escrita agora:** pelo E3c, o ganho do Eikos vem em boa parte do corpus em geral, então esperamos
  E3b − controle > 0 também aqui, menor que os +3,4 das programáticas.

## Emenda ao protocolo das famílias com professor (22/11/2026, antes da geração definitiva)

O primeiro lote de teste (job 37796, índices 5000–5011, fora da faixa 0–999 da geração definitiva) rodou de ponta a
ponta. Nenhum modelo avaliado viu esses textos; só o escritor, o conferente e esta leitura manual. Grupos que passam no
filtro de forma e depois no conferente:

| família | passam na forma | passam no conferente | principal motivo |
|---|---|---|---|
| `postura_alvo` | 12/12 | 12/12 | — |
| `inferencia_pt` | 12/12 | 7/12 | escritor erra o gabarito (ordem no tempo não escrita, "contradiz" que é neutra) |
| `resposta_indireta` | 4/12 | 1/4 | "não" proibido em toda resposta; "sim, se..." para perguntas sobre algo já feito |
| `correferencia` | 8/12 | 1/8 | pares ambíguos de verdade (sujeito composto "X e Y..., e ela...") |

O conferente acertou o diagnóstico nos casos lidos: os erros são do escritor. Mudanças, todas antes de gerar os
índices 0–999:

- **Escritor com raciocínio.** `enable_thinking=true`, temperatura 0,6 e top-p 0,95, que é a amostragem recomendada
  para esse modo, e até 8.000 tokens. O JSON é lido depois do `</think>`, e um raciocínio cortado conta como resposta
  cortada.
- **Inferência.** O pedido manda conferir cada hipótese contra o texto. O mecanismo de ordem no tempo passa a exigir
  ordem "dita explicitamente".
- **Resposta indireta.**
  - "não", "nunca" e "jamais" saem da lista proibida. A negação passa a ser sorteada por resposta (35%),
    independente da classe, como na inferência.
  - Os tipos de pergunta ficam só os que admitem "sim, se...": convite, pedido de ajuda, preferência, plano e pedido
    de permissão ou favor.
  - O uso não condicional de "se" e "quando" ganha exemplos, para evitar frases feitas.
- **Correferência.** O pedido descreve o esquema de Winograd: papéis diferentes na oração principal, nunca sujeito
  composto, e a palavra trocada inverte qual papel combina com o pronome. Vem com um esquema abstrato e a ordem de não
  copiá-lo.
- **Conferente:** sem mudança. A memória do vLLM foi ajustada: 0,85 no escritor, 0,92 no conferente e até 64
  sequências.

Os filtros de forma, o conferente às cegas em 2 ordens, a seleção, a revisão humana, a auditoria e o filtro de
dificuldade continuam como registrados. Um segundo lote de teste (índices 5100–5123) mede o efeito destas mudanças
antes da geração definitiva.
