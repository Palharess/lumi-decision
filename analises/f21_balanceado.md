# F2.1: mistura balanceada (13/11/2026)

**Jobs:** 36911–36914, protocolo em `plano_fase2.md`.
- **Braço balanceado:** a receita do E3 com teto de 1.300 exemplos por família, 2 épocas cortadas nas mesmas
  508 atualizações (cerca de 20,7 mil exemplos vistos, 62% do Felhen).
- **Comparação:** contra o E3 da mesma semente (20.350 exemplos, 80% do Felhen).
- **Leitura das células:** acertos/respostas no dev, nas duas ordens do protocolo.

**Correção do orçamento (revisão 11, conferida em 14/11 nos `treino_meta.json`):** "mesmo orçamento" vale só
em parte.
- **Atualizações:** 508 contra 508 na semente 0, mas **508 contra 502 na semente 1**. O E3 da semente 1 agrupa
  os lotes de outro jeito e fecha a época em 502; o balanceado é cortado em 508 pelo `--max-steps`.
- **Contagem exata** (job 36958, `scripts/contar_orcamento.py`, que refaz os planos de lotes; reproduz as 508
  atualizações e os 8,11 M tokens do E3):

| Semente | Braço | Atualizações | Exemplos vistos | Tokens de conteúdo |
|---|---|---|---|---|
| 0 | E3 | 508 | 20.350 | 8.108.986 |
| 0 | Balanceado | 508 | 20.686 | 7.808.733 (−3,7%) |
| 1 | E3 | 502 | 20.350 | 8.108.986 |
| 1 | Balanceado | 508 | 20.775 | 7.911.249 (−2,4%) |

- O balanceado vê menos tokens porque os exemplos repetidos do Telepatia e do Traces são mais curtos que os do
  Felhen que saíram.
- As diferenças (6 atualizações na semente 1; 2–4% dos tokens) são pequenas perto do ruído entre sementes e não
  mudam o veredito (inconclusivo). Mas o texto não deve dizer "mesmo orçamento" sem essa ressalva. No piloto do
  F3, o braço diverso e o controle serão casados também em tokens, conferidos com esse script antes de submeter.

## Acertos

| Grupo | Congelado | E3 s0 | Balanceado s0 | E3 s1 | Balanceado s1 |
|---|---|---|---|---|---|
| `factck` (reservada) | 118/204 | 124/204 | 107/204 | 118/204 | 130/204 |
| `juristcu` (reservada) | 285/600 | 286/600 | 281/600 | 291/600 | 294/600 |
| `invoice` (reservada) | 1.672/3.000 | 1.817/3.000 | 1.748/3.000 | 1.871/3.000 | 1.796/3.000 |
| Felhen vistas | 2.165/2.996 | 2.605 | 2.629 | 2.608 | 2.622 |
| Telepatia vistas | 437/840 | 607 | 646 | 616 | 639 |
| Traces vistas | 63/74 | 65 | 66 | 64 | 67 |

**Casos em que a resposta muda com a ordem:**

| Família | E3 s0 | Balanceado s0 | E3 s1 | Balanceado s1 |
|---|---|---|---|---|
| `factck` (102) | 11 | 20 | 14 | 14 |
| `juristcu` (300) | 39 | 47 | 47 | 41 |
| `invoice` (1.500) | 413 | 213 | 275 | 203 |

## Critério registrado (balanceado − E3)

| Semente | Reservadas (IC 95%) | factck | juristcu | invoice | Felhen vistas | Telepatia vistas |
|---|---|---|---|---|---|---|
| 0 | −3,8 [−6,0; −1,7] | −8,3 | −0,8 | −2,3 | +0,8 [−0,1; +1,7] | **+4,6 [+2,6; +6,7]** |
| 1 | +1,3 [−0,9; +3,4] | +5,9 | +0,5 | −2,5 | +0,5 [−0,4; +1,3] | **+2,7 [+0,4; +5,2]** |

**Veredito pela regra registrada: inconclusivo.** As sementes divergem: a s0 tem o IC abaixo de zero e a s1
cruza zero. A estabilidade não melhora em nenhuma (−0,6 [−4,0; +2,9] e −2,3 [−5,6; +1,2]).

## O que isso diz

1. **Rebalancear os tipos de decisão que já temos não melhora a transferência.** O saldo nas reservadas muda de
   sinal entre as sementes, puxado de novo por `factck` (107 e 130 de 204), a família mais instável.
2. **`invoice` piora nas duas sementes** (−2,3 e −2,5), mesmo com o dobro de exposições do Telepatia, a fonte
   de onde ela vem. Mais repetição das famílias vistas do Telepatia ajudou essas famílias (+4,6 e +2,7) e não
   ajudou a família nova do mesmo domínio. É compatível com o modelo aprendendo as convenções de cada família, e
   não uma habilidade que transfere.
3. **O Felhen não precisa de 16 mil exemplos.** Com 6.500 únicos (12,8 mil exposições), as famílias vistas do
   Felhen não pioraram: ficaram até um pouco acima, +0,8 e +0,5. Isso libera orçamento para dados novos.
4. **Para o plano:** este é um resultado nulo do **balanceamento**, que deixa aberta a pergunta da
   **diversidade** (F3). Ele reforça que o caminho são tipos de decisão novos, não mais repetição dos atuais, e
   mostra que dá para cortar o Felhen pela metade e usar esse orçamento em dados novos.
