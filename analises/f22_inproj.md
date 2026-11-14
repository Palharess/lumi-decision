# F2.2: E3 sem treinar o LoRA do `in_proj` do Mamba (14/11/2026)

**Jobs:** treinos 36940 e 36941, comparações 36942 e 36943, protocolo em `plano_fase2.md`.

**Braço:** a receita do E3 com o LoRA do `in_proj` criado e congelado com B = 0. A saída do `in_proj` é a do
modelo base, e todo o resto é igual ao E3 da mesma semente.

**Conferências do treino:**
- **Parâmetros treináveis:** 10.119.168, contra 17.054.208 do E3. Os 6.935.040 que saíram são os 21 módulos
  `in_proj`, todos com B = 0 no fim.
- **Atualizações:** 508 na semente 0 e 502 na semente 1, iguais às do E3 de cada semente. A comparação confere
  isso e recusaria rodar se não fossem.

**Leitura das células:** acertos/respostas no dev, nas duas ordens do protocolo.

## Acertos

| Grupo | Congelado | E3 s0 | Sem `in_proj` s0 | E3 s1 | Sem `in_proj` s1 |
|---|---|---|---|---|---|
| `factck` (reservada) | 118/204 | 124/204 | 112/204 | 118/204 | 117/204 |
| `juristcu` (reservada) | 285/600 | 286/600 | 286/600 | 291/600 | 302/600 |
| `invoice` (reservada) | 1.672/3.000 | 1.817/3.000 | 1.787/3.000 | 1.871/3.000 | 1.858/3.000 |
| Felhen vistas | 2.165/2.996 | 2.605/2.996 | 2.593/2.996 | 2.608/2.996 | 2.597/2.996 |
| Telepatia vistas | 437/840 | 607/840 | 616/840 | 616/840 | 610/840 |
| Traces vistas | 63/74 | 65/74 | 65/74 | 64/74 | 64/74 |

**Casos em que a resposta muda com a ordem:**

| Família | E3 s0 | Sem `in_proj` s0 | E3 s1 | Sem `in_proj` s1 |
|---|---|---|---|---|
| `factck` (102) | 11 | 17 | 14 | 19 |
| `juristcu` (300) | 39 | 53 | 47 | 51 |
| `invoice` (1.500) | 413 | 478 | 275 | 270 |

## Critério registrado (sem `in_proj` − E3)

| Semente | Reservadas (IC 95%) | factck | juristcu | invoice | Estabilidade | Felhen vistas | Telepatia vistas |
|---|---|---|---|---|---|---|---|
| 0 | **−2,3 [−4,2; −0,6]** | −5,9 | 0,0 | −1,0 | piora, +5,0 [+2,2; +7,8] | −0,4 [−1,0; +0,2] | +1,1 [−0,5; +2,6] |
| 1 | +0,3 [−1,2; +1,8] | −0,5 | +1,8 | −0,4 | +2,0 [−1,1; +4,8] | −0,4 [−0,9; +0,2] | −0,7 [−2,1; +0,6] |

**Veredito pela regra registrada: inconclusivo.**
- Na semente 0, tirar o `in_proj` piora, com o IC inteiro abaixo de zero.
- Na semente 1, empata, com o IC inteiro dentro de ±2.
- Para "o `in_proj` ajuda" o IC teria de ficar abaixo de zero nas duas sementes. Para "dispensável" teria de
  ficar dentro de ±2 nas duas. Nenhum dos dois aconteceu.

## O que isso diz

1. **Tirar o `in_proj` nunca ajudou.** Nas reservadas, uma semente piora e a outra empata. Nas vistas do Felhen,
   as duas sementes perdem um pouco: 12 e 11 acertos a menos em 2.996, e a NLL sobe de 0,365 para 0,375 e de
   0,363 para 0,378. A queda da semente 0 vem de novo de `factck`: 112 contra 124 de 204, ou seja, 12 respostas.
   É a família mais instável entre sementes.
2. **A decisão fica um pouco menos estável sem o `in_proj`.** Na semente 0 a resposta muda com a ordem em mais
   casos nas três reservadas (+5,0 [+2,2; +7,8]). Na semente 1 a tendência é a mesma, com o IC cruzando zero.
3. **Para o projeto:** o `in_proj` fica na receita. O A1 não muda, e no piloto do F3 os dois braços usam os
   mesmos alvos do E3, com `in_proj`. Os 41% de parâmetros no Mamba não são o gargalo da transferência. Também
   não são peso morto: retirá-los só piora ou empata.

## Diagnóstico do F2.3a: o residual lido só pelas letras

Job 36944. É o mesmo adaptador residual, com a cabeça desligada (`--sem-ponteiro`), no `factck` do dev.

| Semente | Residual | Residual só pelas letras | Controle | Residual × só letras | Só letras × controle |
|---|---|---|---|---|---|
| 0 | 125/204 | 127/204 | 125/204 | 2 discordâncias | 15 discordâncias |
| 1 | 114/204 | 115/204 | 115/204 | 1 discordância | 15 discordâncias |

**A cabeça quase não age na hora de responder.** Desligá-la muda só 1 ou 2 das 204 respostas. As 13 e 16
discordâncias entre residual e controle vêm do LoRA, que treinou por outro caminho com a cabeça presente, e não
da nota que a cabeça soma. Isso reforça a leitura do F2.3a: a cabeça em cima das letras fica inerte.
