# F1: réplicas pareadas do E3 e do E4 (13/11/2026)

Sementes 1 e 2 (jobs 36898–36903), somadas à semente 0 (E3 job 36852, E4 job 36885).
- **Mesmo conjunto de treino:** os mesmos 20.350 exemplos em todas as sementes (`--data-seed 0`).
- **Pareamento:** dentro de cada semente, E3 e E4 têm os mesmos lotes, as mesmas ordens e o mesmo LoRA inicial.
- **Avaliação:** dev, nas duas ordens do protocolo (canônica e invertida).
- **Leitura das células:** acertos/respostas (%).

## Acertos por grupo

| Grupo | Congelado | E3 s0 | E3 s1 | E3 s2 | E4 s0 | E4 s1 | E4 s2 |
|---|---|---|---|---|---|---|---|
| `factck` (reservada) | 118/204 (57,8) | 124/204 (60,8) | 118/204 (57,8) | 116/204 (56,9) | 88/204 (43,1) | 113/204 (55,4) | 85/204 (41,7) |
| `juristcu` (reservada) | 285/600 (47,5) | 286/600 (47,7) | 291/600 (48,5) | 315/600 (52,5) | 309/600 (51,5) | 331/600 (55,2) | 295/600 (49,2) |
| `invoice` (reservada) | 1.672/3.000 (55,7) | 1.817/3.000 (60,6) | 1.871/3.000 (62,4) | 1.653/3.000 (55,1) | 1.839/3.000 (61,3) | 1.681/3.000 (56,0) | 1.735/3.000 (57,8) |
| Felhen vistas | 2.165/2.996 (72,3) | 2.605 (86,9) | 2.608 (87,0) | 2.608 (87,0) | 2.617 (87,3) | 2.611 (87,1) | 2.614 (87,2) |
| Telepatia vistas | 437/840 (52,0) | 607 (72,3) | 616 (73,3) | 625 (74,4) | 629 (74,9) | 613 (73,0) | 622 (74,0) |
| Traces (todas) | 81/92 (88,0) | 83 (90,2) | 82 (89,1) | 83 (90,2) | 83 (90,2) | 84 (91,3) | 84 (91,3) |

## Critérios registrados, por semente

| Semente | H2: E3 − congelado (IC 95%) | factck | juristcu | invoice | H3: E4 − E3, qualidade (IC 95%) | factck | juristcu | invoice | Veredito H3 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | +2,6 [−0,3; +5,9] | +2,9 | +0,2 | +4,8 | −4,4 [−7,2; −1,5] | −17,6 | +3,8 | +0,7 | piora |
| 1 | +2,5 [−0,5; +5,7] | 0,0 | +1,0 | +6,6 | −0,7 [−3,2; +1,6] | −2,5 | +6,7 | −6,3 | inconclusivo |
| 2 | +1,1 [−1,7; +4,2] | −1,0 | +5,0 | −0,6 | −5,3 [−7,9; −2,9] | −15,2 | −3,3 | +2,7 | piora |
| **média** | **+2,1** | +0,6 | +2,1 | +3,6 | **−3,5** | −11,8 | +2,4 | −1,0 | 2 de 3 pioram |

Estabilidade (H3, secundária): +4,7 [+0,6; +8,7] na s0, −2,7 [−6,1; +0,6] na s1, +4,3 [−0,2; +8,8] na s2. Não há
melhora em nenhuma semente.

## O que as réplicas mostram

1. **O ganho nas famílias vistas é sólido.**
   - Felhen: de 2.165 para 2.605–2.617 acertos (de 72% para 87%) nas 6 réplicas, com variação de ±0,2 ponto.
   - Telepatia: de 437 para 607–629 (de 52% para 72–75%).
2. **A transferência do E3 é pequena, positiva e nunca conclusiva:** +2,6, +2,5 e +1,1 nas três sementes. O que
   muda muito de semente para semente é a **composição** por família: em `invoice`, +4,8, +6,6 e −0,6 (até 7
   pontos de diferença); em `juristcu`, +0,2, +1,0 e +5,0. Conclusões por família com uma semente só (como o
   "+4,8 em invoice" do E3 original) não são confiáveis.
3. **A pointer head piora em 2 de 3 sementes,** e a piora vem do mesmo lugar, `factck`: 88, 113 e 85 acertos em
   204, contra 116–124 do E3. Na semente 1 ela quase não piorou. **O colapso em `factck` depende da semente:** é
   uma instabilidade da pointer head, não um efeito fixo. O E3 fica entre 116 e 124 nas três sementes.
4. **Ruído de referência para as próximas variantes:**
   - na média das reservadas, as sementes do mesmo braço variam cerca de 1,5 ponto;
   - numa família isolada, até 7 pontos (E3) e 15 pontos (E4 em `factck`);
   - diferenças menores que isso, com uma semente, não sustentam conclusão; as variantes do F2 precisam de pelo
     menos 2 sementes por braço.

## Consequências

- **H2:** "transferência pequena e positiva, consistente em 3 sementes, não demonstrada pelo critério registrado
  em nenhuma delas".
- **H3:** "a pointer head piora a qualidade nas reservadas em 2 de 3 sementes. A piora vem de um colapso em
  `factck` que depende da semente".
- **Pointer residual sobre o E3:** é o teste certo para saber se preservar as letras evita esse colapso.
