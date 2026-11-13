# F2.3a: pointer residual sobre o E3 (13/11/2026)

**Jobs:** 36904–36909, protocolo em `plano_fase2.md`.

**Braços:** os dois partem do E3 da mesma semente e recebem o mesmo treino adicional (mesmos exemplos, ordens da
2ª época, mesmo cronograma).
- **Residual:** letras + pointer head, com a cabeça começando em zero.
- **Controle:** só letras.

**Atualizações:** 501 na semente 0 e 499 na semente 1, iguais nos dois braços. O protocolo previa 508; o plano
de lotes da 2ª época agrupa os exemplos de outro jeito. A checagem de partida passou nas duas sementes: a leitura
residual começou idêntica à do E3 (diferença 0,0).

**Leitura das células:** acertos/respostas no dev, nas duas ordens do protocolo.

## Acertos

| Grupo | Congelado | E3 s0 | E4 s0 | Controle s0 | Residual s0 | E3 s1 | E4 s1 | Controle s1 | Residual s1 |
|---|---|---|---|---|---|---|---|---|---|
| `factck` | 118/204 | 124/204 | 88/204 | 125/204 | 125/204 | 118/204 | 113/204 | 115/204 | 114/204 |
| `juristcu` | 285/600 | 286/600 | 309/600 | 294/600 | 283/600 | 291/600 | 331/600 | 287/600 | 293/600 |
| `invoice` | 1.672/3.000 | 1.817 | 1.839 | 1.870 | 1.877 | 1.871 | 1.681 | 1.799 | 1.814 |
| Felhen vistas | 2.165/2.996 | 2.605 | 2.617 | 2.620 | 2.620 | 2.608 | 2.611 | 2.618 | 2.616 |
| Telepatia vistas | 437/840 | 607 | 629 | 638 | 639 | 616 | 613 | 650 | 646 |

**Casos em que a resposta muda com a ordem:**

| Família | E3 s0 | Controle s0 | Residual s0 | E3 s1 | Controle s1 | Residual s1 |
|---|---|---|---|---|---|---|
| `factck` (102) | 11 | 8 | 14 | 14 | 15 | 19 |
| `juristcu` (300) | 39 | 44 | 49 | 47 | 61 | 53 |
| `invoice` (1.500) | 413 | 295 | 212 | 275 | 271 | 306 |

## Critério registrado (residual − controle)

| Semente | Qualidade (IC 95%) | factck | juristcu | invoice | Estabilidade | Regra |
|---|---|---|---|---|---|---|
| 0 | −0,5 [−1,7; +0,6] | 0,0 | −1,8 | +0,2 | +0,7 [−1,7; +3,0] | **empate** (IC dentro de ±2) |
| 1 | +0,3 [−0,8; +1,6] | −0,5 | +1,0 | +0,5 | +1,2 [−1,1; +3,7] | **empate** (IC dentro de ±2) |

## O que isso diz

1. **Somar a pointer head às letras não muda nada.** Nas duas sementes o residual empata com o controle, com o
   IC inteiro dentro de ±2 pontos. Também não ficou mais estável com a ordem.
2. **Preservar as letras evita o colapso em `factck`:** o residual fica em 125 e 114 de 204, igual ao controle,
   contra 88 e 113 da pointer head pura. O ganho de manter as letras vem do conhecimento prévio do LM head, como
   no *Small Heads*; a cabeça em cima dele não acrescenta.
3. **Treinar uma época a mais (controle contra E3) ajuda nas vistas e pouco nas reservadas.**
   - Telepatia vista: de 607 para 638 (s0) e de 616 para 650 (s1).
   - Reservadas: mistura de pequenas altas e baixas, como em `invoice`, que vai de 1.817 para 1.870 na s0 e de
     1.871 para 1.799 na s1.
4. **Para o projeto:** a forma de ler a decisão (letras, pointer head ou as duas) não é o que limita a
   transferência. Se a cabeça entra sozinha, piora e fica instável; se entra junto com as letras, é inócua. O
   próximo ganho tem de vir de onde a literatura aponta: os dados (variedade de tipos de decisão) e a capacidade
   de adaptação (ablação do `in_proj`).
