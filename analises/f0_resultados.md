# F0: resultados dos diagnósticos sem treino (13/11/2026)

Jobs 36893 (pointer head: `factck` em 6 permutações; `juristcu` e `invoice` em 5 ordens) e 36894 (ENEM, metade
de exploração). Tudo exploratório: dev e provas, uma semente por modelo. Nas binárias de `invoice`, cada ordem
única conta uma vez. ICs reamostram grupos (em `invoice`, as 300 faturas inteiras). Em `invoice` a porcentagem por contagem (acertos/respostas) é menor que a
média por caso usada antes, porque as perguntas binárias têm menos ordens únicas. Script: `scripts/analise_f0.py`.

## Acertos com mais ordens

Cada célula: acertos / respostas (porcentagem). Uma resposta = um caso numa ordem; ordens que repetem a mesma
sequência contam uma vez.

| | Casos × ordens | Congelado (A0) | E3 (A1) | E4 pela cabeça (B) | E4 pelas letras |
|---|---|---|---|---|---|
| `factck` | 102 × 6 | 270/612 (44,1%) | **337/612 (55,1%)** | 248/612 (40,5%) | 267/612 (43,6%) |
| `juristcu` | 300 × 5 | 739/1.500 (49,3%) | 729/1.500 (48,6%) | **769/1.500 (51,3%)** | 554/1.500 (36,9%) |
| `invoice` | 1.500 perguntas, 2 a 5 ordens | 2.792/5.700 (49,0%) | 3.223/5.700 (56,5%) | 3.262/5.700 (57,2%) | **3.349/5.700 (58,8%)** |
| ENEM | 763 × 2 | **1.025/1.526 (67,2%)** | 1.015/1.526 (66,5%) | 983/1.526 (64,4%) | 964/1.526 (63,2%) |

**Diferenças** (pontos percentuais na mesma base de acertos/respostas, ou seja, cada resposta numa ordem única pesa igual; IC 95% reamostrando grupos, com as 5 perguntas de cada fatura juntas). Em `invoice`, dando peso igual a cada pergunta, o ganho do E3 é +5,4; as duas medidas são válidas:

| | E3 − congelado | E4 cabeça − E3 |
|---|---|---|
| `factck` | +10,9 [+3,9; +18,3] | **−14,5 [−21,1; −8,2]** |
| `juristcu` | −0,7 [−3,7; +2,5] | +2,7 [+0,5; +5,0] |
| `invoice` | +7,6 [+5,5; +9,6] (por pergunta: +5,4) | +0,7 [−0,5; +1,9] |
| ENEM | −0,7 [−2,5; +1,2] | −2,1 [−4,0; −0,3] |

**Casos em que a resposta muda com a ordem:**

| | Congelado | E3 | E4 cabeça | E4 letras |
|---|---|---|---|---|
| `factck` (102) | 82 | 41 | 60 | 102 |
| `juristcu` (300) | 131 | 62 | 58 | 272 |
| `invoice` (1.500) | 505 | 508 | **237** | 464 |
| ENEM (763) | 227 | 203 | 230 | 247 |

## O que isso diz

1. **Não detectamos regressão do E3 no ENEM.** No ENEM (5 alternativas, acaso 20%), o Lumi congelado acerta
   1.025 de 1.526 respostas (67,2%), e o E3 1.015 (66,5%; −0,7 [−2,5; +1,2]). O intervalo ainda admite perda de até 2,5 pontos, e o teste
   cobre só múltipla escolha numa passada. A pointer head perde um pouco (983, ou −2,1 contra o E3, [−4,0; −0,3]).
2. **A pointer head não é pior em todo lugar.** Contra o E3:
   - em `factck` o déficit é grande (−14,5);
   - no ENEM é pequeno (−2,1);
   - em `juristcu` ela ganha (+2,7);
   - em `invoice` empata (+0,7) e muda de resposta com a ordem em 237 das 1.500 perguntas, contra 508 no E3.
   O problema está concentrado em `factck`, onde as descrições das opções se referem umas às outras.
3. **O E3 transfere em `factck` e em `invoice` (exploratório; a H2 registrada segue inconclusiva):** em `factck` ganha +10,9 [+3,9; +18,3] sobre o congelado (com as 2 ordens do
   protocolo, eram +2,9); em `invoice`, +7,6 [+5,5; +9,6]; em `juristcu`, nada (−0,7 [−3,7; +2,5]).
4. **O adaptador do E4 lido pelas letras é instável.** Ele quebra nas perguntas de muitas opções: em `factck`
   escolhe a 1ª opção em 75% dos casos, e em `juristcu` muda de resposta em 272 de 300 casos e acerta 554 de
   1.500. Mas em `invoice`, cujas perguntas têm 2 a 4 opções, é a melhor leitura (3.349 de 5.700). Sem uma sonda linear não dá para separar a leitura
   quebrada da qualidade das representações.
5. **A ordem das alternativas pesa até no ENEM:** a resposta muda em 203 a 247 das 763 questões só invertendo a ordem,
   em todos os modelos.

## Consequências para o plano

- O resultado do E4 na H3 depende muito de `factck`. Pelo critério registrado (duas ordens, peso igual por
  família) ele continua uma piora, mas a pointer head não é uniformemente pior. Isso reforça o teste da
  **pointer residual sobre o E3** e as **réplicas** antes de fechar a H3.
- O ENEM serve bem como avaliação externa: o nível de acerto é informativo e não satura. O próximo passo é
  estendê-lo às outras provas (BLUEX, FUVEST, OAB, POSCOMP) e aos outros modelos, mantendo a metade
  "confirmar" intocada.
- A sensibilidade à ordem no ENEM (cerca de 30% em todos) é um alvo natural para a variante de consistência
  entre ordens (F2.5).
