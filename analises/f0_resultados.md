# F0: resultados dos diagnósticos sem treino (13/11/2026)

Jobs 36893 (pointer head: `factck` em 6 permutações; `juristcu` e `invoice` em 5 ordens) e 36894 (ENEM, metade
de exploração). Tudo exploratório: dev e provas, uma semente por modelo. Nas binárias de `invoice`, cada ordem
única conta uma vez. ICs reamostram casos; em `invoice` as 5 perguntas do mesmo estado não andam juntas, então
os intervalos ali são um pouco estreitos. Script: `scripts/analise_f0.py`.

## Acurácia com mais ordens

| | Congelado (A0) | E3 (A1) | E4 pela cabeça (B) | E4 pelas letras | B − A1 (IC 95%) |
|---|---|---|---|---|---|
| `factck` (6 permutações) | 44,1 | **55,1** | 40,5 | 43,6 | **−14,5 [−21,1; −8,2]** |
| `juristcu` (5 rotações de 10) | 49,3 | 48,6 | **51,3** | 36,9 | +2,7 [+0,5; +4,9] |
| `invoice` (ordens únicas) | 55,0 | 60,5 | 61,1 | **63,7** | +0,6 [−0,7; +2,0] |
| ENEM, 763 questões (2 ordens) | **67,2** | 66,5 | 64,4 | 63,2 | −2,1 [−4,0; −0,3] |

**Decisão muda com a ordem (%):**

| | A0 | A1 | B | E4 pelas letras |
|---|---|---|---|---|
| `factck` | 80 | 40 | 59 | 100 |
| `juristcu` | 44 | 21 | 19 | 91 |
| `invoice` | 34 | 34 | **16** | 31 |
| ENEM | 30 | 27 | 30 | 32 |

## O que isso diz

1. **O LoRA não estragou o conhecimento geral.** No ENEM (5 alternativas, acaso 20%), o Lumi congelado acerta
   67,2%, e o E3 fica em 66,5 (−0,7 [−2,5; +1,2]). O treino de decisão não trouxe ganho nem perda detectável
   numa tarefa de conhecimento fora do treino. A pointer head perde um pouco (−2,8 contra o congelado,
   [−5,0; −0,6]).
2. **A pointer head não é pior em todo lugar.** Contra o E3:
   - em `factck` o déficit é grande (−14,5);
   - no ENEM é pequeno (−2,1);
   - em `juristcu` ela ganha (+2,7);
   - em `invoice` empata (+0,6) e muda muito menos de resposta com a ordem (16% contra 34%).
   O problema está concentrado em `factck`, onde as descrições das opções se referem umas às outras.
3. **O E3 com mais ordens:** em `factck` ganha +10,9 [+3,9; +18,3] sobre o congelado (com as 2 ordens do
   protocolo, eram +2,9); em `invoice`, +5,4 [+3,4; +7,4]; em `juristcu`, nada (−0,7 [−3,7; +2,3]).
4. **O adaptador do E4 lido pelas letras é instável.** Ele quebra nas perguntas de muitas opções: em `factck`
   escolhe a 1ª opção em 75% dos casos, e em `juristcu` muda de resposta em 91% e cai 12 pontos. Mas em `invoice`,
   cujas perguntas têm 2 a 4 opções, é a melhor leitura (63,7). Sem uma sonda linear não dá para separar a leitura
   quebrada da qualidade das representações.
5. **A ordem das alternativas pesa até no ENEM:** a resposta muda em 27% a 32% das questões só invertendo a ordem,
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
