# E3c: o Eikos separado por tipo de decisão (16/11/2026)

**Jobs:**
- treinos 37207–37212 (próximo, resto e controle, sementes 0 e 1);
- pareamento 37213 e 37214;
- reservadas v2: 37215 e 37217;
- `factck` em 6 ordens: 37216 e 37218.

O protocolo foi registrado antes de rodar em `plano_fase2.md` (commit 447b404). Análise:
`scripts/analise_m2.py`, `scripts/analise_m1.py` e `scripts/analise_e3c_vistas.py`.

**Os braços:**
- **Próximo:** E3 + as 8 famílias do Eikos de regras, políticas e datas (6.428 linhas).
- **Resto:** E3 + 6.428 linhas sorteadas das outras 19 famílias, estratificadas por idioma.
- **Controle:** só os dados do E3.

**Pareamento:** 654 (s0) e 658 (s1) atualizações nos três braços de cada semente, conferidas no
`treino_meta.json`. Tokens de conteúdo reais:

| Semente | Próximo | Resto | Controle |
|---|---|---|---|
| 0 | 11,05 M | 11,32 M | 10,48 M |
| 1 | 10,99 M | 11,51 M | 10,46 M |

## Critério principal: próximo − resto nas 4 reservadas v2

| Semente | Média das 4 novas | `restricoes` | `regra` | `custo` | `ordem` |
|---|---|---|---|---|---|
| 0 | +0,8 [−0,5; +2,1] | **−2,3** | **+8,5** | **−3,3** | +0,3 [−2,3; +3,0] |
| 1 | **+1,3 [+0,4; +2,3]** | −0,5 [−2,4; +1,4] | **+4,4** | −0,9 [−2,6; +0,8] | **+2,3** |

(Em negrito, IC inteiro de um lado do zero.)

**Veredito pela regra registrada: inconclusivo.** Pela regra, "o ganho vem do tipo de decisão" exigia IC acima
de zero nas duas sementes. Na s1 o IC fica acima de zero; na s0, cruza o zero.

**A previsão por família só se confirmou em `regra_beneficio`.**
- `regra_beneficio`: confirmada nas duas sementes, com folga.
- `ordem_eventos`: quase nada (+0,3 e +2,3).
- `restricoes_oferta`: falhou, foi na direção contrária (−2,3 e −0,5).

**Acertos (dev, todas as rotações):**

| Braço | `restricoes` /1.252 | `regra` /1.232 | `custo` /972 | `ordem` /1.284 | `factck` /612 (6 ordens) |
|---|---|---|---|---|---|
| Controle s0 | 945 | 913 | 483 | 860 | 339 |
| Próximo s0 | 953 | 958 | 475 | 974 | 289 |
| Resto s0 | 982 | 853 | 507 | 970 | 337 |
| Controle s1 | 963 | 898 | 489 | 897 | 335 |
| Próximo s1 | 964 | 957 | 480 | 1.002 | 353 |
| Resto s1 | 970 | 903 | 489 | 972 | 360 |
| *E3b s0 / s1 (Eikos inteiro, 891 / 900 atualizações)* | *982 / 997* | *949 / 954* | *469 / 468* | *994 / 998* | *342 / 350* |

## Secundários: cada metade contra o controle

| Comparação | 4 novas | `restricoes` | `regra` | `custo` | `ordem` | 7 reservadas, famílias fixas | 7, dois estágios |
|---|---|---|---|---|---|---|---|
| Próximo − controle s0 | **+3,1 [+1,9; +4,2]** | +0,6 | **+3,7** | −0,8 | **+8,9** | +0,2 [−0,9; +1,3] | +0,2 [−3,9; +4,0] |
| Próximo − controle s1 | **+3,0 [+1,9; +4,2]** | +0,1 | **+4,8** | −0,9 | **+8,2** | **+1,6 [+0,6; +2,5]** | +1,6 [−1,5; +4,7] |
| Resto − controle s0 | **+2,3 [+1,1; +3,5]** | **+3,0** | **−4,9** | **+2,5** | **+8,6** | +0,4 [−0,6; +1,3] | +0,4 [−3,0; +3,9] |
| Resto − controle s1 | **+1,7 [+0,6; +2,7]** | +0,6 | +0,4 | +0,0 | **+5,8** | **+1,7 [+0,7; +2,6]** | +1,7 [−1,0; +4,3] |
| Próximo − resto s0 / s1 (7 reservadas) | — | — | — | — | — | −0,2 [−1,5; +1,1] / −0,1 [−0,9; +0,7] | −0,2 / −0,1 |

- **As duas metades ajudam nas 4 novas, nas duas sementes.**
  - O próximo dá cerca de +3, quase o mesmo que o Eikos inteiro no E3b (+3,4), com 27% menos atualizações.
  - O resto dá cerca de +2.
- **O ganho em `ordem_eventos` não depende do tipo temporal.** O resto, sem `temporal_numeric`, ganha +8,6 e
  +5,8, quase o mesmo que o próximo (+8,9 e +8,2).
- **O ganho em `regra_beneficio` depende do tipo.**
  - Só o próximo ganha (+3,7 e +4,8), e chega a `conceder` em 331/332 nas duas sementes.
  - O resto empata ou piora: na s0, `conceder` cai para 255/332.
  - Nenhum braço pede o documento que falta: de 0/272 a 3/272.
- **`restricoes_oferta` e `custo_plano`:** o resto ajuda um pouco na s0 e empata na s1; o próximo empata. O
  E3b inteiro ganhava +5 em `restricoes`, mais do que qualquer metade.
- **Nas 7 reservadas, as duas metades se equivalem** (próximo − resto −0,2 e −0,1).
- **`factck` (6 ordens):**
  - o próximo s0 perde para o controle: 289 contra 339 de 612, −8,2 [−13,6; −3,1], trocando `falso` por
    `impreciso`;
  - na s1, o mesmo braço ganha: 353 contra 335, +2,9 [−1,3; +7,4].
  - É a mesma instabilidade de semente já vista no `factck`.
- **`invoice` piora nos 4 braços com Eikos** (−4,4 a −5,8 contra o controle). Toda a perda vem de
  `duplicate`:
  - os braços com Eikos respondem "é duplicata" em 476 a 588 de 600, e só 56 são;
  - o controle responde isso em 243 e 184 de 600;
  - nas outras 4 perguntas da `invoice`, os braços com Eikos empatam ou ganham.
  - É o mesmo deslocamento de limiar visto no E3b s0, agora repetido em 4 treinos. Observação, não critério:
    a família continua na conta.

**Famílias vistas: nenhum braço custa nada.** Todos os ICs cruzam zero.

| Semente | Felhen /2.996 (próximo / resto / controle) | Telepatia /840 (próximo / resto / controle) |
|---|---|---|
| 0 | 2.635 / 2.624 / 2.612 | 619 / 626 / 626 |
| 1 | 2.616 / 2.611 / 2.618 | 633 / 628 / 624 |

## O que isso diz

1. **A leitura "transferência próxima" do E3b-r estava só parcialmente certa.**
   - O ganho do Eikos nas famílias novas vem em boa parte do corpus em geral: o resto, sem nenhuma família de
     regras ou de datas, também ajuda (+2,3 e +1,7).
   - O tipo de decisão acrescenta pouco na média (+0,8 e +1,3), e quase só numa família.
2. **Onde o tipo importa, importa muito: `regra_beneficio`.** É a reservada cujo formato é mais parecido com
   as famílias `rules_*` (regra escrita + caso + conceder/negar). Ali o próximo ganha +4 a +8 do resto, nas
   duas sementes.
3. **`ordem_eventos` melhora com qualquer metade.** Não é o tipo temporal.
   - Hipótese, depois de ver os dados: o que ajuda é ler documentos longos com datas e eventos. O resto tem
     `multi_hop`, `trap` e `ambiguous`, com logs e registros datados.
4. **Por esse critério, o Eikos inteiro é melhor que qualquer metade.**
   - O E3b ganhava em `restricoes` (+5) e em `ordem` (+8 a +9) ao mesmo tempo; cada metade só faz uma das
     duas coisas.
   - É descritivo: o E3b teve 891–900 atualizações contra 654–658.
5. **Ponto cego mantido:** nenhum braço aprende a pedir o que falta. O `ambiguous` do resto (155 linhas com
   "não dá para determinar") não basta.

**Limites:**
- os mesmos casos de dev que motivaram a pergunta, e o teste continua intocado;
- a separação é por nome de família;
- o resto viu 2–5% mais tokens que o próximo (o volume favorece o resto e, mesmo assim, o próximo empata ou
  ganha na média).

## Próximos passos possíveis (cada um registrado antes)

- **Usar o Eikos inteiro na receita?**
  - É o que mais ajuda nas 4 novas, sem custo nas vistas.
  - Custa a `invoice` (`duplicate`) e deixa as reservadas v2 de regras e de ordem menos independentes.
  - Pede reservadas novas, de tipos que o Eikos não tem: as 4 famílias com professor.
- **Testar a hipótese dos documentos longos com datas** em `ordem_eventos`, separando `multi_hop`, `trap` e
  `ambiguous` do resto.
- **Calibrar o limiar de `duplicate`** no calib da `invoice`, se a família for usada como vista em algum
  momento (hoje é reservada: não se ajusta nada nela).
- **"Falta informação":** famílias vistas com essa classe; nem o Eikos nem o `ambiguous` resolvem.
