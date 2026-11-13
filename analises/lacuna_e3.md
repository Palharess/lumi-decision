# F0c: onde o E3 ganha e onde não ganha (13/11/2026)

Ganho do E3 (LoRA, letras) sobre o Lumi congelado (E1), por pergunta, na acurácia média das duas ordens. Dados
em `lacuna_e3.json`; script `scripts/analise_lacuna.py`. Exploratório: dev, uma semente.

| Pergunta reservada | Tipo, nº de opções | Maioria | Congelado | E3 | Ganho | Algo parecido no treino? |
|---|---|---|---|---|---|---|
| invoice: urgency | escala, 4 | 0,44 | 0,44 | 0,61 | **+16,5** | sim: `urgency` em 3 famílias vistas do Telepatia |
| invoice: matches_order | sim/não | 0,50 | 0,58 | 0,71 | **+12,8** | sim/não é o tipo mais comum do treino |
| invoice: disposition | escolha, 4 | 0,40 | 0,39 | 0,45 | +6,3 | `disposition` em `security_incidents` (lá o ganho foi 0) |
| invoice: discrepancy_severity | escala, 4 | 0,44 | 0,48 | 0,52 | +4,0 | `severity` só em `security_incidents` |
| factck_veracidade | escolha, 3 | 0,67 | 0,58 | 0,61 | +2,9 | 3 opções só no `olid_alvo`; nada sobre veracidade |
| juristcu_area | escolha, 10 | 0,14 | 0,48 | 0,48 | +0,2 | nenhum exemplo com 10 opções; "área" só no `scielo_area` |
| invoice: duplicate | sim/não | 0,91 | 0,90 | 0,75 | −15,5 | o congelado acertava por dizer sempre "não" |

## O que aparece

1. **A transferência acompanha a semelhança com o treino.** As perguntas de fatura com conceitos que existem
   nas famílias vistas (`urgency`, sim/não sobre o documento) ganham +13 a +17, perto do ganho nas vistas. As
   que trazem um conceito e um conjunto de rótulos novos (veracidade de alegações, área jurídica do TCU) quase
   não ganham.
2. **Aprender "classificar por área" não transferiu.** O `scielo_area` (área de artigos científicos, 8 opções)
   ganhou +36 nas vistas, e o `juristcu_area` (área de acórdãos do TCU, 10 opções) ganhou +0,2. O modelo aprendeu
   a taxonomia do SciELO, não a habilidade geral de classificar por área.
3. **Parte do ganho nas vistas é aprender as convenções de rótulo de cada família.** Nas perguntas vistas em que
   o congelado ficava abaixo da maioria, o ganho médio foi +23 (10 perguntas), contra +13,5 nas demais (10). A
   correlação entre o ganho e "congelado − maioria" é −0,37 nas vistas e ≈0 nas reservadas. Nas famílias novas,
   não há convenção aprendida para aproveitar.

Isso apoia a hipótese de **pouca variedade de tipos de decisão** do plano da fase 2. O modelo transfere quando
a pergunta nova se parece com uma que ele já viu, e aprende pouco de habilidade geral com 83 instruções. A
mistura balanceada (F2.1) e a busca de dados (F3) testam isso diretamente.
