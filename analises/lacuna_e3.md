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
3. ~~Parte do ganho nas vistas é aprender as convenções de cada família~~ (retirado na revisão 10). A correlação
   entre "E3 − congelado" e "congelado − maioria" usa a acurácia do congelado nos dois lados, com sinais opostos,
   o que já produz correlação negativa sem nenhum efeito real. Para testar convenções de rótulo, é preciso usar
   casos em que a mesma evidência exige outra resposta quando a regra do menu muda.

**Alcance:** tudo aqui é exploratório e serve para formular hipóteses. A comparação SciELO × TCU muda ao mesmo
tempo o domínio, a taxonomia e o número de opções. E com todas as 6 ordens (F0a), o E3 ganha **+10,9 [+3,6;
+18,3]** em `factck`, contra +2,9 nas duas ordens do protocolo: as duas ordens do protocolo são justamente as
mais favoráveis ao congelado (62% e 54%, contra 23% a 43% nas outras quatro). O critério registrado da H2 não
muda, mas o ganho do E3 em `factck` provavelmente está subestimado por ele.

Isso apoia a hipótese de **pouca variedade de tipos de decisão** do plano da fase 2. O modelo transfere quando
a pergunta nova se parece com uma que ele já viu, e aprende pouco de habilidade geral com 83 instruções. A
mistura balanceada (F2.1) e a busca de dados (F3) testam isso diretamente.
