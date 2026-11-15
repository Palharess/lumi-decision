# M2: os checkpoints existentes nas reservadas v2 (15/11/2026)

**Jobs:** 37124 e 37125 (só avaliação). Protocolo registrado antes de rodar em `reservadas_v2.md` (commit d70c8e7).
Script: `scripts/analise_m2.py`.

**Leitura das células:** acertos/respostas no `dev` das 4 famílias programáticas. Cada caso é lido em todas as
rotações: 4 em `restricoes_oferta`, `regra_beneficio` e `ordem_eventos`; 3 em `custo_plano`. Diferenças em
pontos na mesma base, com IC 95% pareado reamostrando grupos. As "7 reservadas" são as 4 novas mais as 3
atuais, com o `factck` em 6 ordens (regra da M1). Nenhum veredito registrado muda.

## Por checkpoint

| Checkpoint | `restricoes` /1.252 | `regra` /1.232 | `custo` /972 | `ordem` /1.284 |
|---|---|---|---|---|
| Lumi congelado | 1.002 | 905 | 422 | 835 |
| E3 s0 / s1 / s2 | 952 / 968 / 934 | 907 / 906 / 907 | 487 / 486 / 479 | 863 / 890 / 884 |
| E4 s0 / s1 / s2 (cabeça) | 900 / 928 / 911 | 716 / 714 / 645 | 497 / 489 / 461 | 785 / 766 / 807 |
| E3b / controle do E3b | 982 / 916 | 949 / 913 | 469 / 502 | 994 / 880 |
| F2.1 s0 / s1 | 959 / 968 | 897 / 883 | 513 / 511 | 928 / 883 |
| F2.2 s0 / s1 | 942 / 947 | 907 / 894 | 474 / 472 | 876 / 888 |
| F2.3a residual s0 / s1 | 944 / 957 | 886 / 898 | 513 / 495 | 902 / 938 |
| F2.3a controle s0 / s1 | 935 / 943 | 873 / 902 | 516 / 495 | 901 / 927 |
| F3 diverso s0 / s1 | 951 / 966 | 826 / 856 | 503 / 501 | 861 / 883 |

**Um ponto cego comum:** na `regra_beneficio`, quase nenhum modelo pede o documento que falta.
- O congelado acerta 18/272 nessa classe; os adaptadores, de 0/272 a 19/272.
- Nas outras três classes, todos passam de 85%, exceto o E4 em `negar`.
- É o mesmo padrão da abstenção, que nunca transferiu para as reservadas (H1, H2): nenhum modelo aprendeu a
  decidir "falta informação".

## Comparações registradas

| Comparação | 4 novas | 7 reservadas, famílias fixas | 7 reservadas, dois estágios | Famílias que ganham / perdem |
|---|---|---|---|---|
| H2 s0: E3 − congelado | +1,3 [+0,0; +2,4] | **+3,0 [+1,7; +4,3]** | +3,0 [−0,6; +6,9] | 6 / 1 |
| H2 s1 | **+2,1 [+0,8; +3,3]** | **+3,4 [+2,2; +4,7]** | **+3,4 [+0,3; +6,5]** | 6 / 1 |
| H2 s2 | +1,1 [−0,1; +2,2] | **+2,5 [+1,2; +3,8]** | +2,5 [−1,0; +6,2] | 5 / 2 |
| H3 s0: E4 − E3 | **−6,2 [−7,8; −4,5]** | **−5,0 [−6,4; −3,6]** | −5,0 [−10,3; +0,1] | 3 / 4 |
| H3 s1 | **−7,0 [−8,7; −5,4]** | **−4,0 [−5,3; −2,8]** | −4,0 [−9,2; +1,4] | 2 / 5 |
| H3 s2 | **−7,7 [−9,5; −5,9]** | **−6,6 [−8,1; −5,3]** | **−6,6 [−12,5; −1,4]** | 1 / 6 |
| E3b − controle | **+3,4 [+2,1; +4,8]** | **+1,7 [+0,6; +2,8]** | +1,7 [−1,3; +5,0] | 5 / 2 |
| F2.3a s0: residual − controle | +0,4 [−0,2; +1,0] | +0,3 [−0,3; +0,8] | +0,3 [−0,7; +1,2] | 5 / 2 |
| F2.3a s1 | +0,4 [−0,1; +0,9] | **+0,6 [+0,1; +1,1]** | +0,6 [−0,0; +1,2] | 5 / 1 |
| F2.1 s0: balanceado − E3 | **+1,9 [+1,0; +2,8]** | −0,1 [−1,0; +0,7] | −0,1 [−2,6; +2,1] | 3 / 4 |
| F2.1 s1 | +0,0 [−0,8; +0,9] | +0,1 [−0,7; +0,9] | +0,1 [−1,4; +1,8] | 3 / 3 |
| F2.2 s0: sem `in_proj` − E3 | −0,3 [−1,0; +0,4] | −0,6 [−1,2; +0,1] | −0,6 [−1,5; +0,4] | 1 / 4 |
| F2.2 s1 | **−1,1 [−1,8; −0,3]** | −0,4 [−1,0; +0,3] | −0,4 [−1,4; +0,7] | 1 / 5 |
| F3 s0: diverso − controle | **−3,2 [−4,2; −2,1]** | **−2,2 [−3,2; −1,3]** | −2,2 [−5,3; +0,6] | 2 / 5 |
| F3 s1 | −0,8 [−1,7; +0,0] | **−1,6 [−2,5; −0,6]** | −1,6 [−5,3; +1,0] | 2 / 4 |

**Por família nova:**

| Comparação | `restricoes` | `regra` | `custo` | `ordem` |
|---|---|---|---|---|
| H2 s0 / s1 / s2 | **−4,0 / −2,7 / −5,4** | +0,2 / +0,1 / +0,2 | **+6,7 / +6,6 / +5,9** | +2,2 / **+4,3 / +3,8** |
| H3 s0 / s1 / s2 | **−4,2 / −3,2** / −1,8 | **−15,5 / −15,6 / −21,3** | +1,0 / +0,3 / −1,9 | **−6,1 / −9,7 / −6,0** |
| E3b − controle | **+5,3** | **+2,9** | **−3,4** | **+8,9** |
| F3 s0 / s1 | −0,6 / −0,2 | **−5,8 / −2,2** | −1,0 / −1,0 | **−5,2** / +0,0 |

(Em negrito, IC inteiro de um lado do zero.)

**Ruído de semente, agora pequeno:** na média das 4 novas, as sementes do mesmo braço diferem +0,8, −0,2 e −1,0.
No `factck`, a mesma comparação ia de −2,8 a +5,4 com 6 ordens. Com 7 famílias fixas, fica dentro de ±0,7.

## O que isso diz

1. **O E3 transfere, de forma modesta e desigual.**
   - Na média das 7 reservadas: +3,0, +3,4 e +2,5, com IC acima de zero nas três sementes (famílias fixas).
   - Ganha em 5 ou 6 das 7 famílias: `custo` (cerca de +6), `ordem` (+2 a +4), `factck` e `invoice`.
   - Perde nas três sementes em `restricoes_oferta` (−2,7 a −5,4).
   - Tratando a família como amostra (dois estágios), o IC só fica acima de zero na s1. Com 7 famílias, ainda
     não dá para afirmar que o ganho vale para uma família nova qualquer, mas a direção é consistente.
2. **A pointer head piora também nas famílias novas** (−6,2 a −7,7 nas 4 novas). Ela colapsa de novo numa
   classe: na `regra_beneficio`, acerta `negar` em 97, 85 e 11 de 312. É o mesmo padrão do `factck`.
3. **O piloto F3 piora a transferência.** Nas 7 reservadas (famílias fixas), o IC fica abaixo de zero nas duas
   sementes (−2,2 e −1,6).
   - O padrão é o mesmo do `factck`: o braço diverso dá menos respostas negativas. Ele acerta `negar` em 222 e
     256 de 312, contra 280 e 273 no controle, assim como acertava menos `falso`.
   - Isso é uma observação depois de ver os dados. Hipótese: a troca de metade do Felhen (que tem decisões de
     rejeição, como `olid_ofensivo`) por intenção e emoção (sem classe negativa) tirou exemplos de
     "dizer não".
4. **O E3b (Eikos) é o único braço que melhora nas famílias novas:** +3,4 [+2,1; +4,8] nas 4.
   - Ele ganha em `ordem` (+8,9), `restricoes` (+5,3) e `regra` (+2,9), e perde em `custo` (−3,4).
   - **Ressalva:** o Eikos não tem texto em comum com as reservadas v2 (0 de 16.191 linhas), mas tem tipos de
     decisão próximos: `rules_permit`, `rules_book`, `policy` e `long_policy` (aplicar regras), e
     `temporal_numeric`. É transferência próxima, e de uma semente só.
   - Isso também é evidência a favor da hipótese do F3: **ter no treino o tipo de decisão ajuda**. Já as duas
     famílias do piloto (intenção e emoção) não ajudaram, porque não têm relação com as reservadas.
5. **O F2.1, o F2.2 e o F2.3a ficam perto de zero.** O balanceamento ajuda nas 4 novas só na s0, a cabeça
   residual empata e tirar o `in_proj` não ajuda. As decisões de receita (A1, com `in_proj`) não mudam.
6. **As reservadas v2 medem melhor que o `factck`:** o ruído de semente na média das 4 novas é de até
   1 ponto.

## Consequências para o plano

- **Critério dos próximos testes:** as 7 reservadas, com famílias fixas e IC em dois estágios lado a lado, como
  proposto no desenho. Falta decidir se ele vira o principal agora.
- **O E3b merece réplica:** é o único sinal positivo em famílias novas, e numa direção que faz sentido (tipos
  de decisão próximos). Uma réplica com a semente 1, com controle, custa 2 treinos.
- **Ponto cego "falta informação":** nenhuma receita aprendeu a pedir o que falta nem a se abster nas
  reservadas. É um alvo direto para dados de treino novos, com famílias vistas que tenham essa classe.
- **Antes de usar o Eikos ou famílias de regras no treino:** as reservadas v2 de regras e de ordem deixam de
  ser independentes. Isso precisa ser dito em cada leitura, e as 4 famílias com professor devem cobrir tipos
  que o Eikos não tem.
