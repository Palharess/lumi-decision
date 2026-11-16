# E3b-r: réplica do E3b na semente 1 (16/11/2026)

**Jobs:** treinos 37162 (E3b s1) e 37163 (controle s1), critério original 37164, `factck` em 6 ordens 37165,
reservadas v2 37166. Protocolo registrado antes de rodar em `plano_fase2.md` (commit 172ca0d). Análise:
`scripts/analise_m2.py` e `scripts/analise_m1.py`.

**Pareamento:** 900 atualizações nos dois braços, conferidas no `treino_meta.json`. O nº foi contado antes,
sem GPU; a mesma contagem na semente 0 reproduz as 891 do E3b original. O E3b s1 viu 36.541 exemplos
(16,1 M tokens de conteúdo).

## Critério principal: E3b − controle nas 4 reservadas v2

| Semente | Média das 4 novas | `restricoes` | `regra` | `custo` | `ordem` |
|---|---|---|---|---|---|
| 0 (M2) | **+3,4 [+2,1; +4,8]** | **+5,3** | **+2,9** | **−3,4** | **+8,9** |
| 1 (réplica) | **+3,4 [+2,1; +4,7]** | **+4,3** | **+4,5** | −2,5 [−5,0; +0,1] | **+7,2** |

**Veredito pela regra registrada: replica.** O IC fica inteiro acima de zero na s1, como na s0, então o Eikos
ajuda nas famílias novas nas duas sementes. O efeito por família quase se repete: ganha em `ordem`,
`restricoes` e `regra`, e perde um pouco em `custo`.

**Acertos (dev, todas as rotações):**

| Braço | `restricoes` /1.252 | `regra` /1.232 | `custo` /972 | `ordem` /1.284 | `factck` /612 (6 ordens) |
|---|---|---|---|---|---|
| Controle s0 | 916 | 913 | 502 | 880 | 340 |
| E3b s0 | 982 | 949 | 469 | 994 | 342 |
| Controle s1 | 943 | 898 | 492 | 905 | 326 |
| E3b s1 | 997 | 954 | 468 | 998 | 350 |

## Secundários

| Leitura | s0 | s1 |
|---|---|---|
| 7 reservadas, famílias fixas | **+1,7 [+0,6; +2,8]** | **+2,8 [+1,7; +3,9]** |
| 7 reservadas, dois estágios | +1,7 [−1,3; +5,0] | **+2,8 [+0,4; +5,2]** |
| Famílias que ganham / perdem (de 7) | 5 / 2 | 5 / 2 |
| Critério original do E3b (3 reservadas, 2 ordens) | −0,8 [−3,1; +1,3] | +1,1 [−1,5; +3,7] |
| `factck` em 6 ordens | +0,3 [−4,1; +4,7] | +3,9 [−1,5; +9,3] |
| `invoice` | −4,3 | −0,2 |
| Felhen vistas | 0 (sem diferença) | −0,3 [−1,1; +0,5] |
| Telepatia vistas | 0 (sem diferença) | −0,7 [−2,9; +1,5] |

- **Nas famílias vistas, o Eikos não custa nada.**
- **Nas 3 reservadas antigas, segue sem efeito claro.** O critério original continua inconclusivo nas duas
  sementes.
- **O ganho está nas famílias novas cujo tipo de decisão o Eikos tem.**
- **O ponto cego continua:** o E3b s1 acerta `pedir_documento` em 0/272, o controle também. O Eikos tem uma
  família `ambiguous` (236 linhas), mas ela não ensinou a pedir o que falta.

## O que isso diz

1. **O ganho do Eikos nas famílias novas é real nas duas sementes,** e com o mesmo perfil por família. Não é
   ruído de semente.
2. **É transferência próxima, como o protocolo já previa:**
   - `ordem_eventos` (+7 a +9) combina com `temporal_numeric` no Eikos;
   - `regra_beneficio` e `restricoes_oferta` (+3 a +5) combinam com `rules_permit`, `rules_book`, `policy` e
     `long_policy`.
   - Leitura direta: **ter o tipo de decisão no treino ajuda numa família nova do mesmo tipo**, mesmo com texto,
     domínio e idioma diferentes (o Eikos é 87% inglês; as reservadas v2 são PT-BR).
   - As famílias de intenção e emoção do piloto F3 não ajudaram porque não têm tipo parecido com nenhuma
     reservada.
3. **O ganho não aparece em tipos que o Eikos não cobre:** `juristcu` +2,3 e +2,2 sem IC conclusivo;
   `invoice` e `custo` perdem um pouco.
4. **Limites:**
   - réplica de semente nos mesmos casos de dev que levantaram a pergunta;
   - teste intocado;
   - o Eikos muda ao mesmo tempo a variedade, o idioma e o volume (36,5 mil exemplos contra 20 mil).

## Próximos passos (cada um registrado antes)

- **Separar por tipo:** um braço com só as famílias de regras e temporais do Eikos, contra um com o resto, no
  mesmo nº de atualizações. Isso diz se o ganho vem dos tipos próximos ou do corpus em geral.
- **Idioma:** um braço só com o `eikos_pt` (2.158 linhas em PT-BR).
- **Reservadas com professor:** cobrir tipos que o Eikos não tem, para medir transferência longe do treino.
  Hoje, as reservadas v2 de regras e de ordem já não são independentes do Eikos.
- **Ponto cego "falta informação":** precisa de famílias vistas com essa classe; o Eikos não resolve.
