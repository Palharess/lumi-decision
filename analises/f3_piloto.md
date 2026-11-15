# Piloto F3: diversidade a orçamento fixo (14–15/11/2026)

**Jobs:** treinos 36959 e 36960, avaliação no `f3` 36961 e 36962, comparações 36963 e 36964, orçamento 36965.
Protocolo em `plano_fase2.md`; fontes e auditoria em `f3_fontes.md`.

**Braços (sementes 0 e 1):**
- **Controle:** o braço do F2.1. Felhen com 1.300 por família, Telepatia e Traces inteiros.
- **Diverso:** metade do Felhen (650 por família) trocada por `massive_intencao` e `goemo_emocao` (1.625 cada).
  Telepatia e Traces iguais ao controle.

**Orçamento real:**

| Semente | Braço | Exemplos únicos | Atualizações | Exemplos vistos | Tokens de conteúdo |
|---|---|---|---|---|---|
| 0 | Controle | 10.516 | 508 | 20.686 | 7.808.733 |
| 0 | Diverso | 10.516 | 508 | 20.883 | 6.235.568 (−20%) |
| 1 | Controle | 10.516 | 508 | 20.775 | 7.911.249 |
| 1 | Diverso | 10.516 | 508 | 20.712 | 6.193.897 (−22%) |

Os textos novos são curtos, então o braço diverso viu cerca de 20% menos tokens, como previsto no protocolo. O
viés é contra ele.

**Leitura das células:** acertos/respostas no dev, nas duas ordens do protocolo.

## Reservadas atuais e famílias vistas antigas

| Grupo | Congelado | Controle s0 | Diverso s0 | Controle s1 | Diverso s1 |
|---|---|---|---|---|---|
| `factck` (reservada) | 118/204 | 107/204 | 98/204 | 130/204 | 100/204 |
| `juristcu` (reservada) | 285/600 | 281/600 | 290/600 | 294/600 | 303/600 |
| `invoice` (reservada) | 1.672/3.000 | 1.748/3.000 | 1.860/3.000 | 1.796/3.000 | 1.861/3.000 |
| Felhen vistas | 2.165/2.996 | 2.629/2.996 | 2.590/2.996 | 2.622/2.996 | 2.570/2.996 |
| Telepatia vistas | 437/840 | 646/840 | 636/840 | 639/840 | 637/840 |
| Traces vistas | 63/74 | 66/74 | 66/74 | 67/74 | 65/74 |

**Casos em que a resposta muda com a ordem:**

| Família | Controle s0 | Diverso s0 | Controle s1 | Diverso s1 |
|---|---|---|---|---|
| `factck` (102) | 20 | 21 | 14 | 26 |
| `juristcu` (300) | 47 | 65 | 41 | 41 |
| `invoice` (1.500) | 213 | 228 | 203 | 185 |

## Critério registrado (diverso − controle)

| Semente | Reservadas (IC 95%) | factck | juristcu | invoice | Estabilidade | Felhen vistas | Telepatia vistas |
|---|---|---|---|---|---|---|---|
| 0 | +0,3 [−1,6; +2,3] | −4,4 | +1,5 | **+3,7** | +2,7 [−1,1; +6,4] | **−1,3 [−2,2; −0,4]** | −1,2 [−3,3; +0,8] |
| 1 | **−3,7 [−6,1; −1,4]** | **−14,7** | +1,5 | **+2,2** | +3,5 [−0,2; +7,0] | **−1,7 [−2,6; −0,8]** | −0,2 [−2,0; +1,5] |

**Veredito pela regra registrada: inconclusivo.** A semente 0 cruza zero e a semente 1 fica inteira abaixo.
"Ajuda" exigiria IC acima de zero nas duas sementes; "atrapalha", abaixo nas duas.

## Famílias novas (dev do `f3`)

| Família | Congelado | Controle s0 | Diverso s0 | Controle s1 | Diverso s1 |
|---|---|---|---|---|---|
| `pss_mais_simples` (reservada nova) | 2.988/4.126 | 3.175/4.126 | 3.148/4.126 | 3.112/4.126 | 3.161/4.126 |
| ... só os pares sem o atalho do tamanho | 386/892 | 357/892 | 364/892 | 340/892 | 360/892 |
| `massive_intencao` (vista no diverso) | 2.102/2.542 | 2.217/2.542 | 2.370/2.542 | 2.223/2.542 | 2.349/2.542 |
| `goemo_emocao` (vista no diverso) | 5.266/10.638 | 5.558/10.638 | 6.934/10.638 | 5.617/10.638 | 6.956/10.638 |

**`pss_mais_simples`, diverso − controle:**
- **Toda:** −0,7 [−1,7; +0,3] na semente 0 e +1,2 [+0,3; +2,1] na semente 1.
- **Sem o atalho:** +0,8 [−1,7; +3,4] e +2,2 [−0,3; +4,7].

## O que isso diz

1. **O piloto não mostra ganho de transferência.** Pela regra registrada é inconclusivo, e a semente 1 piora.
2. **Toda a perda nas reservadas vem do `factck` de novo.**
   - O diverso perde 9 e 30 das 204 respostas em `factck`.
   - Ao mesmo tempo, ganha nas duas sementes em `juristcu` (+9 e +9 de 600) e em `invoice` (+112 e +65 de 3.000).
   - Esta é uma **observação depois de ver os dados**, não uma conclusão. O critério fixado dá peso de 1/3 ao
     `factck`, que tem só 102 casos e já oscilou entre 107/204 e 130/204 no mesmo braço de controle, só trocando
     a semente.
3. **O `factck` com 102 casos virou o gargalo de medida do projeto.** Em F1, F2.1, F2.2 e agora no piloto, ele
   decidiu o veredito. Uma correção registrada **antes** do próximo teste seria avaliar o `factck` nas 6
   permutações do F0a, em vez de 2 ordens. Isso não muda o veredito deste piloto.
4. **Cortar o Felhen para 3.250 custa nas vistas dele:** 39 e 52 acertos a menos em 2.996. No F2.1, 6.500
   exemplos ainda não custavam nada. O limite está entre os dois.
5. **As famílias novas foram aprendidas.**
   - `goemo_emocao`: de 5.558 para 6.934 de 10.638 (s0); de 5.617 para 6.956 (s1).
   - `massive_intencao`: de 2.217 para 2.370 de 2.542 (s0); de 2.223 para 2.349 (s1).
   - Mas isso não virou transferência para a reservada nova.
6. **Todos os modelos leem a `pss_mais_simples` pelo tamanho da frase.**
   - Nos pares em que a frase simples não é a mais curta, todos acertam menos da metade: de 340/892 a 386/892.
   - Escolher "a mais curta" explica de 3.108/4.126 a 3.353/4.126 das respostas.
   - A diversidade não mudou esse atalho.
7. **Leitura para o plano:**
   - Duas famílias novas, ambas vindas do inglês, com 20% menos tokens, não bastam para separar "diversidade"
     de ruído.
   - Se o F3 continuar, precisa de muito mais famílias (provavelmente sintéticas, em PT-BR) e de uma medida
     menos ruidosa nas reservadas, registrada antes.
