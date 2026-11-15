# M1: `factck` nas 6 permutações, para todos os checkpoints (15/11/2026)

**Jobs:** 37119 e 37120 (só avaliação). Protocolo registrado antes de rodar em `plano_fase2.md` (commit 5ed2e0f).
Scripts: `slurm/m1_factck6.sbatch`, `scripts/analise_m1.py` e `scripts/analise_m1_criterio.py` (leitura
exploratória do critério completo).

**Leitura das células:** acertos/respostas no dev de `factck_veracidade` (102 casos × 6 ordens = 612 respostas;
por classe: `falso` 68 casos = 408 respostas, `impreciso` 24 = 144, `verdadeiro` 10 = 60). Diferenças em pontos
na mesma base de 612; IC 95% pareado reamostrando os 102 casos com as 6 ordens juntas.

## Sanidade

- O congelado e o E3 s0 repetem o F0a: 270/612 e 337/612.
- Em `perm012`/`perm210` (= canônica/invertida), 13 dos 20 checkpoints repetem as 204 respostas originais.
  - Nos outros 7, mudam de 1 a 6 respostas, todas em casos quase empatados: a margem entre as duas opções mais
    prováveis era de no máximo 0,016.
  - É ruído numérico do bf16 com outra composição de lote. A leitura de 2 ordens carrega esse ruído de cerca
    de ±3/204.

## Por checkpoint

| Checkpoint | 6 ordens | `falso` | `impreciso` | `verdadeiro` | Muda entre as ordens (de 102) |
|---|---|---|---|---|---|
| Lumi congelado | 270/612 | 165/408 | 101/144 | 4/60 | 82 |
| E3 s0 | 337/612 | 245/408 | 77/144 | 15/60 | 41 |
| E3 s1 | 320/612 | 206/408 | 95/144 | 19/60 | 47 |
| E3 s2 | 324/612 | 224/408 | 83/144 | 17/60 | 53 |
| E4 s0 (cabeça) | 248/612 | 141/408 | 87/144 | 20/60 | 60 |
| E4 s1 (cabeça) | 319/612 | 219/408 | 88/144 | 12/60 | 39 |
| E4 s2 (cabeça) | 234/612 | 126/408 | 89/144 | 19/60 | 55 |
| E3b | 342/612 | 242/408 | 75/144 | 25/60 | 48 |
| controle do E3b | 340/612 | 248/408 | 73/144 | 19/60 | 41 |
| F2.3a residual s0 | 368/612 | 280/408 | 73/144 | 15/60 | 36 |
| F2.3a controle s0 | 356/612 | 260/408 | 82/144 | 14/60 | 41 |
| F2.3a residual s1 | 322/612 | 208/408 | 98/144 | 16/60 | 55 |
| F2.3a controle s1 | 316/612 | 195/408 | 105/144 | 16/60 | 55 |
| F2.1 balanceado s0 | 304/612 | 192/408 | 95/144 | 17/60 | 44 |
| F2.1 balanceado s1 | 337/612 | 242/408 | 86/144 | 9/60 | 54 |
| F2.2 sem `in_proj` s0 | 326/612 | 227/408 | 83/144 | 16/60 | 36 |
| F2.2 sem `in_proj` s1 | 320/612 | 210/408 | 93/144 | 17/60 | 43 |
| F3 diverso s0 | 255/612 | 137/408 | 106/144 | 12/60 | 32 |
| F3 diverso s1 | 268/612 | 159/408 | 100/144 | 9/60 | 50 |

## Comparações registradas, só `factck`

| Comparação | 2 ordens (nesta rodada) | 6 ordens |
|---|---|---|
| H2 s0: E3 − congelado | +2,9 [−4,9; +11,8] | **+10,9 [+4,2; +18,8]** |
| H2 s1 | +0,0 [−7,8; +8,8] | **+8,2 [+2,3; +14,7]** |
| H2 s2 | −1,0 [−8,8; +7,4] | **+8,8 [+2,5; +15,8]** |
| H3 s0: E4 − E3 | −17,6 [−25,5; −9,8] | **−14,5 [−21,4; −8,5]** |
| H3 s1 | −2,5 [−9,3; +3,9] | −0,2 [−5,2; +4,6] |
| H3 s2 | −15,2 [−23,0; −8,3] | **−14,7 [−20,9; −9,0]** |
| E3b − controle | −1,0 [−6,9; +4,9] | +0,3 [−4,1; +4,7] |
| F2.3a s0: residual − controle | +0,5 [−2,5; +3,4] | +2,0 [−0,5; +4,4] |
| F2.3a s1 | −0,5 [−3,9; +2,9] | +1,0 [−1,3; +3,1] |
| F2.1 s0: balanceado − E3 | −8,3 [−14,7; −2,5] | **−5,4 [−9,6; −1,6]** |
| F2.1 s1 | +6,4 [+0,5; +12,7] | +2,8 [−1,1; +6,7] |
| F2.2 s0: sem `in_proj` − E3 | −5,9 [−10,8; −1,5] | −1,8 [−4,7; +1,0] |
| F2.2 s1 | −0,5 [−4,9; +3,9] | +0,0 [−3,1; +3,1] |
| F3 s0: diverso − controle | −5,9 [−11,3; −0,5] | **−8,0 [−13,2; −3,1]** |
| F3 s1 | −15,2 [−22,1; −8,8] | **−11,3 [−16,7; −6,2]** |
| Semente: E3 s1 − s0 | −2,9 [−8,8; +2,9] | −2,8 [−6,5; +0,8] |
| Semente: E3 s2 − s0 | −3,9 [−10,3; +2,0] | −2,1 [−5,6; +1,0] |
| Semente: controle F3 s1 − s0 | +11,8 [+5,4; +19,1] | **+5,4 [+1,3; +9,6]** |

**Leitura registrada da queda do F3: persiste.** O IC de diverso − controle fica inteiro abaixo de zero nas duas
sementes. A perda está em `falso` (192 → 137 e 242 → 159 de 408), com um ganho pequeno em `impreciso`
(95 → 106 e 86 → 100 de 144): o braço diverso desloca respostas de `falso` para `impreciso`.

## Exploratório: o critério principal com a regra nova aplicada aos pares antigos

A regra fixada antes desta rodada vale para os **próximos** testes: `factck` nas 6 ordens, `juristcu` e `invoice`
nas 2. Aplicá-la aos pares já julgados é uma releitura e não muda os vereditos registrados.

| Comparação | Registrado (2 ordens) | Regra nova (`factck` em 6) |
|---|---|---|
| H2 s0: E3 − congelado | +2,6 [−0,3; +5,9] | **+5,3 [+2,7; +8,1]** |
| H2 s1 | +2,5 [−0,5; +5,7] | **+5,3 [+2,9; +7,7]** |
| H2 s2 | +1,1 [−1,7; +4,2] | **+4,4 [+1,9; +7,1]** |
| H3 s0: E4 − E3 | **−4,4 [−7,2; −1,5]** | **−3,3 [−5,7; −0,9]** |
| H3 s1 | −0,7 [−3,2; +1,6] | +0,1 [−2,0; +2,2] |
| H3 s2 | **−5,3 [−7,9; −2,9]** | **−5,1 [−7,3; −2,9]** |
| E3b − controle | −0,8 [−3,1; +1,3] | −0,5 [−2,4; +1,2] |
| F2.3a s0 | −0,5 [−1,7; +0,6] | +0,1 [−0,9; +1,1] |
| F2.3a s1 | +0,3 [−0,8; +1,6] | +0,8 [+0,0; +1,7] |
| F2.1 s0 | **−3,8 [−6,0; −1,7]** | **−2,8 [−4,4; −1,3]** |
| F2.1 s1 | +1,3 [−0,9; +3,4] | +0,3 [−1,4; +1,8] |
| F2.2 s0 | **−2,3 [−4,2; −0,6]** | −0,9 [−2,1; +0,2] |
| F2.2 s1 | +0,3 [−1,2; +1,8] | +0,5 [−0,8; +1,7] |
| F3 s0 | +0,3 [−1,6; +2,3] | −0,9 [−2,7; +0,8] |
| F3 s1 | **−3,7 [−6,1; −1,4]** | **−2,5 [−4,6; −0,6]** |

(Componentes por família no script; `juristcu` e `invoice` são os mesmos dos vereditos originais.)

## O que isso diz

1. **Com 6 ordens, o E3 transfere para o `factck` nas três sementes:** +10,9, +8,2 e +8,8 pontos, todos com IC
   acima de zero. Com 2 ordens, o mesmo par dava +2,9, 0,0 e −1,0.
   - A H2 registrada continua inconclusiva.
   - Pela regra nova, o critério principal da H2 teria IC acima de zero nas três sementes (+5,3, +5,3, +4,4).
   - A confirmação tem de vir do teste oficial, que será lido com a regra nova (fixada antes desta rodada).
2. **As 2 ordens punham ruído de posição no `factck`.**
   - Com 6 ordens, os ICs das 18 comparações ficam em média 27% mais estreitos (de 6% a 46%).
   - A diferença entre os dois controles do F3 (só a semente muda) cai de +11,8 para +5,4. Mesmo assim, segue
     acima de zero: a mistura balanceada do F2.1 é instável no `factck` entre sementes.
   - Entre as sementes do E3, a diferença fica entre −2,1 e −2,8, com IC cruzando zero.
3. **Vereditos que a releitura mantém:**
   - **H3 piora em 2 de 3 sementes.** O colapso da cabeça no `factck` é real e depende da semente (s1 empata).
   - **F2.1 s0 piora.**
   - **O F2.3a empata.**
   - **O E3b não ajuda.**
   - **O piloto F3 segue inconclusivo:** s0 cruza zero, s1 piora. A queda no `factck` persiste.
4. **O F2.2 fica mais neutro:** com 6 ordens, tirar o `in_proj` não mexe no `factck` (−1,8 e 0,0, ambos
   cruzando zero). A piora da s0 vinha em boa parte das 2 ordens. A decisão de manter o `in_proj` não muda: nada
   indica ganho em tirá-lo.
5. **O braço diverso do F3 desloca o `factck` de `falso` para `impreciso`.** O congelado já preferia
   `impreciso` (101/144 contra 165/408 em `falso`). O LoRA do E3 aprende a dizer `falso`, e trocar metade do
   Felhen por intenção e emoção desfaz parte disso.
