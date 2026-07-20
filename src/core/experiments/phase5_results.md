# Resultados corrigidos da Fase 5: escalabilidade até 50 sub-QUBOs

Data da análise: 2026-07-06

## Resumo executivo

A Fase 5 contém **200 runs únicos** em 13 dimensões, de `N=10` a `N=200`.
O maior caso tem 200 processos, 400 variáveis globais e 50 sub-QUBOs de até
oito variáveis. Os 200 runs terminaram com sucesso, produziram atribuições
globais one-hot viáveis e mantiveram todos os 2728 sub-QUBOs viáveis.

Uma auditoria posterior identificou que o gap relativo inicialmente usado não
é apropriado para medir qualidade. A energia QUBO de uma solução viável contém
um offset negativo que cresce com N. Dividir a diferença energética por
`abs(E_baseline)` fazia o gap diminuir artificialmente e também tornava a
tolerância de `is_optimal` demasiado permissiva.

Os JSON originais preservam pesos e assignments, pelo que os 200 runs foram
reanalisados sem repetir QAOA. A análise corrigida calcula diretamente:

```text
objetivo_balanceamento = sum((load_k - mean_load)^2)
desequilíbrio_normalizado = (max(loads)-min(loads))/mean(loads)
regret = max(0, objetivo_pipeline-objetivo_baseline)
```

As conclusões corrigidas são:

1. **A escalabilidade de execução é real.** A pipeline construiu o QUBO global
   de 400 variáveis, mas o QAOA nunca recebeu mais de oito variáveis; executou
   50 subproblemas sequenciais.
2. **A viabilidade foi preservada em 200/200 runs.** Não houve crashes,
   assignments incompletos ou violações one-hot.
3. **O balanceamento melhora com N nesta família.** O desequilíbrio normalizado
   médio desce de 3,217% em `N=10` para 0,00792% em `N=200`.
4. **Os matches energéticos anteriores estavam inflacionados.** O critério
   antigo classificava 26/200 runs como equivalentes à referência. Sem o offset,
   apenas **1/200** satisfaz a tolerância de `1e-9`.
5. **Não foi encontrado nenhum ótimo global certificado.** Nos 60 runs com
   brute-force (`N<=20`), a pipeline ficou ligeiramente acima do ótimo em 60/60.
   Acima de N=20, a referência é simulated annealing e não certifica o ótimo.
6. **O custo cresce aproximadamente de forma linear.** O tempo médio passou de
   6,041 s em `N=10` para 136,060 s em `N=200`.

O resultado cientificamente defensável é que a decomposição preservou
viabilidade e produziu balanceamento muito próximo da referência até 50
sub-QUBOs. Não demonstra optimalidade global, resolução QAOA de 400 qubits nem
vantagem quântica.

## Proveniência e cobertura

| Fonte | Conteúdo | Runs usados |
|:---|:---|---:|
| `sweep_20260705_142459_e5a8c292` | Piloto, `N in {10,15}` | 40 |
| `sweep_20260705_151529_715b9a8d` | Sweep principal, `N=20..200` | 160 |
| **Total analítico** | 13 valores de N | **200** |

O ficheiro `sweep_20260705_150912_71b34970` contém uma tentativa incompleta com
19 runs de `N=20`. Repete as seeds 0 a 18 do sweep principal e foi excluído para
evitar dupla contagem.

Os runs aceites demoraram aproximadamente 1 h 58 min 55 s de relógio. Incluindo
a tentativa incompleta, a campanha consumiu cerca de 2 h 4 min 32 s.

### Configuração

- dois cores e pesos uniformes aleatórios em `(0,1]`;
- `qubit_max=8`, até quatro processos por sub-QUBO;
- pipeline iterativa e ordenação `COUPLING_DESCENDING`;
- QAOA com mixer XY, `p=2`, 100 passos e `top_k=10`;
- `top_k` foi efetivamente aplicado pelo `InvestigativePennylaneSolver`;
- 20 repetições até `N=50`, 12 até `N=100` e oito até `N=200`.

## Qualidade corrigida por dimensão

`Imbalance` é a diferença entre a maior e a menor carga dividida pela carga
média. `Regret` é a diferença entre os objetivos de balanceamento, calculada
diretamente das cargas. “Match” significa `regret <= 1e-9`; para SA continua a
ser apenas equivalência a uma referência heurística.

| N | Vars | Sub-QUBOs | Runs | Referência | Imbalance médio | Mediana | Máximo | Regret médio | Regret máximo | Matches novos | Matches antigos |
|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 10 | 20 | 3 | 20 | BF* | 3,2174% | 2,8087% | 7,0231% | 5.790e-03 | 2.388e-02 | 0/20 | 0/20 |
| 15 | 30 | 4 | 20 | BF* | 1,1222% | 0,8488% | 2,6350% | 1.461e-03 | 7.293e-03 | 0/20 | 0/20 |
| 20 | 40 | 5 | 20 | BF* | 0,9633% | 0,6951% | 3,2695% | 3.061e-03 | 1.806e-02 | 0/20 | 0/20 |
| 25 | 50 | 7 | 20 | SA | 0,2930% | 0,1556% | 1,7398% | 5.944e-04 | 8.250e-03 | 0/20 | 2/20 |
| 30 | 60 | 8 | 20 | SA | 0,4177% | 0,3623% | 1,1670% | 7.709e-04 | 2.905e-03 | 0/20 | 1/20 |
| 40 | 80 | 10 | 20 | SA | 0,1409% | 0,1014% | 0,4050% | 1.587e-04 | 8.781e-04 | 0/20 | 1/20 |
| 50 | 100 | 13 | 20 | SA | 0,1154% | 0,0963% | 0,3016% | 1.730e-04 | 6.924e-04 | 0/20 | 2/20 |
| 65 | 130 | 17 | 12 | SA | 0,0954% | 0,0368% | 0,3068% | 2.860e-04 | 1.437e-03 | 0/12 | 0/12 |
| 80 | 160 | 20 | 12 | SA | 0,0563% | 0,0192% | 0,2144% | 1.564e-04 | 1.016e-03 | 0/12 | 3/12 |
| 100 | 200 | 25 | 12 | SA | 0,0329% | 0,0136% | 0,1295% | 7.623e-05 | 4.280e-04 | 0/12 | 4/12 |
| 130 | 260 | 33 | 8 | SA | 0,0222% | 0,0109% | 0,0902% | 7.635e-05 | 4.794e-04 | 1/8 | 3/8 |
| 160 | 320 | 40 | 8 | SA | 0,0200% | 0,0033% | 0,0730% | 9.408e-05 | 5.088e-04 | 0/8 | 5/8 |
| 200 | 400 | 50 | 8 | SA | 0,0079% | 0,0036% | 0,0287% | 1.987e-05 | 1.130e-04 | 0/8 | 5/8 |

`BF*` é brute-force certificado. SA é uma referência heurística com 10 restarts.
O único match corrigido ocorreu em `N=130`, seed 2, contra SA; não é certificado.

No conjunto completo, o desequilíbrio normalizado da pipeline teve:

- média de 0,6401%;
- mediana de 0,1564%;
- máximo de 7,0231%, em `N=10`, seed 7;
- 169/200 runs até 1%;
- 147/200 até 0,5%;
- 84/200 até 0,1%;
- 30/200 até 0,01%.

A baseline teve desequilíbrio normalizado médio de 0,02027% e mediana de
`4.93e-7`. O regret médio foi `1.240e-3`, a mediana `1.195e-4` e o máximo
`2.388e-2`.

### Por que o gap antigo falhou

Para dois cores e assignments one-hot viáveis, a energia pode ser escrita como:

```text
E_QUBO = imbalance^2/2 - total_weight^2/2 - penalty*N
```

Os dois últimos termos não alteram qual assignment é melhor, mas tornam a
energia cada vez mais negativa. Em `N=200`, `abs(E)` ronda 6000. Assim, dividir
uma diferença de ordem `1e-6` por 6000 produz um gap aparentemente extraordinário.
O mesmo offset entrava na tolerância relativa e permitia que soluções distintas
fossem classificadas como matches.

A reanálise evita subtrair duas energias grandes e próximas. Reconstrói as
cargas diretamente dos pesos e assignments, eliminando também perda de precisão
por cancelamento numérico.

## Escalabilidade temporal

| N | Total médio | Desvio | QAOA médio | Overhead médio | Fração QAOA |
|---:|---:|---:|---:|---:|---:|
| 10 | 6,041 s | 0,115 s | 6,034 s | 0,006 s | 99,90% |
| 15 | 9,136 s | 0,278 s | 8,954 s | 0,182 s | 98,01% |
| 20 | 17,253 s | 0,269 s | 11,511 s | 5,742 s | 66,72% |
| 25 | 16,340 s | 0,207 s | 14,492 s | 1,848 s | 88,69% |
| 30 | 19,049 s | 0,115 s | 16,828 s | 2,221 s | 88,34% |
| 40 | 25,568 s | 0,162 s | 22,552 s | 3,016 s | 88,21% |
| 50 | 32,097 s | 0,162 s | 28,263 s | 3,834 s | 88,05% |
| 65 | 42,206 s | 0,160 s | 37,027 s | 5,179 s | 87,73% |
| 80 | 51,835 s | 0,293 s | 45,338 s | 6,498 s | 87,47% |
| 100 | 65,996 s | 2,169 s | 57,555 s | 8,441 s | 87,21% |
| 130 | 95,559 s | 2,758 s | 83,224 s | 12,336 s | 87,09% |
| 160 | 105,881 s | 7,105 s | 91,368 s | 14,513 s | 86,29% |
| 200 | 136,060 s | 2,146 s | 117,649 s | 18,411 s | 86,47% |

O QAOA simulado consumiu 1,719 h, 86,81% das 1,981 h instrumentadas. Um ajuste
linear às médias produz aproximadamente 2,39 s de QAOA e 2,76 s totais por
sub-QUBO adicional (`R²=0,997` e `R²=0,995`). Isto demonstra escalabilidade no
número de chamadas de oito variáveis, não no tamanho do circuito individual.

## Sweep adaptativo corrigido

A execução histórica usou o gap energético antigo como sinal. O runner foi
alterado para usar:

```toml
quality_metric = "normalized_load_imbalance"
quality_ceiling_mean = 0.01
monotonic_min_delta = 1e-4
```

Uma execução futura sinaliza transição quando o desequilíbrio normalizado médio
ultrapassa 1% ou sobe de forma material em três valores consecutivos. Os tempos
e gaps energéticos antigos continuam registados para auditoria, mas não controlam
a execução.

Aplicando retrospetivamente este critério aos pontos novos, `N=20` tem média de
0,9633% e todos os pontos seguintes ficam abaixo de 0,5%. Logo, o sweep principal
também teria percorrido a grelha sem sinal de degradação. Os pilotos `N=10/15`
ficam acima de 1%, mas são anteriores ao ponto de retoma em `N=20`.

## Interpretação e limites

Os dados suportam:

- viabilidade global em 200/200 runs;
- execução de 50 sub-QUBOs sequenciais para representar 400 variáveis globais;
- balanceamento relativo progressivamente melhor nesta família aleatória;
- custo aproximadamente linear quando o tamanho de cada sub-QUBO é fixo.

Os dados não suportam:

- ótimo global em `N>20`, porque a baseline é SA;
- resolução de um circuito QAOA com 400 qubits;
- vantagem quântica, pois o QAOA foi simulado localmente;
- generalização para mais cores, distribuições adversariais ou sub-QUBOs maiores;
- equivalência energética em 26 runs: 25 desses matches desaparecem sem o offset.

O facto de o balanceamento melhorar com N é plausível para number partitioning
com muitos pesos contínuos e ordenação descendente: existem mais combinações e
mais pesos pequenos no fim para corrigir a carga. A experiência não isola este
mecanismo causalmente.

## Artefactos

- `src/experiments/phase5_reanalysis.py` reproduz a análise corrigida;
- `src/experiments/results/phase5/analysis/phase5_offset_free_raw.csv` contém os
  200 runs com cargas, objetivo, regret, match novo e classificação antiga;
- `src/experiments/results/phase5/analysis/phase5_offset_free_summary.csv` contém
  a agregação por N;
- `src/experiments/results/phase5/analysis/manifest.json` regista entradas e
  definições;
- os JSONL originais permanecem inalterados.

Reprodução:

```bash
uv run python src/experiments/phase5_reanalysis.py
```
