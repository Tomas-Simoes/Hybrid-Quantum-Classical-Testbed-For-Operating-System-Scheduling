# Resultados preliminares da fase experimental 2

Data: 2026-06-26

## Artefactos gerados

- Sweep agregado da fase 2: `src/experiments/results/sweep_20260626_163548_3b2fdff6_results.jsonl`
- Output legivel do sweep: `src/experiments/results/sweep_20260626_163548_3b2fdff6_output.txt`
- Tabela agregada fase 1 + fase 2: `src/experiments/results/analysis_20260626_phase2/phase2_table.csv`
- Graficos: `src/experiments/results/analysis_20260626_phase2/*.png`
- Runs fase 1 reutilizados:
  - Sanidade A: `run_20260626_152701_7a132ebd_results.jsonl`
  - B1 fixo: `sweep_20260626_152713_c32d7ee9_results.jsonl`
  - C1: `sweep_20260626_152916_694c56fc_results.jsonl`
  - D2: `sweep_20260626_153039_2c23dc71_results.jsonl`

Comando principal:

```bash
uv run python src/experiments/sweep_runner.py \
  research_b1_random research_b2 research_b3 research_b4_x research_b4_xy \
  research_c2 research_d1 research_d3 research_d4 research_d6
```

Resumo global da fase 2: 340 execucoes, 340 sucessos, 340/340 viaveis, 335/340 otimas contra brute-force. As 5 execucoes nao otimas sao todas de D6 com `top_k=1`.

## Alteracoes ao harness para fase 2

- `QAOAConfig` aceita agora `init_strategy = "fixed" | "random"` e `random_seed`.
- O solver PennyLane regista `initial_gamma`, `initial_beta`, `final_gamma` e `final_beta`.
- O runner grava `solver_params` e metricas operacionais de convergencia.
- `sweep_runner.py` suporta `sweep.cases`, necessario para casos pareados em C2.

As metricas de convergencia devem ser lidas como diagnostico operacional: `convergence_iterations_to_final_tol` e a primeira iteracao dentro de uma tolerancia relativa de `1e-4` face ao objectivo final observado, nao uma prova formal de convergencia.

## B1R - Profundidade QAOA com inicializacao aleatoria

Instancia: N=4, K=2, pesos `[0.15, 0.35, 0.25, 0.25]`, mixer XY, P=5.0, steps=100, sementes 101..105.

| layers | Execucoes | Feasible | Otimo | Gap medio | Tempo medio ms | Iter. ate tol. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 100% | 100% | 0.000000 | 1325.1 | 6.4 |
| 2 | 5 | 100% | 100% | 0.000000 | 1775.9 | 5.8 |
| 3 | 5 | 100% | 100% | 0.000000 | 2385.5 | 40.8 |
| 4 | 5 | 100% | 100% | 0.000000 | 3084.8 | 53.4 |
| 5 | 5 | 100% | 100% | 0.000000 | 3805.4 | 59.8 |
| 6 | 5 | 100% | 100% | 0.000000 | 4350.9 | 66.6 |
| 7 | 5 | 100% | 100% | 0.000000 | 5076.8 | 64.6 |

Interpretacao: a conclusao de B1 mantem-se com inicializacao aleatoria. Nesta instancia pequena, p=1 ja chega ao otimo em todas as sementes. A profundidade adicional aumenta tempo e, em media, tambem atrasa a estabilizacao do objectivo. Nao ha evidencia de barren plateau nesta escala; ha saturacao precoce da qualidade.

## B2 - Passos e learning rate

Instancia: mesma de B1R, mixer XY, p=2, 5 repeticoes por combinacao.

| steps | Execucoes | Feasible | Otimo | Gap medio | Tempo medio ms | Iter. ate tol. |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 20 | 100% | 100% | 0.000000 | 906.4 | 12.5 |
| 100 | 20 | 100% | 100% | 0.000000 | 1779.3 | 19.8 |
| 200 | 20 | 100% | 100% | 0.000000 | 3592.2 | 19.8 |
| 500 | 20 | 100% | 100% | 0.000000 | 9211.8 | 19.0 |

| learning_rate | Execucoes | Feasible | Otimo | Gap medio | Iter. ate tol. |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 20 | 100% | 100% | 0.000000 | 22.0 |
| 0.05 | 20 | 100% | 100% | 0.000000 | 15.0 |
| 0.10 | 20 | 100% | 100% | 0.000000 | 10.0 |
| 0.50 | 20 | 100% | 100% | 0.000000 | 24.0 |

Interpretacao: neste benchmark, 50 passos ja bastam. Aumentar para 500 passos multiplica o custo sem ganho de qualidade. Entre os learning rates testados, `0.1` estabilizou mais cedo em media, mas todos atingiram o otimo.

## B3 - Grid de inicializacao fixa

Instancia: mesma de B1R, mixer XY, `gamma,beta in {0.1, 0.3, 0.5, 0.7, 1.0}`, p=2 e p=3.

| layers | Execucoes | Feasible | Otimo | Gap medio | Tempo medio ms | Iter. ate tol. |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 25 | 100% | 100% | 0.000000 | 1857.8 | 10.7 |
| 3 | 25 | 100% | 100% | 0.000000 | 2466.4 | 35.4 |

Interpretacao: nenhum ponto da grelha mostrou degradacao de viabilidade ou otimalidade. A inicializacao `gamma=beta=0.5` nao aparece como minimo local problematico nesta workload. Tal como em B1/B2, p=3 custa mais e nao melhora a qualidade observada.

## B4 - Mixer e penalidade

Instancia: mesma de B1R. Para esta workload, `P_safe = 0.35`.

### Mixer X

| P | layers | Execucoes | Feasible | Otimo | Gap medio | Tempo medio ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0.175 | 1 | 5 | 100% | 100% | 0.000000 | 1393.7 |
| 0.175 | 2 | 5 | 100% | 100% | 0.000000 | 2369.8 |
| 0.175 | 3 | 5 | 100% | 100% | 0.000000 | 3155.8 |
| 0.350 | 1 | 5 | 100% | 100% | 0.000000 | 1438.2 |
| 0.350 | 2 | 5 | 100% | 100% | 0.000000 | 2298.3 |
| 0.350 | 3 | 5 | 100% | 100% | 0.000000 | 3282.3 |
| 0.700 | 1 | 5 | 100% | 100% | 0.000000 | 1399.7 |
| 0.700 | 2 | 5 | 100% | 100% | 0.000000 | 2245.2 |
| 0.700 | 3 | 5 | 100% | 100% | 0.000000 | 3155.0 |

### Mixer XY

| layers | Execucoes | Feasible | Otimo | Gap medio | Tempo medio ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 100% | 100% | 0.000000 | 1229.0 |
| 2 | 5 | 100% | 100% | 0.000000 | 1798.7 |
| 3 | 5 | 100% | 100% | 0.000000 | 2452.5 |

Interpretacao: o mixer X continuou a encontrar solucoes viaveis e otimas mesmo em `P_safe/2`, e o XY tambem foi perfeito. Isto reforca a leitura de fase 1: esta instancia e demasiado regular para stressar P. O mixer XY continua preferivel como default porque preserva one-hot estruturalmente.

## C2 - Distribuicao de pesos e penalidade segura

Instancias: 10 uniformes e 10 enviesadas, cada uma com `P_safe` e `2*P_safe`, mixer X, p=2, top_k=16.

| Distribuicao | Multiplo P_safe | Execucoes | Feasible | Otimo | Gap medio | Tempo medio ms |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 1.0 | 10 | 100% | 100% | 0.000000 | 2230.4 |
| uniform | 2.0 | 10 | 100% | 100% | 0.000000 | 2291.1 |
| skewed | 1.0 | 10 | 100% | 100% | 0.000000 | 2248.7 |
| skewed | 2.0 | 10 | 100% | 100% | 0.000000 | 2268.2 |

Interpretacao: `P_safe` foi suficiente em todos os casos C2, incluindo pesos enviesados. Duplicar P nao melhorou qualidade, porque a qualidade ja estava saturada. Ainda nao ha evidencia contra a heuristica de P, mas tambem nao ha stress adversarial forte: top_k=16 numa instancia de 8 variaveis continua a dar bastante cobertura.

## D1 - Escalabilidade com N

Workloads geradas uniformemente com peso total 1.0, K=2, `qubit_max=6`, mixer XY, p=2, steps=100.
Nos pontos directos, o load imbalance foi inferido dos pesos uniformes e das atribuicoes otimas, porque a metrica so e gravada directamente pela pipeline iterativa.

| N | Pipeline | Sub-QUBOs medios | Execucoes | Feasible | Otimo | Load imbalance medio | Tempo medio ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | directa | 0 | 5 | 100% | 100% | 0.000 | 896.1 |
| 3 | directa | 0 | 5 | 100% | 100% | 0.333 | 1279.6 |
| 4 | iterativa | 2 | 5 | 100% | 100% | 0.000 | 2401.3 |
| 5 | iterativa | 2 | 5 | 100% | 100% | 0.200 | 2869.1 |
| 6 | iterativa | 2 | 5 | 100% | 100% | 0.000 | 3279.0 |
| 8 | iterativa | 3 | 5 | 100% | 100% | 0.000 | 4202.9 |
| 10 | iterativa | 4 | 5 | 100% | 100% | 0.000 | 5755.9 |

Interpretacao: a decomposicao manteve otimalidade global ate N=10 nestas workloads uniformes. Os desequilibrios de N=3 e N=5 sao esperados por indivisibilidade dos pesos, nao por falha da pipeline. O tempo cresce com o numero de sub-QUBOs e com a validacao brute-force global.

## D3 - Sobreposicao directa vs iterativa

Instancia: N=6, K=2, pesos `[0.20, 0.30, 0.15, 0.35, 0.10, 0.40]`, p=2, steps=100.

| qubit_max | Pipeline | Sub-QUBOs medios | Execucoes | Feasible | Otimo | Load imbalance medio | Tempo medio ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 6 | iterativa | 2 | 5 | 100% | 100% | 0.000 | 3321.3 |
| 12 | directa | 0 | 5 | 100% | 100% | 0.000 | 3581.7 |

Interpretacao: nesta instancia, directo e iterativo concordam e atingem o otimo global. Isto mostra que a anomalia D2 de fase 1 (`qubit_max=16` directo subotimo em N=8) nao e inevitavel para qualquer execucao directa; e dependente da instancia e do orcamento algoritmico.

## D4 - Estrategia de ordenacao

Instancia: N=8, K=2, pesos `[0.30, 0.27, 0.22, 0.18, 0.15, 0.11, 0.08, 0.05]`, `qubit_max=6`, p=2, steps=100.

| sorting_strategy | Sub-QUBOs medios | Execucoes | Feasible | Otimo | Load imbalance medio | Tempo medio ms |
|---|---:|---:|---:|---:|---:|---:|
| COUPLING_DESCENDING | 3 | 5 | 100% | 100% | 0.000 | 4486.8 |
| WEIGHT_DESCENDING | 3 | 5 | 100% | 100% | 0.000 | 4495.0 |

Interpretacao: nao houve diferenca observavel nesta instancia. A estrategia `CORE_BALANCE` nao foi testada porque nao ha suporte implementado no codigo actual; inclui-la no TOML seria apenas metadado ou falharia sem alteracao funcional.

## D6 - Sensibilidade a top_k

Instancia: mesma de D4, `qubit_max=6`, p=2, steps=100.

| top_k | Execucoes | Feasible | Otimo | Gap medio | Load imbalance medio | Tempo medio ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 100% | 0% | 0.000591 | 0.220 | 4514.8 |
| 3 | 5 | 100% | 100% | 0.000000 | 0.000 | 4541.6 |
| 5 | 5 | 100% | 100% | 0.000000 | 0.000 | 4183.6 |
| 10 | 5 | 100% | 100% | 0.000000 | 0.000 | 4126.4 |

Detalhe do caso `top_k=1`: a solucao continua one-hot e viavel, mas fica subotima com energia candidata `-40.9006` contra otimo global `-40.9248` (`delta = 0.0242`) e load imbalance `0.22`.

Interpretacao: D6 e o primeiro sweep desta fase que revela um parametro claramente sensivel. Escolher apenas o estado mais provavel e demasiado miope para a decomposicao iterativa; `top_k>=3` bastou nesta instancia para recuperar o otimo global. Recomendacao preliminar: nao usar `top_k=1` em experiencias de qualidade; manter pelo menos `top_k=3`, e preferir `top_k>=10` quando o custo de decodificacao for aceitavel.

## D5 e live trace

D5 nao foi executado de forma deterministica nesta fase. Os parametros `io_alpha`, `affinity_alpha`, `homogeneity_threshold` e `zscore_threshold` so afectam a construcao de clusters quando a pipeline passa por dados live/adaptativos; as workloads `preset` e `generated` usadas nestes sweeps bypassam essa dinamica. Executar D5 agora com TOML sintetico daria uma falsa sensacao de cobertura.

O live trace continua recomendado apenas como demonstracao/diagnostico, nao como evidencia estatistica, porque depende do estado da maquina no momento da captura.

## Conclusoes preliminares

- A fase 2 confirma robustez nos casos pequenos: todos os sweeps B, C, D1, D3 e D4 foram viaveis e otimos contra brute-force.
- Para a instancia N=4/K=2, aumentar `layers`, `steps` ou variar inicializacao nao melhorou qualidade; so aumentou custo.
- `P_safe` funcionou em C2 para distribuicoes uniformes e enviesadas, mas ainda falta stress adversarial mais forte.
- A decomposicao iterativa teve bom comportamento em workloads uniformes ate N=10 e na instancia N=8 de D4/D6 quando `top_k>=3`.
- `top_k` e criticamente relevante: `top_k=1` produziu uma solucao viavel mas subotima em 5/5 repeticoes.
- A anomalia D2 de fase 1 permanece importante: mais qubits nao implicam monotonicamente melhor qualidade se o problema global ficar mais dificil para o QAOA com o mesmo p/steps.

## Recomendacoes para a proxima ronda

- Usar mixer XY como default para workloads one-hot.
- Para benchmarks pequenos, usar `layers=1` ou `2` e `steps=50` ou `100`; valores maiores so devem ser justificados por instancia mais dificil.
- Evitar `top_k=1`; usar `top_k>=3` como minimo e `top_k>=10` como baseline mais conservador.
- Repetir a falha D2 directa N=8 com maior orcamento algoritmico (`layers`, `steps`, inicializacao multi-start) antes de tirar conclusoes sobre hardware maior.
- Criar instancias adversariais para P: pesos mais extremos, `top_k` menor e comparacao X vs XY.
- Implementar ou expor um caminho deterministico para D5 se a analise de clustering for parte central da tese.

## Ameacas a validade

- Todas as instancias desta fase ainda sao pequenas o suficiente para brute-force; isto e bom para validacao, mas limitado para escalabilidade real.
- Muitas repeticoes sao deterministicas quando a configuracao fixa a inicializacao; as percentagens nao devem ser lidas como estimativa estatistica independente.
- As workloads sinteticas sao regulares; faltam workloads com distribuicoes Pareto, cargas correlacionadas e constraints adicionais.
- Os tempos medem simulacao PennyLane/local e incluem overheads do runner e validacao; nao sao tempos de hardware quantico.
- Nao ha evidencia de vantagem quantica. Os resultados avaliam a qualidade operacional da pipeline e das heuristicas de decomposicao/decodificacao.
