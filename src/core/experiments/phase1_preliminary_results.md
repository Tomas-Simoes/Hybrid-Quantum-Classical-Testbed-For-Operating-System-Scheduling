# Resultados preliminares da fase experimental

Data: 2026-06-26

## Artefactos gerados

- Tabela agregada: `src/experiments/results/analysis_20260626_phase1/phase1_table.csv`
- Graficos: `src/experiments/results/analysis_20260626_phase1/*.png`
- Runs usados:
  - Sanidade A: `run_20260626_152701_7a132ebd_results.jsonl`
  - B1: `sweep_20260626_152713_c32d7ee9_results.jsonl`
  - C1: `sweep_20260626_152916_694c56fc_results.jsonl`
  - D2: `sweep_20260626_153039_2c23dc71_results.jsonl`

## A - Sanidade minima

| Cenario | N | K | Variaveis | Feasible | Otimo brute-force |
|---|---:|---:|---:|---:|---:|
| T1.1 | 2 | 2 | 4 | sim | sim |
| T1.2 | 3 | 2 | 6 | sim | sim |
| T1.3 | 4 | 2 | 8 | sim | sim |

Interpretacao: a pipeline directa esta operacional em instancias minimas e o check one-hot produz solucoes viaveis. O brute-force esta disponivel nestas dimensoes e confirma otimalidade global. Estes resultados validam o harness para experiencias pequenas, mas nao dizem nada sobre escalabilidade. Devem ser tratados como teste de sanidade, nao como evidencia de vantagem.

## B1 - Profundidade QAOA

Instancia: N=4, K=2, pesos `[0.15, 0.35, 0.25, 0.25]`, mixer XY, P=5.0, steps=100, 5 repeticoes.

| layers | Gap medio | Feasibility | Otimo | Tempo medio ms |
|---:|---:|---:|---:|---:|
| 1 | 0.000000 | 100% | 100% | 1277.5 |
| 2 | 0.000000 | 100% | 100% | 1941.9 |
| 3 | 0.000000 | 100% | 100% | 2469.2 |
| 4 | 0.000000 | 100% | 100% | 3169.6 |
| 5 | 0.000000 | 100% | 100% | 3759.5 |
| 6 | 0.000000 | 100% | 100% | 4391.6 |
| 7 | 0.000000 | 100% | 100% | 5110.9 |

Interpretacao: nesta instancia, p=1 ja e suficiente para atingir o otimo global em todas as repeticoes. Aumentar `layers` nao melhora a qualidade observada e aumenta quase linearmente o tempo. Nao ha sinal de barren plateau nesta escala; ha apenas saturacao precoce da qualidade. Para esta classe de instancia pequena, p>1 nao se justifica empiricamente.

## C1 - Penalidade P com mixer X

Instancia: N=4, K=2, mesmos pesos de B1, mixer X, p=2, steps=100, 5 repeticoes. Para esta workload, `P_safe = 2*w_max*(W_total/K) = 0.35`.

| P | Multiplo de P_safe | Gap medio | Feasibility | Otimo |
|---:|---:|---:|---:|---:|
| 0.0875 | 0.25x | 0.000000 | 100% | 100% |
| 0.1750 | 0.50x | 0.000000 | 100% | 100% |
| 0.3500 | 1.00x | 0.000000 | 100% | 100% |
| 0.7000 | 2.00x | 0.000000 | 100% | 100% |
| 1.4000 | 4.00x | 0.000000 | 100% | 100% |

Interpretacao: o mixer X encontrou solucoes viaveis e otimas mesmo abaixo de `P_safe` nesta instancia. Isto nao invalida a heuristica: apenas indica que esta workload pequena e regular nao e adversarial para a restricao one-hot. Como o solver agora escolhe dentro do `top_k`, o resultado tambem depende de `top_k=64`, que cobre muitos estados de uma instancia de 8 variaveis. Para testar robustez de P, e necessario repetir C1 com instancias skewed e top-k menor.

## D2 - Sweep qubit_max

Instancia: N=8, K=2, pesos `[0.30, 0.27, 0.22, 0.18, 0.15, 0.11, 0.08, 0.05]`, mixer XY, p=2, steps=100, 5 repeticoes.

| qubit_max | Pipeline | Sub-QUBOs medios | Gap medio | Feasibility | Otimo | Tempo medio ms |
|---:|---|---:|---:|---:|---:|---:|
| 4 | iterativa | 4 | 0.000000 | 100% | 100% | 4550.3 |
| 6 | iterativa | 3 | 0.000000 | 100% | 100% | 4546.8 |
| 8 | iterativa | 2 | 0.000000 | 100% | 100% | 4777.0 |
| 10 | iterativa | 2 | 0.000000 | 100% | 100% | 4792.5 |
| 12 | iterativa | 2 | 0.000000 | 100% | 100% | 5648.0 |
| 16 | directa | 0 | 0.000591 | 100% | 0% | 6911.1 |

Interpretacao: a pipeline iterativa manteve feasibility e atingiu o otimo brute-force em todos os pontos decompostos. O ponto directo `qubit_max=16` foi sempre viavel, mas nao atingiu o otimo com p=2 e 100 passos. Isto e uma anomalia importante face a uma leitura monotonicamente favoravel a mais qubits: maior orcamento reduz a decomposicao, mas tambem aumenta a dificuldade de optimizacao QAOA da instancia global. A conclusao correcta e preliminar: hardware maior so melhora a solucao se o orcamento algoritmico/classico for suficiente para explorar a instancia maior.

## Recomendacoes praticas preliminares

- Usar mixer XY como default para preservar one-hot estruturalmente.
- Para instancias pequenas N<=4, usar `layers=1` ou `2`; p maior apenas aumenta tempo neste conjunto.
- Manter `steps=100` como baseline inicial, mas testar `steps=200/500` no caso directo N=8, porque `qubit_max=16` falhou otimalidade com p=2/100.
- Usar `P_safe` como baseline conservador para mixer X, mas nao concluir robustez sem workloads skewed.
- Para decomposicao, `qubit_max` deve ser avaliado em conjunto com qualidade QAOA: mais qubits nao implica automaticamente melhor resultado se a instancia global ficar mais dificil de optimizar.

## Ameacas a validade

- As repeticoes sao deterministicas para estas configuracoes; a estabilidade observada pode reflectir inicializacao fixa e nao variabilidade real.
- O conjunto de workloads sinteticos e pequeno e regular; ainda falta testar distribuicoes skewed/Pareto e instancias adversariais.
- Brute-force so e viavel ate ao limite implementado; para instancias maiores, comparacoes devem usar SA como baseline heuristico, sem chamar otimo ao resultado.
- Nao ha evidencia de vantagem quantica; os resultados medem comportamento de uma simulacao QAOA/PennyLane e de uma pipeline de decomposicao.
- Resultados live via psutil devem ser tratados apenas como demonstracao de integracao, por dependerem do estado da maquina no momento da captura.
