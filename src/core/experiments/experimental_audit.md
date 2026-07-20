# Auditoria experimental da pipeline hibrida

Data: 2026-06-26

> Documento histórico. A execução descrita abaixo está agora encapsulada em
> `investigative_runtime.py`; o núcleo de produção foi restaurado ao commit `076326b`.

## Parametros TOML realmente usados pelo runner

O ficheiro `scenario_runner.py` le todos os TOML, mas so alguns campos alteram a execucao:

- `workload.mode`: `preset`, `generated`, `live_trace`.
- `workload.weights`: usado em `preset`.
- `workload.num_cores`: usado para construir snapshot e QUBO.
- `workload.num_processes`: validado em `generated`; em `preset` e praticamente documental.
- `workload.weight_strategy`: suporta apenas `uniform_total`.
- `workload.total_weight`: usado apenas com `generated` e `uniform_total`.
- `qaoa.layers`, `qaoa.steps`, `qaoa.learning_rate`, `qaoa.top_k`, `qaoa.mixer_type`, `qaoa.init_gamma`, `qaoa.init_beta`.
- `qubo.penalty`, `qubo.target_load`.
- `tracer.min_rss`, `tracer.min_cpu`, `tracer.cpu_interval`, `tracer.num_samples`, `tracer.live_mode`.
- `decomposition.qubit_max`, `decomposition.num_cores`, `decomposition.io_alpha`, `decomposition.affinity_alpha`, `decomposition.homogeneity_threshold`, `decomposition.zscore_threshold`, `decomposition.sorting_strategy`.
- `execution.repeats`.

Campos presentes em TOML mas nao usados para controlar a execucao:

- `execution.pipeline`: o codigo escolhe a pipeline por `num_entities * num_cores <= qubit_max`; o valor do TOML e documental.
- `execution.solvers`: documental; o runner executa sempre QAOA e a validacao chama brute-force ou SA conforme o limite.
- `execution.brute_force_policy`: documental; o limite real esta em `BRUTE_FORCE_VAR_LIMIT = 22`.
- `checks.*`: documental; nao ha assercoes automaticas no runner.
- `record.metrics`: documental; nao filtra nem altera as metricas gravadas.
- `pre_run.operator_action`, `workload.process_source`, `qubo.p_critical_note`, `qubo.penalty_intent`: metadados.
- `qubo.extra_couplings`: explicitamente ignorado com aviso.
- `sweep.axes`: o `scenario_runner.py` nao expande sweeps. O novo `sweep_runner.py` expande estes eixos e injecta variantes em memoria.

## Metricas actualmente gravadas

Para a `DefaultPipeline`, `summarize_output()` grava:

- `pipeline`, `output_type`, `num_variables`, `num_entities`, `num_cores`.
- `energy`, `feasible`, `optimal`, `optimality_gap`.
- `solve_time_ms`, `solver_backend`, `max_probability`, `assignments`.
- `annealing_energy`, `annealing_gap`, `beats_annealing`, `annealing_solve_time_ms`.
- `validation`, incluindo energia candidata, energia brute-force quando disponivel, assignments, erros de viabilidade, minimo unconstrained e dados de SA quando brute-force e recusado.

Para a `IterativePipeline`, grava tambem:

- `num_sub_qubos`, `num_feasible_sub_qubos`, `all_sub_qubos_feasible`.
- `load_imbalance`, `final_phi`.

Metricas ausentes ou incompletas para a matriz experimental proposta:

- Iteracoes ate convergencia ou criterio formal de convergencia.
- Tempo por circuito/camada separado do tempo total do solver.
- Distribuicao completa de probabilidades e lista top-k no JSON final.
- Feasibility rate agregada entre repeticoes; e derivavel no CSV, mas nao e calculada no runner.
- Gaps relativos contra SA em todos os casos; hoje `annealing_gap` so aparece quando brute-force e recusado.
- Metricas intra/inter-cluster da decomposicao.
- Parametros finais optimizados `gamma` e `beta`.
- Registo explicito de pesos gerados/live e de composicao dos clusters por variante.

## Estado dos bugs assinalados

1. `psutil cpu_percent=0.0`: mitigado parcialmente. O tracer chama `proc.cpu_percent()` para inicializar a medicao, dorme `cpu_interval`, e so depois chama `proc.cpu_percent()` novamente. Portanto nao e uma leitura imediata unica. O valor de `min_cpu`, contudo, nao esta a filtrar processos porque a linha correspondente esta comentada.

2. Capacidade residual negativa: confirmado como estado possivel. A pipeline imprime `L_avg - phi`; `phi` e actualizado com `phi[assigned_core] += entity.cpu_weight`, logo uma decisao anterior pode sobrecarregar um core e tornar a capacidade residual negativa. O sinal da propagacao de bias esta coerente com o builder: o termo diagonal soma `+2*w_i*phi_k`, penalizando atribuicoes futuras a cores ja carregados. O problema e interpretativo/experimental: residuo negativo nao e erro aritmetico por si so, mas indica degradacao local da decomposicao.

3. `alpha` metric inversion: nao ha inversao no calculo. `SchedulingEngine.optimality_gap()` devolve `(candidate_energy - optimal_energy) / escala`; zero e melhor, valores positivos sao piores. A ambiguidade vem do nome `alpha`, que pode ser lido como qualidade. Nos resultados e na tese deve ser chamado `optimality_gap`.

4. Feasibility check: no núcleo, `SolverValidator` e `BruteForceSolver` mantêm o comportamento do commit `076326b`. A semântica experimental estrita por grupos one-hot está isolada em `investigative_runtime.py`.

5. `top_k`: antes desta auditoria, `qaoa.top_k` era aceite no TOML mas o solver percorria todos os estados por probabilidade. Foi corrigido para escolher a melhor solucao viavel apenas dentro dos `top_k` estados mais provaveis, com fallback para o estado mais provavel quando nenhum candidato top-k e viavel.

## Suporte a sweeps

O runner historico nao expande `sweep.axes`. Foi adicionado `src/experiments/sweep_runner.py` para:

- expandir eixos por produto cartesiano;
- injectar valores por caminhos pontilhados, por exemplo `qaoa.layers`;
- preservar `execution.repeats`;
- registar `config.sweep_context` em cada resultado.

Foi tambem adicionado `src/experiments/analyze_results.py` para produzir uma tabela CSV e graficos PNG simples a partir dos JSONL gerados.

## Ordem de prioridade recomendada

1. Correr sanidade curta: `T1.1`, `T1.2`, `T1.3`.
2. Correr `research_b1_layers_sweep` para profundidade QAOA.
3. Correr `research_c1_penalty_x_sweep` para calibracao de P com mixer X.
4. Correr `research_d2_qubit_max_sweep` para a curva qualidade-vs-orcamento de qubits.
5. So depois correr live trace, assinalando explicitamente nao determinismo e dependencia do estado da maquina.
