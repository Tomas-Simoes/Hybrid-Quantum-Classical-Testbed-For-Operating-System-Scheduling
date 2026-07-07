# Auditoria das alterações de código introduzidas para a campanha experimental

> Nota histórica: este relatório descreve a implementação antes da encapsulação
> investigativa. A implementação atual restaurou o núcleo ao commit `076326b` e
> concentrou configurações, baselines, instrumentação e métricas em
> `src/experiments/investigative_runtime.py`. As referências abaixo a alterações em
> `main.py`, pipelines, contratos e solvers documentam o estado anterior, não o atual.

Data da auditoria: 2026-07-04

## 1. Escopo e limitação de autoria

Este relatório compara o estado atual do diretório de trabalho com o `HEAD` do Git.
O Git demonstra quais linhas foram acrescentadas, removidas ou substituídas desde o
último commit, mas não consegue atribuir autoria individual num diretório partilhado e
com alterações sem commit. Assim, “linhas introduzidas” significa neste documento
“linhas presentes no trabalho experimental e ausentes do `HEAD`”.

Foram excluídos desta contagem:

- JSONL, TXT, logs, imagens, CSV e outros resultados gerados;
- relatórios Markdown de resultados;
- ficheiros LaTeX;
- cenários TOML, que são inventariados no fim como configuração, não como código Python.

Os números abaixo referem-se ao código Python e aos testes:

| Grupo | Linhas adicionadas | Linhas removidas |
|---|---:|---:|
| Sete ficheiros Python já existentes | 560 | 57 |
| Cinco ficheiros Python novos de execução/análise | 1929 | 0 |
| Dois ficheiros de testes novos | 252 | 0 |
| **Total de código e testes** | **2741** | **57** |

As linhas indicadas são as linhas atuais e podem deslocar-se se os ficheiros forem
posteriormente editados.

## 2. Classificação das alterações

### Alterações que mudam resultados ou decisões algorítmicas

1. `top_k` passou a limitar realmente o conjunto de estados examinados.
2. Foi adicionada inicialização QAOA aleatória com semente.
3. A pipeline pode agora ser forçada para `default` ou `iterative`.
4. A descodificação passou a validar explicitamente cada grupo one-hot.
5. A classificação de ótimo passou a usar uma tolerância numérica mais apertada.
6. Snapshots sintéticos podem ser submetidos a clustering antes do scheduling.
7. Foi adicionado um solver de simulated annealing como baseline.

Estas alterações significam que a campanha experimental não avaliou exatamente o
código do `HEAD`; avaliou esta versão alterada.

### Instrumentação que não deve alterar a solução escolhida

1. Tempos por componente com `time.perf_counter()`.
2. Metadados de clustering.
3. Curvas e métricas de convergência.
4. Diagnósticos de massa de probabilidade e ranking.
5. Registo de parâmetros QAOA iniciais e finais.
6. Opção de desativar figuras durante sweeps.

### Infraestrutura experimental

1. Expansão cartesiana de sweeps TOML.
2. Criação de CSV, gráficos e manifestos.
3. Testes do baseline SA, sweeps, clustering e tolerância.

## 3. `src/main.py`: todas as adições explicadas

O ficheiro passou de um orquestrador essencialmente automático para um orquestrador
configurável e instrumentado. O diff contém 162 adições e 27 remoções.

### Linhas 34–35 — configurações de `SchedulingEngine.run_job`

- `RunConfig`: agrupa modo de pipeline, clustering de snapshots preset e visualização.
- `ValidationConfig`: agrupa método de baseline, limites de brute force, seed e
  parâmetros de simulated annealing e tolerância.

Impacto: `RunConfig.pipeline_mode` e `cluster_preset_snapshot` podem mudar a solução produzida.
Os restantes controlam validação, reprodutibilidade e efeitos laterais.

### Linhas 43–52 — estado de instrumentação e validação do modo

- Cria `component_timings_ms` com tempos de workload e clustering inicializados a zero.
- Cria `experiment_metadata` para anexar diagnósticos ao output.
- Rejeita valores de pipeline diferentes de `auto`, `default` e `iterative`.

Impacto: instrumentação, exceto a validação que transforma erros silenciosos em exceção.

### Linha 64 — teste explícito `preset_snapshot is None`

Substitui `if not preset_snapshot`. A nova forma distingue corretamente ausência de
snapshot de um objeto eventualmente falso por sobrecarga de truthiness.

Impacto: correção defensiva do controlo de fluxo.

### Linhas 67–72 — tempo do live tracer

Mede `ProcessTracer.trace()` com relógio monotónico e guarda milissegundos em
`workload_ms`.

Impacto: apenas medição.

### Linhas 80–90 — tempo e diagnósticos do clustering live

- Mede `adaptive_cluster.decompose(snapshot)`.
- Guarda o tempo em `clustering_ms`.
- Chama `cluster_diagnostics()` e guarda o resultado em metadados.

Impacto: a chamada de clustering já existia; as novas linhas medem e descrevem o
resultado, sem mudar a decomposição.

### Linhas 95–119 — clustering opcional de snapshots preset

- Se `run_cfg.cluster_preset_snapshot=True`, cria `AdaptiveCluster`, decompõe o snapshot,
  converte bundles em workload, mede o tempo e guarda diagnósticos.
- Caso contrário, mantém a conversão direta `snapshot.to_workload()`, agora medida.

Impacto: mudança funcional importante. No modo sintético E1c, os processos deixam de
ser entidades individuais e passam a bundles antes da construção do QUBO.

### Linhas 158–161 — configuração do validador

Passa um único `ValidationConfig` para `SolverValidator`.

Impacto: adiciona um baseline e tempo de validação; não altera o candidato QAOA.

### Linhas 167–179 — decisão de pipeline configurável

- `default` força a pipeline direta.
- `iterative` força a pipeline decomposta, mesmo abaixo de `qubit_max`.
- `auto` conserva a regra histórica baseada no número de variáveis.
- Forçar `default` acima de `qubit_max` lança uma exceção.

Impacto: mudança funcional importante. Foi necessária para comparar diretamente as
duas pipelines e para E3 executar sempre o caminho iterativo.

### Linhas 196–212 — visualização direta opcional

Envolve `Visualizer(...)` em `if run_cfg.enable_visualization`.

Impacto: não muda o resultado matemático; evita custo gráfico, ficheiros e consumo de
memória em execuções repetidas.

### Linhas 232–238 — agregação de tempos da pipeline direta

- Copia os tempos produzidos pela `DefaultPipeline`.
- Calcula `instrumented_total_ms`.

Impacto: apenas instrumentação.

### Linhas 247–248 — tempos e metadados no output direto

Acrescenta `component_timings_ms` e `experiment_metadata` ao `SchedulingOutput`.

Impacto: altera o contrato de saída, não a solução.

### Linhas 303–321 — visualização iterativa e fecho da figura

- Torna `IterativeVisualizer` opcional.
- Guarda a figura apenas quando ativado.
- Fecha explicitamente a figura com Matplotlib para evitar acumulação de memória.
- Agrega tempos da pipeline iterativa e calcula o total instrumentado.

Impacto: controlo de efeitos laterais e instrumentação.

### Linhas 335–336 — tempos e metadados no output iterativo

Acrescenta os mesmos dois campos ao `IterativeSchedulingOutput`.

Impacto: contrato de saída.

### Linhas 339–393 — novo método `cluster_diagnostics`

O método:

1. reconstrói a matriz de features e a matriz de afinidade;
2. mapeia PID para processo, índice e bundle;
3. percorre todos os pares de processos;
4. separa couplings intra-cluster e inter-cluster;
5. calcula contagem, soma e média de cada grupo;
6. regista configuração, número de processos, bundles, PIDs, comandos, peso e RSS.

Impacto: observacional. Contudo, volta a calcular features e afinidades, acrescentando
custo de pós-processamento ao caminho de clustering.

## 4. `src/data_contracts.py`: todas as adições explicadas

O diff contém 10 linhas adicionadas.

### Linhas 341–342

Adiciona `component_timings_ms` e `experiment_metadata` a `SchedulingOutput`.

### Linhas 372–373

Adiciona os mesmos campos a `IterativeSchedulingOutput`.

### Linhas 411–412

Adiciona `init_strategy` (`fixed` ou `random`) e `random_seed` a `QAOAConfig`.

### Linhas 417–420

Valida `init_strategy` e lança `ValueError` para valores desconhecidos.

### Configuração de execução e validação

Adiciona `RunConfig` e `ValidationConfig`, evitando propagar cada opção experimental
como argumento individual de `SchedulingEngine.run_job` e `SolverValidator`.

Impacto: os campos de output são instrumentação; os campos QAOA permitem alterar
materialmente a trajetória de otimização.

## 5. `src/solver/pennylane_solver.py`: todas as adições explicadas

O diff contém 36 adições e 16 remoções.

### Linhas 21–22

Copia `init_strategy` e `random_seed` da configuração para o solver.

### Linhas 94–105

- Em modo `random`, cria `numpy.random.default_rng(seed)`.
- Inicializa cada gamma e beta uniformemente em `[0,1)`.
- Em modo `fixed`, conserva os valores constantes históricos.

Impacto: mudança algorítmica. Diferentes sementes podem produzir diferentes mínimos.

### Linha 129 e linha 136

- Limita os índices candidatos aos `top_k` estados mais prováveis.
- O ciclo de seleção percorre apenas esses candidatos.

Antes destas linhas, `top_k` existia na configuração mas o solver percorria todos os
estados e escolhia o melhor viável globalmente. Esta é uma correção funcional decisiva
e explica por que `top_k` passou a afetar a qualidade.

### Linhas 170–180 — novos parâmetros registados

Regista estratégia, semente, gamma/beta iniciais, gamma/beta finais, `top_k` e identifica
o pool como `top_k_probable_states`.

Impacto: rastreabilidade; não muda a solução para além das mudanças anteriores.

### Linhas 188–199 — nova descodificação one-hot

- Divide o bitstring em grupos de `K` bits por entidade.
- Só descodifica um grupo se a soma for exatamente 1.
- Obtém o core pelo único bit ativo.
- A solução é viável apenas se todas as entidades forem descodificadas.

Antes, o código percorria todos os bits ativos e criava strings `CONFLICT(...)` quando
uma entidade tinha mais de um core. A nova versão alinha a descodificação com a condição
one-hot usada pelo validador e pelo brute force.

Impacto: correção funcional; pode mudar o conteúdo de assignments inválidos e a decisão
de viabilidade em casos-limite.

## 6. `src/solver/solver_validator.py`: todas as adições explicadas

O diff contém 52 adições e uma remoção.

### Linhas 4–5

Importa o novo `AnnealingSolver`.

### Construtor

Recebe um único `ValidationConfig`; os controlos do baseline deixam de ser argumentos
individuais do validador.

### Linhas 34–42

Inicializa todos os campos de tempo, energia, assignments, backend, gap e erro do SA.

### Linha 47

Guarda o tempo reportado pelo brute force.

### Linhas 56–61 — tolerância de ótimo

Substitui `np.isclose()` com defaults por `np.isclose(rtol=1e-9, atol=1e-9)`.

Impacto: torna a classificação mais restrita. Nota metodológica: isto não é exatamente
o mesmo que `abs(E-E*) <= 1e-9`, porque `np.isclose` também aceita o termo relativo
`1e-9 * abs(E*)`. A análise final da Fase 4 recalcula separadamente a condição absoluta.

### Linhas 65–81 — execução do simulated annealing

Executa SA quando o brute force não está disponível ou quando foi explicitamente
pedido. Guarda energia, assignments, viabilidade, tempo, backend, gap e se o candidato
QAOA iguala/supera SA. Exceções do baseline são capturadas em `annealing_error`.

Impacto: baseline de validação. Não altera o candidato QAOA, mas aumenta o tempo total
da experiência.

### Linhas 94–102

Expõe todos os novos campos no dicionário de validação.

## 7. `src/pipeline/default_pipeline.py`: todas as adições explicadas

O diff contém 24 adições e 7 remoções.

### Linhas 23–31

Substitui `time.time()` por `time.perf_counter()` e mede separadamente construção QUBO
e execução QAOA em milissegundos.

### Linhas 42–55

Mede a validação e grava um dicionário uniforme de tempos, usando zero para componentes
que não existem na pipeline direta.

### Linhas 58–64

Evita formatar `None` quando o brute force é recusado. Nesse caso imprime a razão e,
quando disponível, a energia SA e a comparação com QAOA.

Impacto: instrumentação e robustez de output; não muda a otimização.

## 8. `src/pipeline/iterative_pipeline.py`: todas as adições explicadas

O diff contém 42 adições e 3 remoções.

### Linhas 73–79

Inicializa acumuladores para construção, decomposição, QAOA, reconstrução e validação.

### Linhas 82–95

Mede construção do QUBO global e particionamento em sub-QUBOs.

### Linhas 121–135

Mede extração de cada sub-QUBO e acumula o tempo de cada chamada QAOA.

### Linhas 152–161

Mede acumulação de assignments, atualização de `phi` e histórico.

### Linhas 185–208

Mede construção do QUBO de validação, reconstrução do resultado global e validação;
anexa o dicionário ao resultado global.

Impacto: instrumentação. O fluxo e as decisões da pipeline permanecem iguais.

## 9. `src/experiments/scenario_runner.py`: todas as adições explicadas

O diff contém 234 adições e 3 remoções.

### Linhas 167–204 — `build_synthetic_cluster_snapshot`

Constrói um `SystemSnapshot` determinístico a partir de processos descritos no TOML:
PID, comando, peso CPU, core atual, RSS, prioridade, I/O wait e classe de prioridade.
Valida que existe pelo menos um processo e define defaults explícitos.

Impacto: novo modo de workload usado por E1c.

### Linhas 206–211 — novo contrato de `build_run_inputs`

O retorno passa a incluir `RunConfig` e `ValidationConfig` como sétimo e oitavo elementos.

### Linhas 214–215

Lê as secções TOML `execution` e `validation`, que antes eram em grande parte
documentais.

### Linhas 238–240

Reconhece `workload.mode="synthetic_cluster"` e constrói o snapshot sintético.

### Linhas 255–260

Passa `init_strategy` e `random_seed` para `QAOAConfig`.

### Linhas 288–310 — opções de runtime

Constrói e devolve:

- modo de pipeline;
- flag de clustering do snapshot;
- flag de SA obrigatório;
- semente SA explícita, ou semente QAOA, ou 42;
- flag de visualização.

Impacto: liga os novos campos TOML ao comportamento real da aplicação.

### Linhas 318–371 — `probability_diagnostics`

Quando existem probabilidades:

1. ordena todos os estados por probabilidade;
2. reconstrói cada bitstring;
3. testa viabilidade one-hot;
4. calcula energia dos estados viáveis;
5. identifica massa viável/inválida, ranking do primeiro viável e ótimo, massa ótima e
   candidatos viáveis dentro de `top_k`.

Quando não existem probabilidades, devolve os mesmos campos com `None`.

Impacto: diagnóstico. O custo cresce como `2^n`, porque enumera todo o espaço de
estados. Pode aumentar significativamente o pós-processamento em QUBOs diretos maiores.

Nota: o teste de energia aqui usa `np.isclose(rtol=1e-9, atol=1e-9)`, não a igualdade
absoluta pura usada na análise final.

### Linhas 373–406 — `convergence_metrics`

Regista comprimento da curva, objetivo inicial/final e primeira iteração que fica a
uma tolerância `1e-4 * max(1, abs(final))` do valor final.

Impacto: métrica descritiva; não é critério de paragem do otimizador.

### Linhas 427–445 — métricas extra no output direto

Acrescenta SA, tempos BF/SA, acordo QAOA-SA, tempos por componente, metadados, parâmetros
do solver, diagnósticos de probabilidade e convergência.

### Linhas 452–460

Passa o resultado direto e o QUBO aos dois auxiliares de diagnóstico.

### Linhas 478–496 — métricas extra no output iterativo

Acrescenta os mesmos campos aplicáveis e agrega passos/iterações das curvas dos
sub-QUBOs. Diagnósticos completos de probabilidade não são calculados para o resultado
global iterativo.

### Linhas 569–583

Inclui os novos campos no output legível TXT.

### Linhas 676–684, 692 e 702

- Desempacota `run_cfg` e `validation_cfg`.
- Guarda ambos em `resolved_config`.
- Passa-os para `SchedulingEngine.run_job()`.

Impacto: fecha a ligação entre TOML, runner e aplicação.

## 10. Novo `src/solver/simulated_annealing_solver.py` — 197 linhas

Todas as linhas deste ficheiro são novas.

### Linhas 1–8

Imports de tempo, NumPy, contrato abstrato e tipos de dados.

### Linhas 9–27 — configuração

Define `AnnealingSolver`, valida `sweeps` e `restarts` e guarda temperaturas e semente.

### Linhas 29–127 — algoritmo principal

- Valida a estrutura one-hot do QUBO.
- Cria RNG reproduzível.
- Estima temperaturas quando omitidas.
- Executa vários restarts.
- Mantém sempre uma atribuição one-hot válida.
- Propõe mudanças de core por entidade.
- Calcula delta de energia sem recomputar todo o produto matricial.
- Aceita melhorias e movimentos piores segundo Metropolis.
- Guarda melhor solução e curva de convergência.
- Devolve `SolverResult` com parâmetros e tempo.

### Linhas 129–132

Estima temperatura inicial pela mediana dos coeficientes QUBO não nulos.

### Linhas 134–143

Implementa arrefecimento geométrico.

### Linhas 145–154

Cria atribuição round-robin no primeiro restart e aleatória nos restantes.

### Linhas 156–161

Converte atribuições em bitstring one-hot.

### Linhas 163–178

Calcula a variação de energia de mover uma entidade entre dois cores.

### Linhas 180–194

Descodifica e verifica a solução.

### Linha 197

Cria o alias `SimulatedAnnealingSolver`.

Impacto: novo baseline clássico; não faz parte do QAOA nem melhora diretamente a
solução QAOA.

## 11. Novo `src/experiments/sweep_runner.py` — 239 linhas

Todas as linhas deste ficheiro são novas.

### Linhas 1–22

Imports e reutilização das funções do runner principal.

### Linhas 25–39 — `set_dotted_path`

Define valores aninhados através de caminhos como `qaoa.layers` e valida conflitos.

### Linhas 42–109 — `expand_sweep`

- Suporta casos nomeados e eixos.
- Calcula produto cartesiano.
- Faz deep copy de cada variante.
- Injeta valores TOML.
- Cria `sweep_context`, IDs e nomes únicos.

### Linhas 112–151 — `run_sweep_scenario`

Carrega um cenário, expande variantes, aplica limite opcional, suporta dry-run e
executa todas as repetições através de `run_scenario_once`.

### Linhas 154–169 — CLI

Define seletores, `--all`, `--list`, `--dry-run`, `--max-variants` e diretórios.

### Linhas 172–235 — programa principal

Seleciona cenários, cria IDs e ficheiros agregados, executa sweeps e apresenta contagens
de sucesso/falha e caminhos de output.

### Linhas 238–239

Entry point do script.

Impacto: infraestrutura experimental. Corrige a limitação do runner antigo, que não
expandia `sweep.axes`.

## 12. Novo `src/experiments/analyze_results.py` — 183 linhas

Todas as linhas são novas e não participam na execução do scheduler.

- Linhas 1–29: imports, backend gráfico não interativo, caminhos e métricas.
- Linhas 30–39: leitura de JSONL.
- Linhas 40–48: acesso a caminhos aninhados.
- Linhas 49–83: normalização e flatten de cada resultado.
- Linhas 84–92: escrita CSV.
- Linhas 95–104: conversão numérica segura.
- Linhas 107–116: médias agrupadas.
- Linhas 119–146: gráficos genéricos por eixo de sweep.
- Linhas 149–156: argumentos CLI.
- Linhas 159–179: orquestra leitura, CSV e gráficos.
- Linhas 182–183: entry point.

Impacto: análise offline; não altera resultados brutos.

## 13. Novo `src/experiments/phase3_analysis.py` — 635 linhas

Todas as linhas são novas e dedicadas à análise offline da Fase 3.

- Linhas 1–52: imports, caminhos, inputs, sementes SA e cores.
- Linhas 53–163: leitura, normalização e conversão de registos em linhas tabulares.
- Linhas 164–231: CSV, campos de agrupamento e estatísticas agregadas.
- Linhas 232–269: reconstrução de workload e QUBO a partir do resultado.
- Linhas 270–335: execução dos baselines D1, incluindo SA.
- Linhas 336–345: agrupamento de registos.
- Linhas 346–384: gráfico tempo versus variáveis.
- Linhas 385–446: grelhas e heatmaps de parâmetros QAOA.
- Linhas 447–485: qualidade versus `qubit_max`.
- Linhas 486–512: pipeline direta versus iterativa.
- Linhas 513–552: QAOA versus brute force versus SA.
- Linhas 553–575: sensibilidade a `top_k`.
- Linhas 576–586: CLI.
- Linhas 587–631: geração de CSV, figuras e manifesto.
- Linhas 634–635: entry point.

Impacto: análise offline. Reconstrói problemas e executa baselines, mas não é chamado
pela aplicação normal.

## 14. Novo `src/experiments/phase4_analysis.py` — 675 linhas

Todas as linhas são novas e dedicadas à análise offline da Fase 4.

- Linhas 1–40: imports, caminhos, cores e componentes temporais.
- Linhas 41–97: leitura e utilitários estatísticos.
- Linhas 98–116: reconstrução de load imbalance quando ausente.
- Linhas 117–132: cálculo de pureza dos clusters.
- Linhas 133–240: transformação de cada run, incluindo optimalidade absoluta estrita,
  acordo com SA e separação do tempo de validação operacional.
- Linhas 241–294: CSV e agregação por configuração.
- Linhas 295–327: resumo temporal E4.
- Linhas 328–361: comparação histórica D2.
- Linhas 362–368: gravação de figuras.
- Linhas 369–397: gráfico E1 adversarial.
- Linhas 398–424: gráfico de clustering E1c.
- Linhas 425–455: heatmap E2.
- Linhas 456–490: tradeoff E2/decomposição.
- Linhas 491–537: generalização de `top_k` E3.
- Linhas 538–569: decomposição temporal E4.
- Linhas 570–577: CLI.
- Linhas 578–671: geração das tabelas, figuras, comparação D2 e manifesto.
- Linhas 674–675: entry point.

Impacto: análise offline. É neste ficheiro que a optimalidade final é recalculada por
`abs(E-E*) <= 1e-9`, independentemente do valor registado pelo validador.

## 15. Testes novos — 252 linhas

### `tests/test_simulated_annealing_solver.py` — 91 linhas

- Linhas 1–19: imports e configuração de paths.
- Linhas 20–43: fábricas de workloads e QUBO.
- Linhas 45–88: testa viabilidade, energia contra brute force, reprodutibilidade e
  integração do SA no validador.
- Linhas 90–91: entry point.

### `tests/test_phase4_experiment_support.py` — 161 linhas

- Linhas 1–24: imports e configuração.
- Linhas 25–50: produto cartesiano entre casos e eixos.
- Linhas 52–87: snapshot sintético e opções de runtime.
- Linhas 89–119: execução simultânea de brute force e SA.
- Linhas 121–157: tolerância mais estrita de optimalidade.
- Linhas 160–161: entry point.

Impacto: não altera produção; documenta comportamento esperado.

## 16. Cenários TOML novos, não contados como código Python

Foram adicionados cenários de investigação para profundidade, inicialização,
otimizador, penalização, mixer, dimensão, decomposição, `top_k`, clustering e Fase 4.
Os sete diretamente associados à Fase 4 são:

1. `phase4_e1a_dominant_process.toml`;
2. `phase4_e1b_near_equal_conflict.toml`;
3. `phase4_e1c_synthetic_clustering.toml`;
4. `phase4_e2_direct_budget.toml`;
5. `phase4_e2b_exact_direct_budget.toml`;
6. `phase4_e3_adversarial_top_k.toml`;
7. `phase4_e3_uniform_top_k.toml`.

Estes ficheiros não implementam algoritmos, mas escolhem parâmetros que mudam a
execução através do runner alterado.

## 17. Alterações que exigem menção explícita na dissertação

1. **Correção de `top_k`:** anteriormente era ignorado na seleção final.
2. **Inicialização aleatória:** as sementes 101–105 passaram a controlar gamma/beta.
3. **Pipeline forçada:** algumas experiências não seguem a decisão automática original.
4. **Clustering preset:** E1c usa um caminho que não existia originalmente.
5. **Descodificação one-hot:** o conceito de solução viável foi alinhado entre solvers.
6. **SA:** novo baseline clássico implementado localmente.
7. **Optimalidade:** o relatório usa uma reclassificação absoluta feita offline.
8. **Tempos:** os tempos operacionais excluem BF e SA através de pós-processamento.

## 18. Riscos e pontos que devem ser revistos

1. O working tree não tem um commit que identifique a versão experimental.
2. `np.isclose(rtol=1e-9, atol=1e-9)` no validador não é literalmente a regra absoluta
   `<=1e-9` usada na análise.
3. `probability_diagnostics` enumera `2^n` estados e pode tornar-se impraticável.
4. A medição de `instrumented_total_ms` soma componentes instrumentados, não é um tempo
   wall-clock independente.
5. Forçar a pipeline iterativa abaixo do limite muda a semântica da aplicação original.
6. O SA preserva one-hot por construção; a comparação com QAOA deve declarar esta
   vantagem estrutural de representação.
7. As alterações de algoritmo, instrumentação, resultados e documentação estão todas
   misturadas no mesmo working tree e deveriam ser separadas em commits auditáveis.

## 19. Como consultar as linhas literais

Este relatório explica todos os blocos adicionados. O patch literal completo dos
ficheiros existentes pode ser obtido sem alterar o repositório com:

```bash
git diff -- src/data_contracts.py \
  src/experiments/scenario_runner.py \
  src/main.py \
  src/pipeline/default_pipeline.py \
  src/pipeline/iterative_pipeline.py \
  src/solver/pennylane_solver.py \
  src/solver/solver_validator.py
```

Os ficheiros Python novos são integralmente linhas adicionadas e estão descritos nas
secções 10–15.
