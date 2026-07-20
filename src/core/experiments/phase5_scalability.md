# Phase 5 — sweep de escalabilidade em N

O cenário `phase5_e5_scalability_n.toml` mede a pipeline iterativa completa com
`K=2`, `qubit_max=8`, mixer XY, `top_k=10`, profundidade 2 e 100 passos. Os
pesos são amostrados uniformemente em `(0, 1]`; a seed da instância e a seed do
QAOA são fluxos separados e ficam registadas nos dados brutos.

Validar o plano sem executar QAOA:

```bash
uv run python src/experiments/sweep_runner.py phase5_e5 --dry-run
```

Executar o sweep:

```bash
uv run python src/experiments/sweep_runner.py phase5_e5
```

O runner reutiliza automaticamente as 40 execuções do piloto (`N=10/15`) e
começa novas execuções em `N=20`. Percorre a grelha grosseira em ordem crescente,
acompanhando o desequilíbrio normalizado
`(max(loads)-min(loads))/mean(loads)`. Um sinal de transição ocorre quando a
média ultrapassa 1% ou quando sobe, acima da margem anti-ruído configurada, em
três valores consecutivos de N. Nesse caso, insere até dois pontos entre os
últimos pontos grosseiros e continua mais três pontos da grelha após o sinal. O
limite prático continua a ser 180 segundos de tempo médio por execução.

O gap energético antigo continua preservado para compatibilidade, mas não
controla a execução. A energia QUBO viável inclui um offset negativo que cresce
com N; dividi-lo por `abs(E_baseline)` fazia o gap parecer artificialmente menor
e aumentava a tolerância de equivalência. As métricas primárias são agora
calculadas diretamente das cargas:

- `load_balance_objective = sum((load_k - mean_load)^2)`;
- `normalized_load_imbalance = (max(loads)-min(loads))/mean(loads)`;
- `objective_regret = max(0, objective_pipeline-objective_baseline)`;
- `baseline_match_offset_free`, separado de `certified_optimal_offset_free`.

Cada execução continua a ser guardada em JSON/JSONL. Adicionalmente, o diretório
`phase5_e5_scalability_n_result` recebe dois CSVs por run:

- `<run_id>_raw.csv`: uma linha por seed, incluindo proveniência (piloto ou run
  atual), tempos, energias, método de referência, métricas sem offset,
  desequilíbrio e estabilidade do annealing;
- `<run_id>_summary.csv`: objetivo e regret médios, desequilíbrio normalizado,
  matches contra a referência, ótimo certificado e estatísticas de tempo por N.

Para `N <= 20`, a referência enumera apenas atribuições one-hot viáveis (`2^N`
com dois núcleos), com time-box de 60 segundos. Se exceder o limite ou a
time-box, e para `N > 20`, a referência passa a simulated annealing com 10
restarts. Nestes casos, `baseline_certified=false` e o resultado é reportado como
match heurístico, nunca como ótimo global certificado.

## Compatibilidade com D1

A alegação anterior de 100% até `N=10`, com `qubit_max=6`, foi conferida nos
JSON originais de D1. Embora o validador usado nessa execução tivesse o
`np.isclose` por omissão, as 35 diferenças absolutas de energia são no máximo
`7.11e-15`; portanto, todas continuam ótimas sob `atol=rtol=1e-9`.

Isso não torna D1 e E5 diretamente comparáveis. D1 usou pesos iguais gerados por
`uniform_total`, `top_k=32`, inicialização fixa e cinco repetições determinísticas;
E5 usa instâncias `uniform_random`, `top_k=10`, inicialização e seeds QAOA
distintas. A diferença observada no piloto não pode ser atribuída isoladamente a
`qubit_max=6` versus `qubit_max=8`; o relatório deve apresentar D1 como validação
num regime mais simples, não como controlo pareado de E5.
