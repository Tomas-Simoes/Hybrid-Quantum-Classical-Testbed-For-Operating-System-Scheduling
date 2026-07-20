# Mixer X versus XY under restricted candidate and optimizer budgets

Date: 2026-07-03

## Objective

Identify a reproducible regime where the non-conserving X mixer struggles more than
the one-hot-preserving XY mixer. The experiments compare paired configurations with the
same workload, QAOA depth, optimizer budget, initialization, penalty and candidate
budget. Exact brute force is used as the quality reference.

## New diagnostics

The experiment runner now records, for direct-pipeline statevector runs:

- total probability mass assigned to feasible one-hot states;
- total probability mass assigned to invalid states;
- whether the most probable state is feasible;
- the rank of the first feasible state;
- the rank and total probability mass of globally optimal states;
- the number of feasible states in the configured `top_k` pool.

These metrics distinguish structural constraint preservation from successful
post-selection.

## Experimental coverage

| Experiment | Purpose | Runs | Successful |
|---|---|---:|---:|
| MX1 | Initial `top_k={1,3}` screen, random initialization | 72 | 72 |
| MX2 | Full probability/rank boundary, `p={1,2,3}` | 120 | 120 |
| MX3 | Optimizer-step boundary with actual `top_k=1` | 240 | 240 |
| MX4 | Fixed-initialization verification | 72 | 72 |
| MX5 | Balanced comparison, `top_k={2,3,5,10}` | 160 | 160 |
| **Total** |  | **664** | **664** |

Two N=4, K=2 workloads were used:

- reference: `[0.15, 0.35, 0.25, 0.25]`, `P_safe=0.35`;
- dominant: `[0.70, 0.10, 0.10, 0.10]`, `P_safe=0.70`.

Random-initialization experiments use seeds 101--105. Penalties of
`0.25*P_safe` and `P_safe` were compared. All runs use exact statevector
probabilities and exact brute-force validation.

## Main boundary: optimizer steps with actual top_k=1

MX3 fixes `p=1`, uses random initialization and aggregates 20 X and 20 XY runs at
each optimizer budget (two workloads, two penalty levels and five seeds).

| Steps | X feasible | XY feasible | X optimal | XY optimal | Mean feasible mass X | Mean feasible mass XY |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 6/20  | 20/20 | 6/20  | 0/20 | 0.0722 | 1.0000 |
| 3  | 12/20 | 20/20 | 12/20 | 0/20 | 0.0974 | 1.0000 |
| 5  | 12/20 | 20/20 | 12/20 | 0/20 | 0.1233 | 1.0000 |
| 10 | 20/20 | 20/20 | 20/20 | 0/20 | 0.1738 | 1.0000 |
| 20 | 20/20 | 20/20 | 19/20 | 0/20 | 0.3498 | 1.0000 |
| 50 | 20/20 | 20/20 | 17/20 | 0/20 | 0.6453 | 1.0000 |

Across all 120 runs per mixer, X produced 90/120 feasible outputs, while XY produced
120/120. The empirical feasibility boundary for X in this experiment is 10 optimizer
steps: below it, the most probable X state is frequently invalid; from 10 steps onward,
all tested X outputs are feasible.

This is not a universal threshold. It is the observed boundary for these workloads,
seeds, learning rate and depth.

## Probability-space efficiency

MX2 uses full statevector diagnostics. A large internal candidate pool is used only to
retain the final result; the recorded ranks allow the outcome for any smaller `top_k`
to be inferred exactly from the same optimized distribution.

### At 0.25 P_safe

| Workload | Mixer | p=1 feasible mass | p=2 feasible mass | p=3 feasible mass |
|---|---|---:|---:|---:|
| Reference | X  | 0.3210 | 0.4901 | 0.7747 |
| Reference | XY | 1.0000 | 1.0000 | 1.0000 |
| Dominant  | X  | 0.7195 | 0.8469 | 0.9074 |
| Dominant  | XY | 1.0000 | 1.0000 | 1.0000 |

### At P_safe

| Workload | Mixer | p=1 feasible mass | p=2 feasible mass | p=3 feasible mass |
|---|---|---:|---:|---:|
| Reference | X  | 0.8624 | 0.9635 | 0.9789 |
| Reference | XY | 1.0000 | 1.0000 | 1.0000 |
| Dominant  | X  | 0.9805 | 0.9931 | 0.9947 |
| Dominant  | XY | 1.0000 | 1.0000 | 1.0000 |

The largest observed separation occurs for the reference workload at
`0.25*P_safe`, `p=1`: X assigns 67.9% of its probability to invalid states, whereas
XY assigns 0%.

## Optimality trade-off

The structural advantage of XY did not translate into better optimal-solution recovery
with a very small candidate pool:

- in MX2, the globally optimal state was ranked first by X in all 60 X runs;
- for XY it appeared at ranks 6--10 on the reference workload and ranks 2--5 on the
  dominant workload in most configurations;
- in MX3 (`p=1`, random initialization, actual `top_k=1`), X obtained 86/120 global
  optima, while XY obtained 0/120;
- in MX4 (fixed initialization), X obtained 32/36 global optima and XY obtained 2/36.

Thus, the experiments demonstrate XY superiority in feasibility robustness and sampling
efficiency, not overall optimality under `top_k=1`. The current XY implementation starts
from a single feasible assignment (all entities on core 0). With the tested shallow
circuits, the probability distribution often remains biased toward feasible but
suboptimal assignments.

## Timing

XY was consistently faster in the new experiments:

| Experiment | Mean X time | Mean XY time | XY reduction |
|---|---:|---:|---:|
| MX2 | 2352.3 ms | 1884.7 ms | 19.9% |
| MX3 | 246.7 ms | 207.4 ms | 15.9% |
| MX4 | 761.3 ms | 597.3 ms | 21.5% |

## Balanced practical top-k comparison

MX5 corrects the deliberately extreme `top_k=1` stress test. It uses
`top_k in {2,3,5,10}`, `p in {2,3}`, 100 optimizer steps, `P_safe` and five random
seeds. All 160 runs were feasible.

| Workload | p | top_k | X optimal | XY optimal |
|---|---:|---:|---:|---:|
| Reference | 2 | 2  | 5/5 | 0/5 |
| Reference | 2 | 3  | 5/5 | 0/5 |
| Reference | 2 | 5  | 5/5 | 0/5 |
| Reference | 2 | 10 | 5/5 | 5/5 |
| Reference | 3 | 2  | 5/5 | 1/5 |
| Reference | 3 | 3  | 5/5 | 2/5 |
| Reference | 3 | 5  | 5/5 | 2/5 |
| Reference | 3 | 10 | 5/5 | 5/5 |
| Dominant  | 2 | 2  | 5/5 | 2/5 |
| Dominant  | 2 | 3  | 5/5 | 2/5 |
| Dominant  | 2 | 5  | 5/5 | 5/5 |
| Dominant  | 2 | 10 | 5/5 | 5/5 |
| Dominant  | 3 | 2  | 5/5 | 2/5 |
| Dominant  | 3 | 3  | 5/5 | 2/5 |
| Dominant  | 3 | 5  | 5/5 | 5/5 |
| Dominant  | 3 | 10 | 5/5 | 5/5 |

Overall, X reached 80/80 global optima and XY reached 43/80. At `top_k=10`, both
mixers reached 20/20 optima. The mean time was 2787.0 ms for X and 2200.7 ms for XY,
making XY 21.0% faster. XY retained feasible probability mass 1.0; X ranged from
0.9635 to 0.9947 in this `P_safe` regime.

The practical comparison therefore does not support overall XY superiority. It supports
three narrower findings: XY guarantees feasibility, concentrates no probability on
invalid states and is faster; X ranks optimal states more aggressively in the current
implementation and performed better when `top_k<10`.

## Defensible conclusion

For restricted classical-optimizer budgets and `top_k=1`, X loses its feasibility
guarantee: it failed one-hot validation in 30/120 random-initialization runs, with all
failures occurring below 10 optimizer steps. XY remained feasible in 120/120 runs and
always kept 100% of its probability inside the feasible subspace. This is a stress-test
result, not the primary comparison.

With practical `top_k` values and 100 optimizer steps, X was always feasible and obtained
more global optima than XY. A thesis claim should therefore be limited to:

> XY is structurally and empirically more robust with respect to feasibility, requires
> no post-selection to recover a one-hot state, and was 15.9--21.5% faster in these
> tests. The present implementation does not show superior optimal-solution recovery:
> with practical candidate pools, X ranked the global optimum more effectively, while
> both mixers reached 100% optimality at `top_k=10`.

## Source files

- `src/experiments/scenarios/research_mixer_low_topk_screen.toml`
- `src/experiments/scenarios/research_mixer_probability_boundary.toml`
- `src/experiments/scenarios/research_mixer_optimizer_budget_boundary.toml`
- `src/experiments/scenarios/research_mixer_fixed_init_boundary.toml`
- `src/experiments/scenarios/research_mixer_balanced_topk.toml`
- `src/experiments/results/sweep_20260703_155754_16435bc7_results.jsonl`
- `src/experiments/results/sweep_20260703_160211_a5d61eec_results.jsonl`
- `src/experiments/results/sweep_20260703_160736_45385d30_results.jsonl`
- `src/experiments/results/sweep_20260703_160928_a766ff1b_results.jsonl`
- `src/experiments/results/sweep_20260703_161638_263cfb4f_results.jsonl`
