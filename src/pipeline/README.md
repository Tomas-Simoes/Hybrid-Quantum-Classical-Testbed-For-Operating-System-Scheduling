# pipeline

This module defines the execution flows that connect builders, solvers,
validators, and decomposition.

## DefaultPipeline

`default_pipeline.py` handles small workloads that fit within the configured
qubit limit.

Flow:

```text
Workload -> CoreAssignmentBuilder -> QUBOInstance -> PennylaneSolver
                                                -> SolverValidator
```

The default pipeline can return an infeasible `SolverResult`. That result is
still useful for debugging and visualization, but it should not be treated as a
valid schedule.

## IterativePipeline

`iterative_pipeline.py` handles workloads that exceed `DecompositorConfig.qubit_max`.

Flow:

```text
Global QUBO
  -> partition workload into groups
  -> extract sub-QUBO
  -> solve sub-QUBO
  -> update final assignments
  -> update phi core-load vector
  -> repeat
```

`phi` stores accumulated load per core. When a sub-QUBO is extracted, previously
fixed assignments are propagated as diagonal bias terms:

```text
Q_diag += 2 * entity_weight * phi_core
```

## Infeasible Sub-QUBO Guard

If a sub-QUBO returns `result.is_feasible == False`, the iterative pipeline raises
`InfeasibleSubQUBOError` before updating:

- `final_assignments`
- `phi`
- later sub-QUBOs

The exception carries partial diagnostic data:

- failed sub-QUBO index,
- failed `SolverResult`,
- assignments completed before failure,
- solver results collected so far,
- phi history collected so far.

This prevents a bad infeasible fallback from contaminating the rest of an
iterative experiment.

