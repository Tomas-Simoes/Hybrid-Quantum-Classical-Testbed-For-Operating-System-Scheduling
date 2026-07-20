# abstract

This module contains small base interfaces.

## BaseSolver

Solvers implement:

```python
solve(qubo: QUBOInstance) -> SolverResult
```

Main implementation today: `solver/pennylane_solver.py`.

## BaseBuilder

Builders implement:

```python
build(snapshot_or_workload) -> QUBOInstance
```

Main implementation today: `builder/builder_core.py`.

