# solver

This module contains QUBO solvers and validation helpers.

## PennylaneSolver

`pennylane_solver.py` is the main QAOA solver.

It:

- converts the QUBO matrix to an Ising Hamiltonian,
- initializes the QAOA circuit,
- optimizes parameters with Adam,
- records a convergence curve,
- samples final probabilities,
- inspects the top-k most probable bitstrings,
- returns the lowest-energy feasible bitstring found in top-k.

The solver tries `lightning.gpu` first in local/development runs. If GPU device
creation or any later QAOA runtime step fails, it retries the full solve on
`lightning.qubit`.

Production-like runtimes, including Render, skip the GPU probe and use
`lightning.qubit` directly. Set `QAOA_FORCE_CPU=true` to force that behavior in
any environment, or `QAOA_FORCE_CPU=false` to explicitly allow the GPU-first
path.

Initial QAOA parameters are configurable through `QAOAConfig`:

```python
init_gamma: float = 0.5
init_beta: float = 0.5
```

The Streamlit sweep can vary a shared initial value `theta0`, applied as
`init_gamma = init_beta = theta0`, so initialization sensitivity can be studied
alongside penalty and circuit depth.

If no top-k bitstring is feasible, the solver returns the best fallback candidate
with `is_feasible=False`. This is useful for diagnostics, but downstream
pipelines must treat it carefully.

## Mixers

`QAOAConfig` supports:

```python
mixer_type: Literal["xy", "x"] = "xy"
```

- `"x"`: standard X mixer with a Hadamard initialization over all qubits.
- `"xy"`: custom XY-style mixer initialized in a feasible one-hot state. The goal
  is to preserve one-hot feasibility while mixing possible core assignments for
  each process.

The XY mixer should be regression-tested before using it for strong experimental
claims about mixer performance.

## BruteForceSolver

`brute_force_solver.py` enumerates every bitstring and returns the global minimum.
It refuses QUBOs above 22 variables to avoid runaway runtimes.

## SolverValidator

`solver_validator.py` compares a candidate `SolverResult` against brute force.
It reports:

- candidate energy,
- global optimum energy,
- candidate/global assignments,
- feasibility errors,
- whether the candidate is optimal.

## Metrics

The project reports **optimality gap**, not confidence:

```text
optimality_gap = (qaoa_energy - feasible_optimum) / (abs(reference_energy - feasible_optimum) + eps)
```

`reference_energy` is the unconstrained brute-force minimum when available,
falling back to zero. An optimality gap of `0.0` means QAOA matched the
feasible brute-force optimum exactly.
Max output probability is a separate sampling concentration signal.

## Other Files

- `qiskit_solver.py`: experimental/placeholder Qiskit path.
- `qubo_solver.py`: older simple brute-force helper, not the main runtime path.
