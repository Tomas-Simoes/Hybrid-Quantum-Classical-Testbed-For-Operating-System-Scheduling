# Tests

This folder contains project tests outside the production `src/` modules. The
goal is to keep the implementation code focused while giving builder, solver,
pipeline, and data-contract tests one shared home.

## `test_builder_core.py`

This suite verifies the process-to-core QUBO produced by
`CoreAssignmentBuilder`.

The tests build tiny static workloads with 2 or 3 processes and 2 cores, then
check that the generated `QUBOInstance` has the expected structure:

- `num_variables` equals `num_processes * num_cores`
- `Q` has shape `(num_variables, num_variables)`
- `variable_map` contains every `(process_id, core_id)` pair exactly once
- QUBO metadata is populated from the workload and config
- a known 2-process workload produces the expected Q matrix coefficients

The suite also checks assignment feasibility through the same decoding logic
used by the brute-force solver:

- a one-hot bitstring decodes into one core assignment per process
- a non-one-hot bitstring is detected as infeasible

Together, these tests protect the contract between the builder and downstream
solvers: every process/core decision variable must be present, the Q matrix must
match the expected encoding, and decoded assignments must respect the one-hot
constraint.

## `test_brute_force_solver.py`

This suite verifies that `BruteForceSolver` can be used as the ground-truth
oracle for small QUBOs.

The tests build a tiny real QUBO through `CoreAssignmentBuilder`, solve it with
`BruteForceSolver`, and check that:

- the solver returns a feasible result
- every process has exactly one active core variable
- decoded assignments agree with the active bits in the returned bitstring
- reported energy equals `bitstring.T @ Q @ bitstring`
- the returned feasible energy is the minimum across all one-hot assignments

The suite also checks the solver size guard. Brute force is exponential, so
`BruteForceSolver` refuses QUBOs above `BRUTE_FORCE_VAR_LIMIT` instead of trying
to enumerate an impractically large search space.

## Running Tests

From the project root, run every test suite:

```sh
make all-tests
```

Run one suite by selector:

```sh
make test test_brute_force
make test test_builder_core
```

The `test` target also accepts the full unittest module name, for example:

```sh
make test tests.test_brute_force_solver
```

You can still run discovery directly without Make:

```sh
.venv/bin/python -m unittest discover -s tests
```
