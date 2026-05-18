# src

This folder contains the runtime code for the scheduling testbed.

## Entry Points

- `main.py`: defines `SchedulingEngine` and the CLI experiment.
- `app.py`: Streamlit interface for single runs and sweeps.
- `data_contracts.py`: dataclasses and config objects shared across modules.

## SchedulingEngine

`SchedulingEngine.run_job(...)` is the main orchestration API. It accepts:

- `QAOAConfig`
- `QUBOConfig`
- `TracerConfig`
- `DecompositorConfig`
- optional `preset_snapshot`

If a preset snapshot is provided, it is converted directly to a workload. If no
preset snapshot is provided, the engine traces the live system, optionally
clusters the process list, and then schedules the resulting workload.

The engine chooses between:

- `DefaultPipeline` when `num_entities * num_cores <= qubit_max`
- `IterativePipeline` when the workload exceeds the configured qubit limit

## Preset Snapshots

Use `SchedulingEngine.build_preset_snapshot(weights, num_cores)` to build a
synthetic `SystemSnapshot` from a list of process weights.

```python
snapshot = SchedulingEngine.build_preset_snapshot(
    weights=[0.4, 0.3, 0.2, 0.1],
    num_cores=2,
)
```

## Core Contracts

Important dataclasses in `data_contracts.py`:

- `ProcessInfo`: raw traced or preset process information
- `SystemSnapshot`: process list plus machine/core metadata
- `Workload` / `WorkloadEntity`: scheduling abstraction used by builders
- `QUBOInstance`: Q matrix and variable metadata
- `SolverResult`: solver output, probabilities, convergence, feasibility
- `SchedulingOutput`: default/single-QUBO output
- `IterativeSchedulingOutput`: decomposed workload output
- `QAOAConfig`, `QUBOConfig`, `TracerConfig`, `DecompositorConfig`

`QAOAConfig` includes QAOA initialization controls:

- `init_gamma`
- `init_beta`

These default to `0.5` and are used to initialize every layer's gamma/beta
parameter before Adam optimization.

## Project Map

```text
src/
  abstract/        Base interfaces
  builder/         QUBO builders
  decomposition/   Clustering, heuristics, sub-QUBO extraction
  pipeline/        Default and iterative scheduling flows
  solver/          QAOA, brute force, validation helpers
  tracer/          Live system process tracing
  visualizer/      Matplotlib and console visualizations
  app.py           Streamlit UI
  main.py          SchedulingEngine and CLI
  data_contracts.py
```

## Notes

- The field `SchedulingOutput.alpha` is currently used as an optimality gap for
  compatibility with earlier code.
- `data_contracts.round_trip_test()` contains older references and should be
  refreshed before relying on it.
- `ProcessInfo.to_dict()` / `from_dict()` may need updates to include all current
  fields.
