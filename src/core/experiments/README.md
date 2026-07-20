# Investigative module

Research execution is isolated from the production scheduling engine.

`investigative_runtime.py` owns the experimental boundary:

- seeded and top-k QAOA variants;
- forced pipeline selection and optional preset clustering;
- exact feasible and simulated-annealing baselines;
- component timing, clustering diagnostics and derived metrics;
- the `InvestigativeEngine` entry point used by `scenario_runner.py`.

The production `main.py`, shared data contracts, pipelines and solver validator
remain identical to the `main` branch baseline. Experiment code must depend on
the production core; the production core must not import from `experiments`.
