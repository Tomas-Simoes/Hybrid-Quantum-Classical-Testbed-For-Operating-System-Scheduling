# visualizer

This module contains Matplotlib and console visualizations.

All Matplotlib visualizers use the non-interactive `Agg` backend so they can run
inside Streamlit and scripts without blocking on GUI windows.

## graph_visualizer.py

`Visualizer` is used for default/single-QUBO runs.

Panels:

- energy landscape with feasible/infeasible states,
- QUBO matrix heatmap,
- QAOA convergence curve,
- top-k output probabilities,
- metadata panel,
- parameter sweep plot.

## iterative_visualizer.py

`IterativeVisualizer` is used for decomposed workloads.

Panels:

- final load balance,
- phi/core-load evolution,
- global Q matrix,
- convergence grid per sub-QUBO,
- probability grid per sub-QUBO.

## snapshot_visualization.py

`SnapshotVisualizer` prints console tables for:

- raw live system snapshots,
- clustered/bundled snapshots.

