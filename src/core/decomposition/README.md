# decomposition

This module contains the logic for reducing large live workloads into smaller
QUBOs.

## AdaptiveCluster

`adaptive_cluster.py` clusters live `SystemSnapshot` processes into bundles.

The clustering path:

1. separates real-time priority processes from normal processes,
2. computes effective CPU demand,
3. builds a feature matrix from effective CPU and memory,
4. builds an RBF affinity matrix,
5. runs spectral clustering,
6. falls back to KMeans if spectral clustering fails,
7. splits bundles that are too heterogeneous or heavy.

Effective CPU demand:

```text
w_eff = cpu_weight * (1 - io_alpha * io_wait_ratio)
```

## SubQUBODecomposer

`subqubo_decomposer.py` partitions a `Workload` into groups and extracts local
sub-QUBOs from the global Q matrix.

Partition heuristics return **0-based positions** into `workload.entities`, not
`entity_id` values. The decomposer maps those positions back to
`WorkloadEntity` objects before extracting sub-QUBOs. This matters because preset
snapshots use IDs like `1000, 1001, ...` and live snapshots use OS PIDs.

Bias propagation accounts for assignments already fixed by previous sub-QUBOs:

```text
Q_diag += 2 * w_i * phi_k
```

where `phi_k` is accumulated load on core `k`.

## Heuristics

`subqubo_heuristics.py` defines:

- `WEIGHT_DESCENDING`: sort by CPU weight and chunk by capacity.
- `COUPLING_DESCENDING`: greedily group high-coupling/high-magnitude entities.
- `CORE_BALANCE`: dynamic placeholder, not fully implemented.
