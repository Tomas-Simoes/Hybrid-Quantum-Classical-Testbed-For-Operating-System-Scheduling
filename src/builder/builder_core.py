import numpy as np
from itertools import product

from abstract.abstract import BaseBuilder
from data_contracts import QUBOConfig, QUBOInstance, Workload

class CoreAssignmentBuilder(BaseBuilder):
    def __init__(self, qubo_cfg: QUBOConfig):
        self.qubo_cfg = qubo_cfg
        self.P = qubo_cfg.penalty

    def build(self, workload: Workload) -> QUBOInstance:
        weights = [e.cpu_weight for e in workload.entities]

        N = len(weights)
        K = workload.num_cores
        Q = np.zeros((N * K, N * K))
        L_avg = sum(weights) / K
        
        if N == 0: raise ValueError("Workload must contain at least one entity.")
        if K <= 0: raise ValueError("Number of cores must be greater than zero.")
        if N < K: raise Warning("Number of entities must be greater than or equal to number of cores.")
        
        for i, j in product(range(N), range(K)):
            for k, l in product(range(N), range(K)):
                idx1, idx2 = i * K + j, k * K + l
                if i == k and j == l:
                    Q[idx1, idx2] = (weights[i] ** 2) - 2 * L_avg * weights[i] - self.P
                elif i != k and j == l:
                    Q[idx1, idx2] = weights[i] * weights[k]
                elif i == k and j != l:
                    Q[idx1, idx2] = self.P

        variable_map = {
            i * K + j: (workload.entities[i].entity_id, j)
            for i in range(N)
            for j in range(K)
        }

        return QUBOInstance(
            Q=Q,
            num_variables=N * K,
            variable_map=variable_map,
            num_entities=N,
            num_cores=K,
            penalty_weight=self.P,
            iteration_index=0,
            source_snapshot_id=workload.snapshot_id,
        )
