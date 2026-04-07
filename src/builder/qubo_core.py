import numpy as np
from itertools import product

from abstract.abstract import BaseBuilder
from data_contracts import QUBOConfig, QUBOInstance, SystemSnapshot

class CoreAssignmentBuilder(BaseBuilder):
    def __init__(self, qubo_cfg: QUBOConfig):
        self.qubo_cfg = qubo_cfg
        self.P = qubo_cfg.penalty

    def build(self, snapshot: SystemSnapshot) -> QUBOInstance:
        weights = [p.cpu_weight for p in snapshot.processes]
        N = len(weights)
        K = snapshot.num_cores

        Q = np.zeros((N * K, N * K))

        for i, j in product(range(N), range(K)):
            for k, l in product(range(N), range(K)):
                idx1, idx2 = i * K + j, k * K + l

                if idx1 <= idx2:
                    if i == k and j == l:
                        Q[idx1, idx2] = (weights[i] ** 2) - self.P
                    elif i != k and j == l:
                        Q[idx1, idx2] = 2 * weights[i] * weights[k]
                    elif i == k and j != l:
                        Q[idx1, idx2] = 2 * self.P

        variable_map = {
            i * K + j: (snapshot.processes[i].pid, j)
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
            source_snapshot_id=snapshot.snapshot_id,
        )