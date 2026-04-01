import numpy as np
from itertools import product
from typing import Dict

from abstract.abstract import BaseBuilder
from data_contracts import QUBOInstance, SystemSnapshot

class TimeAssignmentBuilder(BaseBuilder):
    def __init__(self, penalty: float, core_assignments: Dict[int, int], time_slots: int):
        """
        penalty: QUBO penalty weight
        core_assignments: pid -> core_id (from a prior SolverResult.decoded_assignments)
        time_slots: number of time steps to schedule
        """
        self.P = penalty
        self.core_assignments = core_assignments
        self.T = time_slots

    def build(self, snapshot: SystemSnapshot) -> QUBOInstance:
        processes = snapshot.processes
        N = len(processes)
        T = self.T

        weights = [p.cpu_weight for p in processes]
        core_map = [self.core_assignments[p.pid] for p in processes]

        Q = np.zeros((N * T, N * T))

        for i1, t1 in product(range(N), range(T)):
            for i2, t2 in product(range(N), range(T)):
                idx1 = i1 * T + t1
                idx2 = i2 * T + t2

                if idx1 <= idx2:
                    if i1 == i2 and t1 != t2:
                        Q[idx1, idx2] = 2 * self.P
                    elif i1 != i2 and core_map[i1] == core_map[i2] and t1 == t2:
                        Q[idx1, idx2] = 2 * weights[i1] * weights[i2]
                    elif i1 == i2 and t1 == t2:
                        Q[idx1, idx2] = weights[i1] ** 2 - self.P

        variable_map = {
            i * T + t: (processes[i].pid, t)
            for i in range(N)
            for t in range(T)
        }

        return QUBOInstance(
            Q=Q,
            num_variables=N * T,
            variable_map=variable_map,
            num_entities=N,
            num_cores=T,  # in time context, "cores" dimension is time slots
            penalty_weight=self.P,
            iteration_index=0,
            source_snapshot_id=snapshot.snapshot_id,
        )