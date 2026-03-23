import numpy as np
from itertools import product

from .qubo_solver import QUBOSolver

class TimeAssignmentSolver(QUBOSolver):
    def __init__(self, weights, core_map, time_slots, penalty=3.0):
        """
        weights: list of process weights
        core_map: fixed process → core assignment
        time_slots: number of time steps to schedule
        penalty: QUBO penalty
        """
        self.weights = weights
        self.core_map = core_map
        self.N = len(weights)
        self.T = time_slots
        self.size = self.N * self.T
        self.P = penalty
        self.Q = self._build_matrix()
    
    def _idx(self, i, t):
        """Flatten (process, time) → Q index"""
        return i * self.T + t
    
    def _build_matrix(self):
        size = self.N * self.T
        Q = np.zeros((size, size))
        
        for i1, t1 in product(range(self.N), range(self.T)):
            for i2, t2 in product(range(self.N), range(self.T)):
                idx1, idx2 = self._idx(i1, t1), self._idx(i2, t2)
                if idx1 <= idx2:
                    # same process, different times → penalty
                    if i1 == i2 and t1 != t2:
                        Q[idx1, idx2] = 2 * self.P
                    # same core, same time → collision
                    elif i1 != i2 and self.core_map[i1] == self.core_map[i2] and t1 == t2:
                        Q[idx1, idx2] = 2 * self.weights[i1] * self.weights[i2]
                    # diagonal (reward for assignment)
                    elif i1 == i2 and t1 == t2:
                        Q[idx1, idx2] = self.weights[i1]**2 - self.P
        return Q

    def calculate_energy(self, x):
        return x.T @ self.Q @ x