import numpy as np
from itertools import product

from .qubo_solver import QUBOSolver

class CoreAssignmentSolver(QUBOSolver):
    def __init__(self, weights, cores, penalty):
        """
        weights: list of process weights
        cores: num of cores
        penalty: QUBO penalty
        """
        self.weights = weights
        self.N = len(weights)
        self.K = cores
        self.size = self.N * self.K
        self.P = penalty
        self.Q = self._build_matrix()

    def _build_matrix(self):
        """Internal method to construct the Upper Triangular Q matrix."""
        Q = np.zeros((self.N * self.K, self.N * self.K))
        
        for i, j in product(range(self.N), range(self.K)):
            for k, l in product(range(self.N), range(self.K)):
                idx1, idx2 = i * self.K + j, k * self.K + l

                # for only upper triangle
                if idx1 <= idx2:
                    # diagonal terms (same process, same core)
                    if i == k and j == l: 
                        Q[idx1, idx2] = (self.weights[i] ** 2) - self.P

                    # crossterms (different processes, same cores)
                    elif i != k and j == l: # Collision
                        Q[idx1, idx2] = 2 * self.weights[i] * self.weights[k]

                    # +2P penalty (same process, different cores)
                    elif i == k and j != l: # Wall
                        Q[idx1, idx2] = 2 * self.P
        return Q

    def calculate_energy(self, x):
        """Calculates E = x.T @ Q @ x"""
        return x.T @ self.Q @ x