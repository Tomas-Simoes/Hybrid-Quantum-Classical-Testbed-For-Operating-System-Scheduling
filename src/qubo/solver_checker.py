import numpy as np
from itertools import product

class SolveChecker:
    def __init__(self, weights, core_map, time_slots, penalty=3.0):
        """
        weights: list of process weights
        core_map: list mapping process -> core
        time_slots: number of available time slots
        penalty: QUBO penalty for collisions or invalid assignment
        """
        self.weights = weights
        self.core_map = core_map
        self.N = len(weights)
        self.K = len(set(core_map))
        self.T = time_slots
        self.P = penalty

    def _energy(self, schedule):
        """
        schedule: np.array of shape (N, T), with 0/1 entries
        Returns the QUBO energy for this schedule (matches x^T Q x exactly)
        """
        energy = 0.0

        # Diagonal terms (-P per slot) + off-diagonal same-process pairs (+2P per pair)
        for i in range(self.N):
            a = int(np.sum(schedule[i]))
            energy += a * (self.weights[i]**2 - self.P)
            if a > 1:
                energy += a * (a - 1) * self.P  # 2P * C(a, 2)

        # Collision penalty: same core, same time slot
        for t in range(self.T):
            for c in set(self.core_map):
                processes_on_core = [i for i in range(self.N) if self.core_map[i] == c and schedule[i, t] == 1]
                if len(processes_on_core) > 1:
                    for i in processes_on_core:
                        for j in processes_on_core:
                            if i < j:
                                energy += 2 * self.weights[i] * self.weights[j]

        return energy

    def find_global_optimum(self):
        """
        Exhaustively enumerate all valid schedules
        Returns: (best_schedule, best_energy)
        """
        best_energy = float('inf')
        best_schedule = None

        # Generate all possible 0/1 combinations for N*T matrix
        for x_tuple in product([0, 1], repeat=self.N * self.T):
            schedule = np.array(x_tuple).reshape(self.N, self.T)
            # Only keep schedules with exactly 1 time slot per process
            if np.all(np.sum(schedule, axis=1) == 1):
                energy = self._energy(schedule)
                if energy < best_energy:
                    best_energy = energy
                    best_schedule = schedule.copy()
        return best_schedule, best_energy

    def check_solution(self, candidate_x):
        """
        candidate_x: 1D np.array of length N*T (your solver's solution)
        Returns: dict with validity and energy comparison
        """
        candidate_schedule = np.array(candidate_x).reshape(self.N, self.T)
        valid = True
        errors = []

        # Check one time slot per process
        row_sums = np.sum(candidate_schedule, axis=1)
        for i, s in enumerate(row_sums):
            if s != 1:
                valid = False
                errors.append(f"Process {i} assigned to {s} time slots")

        # Check core collisions
        for t in range(self.T):
            for c in set(self.core_map):
                processes_on_core = [i for i in range(self.N) if self.core_map[i] == c and candidate_schedule[i, t] == 1]
                if len(processes_on_core) > 1:
                    valid = False
                    errors.append(f"Core {c} collision at time {t}: {processes_on_core}")

        candidate_energy = self._energy(candidate_schedule)
        global_schedule, global_energy = self.find_global_optimum()
        is_optimal = candidate_energy == global_energy

        return {
            "valid": valid,
            "errors": errors,
            "candidate_energy": candidate_energy,
            "global_energy": global_energy,
            "is_optimal": is_optimal,
            "global_schedule": global_schedule,
            "candidate_schedule": candidate_schedule
        }

