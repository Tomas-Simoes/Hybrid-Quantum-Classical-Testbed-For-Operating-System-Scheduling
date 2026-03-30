import time
import numpy as np
from itertools import product

from abstract.abstract import BaseSolver
from contracts import QUBOInstance, SolverResult

BRUTE_FORCE_VAR_LIMIT = 22


class BruteForceSolver(BaseSolver):
    def solve(self, qubo: QUBOInstance) -> SolverResult:
        if qubo.num_variables > BRUTE_FORCE_VAR_LIMIT:
            raise RuntimeError(
                f"Brute-force refused: {qubo.num_variables} variables "
                f"exceeds {BRUTE_FORCE_VAR_LIMIT} limit."
            )

        start = time.time()
        best_energy = float("inf")
        best_x = None

        for x_tuple in product([0, 1], repeat=qubo.num_variables):
            x = np.array(x_tuple)
            energy = x.T @ qubo.Q @ x
            if energy < best_energy:
                best_energy = energy
                best_x = x

        elapsed_ms = (time.time() - start) * 1000

        decoded = {}
        for var_idx, val in enumerate(best_x):
            if val == 1:
                entity_id, option_id = qubo.variable_map[var_idx]
                decoded[entity_id] = option_id

        is_feasible = len(decoded) == qubo.num_entities

        return SolverResult(
            bitstring=best_x,
            decoded_assignments=decoded,
            energy=best_energy,
            is_feasible=is_feasible,
            solver_backend="brute_force",
            solve_time_ms=elapsed_ms,
            solver_params={"penalty": qubo.penalty_weight},
        )