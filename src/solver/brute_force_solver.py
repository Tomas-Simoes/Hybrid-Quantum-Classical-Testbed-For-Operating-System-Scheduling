import time
import numpy as np
from itertools import product

from abstract.abstract import BaseSolver
from data_contracts import QUBOInstance, SolverResult

BRUTE_FORCE_VAR_LIMIT = 22


class BruteForceSolver(BaseSolver):
    @staticmethod
    def _decode_assignments(bitstring: np.ndarray, qubo: QUBOInstance) -> tuple[dict, bool]:
        decoded = {}
        K = qubo.num_cores

        for i in range(qubo.num_entities):
            group = bitstring[i * K : (i + 1) * K]
            if group.sum() != 1:
                continue

            core_offset = int(np.argmax(group))
            entity_id, core_id = qubo.variable_map[i * K + core_offset]
            decoded[entity_id] = core_id

        is_feasible = len(decoded) == qubo.num_entities
        return decoded, is_feasible

    def solve(self, qubo: QUBOInstance) -> SolverResult:
        if qubo.num_variables > BRUTE_FORCE_VAR_LIMIT:
            raise RuntimeError(
                f"Brute-force refused: {qubo.num_variables} variables "
                f"exceeds {BRUTE_FORCE_VAR_LIMIT} limit."
            )

        start = time.time()
        best_unconstrained_energy = float("inf")
        best_unconstrained_x = None
        best_unconstrained_decoded = {}
        best_unconstrained_feasible = False

        best_feasible_energy = float("inf")
        best_feasible_x = None
        best_feasible_decoded = {}

        for x_tuple in product([0, 1], repeat=qubo.num_variables):
            x = np.array(x_tuple)
            energy = float(x.T @ qubo.Q @ x)
            decoded, is_feasible = self._decode_assignments(x, qubo)

            if energy < best_unconstrained_energy:
                best_unconstrained_energy = energy
                best_unconstrained_x = x
                best_unconstrained_decoded = decoded
                best_unconstrained_feasible = is_feasible

            if is_feasible and energy < best_feasible_energy:
                best_feasible_energy = energy
                best_feasible_x = x
                best_feasible_decoded = decoded

        elapsed_ms = (time.time() - start) * 1000

        found_feasible = best_feasible_x is not None
        if not found_feasible:
            best_feasible_x = best_unconstrained_x
            best_feasible_energy = best_unconstrained_energy
            best_feasible_decoded = best_unconstrained_decoded

        return SolverResult(
            bitstring=best_feasible_x,
            decoded_assignments=best_feasible_decoded,
            energy=best_feasible_energy,
            is_feasible=found_feasible,
            solver_backend="brute_force",
            solve_time_ms=elapsed_ms,
            solver_params={
                "penalty": qubo.penalty_weight,
                "unconstrained_energy": best_unconstrained_energy,
                "unconstrained_bitstring": best_unconstrained_x.tolist(),
                "unconstrained_assignments": best_unconstrained_decoded,
                "unconstrained_is_feasible": best_unconstrained_feasible,
            },
        )
