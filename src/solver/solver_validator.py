import numpy as np
from contracts import QUBOInstance, SolverResult
from solver.brute_force_solver import BruteForceSolver

class SolverValidator:
    def validate(self, qubo: QUBOInstance, result: SolverResult) -> dict:
        global_optimum = BruteForceSolver().solve(qubo)
        candidate_energy = result.bitstring.T @ qubo.Q @ result.bitstring

        errors = []
        K = qubo.num_cores
        for i in range(qubo.num_entities):
            group = result.bitstring[i * K : (i + 1) * K]
            if group.sum() != 1:
                entity_id = qubo.variable_map[i * K][0]
                errors.append(f"Entity {entity_id} assigned to {int(group.sum())} options")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "candidate_energy": candidate_energy,
            "candidate_assignments": result.decoded_assignments,
            "global_energy": global_optimum.energy,
            "global_assignments": global_optimum.decoded_assignments,
            "is_optimal": np.isclose(candidate_energy, global_optimum.energy),
        }