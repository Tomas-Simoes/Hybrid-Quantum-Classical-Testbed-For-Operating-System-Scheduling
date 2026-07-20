import numpy as np
from data_contracts import QUBOInstance, SolverResult
from solver.brute_force_solver import BruteForceSolver

class SolverValidator:
    def validate(self, qubo: QUBOInstance, result: SolverResult) -> dict:
        candidate_energy = float(result.bitstring.T @ qubo.Q @ result.bitstring)

        errors = []
        K = qubo.num_cores
        for i in range(qubo.num_entities):
            group = result.bitstring[i * K : (i + 1) * K]
            if group.sum() != 1:
                entity_id = qubo.variable_map[i * K][0]
                errors.append(f"Entity {entity_id} assigned to {int(group.sum())} options")

        global_energy = None
        global_assignments = {}
        unconstrained_energy = None
        unconstrained_assignments = {}
        unconstrained_is_feasible = None
        is_optimal = None
        brute_force_error = None

        try:
            global_optimum = BruteForceSolver().solve(qubo)
            global_energy = float(global_optimum.energy)
            global_assignments = global_optimum.decoded_assignments
            unconstrained_energy = global_optimum.solver_params.get("unconstrained_energy")
            unconstrained_assignments = global_optimum.solver_params.get(
                "unconstrained_assignments", {}
            )
            unconstrained_is_feasible = global_optimum.solver_params.get(
                "unconstrained_is_feasible"
            )
            is_optimal = len(errors) == 0 and np.isclose(candidate_energy, global_energy)
        except RuntimeError as e:
            brute_force_error = str(e)

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "candidate_energy": candidate_energy,
            "candidate_assignments": result.decoded_assignments,
            "global_energy": global_energy,
            "global_assignments": global_assignments,
            "unconstrained_energy": unconstrained_energy,
            "unconstrained_assignments": unconstrained_assignments,
            "unconstrained_is_feasible": unconstrained_is_feasible,
            "is_optimal": is_optimal,
            "brute_force_error": brute_force_error,
        }
