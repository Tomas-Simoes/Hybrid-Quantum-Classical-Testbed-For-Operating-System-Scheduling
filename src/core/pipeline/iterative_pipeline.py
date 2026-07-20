from typing import Dict, List

import numpy as np
from builder.builder_core import CoreAssignmentBuilder
from data_contracts import DecompositorConfig, QAOAConfig, QUBOConfig, QUBOInstance, SolverResult, Workload
from decomposition.subqubo_decomposer import SubQUBODecomposer
from decomposition.subqubo_heuristics import Heuristic
from solver.solver_validator import SolverValidator
from solver.pennylane_solver import PennylaneSolver
import time


class InfeasibleSubQUBOError(RuntimeError):
    def __init__(
        self,
        subqubo_index: int,
        result: SolverResult,
        final_assignments: Dict[int, int],
        solver_results: List[SolverResult],
        phi_history: List[np.ndarray],
    ):
        self.subqubo_index = subqubo_index
        self.result = result
        self.final_assignments = dict(final_assignments)
        self.solver_results = list(solver_results)
        self.phi_history = [phi.copy() for phi in phi_history]
        super().__init__(
            f"Sub-QUBO {subqubo_index + 1} produced no feasible top-k assignment; "
            "stopping before updating final assignments or core-load state."
        )


class IterativePipeline:
    def __init__(self, builder: CoreAssignmentBuilder, solver: PennylaneSolver, 
                 solver_validator: SolverValidator, decomposer: SubQUBODecomposer):
        self.builder = builder 
        self.solver = solver 
        self.solver_validator = solver_validator
        self.decomposer = decomposer

    def _build_global_result(
        self,
        workload: Workload,
        qubo: QUBOInstance,
        final_assignments: Dict[int, int],
        solver_results: List[SolverResult],
    ) -> SolverResult:
        bitstring = np.zeros(qubo.num_variables, dtype=int)

        for var_idx, (entity_id, core) in qubo.variable_map.items():
            if final_assignments.get(entity_id) == core:
                bitstring[var_idx] = 1

        K = qubo.num_cores
        is_feasible = all(
            bitstring[i * K : (i + 1) * K].sum() == 1
            for i in range(qubo.num_entities)
        )
        global_energy = float(bitstring.T @ qubo.Q @ bitstring)

        return SolverResult(
            bitstring=bitstring,
            decoded_assignments=dict(final_assignments),
            energy=global_energy,
            is_feasible=is_feasible,
            solver_backend="iterative_qaoa_assembled",
            solve_time_ms=sum(r.solve_time_ms for r in solver_results),
            solver_params={"sub_qubos": len(solver_results)},
        )

    def run(self, filename, workload: Workload, qaoa_cfg: QAOAConfig, qubo_cfg: QUBOConfig, dec_cfg: DecompositorConfig):
        print(f"\n--- Iterative Run Started at {time.ctime()} ---")

        print("Building Q_global...")
        start_time = time.time()
        decomposition_qubo = self.builder.build(workload, include_fixed_bias=False)
        print(f"QUBO Matrix completed in {time.time() - start_time:.4f}s")

        print("Partitioning in Sub-QUBOs...")
        Q_global = decomposition_qubo.Q
        groups = self.decomposer.partition(workload, Q_global, dec_cfg)
        
        print(f"{len(groups)} sub-QUBOs | "
            f"sizes: {[len(g) for g in groups]}")

        print("Starting Iterative Loop...")
        
        K = workload.num_cores
        phi = workload.fixed_load_per_core
        final_assignments: Dict[int, int] = dict(workload.fixed_assignments)
        solver_results: List[SolverResult] = []
        phi_history: List[np.ndarray] = []
        L_avg = workload.total_weight / K

        if final_assignments:
            print(f"Fixed RT assignments: {final_assignments}")
            print(f"Initial phi from RT load: {np.round(phi, 4)}")

        for t, group in enumerate(groups):
            group_weight = sum(e.cpu_weight for e in group)
            print(f"\n[Sub-QUBO {t+1}/{len(groups)}] "
                  f"{len(group)} entities | weight={group_weight:.4f}")
            print(f"  phi before: {np.round(phi, 4)}")
            print(f"  residual capacity: "
                  f"{np.round(L_avg - phi, 4)}")

            # Extract sub-QUBO with bias propagation applied
            sub_qubo = self.decomposer.extract_subqubo(
                Q_global, group, workload, phi, t, qubo_cfg.penalty
            )

            # Solve the sub-QUBO
            t_solve = time.time()
            result = self.solver.solve(sub_qubo)
            print(f"  Solved in {result.solve_time_ms:.1f}ms | "
                  f"energy={result.energy:.6f} | "
                  f"feasible={result.is_feasible}")

            if not result.is_feasible:
                solver_results.append(result)
                print(f"  ERROR: sub-QUBO {t+1} returned no feasible top-k assignment. "
                      f"Best fallback: {result.decoded_assignments}")
                raise InfeasibleSubQUBOError(
                    subqubo_index=t,
                    result=result,
                    final_assignments=final_assignments,
                    solver_results=solver_results,
                    phi_history=phi_history,
                )

            # Accumulate assignments
            final_assignments.update(result.decoded_assignments)

            # update phi with this sub-QUBO's fixed assignments
            self.decomposer.update_phi(phi, group, result.decoded_assignments)
            phi_history.append(phi.copy())

            print(f"  phi after:  {np.round(phi, 4)}")
            solver_results.append(result)

        # Final summary 
        imbalance = phi.max() - phi.min()
        print(f"\n--- Run Complete ---")
        print(f"Final core loads: {np.round(phi, 4)}")
        print(f"Load imbalance:   {imbalance:.6f}  "
              f"(L_avg={L_avg:.4f})")
        movable_assigned = sum(
            1 for entity in workload.entities
            if entity.entity_id in final_assignments
        )
        print(
            f"Movable entities assigned: {movable_assigned}/{len(workload.entities)} | "
            f"Fixed RT assignments: {len(workload.fixed_assignments)}"
        )

        missing = set(e.entity_id for e in workload.entities) - set(final_assignments)
        if missing:
            print(f"WARNING: Unassigned entities: {missing}")

        print("Validating assembled global assignment...")
        validation_qubo = self.builder.build(workload, include_fixed_bias=True)
        global_result = self._build_global_result(
            workload, validation_qubo, final_assignments, solver_results
        )
        global_validation = self.solver_validator.validate(validation_qubo, global_result)
        global_validation["candidate_assignments"] = dict(final_assignments)

        print(f"Global assignment feasible: {global_validation['valid']}")
        print(f"Global candidate energy:   {global_validation['candidate_energy']:.6f}")
        if global_validation["global_energy"] is not None:
            print(f"Global optimum energy:     {global_validation['global_energy']:.6f}")
            print(f"Globally optimal:          {global_validation['is_optimal']}")
        else:
            print(f"Global brute-force skipped: {global_validation['brute_force_error']}")
        if global_validation["errors"]:
            print(f"Global validation errors:  {global_validation['errors']}")

        return (
            final_assignments,
            solver_results,
            phi_history,
            validation_qubo,
            global_result,
            global_validation,
        )
