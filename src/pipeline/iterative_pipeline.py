from typing import Dict, List

import numpy as np
from builder.builder_core import CoreAssignmentBuilder
from data_contracts import DecompositorConfig, QAOAConfig, QUBOConfig, SolverResult, Workload
from decomposition.subqubo_decomposer import SubQUBODecomposer
from decomposition.subqubo_heuristics import Heuristic
from solver.solver_validator import SolverValidator
from solver.pennylane_solver import PennylaneSolver
import time

class IterativePipeline:
    def __init__(self, builder: CoreAssignmentBuilder, solver: PennylaneSolver, 
                 solver_validator: SolverValidator, decomposer: SubQUBODecomposer):
        self.builder = builder 
        self.solver = solver 
        self.solver_validator = solver_validator
        self.decomposer = decomposer

    def run(self, filename, workload: Workload, qaoa_cfg: QAOAConfig, qubo_cfg: QUBOConfig, dec_cfg: DecompositorConfig):
        print(f"\n--- Iterative Run Started at {time.ctime()} ---")

        print("Building Q_global...")
        start_time = time.time()
        qubo_instance = self.builder.build(workload)
        print(f"QUBO Matrix completed in {time.time() - start_time:.4f}s")

        print("Partitioning in Sub-QUBOs...")
        Q_global = qubo_instance.Q
        groups = self.decomposer.partition(workload, Q_global, dec_cfg)
        
        print(f"{len(groups)} sub-QUBOs | "
            f"sizes: {[len(g) for g in groups]}")

        print("Starting Iterative Loop...")
        
        K = workload.num_cores
        phi = np.zeros(K)               # accumulated load per core
        final_assignments: Dict[int, int] = {}
        solver_results: List[SolverResult] = []
        phi_history: List[np.ndarray] = []
        L_avg = workload.total_weight / K

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
                # The solver already falls back to best infeasible bitstring.
                print(f"  WARNING: sub-QUBO {t} returned infeasible solution. "
                      f"Assignments: {result.decoded_assignments}")

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
        print(f"Entities assigned: {len(final_assignments)}/{len(workload.entities)}")

        missing = set(e.entity_id for e in workload.entities) - set(final_assignments)
        if missing:
            print(f"WARNING: Unassigned entities: {missing}")

        return final_assignments, solver_results, phi_history, qubo_instance