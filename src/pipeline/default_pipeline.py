from builder.builder_core import CoreAssignmentBuilder
from data_contracts import QAOAConfig, QUBOConfig, Workload
from data_contracts import SystemSnapshot
from solver.solver_validator import SolverValidator
from solver.pennylane_solver import PennylaneSolver
from visualizer.graph_visualizer import Visualizer
import time
from contextlib import redirect_stdout

class DefaultPipeline:
    def __init__(self, builder: CoreAssignmentBuilder, solver: PennylaneSolver, solver_validator: SolverValidator):
        self.builder = builder 
        self.solver = solver 
        self.solver_validator = solver_validator
    
    def run(self, filename, workload: Workload, qaoa_cfg: QAOAConfig, qubo_cfg: QUBOConfig):
        print(f"\n--- Raw Run Started at {time.ctime()} ---")
        print("Workload ID:", workload.snapshot_id)
        print("Entities:", [(e.entity_id, e.cpu_weight) for e in workload.entities])

        print("Building QUBO...")
    
        start_time = time.time()
        core_qubo = self.builder.build(workload)
        print(f"QUBO Matrix completed in {time.time() - start_time}")

        start_time = time.time()
        core_result = self.solver.solve(core_qubo)
        print(f"QUBO solved in {time.time() - start_time}")

        print(f"\nCore Assignment - Energy: {core_result.energy:.4f}")
        print(f"Core Assignment - Feasible: {core_result.is_feasible}")
        print(f"Core Assignment - Assignments: {core_result.decoded_assignments}")
        print(f"Core Assignment - Solve time: {core_result.solve_time_ms:.1f}ms")

        if not core_result.is_feasible:
            print("WARNING: Infeasible core assignment — increase penalty P")

        # validate core assignment
        start_time = time.time()
        core_validation = self.solver_validator.validate(core_qubo, core_result)
        print(f"Validated in {time.time() - start_time}")

        print(f"Core Assignment - Optimal: {core_validation['is_optimal']}")
        print(f"Core Assignment - Global energy: {core_validation['global_energy']:.4f}")
        if core_validation["errors"]:
            print(f"Core Assignment - Errors: {core_validation['errors']}")
        
        print("\n--- Final Schedule ---")
        for entity in workload.processes:
            core = core_result.decoded_assignments.get(entity.entity_id, "?")
            print(f"  Entity {entity.entity_id} (w={entity.cpu_weight:.3f}) → core {core}")

        print(f"\nTotal solve time: {core_result.solve_time_ms:.3f}ms")

      
        
        return core_qubo, core_result, core_validation
