from builder.qubo_core import CoreAssignmentBuilder
from data_contracts import QAOAConfig, QUBOConfig
from data_contracts import SystemSnapshot
from solver.solver_validator import SolverValidator
from solver.pennylane_solver import PennylaneSolver
from tracer.process_tracer import ProcessTracer
from visualization.visualization import Visualization
import time
from contextlib import redirect_stdout

class DefaultPipeline:
    def __init__(self, tracer: ProcessTracer, builder: CoreAssignmentBuilder, solver: PennylaneSolver, solver_validator: SolverValidator):
        self.tracer = tracer
        self.builder = builder 
        self.solver = solver 
        self.solver_validator = solver_validator
    
    def run(self, filename, snapshot: SystemSnapshot, qaoa_cfg: QAOAConfig, qubo_cfg: QUBOConfig):
        print(f"\n--- Run Started at {time.ctime()} ---")
        print("Snapshot ID:", snapshot.snapshot_id)
        print("Processes:", [(p.pid, p.cpu_weight) for p in snapshot.processes])
        print("Building QUBO...")
    
        start_time = time.time()
        core_qubo = self.builder.build(snapshot)
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
        for proc in snapshot.processes:
            core = core_result.decoded_assignments.get(proc.pid, "?")
            print(f"  PID {proc.pid} (w={proc.cpu_weight:.3f}) → core {core}")

        print(f"\nTotal solve time: {core_result.solve_time_ms:.3f}ms")

        Visualization(
            qubo=core_qubo,
            qaoa_cfg=qaoa_cfg,
            qubo_cfg=qubo_cfg,
            probs=core_result.probs,
            energies_over_time=core_result.convergence_curve,
            global_optimum=core_validation["global_energy"],
        )
        
        return core_result, core_validation
