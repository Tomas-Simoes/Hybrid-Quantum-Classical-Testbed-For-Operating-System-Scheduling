import numpy as np

from tracer.specialized_tracers import MemoryTracer, ProcessTracer
from qubo.qubo_core import CoreAssignmentSolver
from qubo.qubo_time import TimeAssignmentSolver
from qubo.solver_checker import SolveChecker

# simulate data
not_optimal_weights = [0.53, 1.55, 8.13, 0.16, 1.67]

simulated_weights = not_optimal_weights
mem_trace = MemoryTracer(simulated_weights)
proc_trace = ProcessTracer(simulated_weights)

print("Simulated Memory Usage:", mem_trace.get_memory_snapshots())
print("Simulated Processes:", proc_trace.get_process_list())

# core assignment solver
core_solver = CoreAssignmentSolver(proc_trace.weights, cores=2, penalty=150.0)
best_x_core, energy_core = core_solver.solve()

print(f"\nCore Assignment - Optimal Energy: {energy_core:.4f}")
print(f"Core Assignment - Optimal Assignment: {best_x_core}")

# build core map
N = len(proc_trace.weights)
K = 2
core_map = [np.argmax(best_x_core[i*K:(i+1)*K]) for i in range(N)]
for i in range(N):
    if best_x_core[i*K:(i+1)*K].sum() == 0:
        print(f"WARNING: Process {i} was not assigned to any core — increase penalty P")
print("Process → Core Map:", core_map)

# time assignment solver
time_solver = TimeAssignmentSolver(simulated_weights, core_map, time_slots=3, penalty=150.0)
best_x_time, energy_time = time_solver.solve()

print(f"\nTime Assignment - Optimal Energy: {energy_time:.4f}")
print(f"Time Assignment - Optimal Assignment: {best_x_time}")

# solveChecker to find global optimum and validate solver
checker = SolveChecker(proc_trace.weights, core_map, time_slots=3, penalty=150.0)
best_schedule, best_energy = checker.find_global_optimum()
print("\nGlobal optimal energy:", best_energy)
print("Global optimal schedule:\n", best_schedule)

# check solver solution
result = checker.check_solution(best_x_time)

print("\nSolver validation result:")
print("Valid:", result["valid"])
print("Errors:", result["errors"])
print("Candidate energy:", result["candidate_energy"])
print("Global optimal energy:", result["global_energy"])
print("Is candidate globally optimal?", result["is_optimal"])
print("Candidate schedule:\n", result["candidate_schedule"])