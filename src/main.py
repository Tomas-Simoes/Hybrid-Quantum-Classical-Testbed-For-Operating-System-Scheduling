from contracts import SystemSnapshot, ProcessInfo
from builder.qubo_core import CoreAssignmentBuilder
from solver.brute_force_solver import BruteForceSolver
from solver.solver_validator import SolverValidator
from tracer.process_tracer import ProcessTracer

# config
assymetric_weights = [0.2, 0.4, 0.1, 0.2, 0.99]
equal_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
non_squared_weights = [0.2, 0.2, 0.2]
three_cores = [0.3, 0.5, 0.2, 0.8]
simulated_weights = assymetric_weights

penalty = 9
num_cores = 2
time_slots = 1

min_rss_mb = 10.0
cpu_internal = 1.0  # time between 

# modules
tracer = ProcessTracer(min_rss_mb=70.0, min_cpu_weight=0.02, cpu_interval=1.0)
builder = CoreAssignmentBuilder(penalty=penalty)
solver = BruteForceSolver()
solver_validator = SolverValidator()

# build snapshot
system_snapshot = tracer.trace(num_samples=3)
system_snapshot.num_cores = num_cores; # hardcoded value for now

hardcoded_snapshot = SystemSnapshot(
    timestamp=1719500000.0,
    num_cores=num_cores,
    processes=[
        ProcessInfo(
            pid=1000 + i,
            command=f"proc_{i}",
            cpu_weight=w,
            current_core=0,
            rss_mb=w * 1024,
            priority=20,
        )
        for i, w in enumerate(simulated_weights)
    ],
)

snapshot = system_snapshot

print("Snapshot ID:", snapshot.snapshot_id)
print("Processes:", [(p.pid, p.cpu_weight) for p in snapshot.processes])

# --- Core Assignment ---
core_qubo = builder.build(snapshot)
core_result = solver.solve(core_qubo)

print(f"\nCore Assignment - Energy: {core_result.energy:.4f}")
print(f"Core Assignment - Feasible: {core_result.is_feasible}")
print(f"Core Assignment - Assignments: {core_result.decoded_assignments}")
print(f"Core Assignment - Solve time: {core_result.solve_time_ms:.1f}ms")

if not core_result.is_feasible:
    print("WARNING: Infeasible core assignment — increase penalty P")

# validate core assignment
core_validation = solver_validator.validate(core_qubo, core_result)

print(f"Core Assignment - Optimal: {core_validation['is_optimal']}")
print(f"Core Assignment - Global energy: {core_validation['global_energy']:.4f}")
if core_validation["errors"]:
    print(f"Core Assignment - Errors: {core_validation['errors']}")
print("\n--- Final Schedule ---")
for proc in snapshot.processes:
    core = core_result.decoded_assignments.get(proc.pid, "?")
    print(f"  PID {proc.pid} (w={proc.cpu_weight:.3f}) → core {core}")

print(f"\nTotal solve time: {core_result.solve_time_ms:.3f}ms")