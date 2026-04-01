from dataclasses import asdict
from matplotlib import pyplot as plt
import numpy as np
from config_contracts import QAOAConfig, QUBOConfig
from data_contracts import SystemSnapshot, ProcessInfo
from builder.qubo_core import CoreAssignmentBuilder
from solver.pennylane_solver import PennylaneSolver
from solver.solver_validator import SolverValidator
from tracer.process_tracer import ProcessTracer
from pipeline.default_pipeline import DefaultPipeline
from visualization.visualization import Visualization

# config
assymetric_weights = [0.2, 0.4, 0.1, 0.2, 0.99]
equal_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
non_squared_weights = [0.2, 0.2, 0.2]
three_cores = [0.3, 0.5, 0.2, 0.8]
seven_proc = [0.029, 0.058, 0.048, 0.116, 0.029, 0.048, 0.039]
five_proc = [0.29, 0.58, 0.48, 0.116, 0.39]
simulated_weights = five_proc

qubo_cfg = QUBOConfig(penalty=1.8, num_cores=2, snapshot=None)
qaoa_cfg = QAOAConfig(layers=3, steps=200, optimizer_step=0.01);

min_rss_mb = 10.0
cpu_internal = 1.0  # time between 

# build snapshot
tracer = ProcessTracer(min_rss_mb=70.0, min_cpu_weight=0.02, cpu_interval=1.0)
system_snapshot = tracer.trace(num_samples=3)
system_snapshot.num_cores = qubo_cfg.num_cores; 

hardcoded_snapshot = SystemSnapshot(
    timestamp=1719500000.0,
    num_cores=qubo_cfg.num_cores,
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

penalties = np.around(np.arange(1, 3, 0.1), decimals=1)
penalty_results = []

for p in penalties:
    qubo_cfg.penalty = p
    qubo_cfg.snapshot = hardcoded_snapshot

    builder = CoreAssignmentBuilder(penalty=p)
    solver = PennylaneSolver(qaoa_cfg.layers, qaoa_cfg.steps, qaoa_cfg.optimizer_step)
    solver_validator = SolverValidator()

    pipeline = DefaultPipeline(tracer, builder, solver, solver_validator)
    result, validation = pipeline.run( filename=f"dash_pen_{p}", snapshot=qubo_cfg.snapshot, qaoa_cfg=qaoa_cfg, qubo_cfg=qubo_cfg)

    alpha = result.energy / validation['global_energy'] if validation['global_energy'] != 0 else 0

    penalty_results.append({
        "p": p,
        "alpha": alpha,
        "max_p": np.max(result.probs),
        "feasible": result.is_feasible
    })

Visualization._plot_run_results(penalty_results)
plt.show() 