# main.py
import streamlit as st
import numpy as np
from data_contracts import DecompositorConfig, QAOAConfig, QUBOConfig, SystemSnapshot, TracerConfig
from builder.builder_core import CoreAssignmentBuilder
from data_contracts import SchedulingOutput

from decomposition.adaptive_cluster import AdaptiveCluster
from decomposition.subqubo_decomposer import SubQUBODecomposer
from decomposition.subqubo_heuristics import Heuristic

from pipeline.iterative_pipeline import IterativePipeline
from solver.pennylane_solver import PennylaneSolver
from solver.solver_validator import SolverValidator
from pipeline.default_pipeline import DefaultPipeline
from tracer.process_tracer import ProcessTracer
from visualizer.snapshot_visualization import SnapshotVisualizer
cli_mode = False
class SchedulingEngine:
    @staticmethod
    def run_job(
        qaoa_cfg: QAOAConfig, 
        qubo_cfg: QUBOConfig, 
        tracer_cfg: TracerConfig, 
        decompositor_cfg: DecompositorConfig, 
        preset_snapshot: SystemSnapshot | None
    ) -> SchedulingOutput:
        workload = None

        if not preset_snapshot: # then we are using live tracing
            print(f"INITIATING LIVE SYSTEM TRACER")
            proc_tracer = ProcessTracer(tracer_cfg)
            snapshot = proc_tracer.trace()
            snapshot.num_cores = NUM_CORES
            SnapshotVisualizer.print_system_snapshot(snapshot)
            print(f"{'-'*40}\n")

            print(f"INITIATING ADAPTIVE CLUSTERING")
            adaptive_cluster = AdaptiveCluster(decompositor_cfg)
            
            clustered_snapshot = adaptive_cluster.decompose(snapshot)
            workload = clustered_snapshot.to_workload()

            SnapshotVisualizer.print_clustered_snapshot(clustered_snapshot)
            print(f"{'-'*40}\n")
        else:
            snapshot = preset_snapshot
            workload = snapshot.to_workload()  

        print(f"INITIATING QAOA SCHEDULING JOB")
        print(f"WORKLOAD:   {len(workload.entities)} processes on {workload.num_cores} cores")
        print(f"WEIGHTS:    {sum(e.cpu_weight for e in workload.entities):.3f} total CPU load")
        print(f"QUBO CFG:   Penalty (P) = {qubo_cfg.penalty}")
        print(f"QAOA CFG:   Layers (p) = {qaoa_cfg.layers} | Steps = {qaoa_cfg.steps} | η = {qaoa_cfg.learning_rate}")
        print(f"{'-'*40}\n")

       
        # 1. Component Initialization
        builder = CoreAssignmentBuilder(qubo_cfg)
        solver = PennylaneSolver(qaoa_cfg)
        validator = SolverValidator()
        subqubo_decomposer = SubQUBODecomposer()
        
        # 2. Pipeline Decision
        # If we have more qubits than we can support, we have to go onto the IterativePipeline, instead of the DefaultPipeline
        qubit_count = len(workload.entities) * NUM_CORES
        if qubit_count <= decompositor_cfg.qubit_max:
            pipeline = DefaultPipeline(builder, solver, validator) 
            qubo, result, validation = pipeline.run(
                filename="decompositor_test",
                workload=workload, 
                qaoa_cfg=qaoa_cfg, 
                qubo_cfg=qubo_cfg
            )
        else:
            pipeline = IterativePipeline(builder, solver, validator, subqubo_decomposer)
    
            qubo, result, validation = pipeline.run(
                filename="decompositor_test",
                workload=workload, 
                qaoa_cfg=qaoa_cfg, 
                qubo_cfg=qubo_cfg,
                dec_cfg=decompositor_cfg,
        )

        return SchedulingOutput(
            result=result,
            validation=validation,
            used_snapshot=snapshot,
            alpha= (result.energy - validation['global_energy']) / abs(validation['global_energy']),
            qubo_instance=qubo, 
            qaoa_cfg=qaoa_cfg,
            qubo_cfg=qubo_cfg
        )

if __name__ == "__main__":
    print("Running in CLI mode...")
    cli_mode = True
    
    NUM_CORES = 2 

    qaoa_cfg = QAOAConfig(layers=3, steps=1, learning_rate=0.05, top_k=10, mixer_type="X")
    qubo_cfg = QUBOConfig(penalty=1, num_cores=NUM_CORES, snapshot=None, target_load=None)
    tracer_cfg = TracerConfig(min_rss=20, min_cpu=0.005, cpu_interval=1, num_samples=3, live_mode=False)
    decompositor_cfg = DecompositorConfig(qubit_max=12, num_cores=NUM_CORES, io_alpha=0.5, affinity_alpha=0.8, homogeneity_threshold=0.3, zscore_threshold=1.5, sorting_strategy=Heuristic.COUPLING_DESCENDING)

    try:
        output = SchedulingEngine.run_job(
            qaoa_cfg=qaoa_cfg,
            qubo_cfg=qubo_cfg,
            tracer_cfg=tracer_cfg,
            decompositor_cfg=decompositor_cfg,
            preset_snapshot=None
        )

        # 3. Basic CLI Reporting
        print(f"\n{'='*40}")
        print("JOB COMPLETED SUCCESSFULLY")
        print(f"Energy: {output.result.energy:.4f}")
        print(f"Confidence (Alpha): {output.alpha:.4f}")
        print(f"Core Assignments: {output.result.assignments}")
        print(f"{'='*40}")

    except Exception as e:
        print(f"Critical Error during execution: {e}")