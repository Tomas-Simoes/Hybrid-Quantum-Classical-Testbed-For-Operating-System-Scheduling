# main.py
import streamlit as st
import numpy as np
from data_contracts import DecompositorConfig, QAOAConfig, QUBOConfig, SystemSnapshot, TracerConfig
from builder.qubo_core import CoreAssignmentBuilder
from data_contracts import SchedulingOutput
from decompositor.decompositor import Decompositor
from solver.pennylane_solver import PennylaneSolver
from solver.solver_validator import SolverValidator
from pipeline.default_pipeline import DefaultPipeline
from tracer.process_tracer import ProcessTracer

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
        snapshot = preset_snapshot
        
        if not snapshot: # then we are using live tracing
            print(f"\n{'-'*40}")
            print(f"INITIATING LIVE SYSTEM TRACER")
            proc_tracer = ProcessTracer(tracer_cfg)
            snapshot = proc_tracer.trace()
            print(f"Raw Snapshot: {snapshot}")
            print(f"{'-'*40}\n")

            print(f"\n{'-'*40}")
            print(f"INITIATING ADAPTIVE CLUSTERING")
            decompositor = Decompositor(decompositor_cfg)
            decomposed_snapshot = decompositor.decompose(snapshot)
            print(f"Decomposed snapshot: {decomposed_snapshot}")
            print(f"{'-'*40}\n")

        print(f"\n{'-'*40}")
        print(f"INITIATING QAOA SCHEDULING JOB")
        print(f"WORKLOAD:   {len(snapshot.processes)} processes on {snapshot.num_cores} cores")
        print(f"WEIGHTS:    {sum(p.cpu_weight for p in snapshot.processes):.3f} total CPU load")
        print(f"QUBO CFG:   Penalty (P) = {qubo_cfg.penalty}")
        print(f"QAOA CFG:   Layers (p) = {qaoa_cfg.layers} | Steps = {qaoa_cfg.steps} | η = {qaoa_cfg.learning_rate}")
        print(f"{'-'*40}\n")

       
        # 1. Component Initialization
        builder = CoreAssignmentBuilder(qubo_cfg)
        solver = PennylaneSolver(qaoa_cfg)
        validator = SolverValidator()
        
        # 2. Pipeline Execution
        # Note: We pass None to tracer if we already have a hardcoded snapshot
        pipeline = DefaultPipeline(None, builder, solver, validator)
        
        result, validation = pipeline.run(
            filename="decompositor_test",
            snapshot=snapshot, 
            qaoa_cfg=qaoa_cfg, 
            qubo_cfg=qubo_cfg
        )

        # Calculate derived metrics
        alpha = result.energy / validation['global_energy'] if validation['global_energy'] != 0 else 0
        
        return SchedulingOutput(
            result=result,
            validation=validation,
            used_snapshot=snapshot,
            alpha=alpha,
            qubo_instance=builder.build(snapshot), 
            qaoa_cfg=qaoa_cfg,
            qubo_cfg=qubo_cfg
        )

if __name__ == "__main__":
    print("Running in CLI mode...")
    cli_mode = True
    
    qaoa_cfg = QAOAConfig(layers=3, steps=1, learning_rate=0.05, top_k=10)
    qubo_cfg = QUBOConfig(penalty=1, num_cores=2, snapshot=None)
    tracer_cfg = TracerConfig(min_rss=20, min_cpu=0.005, cpu_interval=1, num_samples=3, live_mode=False)
    decompositor_cfg = DecompositorConfig(num_bundles=8, io_alpha=0.5, affinity_alpha=0.8, affinity_sigma=1.0, homogeneity_threshold=0.3)

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