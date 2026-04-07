# main.py
import streamlit as st
import numpy as np
from data_contracts import QAOAConfig, QUBOConfig, SystemSnapshot, TracerConfig
from builder.qubo_core import CoreAssignmentBuilder
from data_contracts import SchedulingOutput
from solver.pennylane_solver import PennylaneSolver
from solver.solver_validator import SolverValidator
from pipeline.default_pipeline import DefaultPipeline
from tracer.process_tracer import ProcessTracer
class SchedulingEngine:
    @staticmethod
    def run_job(qaoa_cfg: QAOAConfig, qubo_cfg: QUBOConfig, tracer_cfg: TracerConfig, preset_snapshot: SystemSnapshot | None) -> SchedulingOutput:
        snapshot = preset_snapshot
        
        if not snapshot: # then we are using live tracing
            print(f"\n{'-'*40}")
            print(f"INITIATING LIVE SYSTEM TRACER")
            proc_tracer = ProcessTracer(tracer_cfg)
            snapshot = proc_tracer.trace()
            print(snapshot)
            st.stop()

        print(f"\n{'-'*40}")
        print(f"INITIATING QAOA SCHEDULING JOB")
        print(f"{'-'*40}")
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
    # Your existing CLI testing logic remains here
    print("Running in CLI mode (not done yet)...")
    #engine = SchedulingEngine()
    #engine.run_job(...)