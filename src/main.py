# main.py
import streamlit as st
import numpy as np
from data_contracts import DecompositorConfig, IterativeSchedulingOutput, QAOAConfig, QUBOConfig, SystemSnapshot, TracerConfig
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
from visualizer.graph_visualizer import Visualizer
from visualizer.iterative_visualizer import IterativeVisualizer
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
            snapshot.num_cores = NUM_CORES # we overwrite system cores to NUM_CORES virtual cores
            
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

       
        # Component Initialization
        builder = CoreAssignmentBuilder(qubo_cfg)
        solver = PennylaneSolver(qaoa_cfg)
        validator = SolverValidator()
        subqubo_decomposer = SubQUBODecomposer()
        
        # Pipeline Decision
        # If we have more qubits than we can support, we have to go onto the IterativePipeline, instead of the DefaultPipeline
        qubit_count = len(workload.entities) * snapshot.num_cores
        if qubit_count <= decompositor_cfg.qubit_max:
            pipeline = DefaultPipeline(builder, solver, validator)
            qubo, result, validation = pipeline.run(
                workload=workload,
                qaoa_cfg=qaoa_cfg,
                qubo_cfg=qubo_cfg,
            )

            Visualizer(
                qubo=qubo,
                qaoa_cfg=qaoa_cfg,
                qubo_cfg=qubo_cfg,
                probs=result.probs,
                energies_over_time=result.convergence_curve,
                global_optimum=validation["global_energy"],
            )

            return SchedulingOutput(
                result=result,
                validation=validation,
                used_snapshot=snapshot,
                alpha=(result.energy - validation["global_energy"]) / abs(validation["global_energy"]),
                qubo_instance=qubo,
                qaoa_cfg=qaoa_cfg,
                qubo_cfg=qubo_cfg,
            )

        else:
            pipeline = IterativePipeline(builder, solver, validator, subqubo_decomposer)
            final_assignments, solver_results, phi_history, qubo_instance = pipeline.run(
                workload=workload,
                qaoa_cfg=qaoa_cfg,
                qubo_cfg=qubo_cfg,
                dec_cfg=decompositor_cfg,
                filename=None
            )

            viz = IterativeVisualizer(
                solver_results=solver_results,
                phi_history=phi_history,
                workload=workload,
                qubo_instance=qubo_instance,
                qaoa_cfg=qaoa_cfg,
                qubo_cfg=qubo_cfg,
            )
            viz.composite(save_path="results/iterative_run.png")

            return IterativeSchedulingOutput(
                final_assignments=final_assignments,
                solver_results=solver_results,
                phi_history=phi_history,
                used_workload=workload,
                qubo_instance=qubo_instance,
                qaoa_cfg=qaoa_cfg,
                qubo_cfg=qubo_cfg,
            )

if __name__ == "__main__":
    print("Running in CLI mode...")
    cli_mode = True
    
    NUM_CORES = 2 

    qaoa_cfg = QAOAConfig(layers=3, steps=10, learning_rate=0.05, top_k=10)
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

        # CLI Reporting
        if isinstance(output, IterativeSchedulingOutput):
            print(f"\n{'='*40}")
            print("ITERATIVE JOB COMPLETED")
            print(f"Sub-QUBOs: {output.num_sub_qubos} | Feasible: {output.num_feasible}/{output.num_sub_qubos}")
            print(f"Total solve time: {output.total_solve_time_ms:.1f}ms")
            print(f"Final core loads: {np.round(output.final_phi, 4)}")
            print(f"Load imbalance:   {output.load_imbalance:.6f}  (L_avg={output.L_avg:.4f})")
            print(f"Assignments: {output.final_assignments}")
            print(f"{'='*40}")
        elif isinstance(output, SchedulingOutput):
            print(f"\n{'='*40}")
            print("JOB COMPLETED SUCCESSFULLY")
            print(f"Energy: {output.result.energy:.4f}")
            print(f"Confidence (Alpha): {output.alpha:.4f}")
            print(f"Core Assignments: {output.result.assignments}")
            print(f"{'='*40}")

    except Exception as e:
        print(f"Critical Error during execution: {e}")