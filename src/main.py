# main.py
import logging
import time
import uuid
import streamlit as st
import numpy as np
from data_contracts import DecompositorConfig, IterativeSchedulingOutput, ProcessInfo, QAOAConfig, QUBOConfig, QUBOInstance, SystemSnapshot, TracerConfig
from builder.builder_core import CoreAssignmentBuilder
from data_contracts import SchedulingOutput

from decomposition.adaptive_cluster import AdaptiveCluster
from decomposition.subqubo_decomposer import SubQUBODecomposer
from decomposition.subqubo_heuristics import Heuristic

from pipeline.iterative_pipeline import InfeasibleSubQUBOError, IterativePipeline
from solver.pennylane_solver import PennylaneSolver
from solver.solver_validator import SolverValidator
from pipeline.default_pipeline import DefaultPipeline
from tracer.process_tracer import ProcessTracer
from visualizer.graph_visualizer import Visualizer
from visualizer.iterative_visualizer import IterativeVisualizer
from visualizer.snapshot_visualization import SnapshotVisualizer

cli_mode = False
logger = logging.getLogger("SchedulingEngine")

class SchedulingEngine:
    @staticmethod
    def run_job(
        qaoa_cfg: QAOAConfig, 
        qubo_cfg: QUBOConfig, 
        tracer_cfg: TracerConfig, 
        decompositor_cfg: DecompositorConfig, 
        preset_snapshot: SystemSnapshot | None
    ) -> SchedulingOutput:
        run_id = str(uuid.uuid4())
        workload = None
        logger.info(
            "run_start run_id=%s live_mode=%s penalty=%s qaoa_backend=%s mixer=%s layers=%s steps=%s",
            run_id,
            tracer_cfg.live_mode,
            qubo_cfg.penalty,
            "pennylane",
            qaoa_cfg.mixer_type,
            qaoa_cfg.layers,
            qaoa_cfg.steps,
        )

        if not preset_snapshot: # then we are using live tracing
            print(f"INITIATING LIVE SYSTEM TRACER")
            proc_tracer = ProcessTracer(tracer_cfg)
            
            snapshot = proc_tracer.trace()
            snapshot.num_cores = qubo_cfg.num_cores # overwrite system cores with configured virtual cores
            
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
        print(
            f"QAOA CFG:   Layers (p) = {qaoa_cfg.layers} | Steps = {qaoa_cfg.steps} | "
            f"η = {qaoa_cfg.learning_rate} | Mixer = {qaoa_cfg.mixer_type} | "
            f"Init γ/β = {qaoa_cfg.init_gamma}/{qaoa_cfg.init_beta}"
        )
        print(f"{'-'*40}\n")

        if not workload.entities:
            fixed_phi = workload.fixed_load_per_core
            print("No movable workload entities; returning fixed RT assignments only.")
            logger.info(
                "run_complete run_id=%s pipeline=none reason=no_movable_entities fixed_assignments=%s",
                run_id,
                len(workload.fixed_assignments),
            )
            return IterativeSchedulingOutput(
                final_assignments=dict(workload.fixed_assignments),
                solver_results=[],
                phi_history=[fixed_phi],
                used_workload=workload,
                qubo_instance=QUBOInstance(
                    Q=np.zeros((0, 0)),
                    num_variables=0,
                    variable_map={},
                    num_entities=0,
                    num_cores=workload.num_cores,
                    penalty_weight=qubo_cfg.penalty,
                    iteration_index=0,
                    source_snapshot_id=workload.snapshot_id,
                ),
                qaoa_cfg=qaoa_cfg,
                qubo_cfg=qubo_cfg,
            )

       
        # Component Initialization
        builder = CoreAssignmentBuilder(qubo_cfg)
        solver = PennylaneSolver(qaoa_cfg)
        validator = SolverValidator()
        subqubo_decomposer = SubQUBODecomposer()
        
        # Pipeline Decision
        # If we have more qubits than we can support, we have to go onto the IterativePipeline, instead of the DefaultPipeline
        qubit_count = len(workload.entities) * workload.num_cores
        if qubit_count <= decompositor_cfg.qubit_max:
            logger.info(
                "pipeline_selected run_id=%s pipeline=default qubits=%s qubit_max=%s entities=%s cores=%s",
                run_id,
                qubit_count,
                decompositor_cfg.qubit_max,
                len(workload.entities),
                workload.num_cores,
            )
            pipeline = DefaultPipeline(builder, solver, validator)
            try:
                qubo, result, validation = pipeline.run(
                    workload=workload,
                    qaoa_cfg=qaoa_cfg,
                    qubo_cfg=qubo_cfg,
                )
            except Exception:
                logger.exception("run_failed run_id=%s pipeline=default", run_id)
                raise

            Visualizer(
                qubo=qubo,
                qaoa_cfg=qaoa_cfg,
                qubo_cfg=qubo_cfg,
                probs=result.probs,
                energies_over_time=result.convergence_curve,
                global_optimum=(
                    validation["global_energy"]
                    if validation["global_energy"] is not None
                    else result.energy
                ),
            )

            global_energy = validation.get("global_energy")
            reference_energy = validation.get("unconstrained_energy")
            alpha = (
                SchedulingEngine.optimality_gap(
                    result.energy,
                    global_energy,
                    reference_energy,
                )
                if global_energy is not None
                else float("nan")
            )
            logger.info(
                "run_complete run_id=%s pipeline=default backend=%s feasible=%s optimal=%s energy=%s global_energy=%s alpha=%s",
                run_id,
                result.solver_backend,
                result.is_feasible,
                validation.get("is_optimal"),
                result.energy,
                global_energy,
                alpha,
            )

            return SchedulingOutput(
                result=result,
                validation=validation,
                used_snapshot=snapshot,
                alpha=alpha,
                qubo_instance=qubo,
                qaoa_cfg=qaoa_cfg,
                qubo_cfg=qubo_cfg,
            )

        else:
            logger.info(
                "pipeline_selected run_id=%s pipeline=iterative qubits=%s qubit_max=%s entities=%s cores=%s",
                run_id,
                qubit_count,
                decompositor_cfg.qubit_max,
                len(workload.entities),
                workload.num_cores,
            )
            pipeline = IterativePipeline(builder, solver, validator, subqubo_decomposer)
            try:
                (
                    final_assignments,
                    solver_results,
                    phi_history,
                    qubo_instance,
                    global_result,
                    validation,
                ) = pipeline.run(
                    workload=workload,
                    qaoa_cfg=qaoa_cfg,
                    qubo_cfg=qubo_cfg,
                    dec_cfg=decompositor_cfg,
                    filename=None
                )
            except Exception:
                logger.exception("run_failed run_id=%s pipeline=iterative", run_id)
                raise

            global_energy = validation.get("global_energy")
            reference_energy = validation.get("unconstrained_energy")
            alpha = (
                SchedulingEngine.optimality_gap(
                    global_result.energy,
                    global_energy,
                    reference_energy,
                )
                if global_energy is not None
                else None
            )
            logger.info(
                "run_complete run_id=%s pipeline=iterative backend=%s feasible=%s optimal=%s sub_qubos=%s energy=%s global_energy=%s alpha=%s",
                run_id,
                global_result.solver_backend,
                global_result.is_feasible,
                validation.get("is_optimal"),
                len(solver_results),
                global_result.energy,
                global_energy,
                alpha,
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
                global_result=global_result,
                validation=validation,
                alpha=alpha,
            )

    @staticmethod
    def optimality_gap(
        candidate_energy: float,
        optimal_energy: float,
        reference_energy: float | None = None,
        eps: float = 1e-12,
    ) -> float:
        gap = candidate_energy - optimal_energy
        if np.isclose(gap, 0.0):
            return 0.0

        if reference_energy is None:
            reference_energy = 0.0

        denominator = abs(reference_energy - optimal_energy)
        if np.isclose(denominator, 0.0):
            denominator = max(abs(candidate_energy), abs(optimal_energy), 1.0)

        return gap / (denominator + eps)

    @staticmethod
    def build_preset_snapshot(weights: list[float], num_cores: int) -> SystemSnapshot:
        return SystemSnapshot(
            timestamp=time.time(),
            num_cores=num_cores,
            processes=[
                ProcessInfo(
                    pid=1000 + i,
                    command=f"proc_{i}",
                    cpu_weight=w,
                    current_core=0,
                    rss_mb=w * 1024,
                    priority=20,
                    io_wait_ratio=None, 
                    priority_class=None   
                )
                for i, w in enumerate(weights)
            ],
            total_ram_mb = None,
            snapshot_id = None
        )

if __name__ == "__main__":
    print("Running in CLI mode...")
    cli_mode = True
    
    NUM_CORES = 2 

    # weights snapshot from system
    weights_iterative = [0.247, 0.217, 0.197, 0.109, 0.069, 0.030, 0.030, 0.020, 0.010,
           0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    weights_default = [0.4, 0.3, 0.2, 0.1] 

    qaoa_cfg = QAOAConfig(layers=5, steps=100, learning_rate=0.05, top_k=10, mixer_type="xy")
    qubo_cfg = QUBOConfig(penalty=1, num_cores=NUM_CORES, snapshot=None, target_load=None)
    tracer_cfg = TracerConfig(min_rss=20, min_cpu=0.005, cpu_interval=1, num_samples=3, live_mode=False)
    decompositor_cfg = DecompositorConfig(qubit_max=12, num_cores=NUM_CORES, io_alpha=0.5, affinity_alpha=0.8, homogeneity_threshold=0.3, zscore_threshold=1.5, sorting_strategy=Heuristic.COUPLING_DESCENDING)

    preset_default = SchedulingEngine.build_preset_snapshot(weights_default, NUM_CORES)
    preset_iterative = SchedulingEngine.build_preset_snapshot(weights_iterative, NUM_CORES)
    try:
        output = SchedulingEngine.run_job(
            qaoa_cfg=qaoa_cfg,
            qubo_cfg=qubo_cfg,
            tracer_cfg=tracer_cfg,
            decompositor_cfg=decompositor_cfg,
            preset_snapshot=preset_default
        )

        # CLI Reporting
        if isinstance(output, IterativeSchedulingOutput):
            print(f"\n{'='*40}")
            print("ITERATIVE JOB COMPLETED")
            print(f"Sub-QUBOs: {output.num_sub_qubos} | Feasible: {output.num_feasible}/{output.num_sub_qubos}")
            if output.global_result is not None:
                print(f"Global energy: {output.global_result.energy:.4f}")
                if output.validation and output.validation.get("global_energy") is not None:
                    print(f"Global optimum: {output.validation['global_energy']:.4f}")
                    print(f"Optimality gap: {output.alpha:.4f}")
            print(f"Total solve time: {output.total_solve_time_ms:.1f}ms")
            print(f"Final core loads: {np.round(output.final_phi, 4)}")
            print(f"Load imbalance:   {output.load_imbalance:.6f}  (L_avg={output.L_avg:.4f})")
            print(f"Assignments: {output.final_assignments}")
            print(f"{'='*40}")
        elif isinstance(output, SchedulingOutput):
            print(f"\n{'='*40}")
            print("JOB COMPLETED SUCCESSFULLY")
            print(f"Energy: {output.result.energy:.4f}")
            print(f"Optimality Gap: {output.alpha:.4f}")
            print(f"Core Assignments: {output.result.decoded_assignments}")
            print(f"{'='*40}")

    except InfeasibleSubQUBOError as e:
        print(f"\n{'='*40}")
        print("ITERATIVE JOB STOPPED")
        print(e)
        print(f"Failed sub-QUBO: {e.subqubo_index + 1}")
        print(f"Failed energy: {e.result.energy:.4f}")
        print(f"Failed fallback assignments: {e.result.decoded_assignments}")
        print(f"Completed assignments before failure: {e.final_assignments}")
        if e.phi_history:
            print(f"Last valid core loads: {np.round(e.phi_history[-1], 4)}")
        else:
            print("Last valid core loads: none")
        print(f"{'='*40}")
    except Exception as e:
        print(f"Critical Error during execution: {e}")
