"""
QAOA Scheduling Testbed — Streamlit Interface
Run with: streamlit run app.py
"""

import os
import time
import copy
import numpy as np
import streamlit as st
from PIL import Image

current_dir = os.path.dirname(__file__)
icon_path = os.path.join(current_dir, "..", "assets", "image", "icon.png")
img = Image.open(icon_path)

st.set_page_config(
    page_title="QAOA Scheduling Testbed",
    page_icon=img,
    layout="wide",
)

from data_contracts import DecompositorConfig
from decomposition.subqubo_heuristics import Heuristic
# ── Local imports ──────────────────────────────────────────────────────────────
try:
    from data_contracts import QAOAConfig, QUBOConfig, TracerConfig
    from data_contracts import IterativeSchedulingOutput, SchedulingOutput
    from main import SchedulingEngine
    from pipeline.iterative_pipeline import InfeasibleSubQUBOError
    from visualizer.graph_visualizer import Visualizer
    from visualizer.iterative_visualizer import IterativeVisualizer
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# ── Pre-Algorithm Config ───────────────────────────────────────────────────────────
proc_preset = {
    "five_proc":  [0.29, 0.58, 0.48, 0.116, 0.39],
    "equal":      [0.2,  0.2,  0.2,  0.2,   0.2 ],
    "asymmetric": [0.2,  0.4,  0.1,  0.2,   0.99],
    "seven_proc": [0.029, 0.058, 0.048, 0.116, 0.029, 0.048, 0.039],
}

num_cores_preset = [2,3,4]
weights = []

qaoa_cfg = QAOAConfig(layers=3, steps=10, learning_rate=0.05, top_k=10)
qubo_cfg = QUBOConfig(penalty=1, num_cores=2, snapshot=None, target_load=None)
tracer_cfg = TracerConfig(min_rss=20, min_cpu=0.005, cpu_interval=1, num_samples=3, live_mode=False)
decompositor_cfg = DecompositorConfig(qubit_max=12, num_cores=2, io_alpha=0.5, affinity_alpha=0.8, homogeneity_threshold=0.3, zscore_threshold=1.5, sorting_strategy=Heuristic.COUPLING_DESCENDING)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def p_critical(weights: list, num_cores: int) -> float:
    if not weights:
        raise ValueError("At least one process weight is required.")
    if num_cores <= 0:
        raise ValueError("Number of cores must be greater than zero.")
    return 2 * max(weights) * (sum(weights) / num_cores)

def parse_weight_list(raw: str) -> list[float]:
    tokens = [x.strip() for x in raw.split(",") if x.strip()]
    return [float(x) for x in tokens]

def validate_workload_inputs(weights: list[float], num_cores: int) -> list[str]:
    errors = []
    if not weights:
        errors.append("Enter at least one process weight.")
    if num_cores <= 0:
        errors.append("Number of cores must be greater than zero.")
    if any(not np.isfinite(w) for w in weights):
        errors.append("Weights must be finite numbers.")
    if any(w < 0 for w in weights):
        errors.append("Weights must be non-negative.")
    return errors

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("QAOA Testbed")
    st.caption("Hybrid Classical-Quantum OS Scheduling")

    # --------------------------------
    # Workload Configuration
    # --------------------------------
    st.subheader("Workload")
    input_errors = []
    p_crit = None
    
    tracer_cfg.live_mode = st.toggle("Live System Snapshot", value=False, help="Toggle between hardcoded presets and real system snapshot.")
   
    if tracer_cfg.live_mode:
        tracer_cfg.min_rss = st.number_input("Min RSS (MB)", value=tracer_cfg.min_rss)
        tracer_cfg.min_cpu = st.number_input("Min CPU Load", value=tracer_cfg.min_cpu)
        tracer_cfg.cpu_interval = st.number_input("Min CPU Load", value=tracer_cfg.cpu_interval)
        tracer_cfg.num_samples = st.number_input("Trace Samples", value=tracer_cfg.num_samples)

        st.info(f"P_critical ≈ **NA**\nNeed system snapshot in order to get P_critical.")
    else:
        preset = st.selectbox("Preset", list(proc_preset.keys()) + ["custom"])
        if preset == "custom":
            raw = st.text_input("Weights (comma-separated)", value="0.29, 0.58, 0.48, 0.116, 0.39")
            try:
                weights = parse_weight_list(raw)
            except ValueError:
                weights = []
                input_errors.append("Invalid weights — use comma-separated floats.")
        else:
            weights = proc_preset[preset]

        st.caption(f"{len(weights)} processes: {[round(w, 3) for w in weights]}")

    st.divider()

    # --------------------------------
    # QUBO Configuration
    # --------------------------------
    st.subheader("QUBO")

    qubo_cfg.num_cores = st.selectbox("Number of cores (K)", num_cores_preset, index=0)
    decompositor_cfg.num_cores = qubo_cfg.num_cores
    qubo_cfg.penalty = st.slider("Penalty weight (P)", 0.5, 5.0, 1.6, 0.05)
    if not tracer_cfg.live_mode:
        input_errors.extend(validate_workload_inputs(weights, qubo_cfg.num_cores))
    
    if tracer_cfg.live_mode: st.warning("Can't check if P is below P_critical — QUBO global minimum may be infeasible.")
    elif input_errors:
        for error in input_errors:
            st.error(error)
    else:
        p_crit = p_critical(weights, qubo_cfg.num_cores)
        st.info(f"P_critical ≈ **{p_crit:.3f}**\nRecommended P ≥ {1.5 * p_crit:.3f}")
        if qubo_cfg.penalty < p_crit:
            st.warning("⚠️ P <= P_critical. QUBO global minimum may be infeasible.")

    st.divider()

    # --------------------------------
    # QAOA Configuration
    # --------------------------------
    st.subheader("QAOA")

    qaoa_cfg.layers = st.slider("Circuit depth (p)", 1, 15, 3)
    qaoa_cfg.steps  = st.slider("Optimizer steps", 10, 500, 100, 10)
    qaoa_cfg.learning_rate = st.select_slider(
        "Learning rate (η)",
        options=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        value=0.01,
    )
    qaoa_cfg.mixer_type = st.selectbox("Mixer", ["xy", "x"], index=0)
    init_theta = st.number_input("Initial θ0 (γ=β)", value=0.5, step=0.05, format="%.3f")
    qaoa_cfg.init_gamma = float(init_theta)
    qaoa_cfg.init_beta = float(init_theta)
    qaoa_cfg.top_k = st.slider("Top-k bitstrings shown", 5, 30, 20)

    st.divider()
    st.caption("Solver: PennyLane · lightning.gpu")
    st.caption("Ground truth: Brute-force (≤22 vars)")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

if not IMPORTS_OK:
    st.error(f"**Import error** — run `streamlit run app.py` from the project root.\n\n`{IMPORT_ERROR}`")
    st.stop()

tab_single, tab_sweep, tab_about = st.tabs(["🔬 Single Run", "📊 Sweep", "📖 About"])


# ── Single Run ─────────────────────────────────────────────────────────
with tab_single:
    st.header("Single Run")

    col_params, col_run = st.columns([3, 1])
    with col_params:
        st.markdown(
            f"**Config:** P={qubo_cfg.penalty} · p={qaoa_cfg.layers} · steps={qaoa_cfg.steps} · "
            f"η={qaoa_cfg.learning_rate} · mixer={qaoa_cfg.mixer_type} · θ0={qaoa_cfg.init_gamma} · "
            f"K={qubo_cfg.num_cores if not tracer_cfg.live_mode else 'Need system snapshot'} · N={len(weights)}"
        )
    with col_run:
        run_btn = st.button(
            "▶ Run QAOA",
            type="primary",
            width="stretch",
            disabled=not tracer_cfg.live_mode and bool(input_errors),
        )

    if run_btn:
        current_snapshot = None if tracer_cfg.live_mode else SchedulingEngine.build_preset_snapshot(weights, qubo_cfg.num_cores)
        run_qaoa_cfg = copy.deepcopy(qaoa_cfg)
        run_qubo_cfg = copy.deepcopy(qubo_cfg)
        run_tracer_cfg = copy.deepcopy(tracer_cfg)
        run_decompositor_cfg = copy.deepcopy(decompositor_cfg)
        
        with st.spinner(f"Running QAOA (p={run_qaoa_cfg.layers}, steps={run_qaoa_cfg.steps})…"):
            t0 = time.time()

            try:
                output = SchedulingEngine.run_job(
                    run_qaoa_cfg,
                    run_qubo_cfg,
                    run_tracer_cfg,
                    run_decompositor_cfg,
                    current_snapshot,
                )
            except InfeasibleSubQUBOError as e:
                st.error(f"Iterative run stopped at sub-QUBO {e.subqubo_index + 1}: no feasible top-k assignment.")
                st.write("Best infeasible fallback:", e.result.decoded_assignments)
                st.write("Completed assignments before failure:", e.final_assignments)
                if e.phi_history:
                    st.write("Last valid core loads:", np.round(e.phi_history[-1], 4))
                st.stop()
            
            elapsed = time.time() - t0

        st.divider()

        if isinstance(output, SchedulingOutput):
            result = output.result
            validation = output.validation
            used_snapshot = output.used_snapshot
            alpha = output.alpha
            qubo = output.qubo_instance

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Energy",         f"{result.energy:.4f}")
            m2.metric("Global optimum", f"{validation['global_energy']:.4f}" if validation["global_energy"] is not None else "N/A")
            m3.metric("Optimality gap", f"{alpha:.4f}" if not np.isnan(alpha) else "N/A")
            m4.metric("Feasible",       "✅ Yes" if result.is_feasible else "❌ No")
            m5.metric("Optimal",        "N/A" if validation["is_optimal"] is None else ("✅ Yes" if validation["is_optimal"] else "❌ No"))
            st.caption(f"Solve time: {result.solve_time_ms:.0f} ms · Wall time: {elapsed:.1f} s")
            if validation.get("brute_force_error"):
                st.info(f"Brute-force optimum skipped: {validation['brute_force_error']}")

            with st.expander("Core assignments", expanded=True):
                rows = []
                for proc in used_snapshot.processes:
                    core     = result.decoded_assignments.get(proc.pid, "?")
                    opt_core = validation["global_assignments"].get(proc.pid, "?")
                    rows.append({
                        "PID":            proc.pid,
                        "Weight":         round(proc.cpu_weight, 3),
                        "QAOA → core":    core,
                        "Optimal → core": opt_core,
                        "Match":          "✅" if core == opt_core else "❌",
                    })
                st.dataframe(rows, width="stretch", hide_index=True)

            viz = Visualizer(
                qubo=qubo,
                qaoa_cfg=output.qaoa_cfg,
                qubo_cfg=output.qubo_cfg,
                probs=result.probs,
                energies_over_time=result.convergence_curve,
                global_optimum=validation["global_energy"] if validation["global_energy"] is not None else result.energy,
                top_k=output.qaoa_cfg.top_k,
            )

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(viz.panel_landscape(), width="stretch")
            with c2:
                st.pyplot(viz.panel_qubo_matrix(), width="stretch")

            c3, c4 = st.columns(2)
            with c3:
                if result.convergence_curve:
                    st.pyplot(viz.panel_convergence(), width="stretch")
            with c4:
                if result.probs is not None:
                    st.pyplot(viz.panel_probabilities(), width="stretch")

            composite_path = f"results/run_P{output.qubo_cfg.penalty}_p{output.qaoa_cfg.layers}.png"
            viz.composite(save_path=composite_path)
            st.caption(f"Composite saved → `{composite_path}`")

        elif isinstance(output, IterativeSchedulingOutput):
            final_phi = output.final_phi
            global_energy = output.global_result.energy if output.global_result else None
            global_optimum = output.validation.get("global_energy") if output.validation else None

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Global energy", f"{global_energy:.4f}" if global_energy is not None else "N/A")
            m2.metric("Global optimum", f"{global_optimum:.4f}" if global_optimum is not None else "N/A")
            m3.metric("Optimality gap", f"{output.alpha:.4f}" if output.alpha is not None else "N/A")
            m4.metric("Sub-QUBOs", output.num_sub_qubos)
            m5.metric("Load imbalance", f"{output.load_imbalance:.4f}")
            global_feasible = (
                "Yes" if output.global_result and output.global_result.is_feasible
                else "No" if output.global_result
                else "N/A"
            )
            st.caption(
                f"Feasible sub-QUBOs: {output.num_feasible}/{output.num_sub_qubos} · "
                f"Global feasible: {global_feasible} · "
                f"Target load: {output.L_avg:.4f} · "
                f"Solve time: {output.total_solve_time_ms:.0f} ms · Wall time: {elapsed:.1f} s"
            )
            if output.validation and output.validation.get("brute_force_error"):
                st.info(f"Brute-force optimum skipped: {output.validation['brute_force_error']}")
            if output.validation and output.validation.get("errors"):
                st.warning(f"Global validation errors: {output.validation['errors']}")

            with st.expander("Core assignments", expanded=True):
                rows = []
                for entity in output.used_workload.entities:
                    rows.append({
                        "Entity": entity.entity_id,
                        "Label": entity.label,
                        "Weight": round(entity.cpu_weight, 3),
                        "QAOA → core": output.final_assignments.get(entity.entity_id, "?"),
                    })
                for entity_id, core in output.used_workload.fixed_assignments.items():
                    rows.append({
                        "Entity": entity_id,
                        "Label": "fixed_rt",
                        "Weight": round(output.used_workload.fixed_loads.get(entity_id, 0.0), 3),
                        "QAOA → core": core,
                    })
                st.dataframe(rows, width="stretch", hide_index=True)

            st.write("Final core loads:", np.round(final_phi, 4))

            viz = IterativeVisualizer(
                solver_results=output.solver_results,
                phi_history=output.phi_history,
                workload=output.used_workload,
                qubo_instance=output.qubo_instance,
                qaoa_cfg=output.qaoa_cfg,
                qubo_cfg=output.qubo_cfg,
                top_k=output.qaoa_cfg.top_k,
            )

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(viz.panel_load_balance(), width="stretch")
            with c2:
                if output.solver_results:
                    st.pyplot(viz.panel_phi_evolution(), width="stretch")
                else:
                    st.info("No movable workload entities were scheduled.")

            c3, c4 = st.columns(2)
            with c3:
                if output.qubo_instance.num_variables:
                    st.pyplot(viz.panel_qubo_matrix(), width="stretch")
            with c4:
                if output.solver_results:
                    st.pyplot(viz.panel_convergence_grid(), width="stretch")

            if output.solver_results:
                st.pyplot(viz.panel_probabilities_grid(), width="stretch")

            if output.solver_results:
                composite_path = "results/iterative_run.png"
                viz.composite(save_path=composite_path)
                st.caption(f"Composite saved → `{composite_path}`")

        else:
            st.error(f"Unexpected scheduling output type: {type(output).__name__}")
            st.stop()

        st.session_state["last_single"] = {"output": copy.deepcopy(output)}

    elif "last_single" in st.session_state:
        st.info("Showing results from last run. Hit **▶ Run QAOA** to rerun.")
    else:
        st.info("Configure parameters in the sidebar, then hit **▶ Run QAOA**.")

# ── Sweep ──────────────────────────────────────────────────────────────
with tab_sweep:
    st.header("Parameter Sweep")

    sweep_mode = st.radio(
        "Sweep over",
        ["Penalty (P)", "Circuit depth (p)", "Initialization (θ0)", "P × p", "P × θ0", "p × θ0", "All (grid)"],
        horizontal=True,
    )

    col_a, col_b, col_c = st.columns(3)

    if sweep_mode in ("Penalty (P)", "P × p", "P × θ0", "All (grid)"):
        with col_a:
            p_min  = st.number_input("P min",  value=1.0, step=0.1, format="%.2f")
            p_max  = st.number_input("P max",  value=2.9, step=0.1, format="%.2f")
            p_step = st.number_input("P step", value=0.1, step=0.05, format="%.2f")
    else:
        p_min = p_max = qubo_cfg.penalty; p_step = 1.0

    if sweep_mode in ("Circuit depth (p)", "P × p", "p × θ0", "All (grid)"):
        with col_b:
            p_layers_raw = st.text_input("p values (comma-sep)", "1, 2, 3, 5, 7, 10")
            try:
                p_layers_list = [int(x.strip()) for x in p_layers_raw.split(",")]
            except ValueError:
                p_layers_list = [3]
    else:
        p_layers_list = [qaoa_cfg.layers]

    with col_c:
        sweep_steps = st.number_input("Steps per run", value=qaoa_cfg.steps, step=10)
        sweep_lr    = st.number_input("Learning rate", value=qaoa_cfg.learning_rate, step=0.005, format="%.4f")
        if sweep_mode in ("Initialization (θ0)", "P × θ0", "p × θ0", "All (grid)"):
            init_raw = st.text_input("θ0 values (comma-sep)", "0.1, 0.25, 0.5, 0.75, 1.0")
            try:
                init_values = [float(x.strip()) for x in init_raw.split(",")]
            except ValueError:
                init_values = [qaoa_cfg.init_gamma]
        else:
            init_values = [qaoa_cfg.init_gamma]
        shots       = st.number_input("Shots per config", value=1, min_value=1, max_value=10,
                                      help="Repeats each config N times — gives mean ± std.")

    penalty_range = np.around(np.arange(p_min, p_max + p_step * 0.5, p_step), decimals=4)
    total_runs    = len(penalty_range) * len(p_layers_list) * len(init_values) * int(shots)
    st.caption(f"Total runs: {total_runs}  ·  estimated time: ~{total_runs * 18:.0f}s")

    sweep_disabled = tracer_cfg.live_mode or bool(input_errors)
    if tracer_cfg.live_mode:
        st.info("Sweep requires a preset or custom workload; live snapshots vary between runs.")

    if st.button("▶ Run Sweep", type="primary", disabled=sweep_disabled):
        if total_runs > 100:
            st.warning(f"⚠️ {total_runs} runs may take a long time.")

        snapshot      = SchedulingEngine.build_preset_snapshot(weights, qubo_cfg.num_cores)
        sweep_results = []
        progress      = st.progress(0, text="Starting sweep…")
        run_index     = 0

        for layer_val in p_layers_list:
            for pen in penalty_range:
                for init_theta in init_values:
                    shot_gaps, shot_max_ps, shot_feasible, shot_optimal = [], [], [], []
                    for shot in range(int(shots)):
                        run_index += 1
                        progress.progress(
                            run_index / total_runs,
                            text=f"P={pen:.2f} · p={layer_val} · θ0={init_theta:.3f} · shot {shot + 1}/{shots}",
                        )
                        try:
                            sweep_qaoa_cfg = QAOAConfig(
                                layers=layer_val,
                                steps=int(sweep_steps),
                                learning_rate=float(sweep_lr),
                                top_k=qaoa_cfg.top_k,
                                mixer_type=qaoa_cfg.mixer_type,
                                init_gamma=float(init_theta),
                                init_beta=float(init_theta),
                            )
                            sweep_qubo_cfg = QUBOConfig(
                                penalty=float(pen),
                                num_cores=qubo_cfg.num_cores,
                                snapshot=None,
                                target_load=None,
                            )

                            output = SchedulingEngine.run_job(
                                qaoa_cfg=sweep_qaoa_cfg,
                                qubo_cfg=sweep_qubo_cfg,
                                tracer_cfg=tracer_cfg,
                                decompositor_cfg=decompositor_cfg,
                                preset_snapshot=snapshot,
                            )

                            if isinstance(output, IterativeSchedulingOutput):
                                st.warning(
                                    f"Skipped iterative run at P={pen}, p={layer_val}, "
                                    f"θ0={init_theta}: sweep metrics require default-pipeline validation."
                                )
                                continue

                            res = output.result
                            val = output.validation
                            shot_gaps.append(output.alpha)
                            shot_max_ps.append(float(np.max(res.probs)) if res.probs is not None else 0.0)
                            shot_feasible.append(res.is_feasible)
                            shot_optimal.append(val["is_optimal"])
                        except InfeasibleSubQUBOError as e:
                            st.error(f"Run stopped at P={pen}, p={layer_val}, θ0={init_theta}, sub-QUBO {e.subqubo_index + 1}: no feasible top-k assignment.")
                        except Exception as e:
                            st.error(f"Run failed at P={pen}, p={layer_val}, θ0={init_theta}: {e}")
                    if shot_gaps:
                        sweep_results.append({
                            "P":             float(pen),
                            "p_layers":      layer_val,
                            "theta0":        float(init_theta),
                            "init_gamma":    float(init_theta),
                            "init_beta":     float(init_theta),
                            "gap_mean":      float(np.mean(shot_gaps)),
                            "gap_std":       float(np.std(shot_gaps)),
                            "max_p_mean":    float(np.mean(shot_max_ps)),
                            "feasible_rate": float(np.mean(shot_feasible)),
                            "optimal_rate":  float(np.mean(shot_optimal)),
                            # keys consumed by Visualizer.plot_sweep
                            "p":       float(pen),
                            "alpha":   float(np.mean(shot_gaps)),
                            "max_p":   float(np.mean(shot_max_ps)),
                            "feasible": float(np.mean(shot_feasible)) == 1.0,
                        })

        progress.empty()
        st.session_state["sweep_results"] = sweep_results
        st.success(f"Sweep complete — {len(sweep_results)} configurations.")

    if "sweep_results" in st.session_state and st.session_state["sweep_results"]:
        sr = st.session_state["sweep_results"]
        st.divider()
        st.subheader("Results")

        import pandas as pd
        df = pd.DataFrame(sr).drop(columns=["p", "alpha", "max_p", "feasible"], errors="ignore")
        st.dataframe(
            df.style.background_gradient(
                subset=["gap_mean", "optimal_rate", "feasible_rate"], cmap="RdYlGn_r"
            ),
            width="stretch", hide_index=True,
        )

        unique_layers = sorted(set(r["p_layers"] for r in sr))
        unique_theta = sorted(set(r["theta0"] for r in sr))
        unique_penalties = sorted(set(r["P"] for r in sr))

        # Single p_layers and theta0: use Visualizer.plot_sweep directly for P sweeps.
        if len(unique_layers) == 1 and len(unique_theta) == 1 and len(unique_penalties) > 1:
            st.pyplot(Visualizer.plot_sweep(sr), width="stretch")
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            if len(unique_theta) > 1 and len(unique_penalties) == 1:
                x_key, x_label = "theta0", "Initial θ0"
            elif len(unique_penalties) > 1:
                x_key, x_label = "P", "Penalty (P)"
            else:
                x_key, x_label = "p_layers", "Circuit depth (p)"

            if x_key == "theta0":
                series = [
                    (f"p={lv}", [r for r in sr if r["p_layers"] == lv])
                    for lv in unique_layers
                ]
            elif x_key == "P":
                series = [
                    (
                        f"p={lv}, θ0={theta:.3f}" if len(unique_theta) > 1 else f"p={lv}",
                        [r for r in sr if r["p_layers"] == lv and r["theta0"] == theta],
                    )
                    for lv in unique_layers
                    for theta in unique_theta
                ]
            else:
                series = [
                    (f"θ0={theta:.3f}", [r for r in sr if r["theta0"] == theta])
                    for theta in unique_theta
                ]

            for label, sub in series:
                if not sub:
                    continue
                sub = sorted(sub, key=lambda r: r[x_key])
                ax.plot([r[x_key] for r in sub], [r["gap_mean"] for r in sub],
                        marker="o", label=label)
            ax.axhline(0.0, color="grey", linestyle="--", alpha=0.5)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Optimality gap (mean)")
            ax.set_title(f"Optimality gap vs {x_label}")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")

        csv = pd.DataFrame(sr).to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", data=csv,
                           file_name="sweep_results.csv", mime="text/csv")


# ── TAB 3: About ──────────────────────────────────────────────────────────────
with tab_about:
    st.header("About this Testbed")
    st.markdown("""
### Hybrid Classical-Quantum Testbed for OS Scheduling

This interface wraps the **QUBO → QAOA pipeline** for process-to-core assignment.
The scientific goal is **characterisation and break-even projection** — measuring
precisely when and how quantum approaches become competitive with classical baselines.

---

#### Pipeline

```
SystemSnapshot → CoreAssignmentBuilder → QUBOInstance
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                       PennylaneSolver               BruteForceSolver
                       (QAOA via lightning.gpu)      (ground truth oracle)
                              │                               │
                              └───────────────┬───────────────┘
                                              ▼
                                      SolverValidator
                             (optimality gap, feasibility, optimality)
```

#### Key parameters

| Parameter | Role |
|-----------|------|
| **P** (penalty) | Constraint enforcement. P < P_critical → infeasible global minimum. |
| **p** (layers)  | Circuit depth. Higher p = more expressive, slower. |
| **steps**       | Optimizer iterations. Rule of thumb: ≥ 50 × p. |
| **η** (lr)      | Adam learning rate. Lower → more stable, slower. |
| **θ0**          | Initial gamma/beta value before Adam optimization. |

#### P_critical

$$P_{\\text{critical}} = 2 \\cdot w_{\\max} \\cdot \\bar{L}, \\quad \\bar{L} = W_{\\text{total}} / K$$

Recommended: $P = 1.5 \\times P_{\\text{critical}}$ for deep circuits (p ≥ 5),
$2 \\times P_{\\text{critical}}$ for shallow (p ≤ 3).
    """)
