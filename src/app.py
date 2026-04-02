"""
QAOA Scheduling Testbed — Streamlit Interface
Run with: streamlit run app.py
"""

import time
import numpy as np
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QAOA Scheduling Testbed",
    page_icon="⚛️",
    layout="wide",
)

# ── Local imports ──────────────────────────────────────────────────────────────
try:
    from config_contracts import QAOAConfig, QUBOConfig
    from data_contracts import SystemSnapshot, ProcessInfo
    from builder.qubo_core import CoreAssignmentBuilder
    from solver.pennylane_solver import PennylaneSolver
    from solver.solver_validator import SolverValidator
    from visualization.visualization import Visualization
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_snapshot(weights: list, num_cores: int) -> SystemSnapshot:
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
            )
            for i, w in enumerate(weights)
        ],
    )


def run_single(snapshot, penalty, layers, steps, lr):
    qubo_cfg   = QUBOConfig(penalty=penalty, num_cores=snapshot.num_cores, snapshot=snapshot)
    qaoa_cfg   = QAOAConfig(layers=layers, steps=steps, optimizer_step=lr)
    builder    = CoreAssignmentBuilder(penalty=penalty)
    solver     = PennylaneSolver(p_layers=layers, steps=steps, optimizer_step=lr)
    qubo       = builder.build(snapshot)
    result     = solver.solve(qubo)
    validation = SolverValidator().validate(qubo, result)
    return result, validation, qubo, qaoa_cfg, qubo_cfg


def p_critical(weights: list, num_cores: int) -> float:
    return 2 * max(weights) * (sum(weights) / num_cores)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚛️ QAOA Testbed")
    st.caption("Hybrid Classical-Quantum OS Scheduling")
    st.divider()

    st.subheader("Workload")
    preset = st.selectbox(
        "Preset",
        ["five_proc", "equal", "asymmetric", "seven_proc", "custom"],
    )
    preset_map = {
        "five_proc":  [0.29, 0.58, 0.48, 0.116, 0.39],
        "equal":      [0.2,  0.2,  0.2,  0.2,   0.2 ],
        "asymmetric": [0.2,  0.4,  0.1,  0.2,   0.99],
        "seven_proc": [0.029, 0.058, 0.048, 0.116, 0.029, 0.048, 0.039],
    }
    if preset == "custom":
        raw = st.text_input("Weights (comma-separated)", value="0.29, 0.58, 0.48, 0.116, 0.39")
        try:
            weights = [float(x.strip()) for x in raw.split(",")]
        except ValueError:
            st.error("Invalid weights — use comma-separated floats.")
            weights = [0.29, 0.58, 0.48, 0.116, 0.39]
    else:
        weights = preset_map[preset]

    st.caption(f"{len(weights)} processes: {[round(w, 3) for w in weights]}")
    num_cores = st.selectbox("Number of cores (K)", [2, 3, 4], index=0)

    p_crit = p_critical(weights, num_cores)
    st.info(f"P_critical ≈ **{p_crit:.3f}**\nRecommended P ≥ {1.5 * p_crit:.3f}")

    st.divider()
    st.subheader("QUBO")
    penalty = st.slider("Penalty weight (P)", 0.5, 5.0, 1.6, 0.05)
    if penalty < p_crit:
        st.warning("⚠️ P is below P_critical — QUBO global minimum may be infeasible.")

    st.divider()
    st.subheader("QAOA")
    layers = st.slider("Circuit depth (p)", 1, 15, 3)
    steps  = st.slider("Optimizer steps", 10, 500, 100, 10)
    lr     = st.select_slider(
        "Learning rate (η)",
        options=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        value=0.01,
    )
    top_k = st.slider("Top-k bitstrings shown", 5, 30, 20)

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


# ── TAB 1: Single Run ─────────────────────────────────────────────────────────
with tab_single:
    st.header("Single Run")

    col_params, col_run = st.columns([3, 1])
    with col_params:
        st.markdown(
            f"**Config:** P={penalty} · p={layers} · steps={steps} · "
            f"η={lr} · K={num_cores} · N={len(weights)}"
        )
    with col_run:
        run_btn = st.button("▶ Run QAOA", type="primary", use_container_width=True)

    if run_btn:
        snapshot = build_snapshot(weights, num_cores)

        with st.spinner(f"Running QAOA (p={layers}, steps={steps})…"):
            t0 = time.time()
            result, validation, qubo, qaoa_cfg, qubo_cfg = run_single(
                snapshot, penalty, layers, steps, lr
            )
            elapsed = time.time() - t0

        # Metrics row
        st.divider()
        alpha = result.energy / validation["global_energy"] if validation["global_energy"] != 0 else 0.0
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Energy",         f"{result.energy:.4f}")
        m2.metric("Global optimum", f"{validation['global_energy']:.4f}")
        m3.metric("Alpha (α)",      f"{alpha:.4f}")
        m4.metric("Feasible",       "✅ Yes" if result.is_feasible else "❌ No")
        m5.metric("Optimal",        "✅ Yes" if validation["is_optimal"] else "❌ No")
        st.caption(f"Solve time: {result.solve_time_ms:.0f} ms · Wall time: {elapsed:.1f} s")

        # Assignment table
        with st.expander("Core assignments", expanded=True):
            rows = []
            for proc in snapshot.processes:
                core     = result.decoded_assignments.get(proc.pid, "?")
                opt_core = validation["global_assignments"].get(proc.pid, "?")
                rows.append({
                    "PID":            proc.pid,
                    "Weight":         round(proc.cpu_weight, 3),
                    "QAOA → core":    core,
                    "Optimal → core": opt_core,
                    "Match":          "✅" if core == opt_core else "❌",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

        # Build Visualization once — reuse across all four panels
        viz = Visualization(
            qubo=qubo,
            qaoa_cfg=qaoa_cfg,
            qubo_cfg=qubo_cfg,
            probs=result.probs,
            energies_over_time=result.convergence_curve,
            global_optimum=validation["global_energy"],
            top_k=top_k,
        )

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(viz.panel_landscape(), use_container_width=True)
        with c2:
            st.pyplot(viz.panel_qubo_matrix(), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            if result.convergence_curve:
                st.pyplot(viz.panel_convergence(), use_container_width=True)
        with c4:
            if result.probs is not None:
                st.pyplot(viz.panel_probabilities(), use_container_width=True)

        # Also save composite to disk (same as the old pipeline behaviour)
        composite_path = f"results/run_P{penalty}_p{layers}.png"
        viz.composite(save_path=composite_path)
        st.caption(f"Composite saved → `{composite_path}`")

        st.session_state["last_single"] = {
            "result": result, "validation": validation,
            "qubo": qubo, "alpha": alpha,
            "qaoa_cfg": qaoa_cfg, "qubo_cfg": qubo_cfg,
        }

    elif "last_single" in st.session_state:
        st.info("Showing results from last run. Hit **▶ Run QAOA** to rerun.")
    else:
        st.info("Configure parameters in the sidebar, then hit **▶ Run QAOA**.")


# ── TAB 2: Sweep ──────────────────────────────────────────────────────────────
with tab_sweep:
    st.header("Parameter Sweep")

    sweep_mode = st.radio(
        "Sweep over",
        ["Penalty (P)", "Circuit depth (p)", "Both (grid)"],
        horizontal=True,
    )

    col_a, col_b, col_c = st.columns(3)

    if sweep_mode in ("Penalty (P)", "Both (grid)"):
        with col_a:
            p_min  = st.number_input("P min",  value=1.0, step=0.1, format="%.2f")
            p_max  = st.number_input("P max",  value=2.9, step=0.1, format="%.2f")
            p_step = st.number_input("P step", value=0.1, step=0.05, format="%.2f")
    else:
        p_min = p_max = penalty; p_step = 1.0

    if sweep_mode in ("Circuit depth (p)", "Both (grid)"):
        with col_b:
            p_layers_raw = st.text_input("p values (comma-sep)", "1, 2, 3, 5, 7, 10")
            try:
                p_layers_list = [int(x.strip()) for x in p_layers_raw.split(",")]
            except ValueError:
                p_layers_list = [3]
    else:
        p_layers_list = [layers]

    with col_c:
        sweep_steps = st.number_input("Steps per run", value=steps, step=10)
        sweep_lr    = st.number_input("Learning rate", value=lr, step=0.005, format="%.4f")
        shots       = st.number_input("Shots per config", value=1, min_value=1, max_value=10,
                                      help="Repeats each config N times — gives mean ± std.")

    penalty_range = np.around(np.arange(p_min, p_max + p_step * 0.5, p_step), decimals=4)
    total_runs    = len(penalty_range) * len(p_layers_list) * int(shots)
    st.caption(f"Total runs: {total_runs}  ·  estimated time: ~{total_runs * 18:.0f}s")

    if st.button("▶ Run Sweep", type="primary"):
        if total_runs > 100:
            st.warning(f"⚠️ {total_runs} runs may take a long time.")

        snapshot      = build_snapshot(weights, num_cores)
        sweep_results = []
        progress      = st.progress(0, text="Starting sweep…")
        run_index     = 0

        for layer_val in p_layers_list:
            for pen in penalty_range:
                shot_alphas, shot_max_ps, shot_feasible, shot_optimal = [], [], [], []
                for shot in range(int(shots)):
                    run_index += 1
                    progress.progress(
                        run_index / total_runs,
                        text=f"P={pen:.2f} · p={layer_val} · shot {shot + 1}/{shots}",
                    )
                    try:
                        res, val, _, _, _ = run_single(
                            snapshot, float(pen), layer_val,
                            int(sweep_steps), float(sweep_lr),
                        )
                        g_e = val["global_energy"]
                        shot_alphas.append(res.energy / g_e if g_e != 0 else 0.0)
                        shot_max_ps.append(float(np.max(res.probs)) if res.probs is not None else 0.0)
                        shot_feasible.append(res.is_feasible)
                        shot_optimal.append(val["is_optimal"])
                    except Exception as e:
                        st.error(f"Run failed at P={pen}, p={layer_val}: {e}")

                if shot_alphas:
                    sweep_results.append({
                        "P":             float(pen),
                        "p_layers":      layer_val,
                        "alpha_mean":    float(np.mean(shot_alphas)),
                        "alpha_std":     float(np.std(shot_alphas)),
                        "max_p_mean":    float(np.mean(shot_max_ps)),
                        "feasible_rate": float(np.mean(shot_feasible)),
                        "optimal_rate":  float(np.mean(shot_optimal)),
                        # keys consumed by Visualization.plot_sweep
                        "p":       float(pen),
                        "alpha":   float(np.mean(shot_alphas)),
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
                subset=["alpha_mean", "optimal_rate", "feasible_rate"], cmap="RdYlGn"
            ),
            use_container_width=True, hide_index=True,
        )

        # Single p_layers: use Visualization.plot_sweep directly
        if len(set(r["p_layers"] for r in sr)) == 1:
            st.pyplot(Visualization.plot_sweep(sr), use_container_width=True)
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            for lv in sorted(set(r["p_layers"] for r in sr)):
                sub = [r for r in sr if r["p_layers"] == lv]
                ax.plot([r["P"] for r in sub], [r["alpha_mean"] for r in sub],
                        marker="o", label=f"p={lv}")
            ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
            ax.set_xlabel("Penalty (P)")
            ax.set_ylabel("Alpha (mean)")
            ax.set_title("Alpha vs P — multi-depth comparison")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

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
                                   (alpha, feasibility, optimality)
```

#### Key parameters

| Parameter | Role |
|-----------|------|
| **P** (penalty) | Constraint enforcement. P < P_critical → infeasible global minimum. |
| **p** (layers)  | Circuit depth. Higher p = more expressive, slower. |
| **steps**       | Optimizer iterations. Rule of thumb: ≥ 50 × p. |
| **η** (lr)      | Adam learning rate. Lower → more stable, slower. |

#### P_critical

$$P_{\\text{critical}} = 2 \\cdot w_{\\max} \\cdot \\bar{L}, \\quad \\bar{L} = W_{\\text{total}} / K$$

Recommended: $P = 1.5 \\times P_{\\text{critical}}$ for deep circuits (p ≥ 5),
$2 \\times P_{\\text{critical}}$ for shallow (p ≤ 3).
    """)