"""
visualization.py — Unified visualization for the QAOA scheduling testbed.

Key change from the original:
  - No plt.show(), no plt.pause(), no TkAgg backend dependency.
  - Every panel is a standalone method returning a plt.Figure.
  - The composite() method assembles all panels into one figure and
    optionally saves it — same visual output as before, no blocking calls.
  - Streamlit usage: pass any Figure directly to st.pyplot(fig).
  - Standalone usage: call fig.show() or plt.show() yourself after composite().
"""

import matplotlib
matplotlib.use("Agg")   # Non-interactive — safe in Streamlit, scripts, and notebooks

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from itertools import product

import numpy as np

from config_contracts import QAOAConfig, QUBOConfig


class Visualization:
    """
    All panel methods are instance methods that return a plt.Figure.
    They never call plt.show() or plt.pause() — the caller decides
    how to display or save the result.

    Usage — standalone script:
        viz = Visualization(qubo, qaoa_cfg, qubo_cfg, probs=..., energies_over_time=...)
        fig = viz.composite(save_path="results/run.png")
        plt.show()          # open a window if a GUI backend is available
        # or: fig.savefig("results/run.png")

    Usage — Streamlit:
        viz = Visualization(qubo, qaoa_cfg, qubo_cfg, probs=..., energies_over_time=...)
        st.pyplot(viz.panel_landscape())
        st.pyplot(viz.panel_qubo_matrix())
        st.pyplot(viz.panel_convergence())
        st.pyplot(viz.panel_probabilities())
        # or just render everything at once:
        st.pyplot(viz.composite())
    """

    def __init__(
        self,
        qubo,
        qaoa_cfg: QAOAConfig,
        qubo_cfg: QUBOConfig,
        probs=None,
        energies_over_time=None,
        global_optimum=None,
        top_k: int = 20,
    ):
        self.qubo               = qubo
        self.qaoa_cfg           = qaoa_cfg
        self.qubo_cfg           = qubo_cfg
        self.probs              = probs
        self.energies_over_time = energies_over_time
        self.global_optimum     = global_optimum
        self.top_k              = top_k

    # ──────────────────────────────────────────────────────────────────────────
    #  Individual panels — each returns a self-contained plt.Figure
    # ──────────────────────────────────────────────────────────────────────────

    def panel_landscape(self, figsize=(6, 4)) -> plt.Figure:
        """Energy landscape: infeasible (red) vs feasible (lime) states."""
        num_vars     = self.qubo.num_variables
        num_entities = self.qubo.num_entities
        num_cores    = self.qubo.num_cores

        all_states = list(product([0, 1], repeat=num_vars))
        energies, feasibility = [], []

        for state in all_states:
            x = np.array(state)
            energies.append(float(x.T @ self.qubo.Q @ x))
            feasibility.append(
                all(sum(state[i * num_cores:(i + 1) * num_cores]) == 1
                    for i in range(num_entities))
            )

        energies    = np.array(energies)
        feasibility = np.array(feasibility)
        indices     = np.arange(len(energies))

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(indices[~feasibility], energies[~feasibility],
                   c="red", s=2, alpha=0.3, label="Infeasible")
        ax.scatter(indices[feasibility], energies[feasibility],
                   c="lime", s=20, edgecolors="black", linewidths=0.5, label="Feasible")
        ax.set_title("Energy Landscape")
        ax.set_xlabel("State Index")
        ax.set_ylabel("Energy  xᵀQx")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def panel_qubo_matrix(self, figsize=(5, 4)) -> plt.Figure:
        """Heatmap of the Q matrix with block separators per process."""
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(self.qubo.Q, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        K = self.qubo.num_cores
        for i in range(1, self.qubo.num_entities):
            ax.axhline(i * K - 0.5, color="black", linewidth=1.5)
            ax.axvline(i * K - 0.5, color="black", linewidth=1.5)

        ax.set_title("QUBO Matrix  Q")
        ax.set_xlabel("Variable index")
        ax.set_ylabel("Variable index")
        fig.tight_layout()
        return fig

    def panel_convergence(self, figsize=(6, 4)) -> plt.Figure:
        """⟨C⟩ per optimizer step with optional global-optimum reference line."""
        if self.energies_over_time is None:
            raise ValueError("No convergence data — pass energies_over_time to the constructor.")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.energies_over_time, label="⟨C⟩ per step", color="steelblue")

        if self.global_optimum is not None:
            ax.axhline(
                self.global_optimum,
                color="lime", linestyle="--",
                label=f"Global optimum ({self.global_optimum:.3f})",
            )

        ax.set_title("QAOA Convergence")
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Expected Energy  ⟨C⟩")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def panel_probabilities(self, figsize=(7, 4)) -> plt.Figure:
        """Bar chart of top-k output bitstring probabilities, coloured by feasibility."""
        if self.probs is None:
            raise ValueError("No probability data — pass probs to the constructor.")

        num_vars = self.qubo.num_variables
        probs    = np.array(self.probs)
        top_k    = min(self.top_k, len(probs))

        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs   = probs[top_indices]

        labels, colors = [], []
        for idx in top_indices:
            bits = tuple(int(b) for b in bin(idx)[2:].zfill(num_vars))
            feasible = all(
                sum(bits[i * self.qubo.num_cores:(i + 1) * self.qubo.num_cores]) == 1
                for i in range(self.qubo.num_entities)
            )
            labels.append(bin(idx)[2:].zfill(num_vars))
            colors.append("lime" if feasible else "tomato")

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(top_k), top_probs, color=colors)
        ax.set_xticks(range(top_k))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_xlabel("Bitstring")
        ax.set_ylabel("Probability")
        ax.set_title(f"Top-{top_k} Output Probabilities")
        ax.legend(
            handles=[Patch(color="lime", label="Feasible"),
                     Patch(color="tomato", label="Infeasible")],
            fontsize=8,
        )
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        return fig

    def panel_metadata(self, figsize=(3, 4)) -> plt.Figure:
        """Text panel showing QUBO/QAOA config and workload weights."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        lines = [
            "=== QUBO CONFIG ===",
            f"Penalty Weight (P): {self.qubo_cfg.penalty}",
            f"Target Cores: {self.qubo_cfg.num_cores}",
            f"Variables (Qubits): {self.qubo.num_variables}",
            f"Entities (Procs): {self.qubo.num_entities}",
            "",
            "=== QAOA CONFIG ===",
            f"Layers (p): {self.qaoa_cfg.layers}",
            f"Max Steps: {self.qaoa_cfg.steps}",
            f"Learning Rate: {self.qaoa_cfg.optimizer_step}",
        ]

        if self.energies_over_time:
            lines.append(f"Actual Iterations: {len(self.energies_over_time)}")

        if hasattr(self.qubo_cfg, "snapshot") and self.qubo_cfg.snapshot is not None:
            lines += ["", "=== WORKLOAD WEIGHTS ==="]
            for proc in self.qubo_cfg.snapshot.processes:
                lines.append(f"PID {proc.pid}: {proc.cpu_weight:.3f}")

        ax.text(
            0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=9,
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.15),
        )
        fig.tight_layout()
        return fig

    # ──────────────────────────────────────────────────────────────────────────
    #  Composite — all panels in one figure (replaces the old __init__ layout)
    # ──────────────────────────────────────────────────────────────────────────

    def composite(self, save_path: str = None) -> plt.Figure:
        """
        Assembles all available panels into a single figure using GridSpec,
        identical layout to the original class.  Returns the Figure.

        Args:
            save_path: if provided, saves to this path at 300 dpi.
        """
        has_convergence = self.energies_over_time is not None
        has_probs       = self.probs is not None
        num_rows        = 1 + int(has_convergence or has_probs)

        fig = plt.figure(figsize=(18, 5 * num_rows))
        gs  = gridspec.GridSpec(
            num_rows, 3,
            figure=fig,
            width_ratios=[1, 1, 0.4],
            hspace=0.4,
            wspace=0.3,
        )

        # Panel 1 — energy landscape
        self._draw_landscape(fig.add_subplot(gs[0, 0]))

        # Panel 2 — QUBO matrix
        self._draw_qubo_matrix(fig.add_subplot(gs[0, 1]))

        # Panel 3 — convergence
        if has_convergence:
            self._draw_convergence(fig.add_subplot(gs[1, 0]))

        # Panel 4 — probabilities
        if has_probs:
            self._draw_probabilities(fig.add_subplot(gs[1, 1]))

        # Panel 5 — metadata (spans all rows)
        self._draw_metadata(fig.add_subplot(gs[:, 2]))

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    # ──────────────────────────────────────────────────────────────────────────
    #  Static sweep plot (no instance data needed)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def plot_sweep(penalty_results: list[dict], save_path: str = None) -> plt.Figure:
        """
        Penalty sweep summary chart.

        Args:
            penalty_results: list of dicts with keys p, alpha, max_p, feasible.
            save_path:       optional path to save the figure.
        """
        ps     = [r["p"]       for r in penalty_results]
        alphas = [r["alpha"]   for r in penalty_results]
        max_ps = [r["max_p"]   for r in penalty_results]
        feas   = [r["feasible"] for r in penalty_results]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color_alpha = "tab:blue"
        ax1.set_xlabel("Penalty Weight (P)")
        ax1.set_ylabel("Approximation Ratio (Alpha)", color=color_alpha)
        ax1.plot(ps, alphas, marker="o", color=color_alpha, linewidth=2, label="Alpha")
        ax1.tick_params(axis="y", labelcolor=color_alpha)
        ax1.grid(True, linestyle="--", alpha=0.6)

        ax2 = ax1.twinx()
        color_p = "tab:orange"
        ax2.set_ylabel("Max Output Probability (Confidence)", color=color_p)
        ax2.plot(ps, max_ps, marker="s", color=color_p, linewidth=2, label="Max Prob")
        ax2.tick_params(axis="y", labelcolor=color_p)

        for i, is_f in enumerate(feas):
            c = "green" if is_f else "red"
            ax1.plot(ps[i], alphas[i], marker="o", markersize=12, color=c, alpha=0.3)

        plt.title("QAOA Sensitivity to Penalty Weight (P)")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    # ──────────────────────────────────────────────────────────────────────────
    #  Private draw helpers — draw onto a provided Axes (used by composite())
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_landscape(self, ax):
        num_vars     = self.qubo.num_variables
        num_entities = self.qubo.num_entities
        num_cores    = self.qubo.num_cores

        all_states = list(product([0, 1], repeat=num_vars))
        energies, feasibility = [], []
        for state in all_states:
            x = np.array(state)
            energies.append(float(x.T @ self.qubo.Q @ x))
            feasibility.append(
                all(sum(state[i * num_cores:(i + 1) * num_cores]) == 1
                    for i in range(num_entities))
            )

        energies    = np.array(energies)
        feasibility = np.array(feasibility)
        indices     = np.arange(len(energies))

        ax.scatter(indices[~feasibility], energies[~feasibility],
                   c="red", s=2, alpha=0.3, label="Infeasible")
        ax.scatter(indices[feasibility], energies[feasibility],
                   c="lime", s=20, edgecolors="black", linewidths=0.5, label="Feasible")
        ax.set_title("Energy Landscape")
        ax.set_xlabel("State Index")
        ax.set_ylabel("Energy  xᵀQx")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _draw_qubo_matrix(self, ax):
        im = ax.imshow(self.qubo.Q, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        K = self.qubo.num_cores
        for i in range(1, self.qubo.num_entities):
            ax.axhline(i * K - 0.5, color="black", linewidth=1.5)
            ax.axvline(i * K - 0.5, color="black", linewidth=1.5)
        ax.set_title("QUBO Matrix  Q")
        ax.set_xlabel("Variable index")
        ax.set_ylabel("Variable index")

    def _draw_convergence(self, ax):
        ax.plot(self.energies_over_time, label="⟨C⟩ per step", color="steelblue")
        if self.global_optimum is not None:
            ax.axhline(
                self.global_optimum,
                color="lime", linestyle="--",
                label=f"Global optimum ({self.global_optimum:.3f})",
            )
        ax.set_title("QAOA Convergence")
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Expected Energy  ⟨C⟩")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _draw_probabilities(self, ax):
        num_vars = self.qubo.num_variables
        probs    = np.array(self.probs)
        top_k    = min(self.top_k, len(probs))

        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs   = probs[top_indices]

        labels, colors = [], []
        for idx in top_indices:
            bits = tuple(int(b) for b in bin(idx)[2:].zfill(num_vars))
            feasible = all(
                sum(bits[i * self.qubo.num_cores:(i + 1) * self.qubo.num_cores]) == 1
                for i in range(self.qubo.num_entities)
            )
            labels.append(bin(idx)[2:].zfill(num_vars))
            colors.append("lime" if feasible else "tomato")

        ax.bar(range(top_k), top_probs, color=colors)
        ax.set_xticks(range(top_k))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_xlabel("Bitstring")
        ax.set_ylabel("Probability")
        ax.set_title(f"Top-{top_k} Output Probabilities")
        ax.legend(
            handles=[Patch(color="lime", label="Feasible"),
                     Patch(color="tomato", label="Infeasible")],
            fontsize=8,
        )
        ax.grid(True, alpha=0.3, axis="y")

    def _draw_metadata(self, ax):
        ax.axis("off")
        lines = [
            "=== QUBO CONFIG ===",
            f"Penalty Weight (P): {self.qubo_cfg.penalty}",
            f"Target Cores: {self.qubo_cfg.num_cores}",
            f"Variables (Qubits): {self.qubo.num_variables}",
            f"Entities (Procs): {self.qubo.num_entities}",
            "",
            "=== QAOA CONFIG ===",
            f"Layers (p): {self.qaoa_cfg.layers}",
            f"Max Steps: {self.qaoa_cfg.steps}",
            f"Learning Rate: {self.qaoa_cfg.optimizer_step}",
        ]
        if self.energies_over_time:
            lines.append(f"Actual Iterations: {len(self.energies_over_time)}")
        if hasattr(self.qubo_cfg, "snapshot") and self.qubo_cfg.snapshot is not None:
            lines += ["", "=== WORKLOAD WEIGHTS ==="]
            for proc in self.qubo_cfg.snapshot.processes:
                lines.append(f"PID {proc.pid}: {proc.cpu_weight:.3f}")
        ax.text(
            0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=9,
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.15),
        )