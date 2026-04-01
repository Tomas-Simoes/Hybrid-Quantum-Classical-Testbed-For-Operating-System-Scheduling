from typing import List
import matplotlib

from config_contracts import QAOAConfig, QUBOConfig
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product

import numpy as np

class Visualization:
    def __init__(
        self,
        qubo,
        qaoa_cfg: QAOAConfig,
        qubo_cfg: QUBOConfig,
        probs=None,
        energies_over_time=None,
        global_optimum=None,
        top_k: int = 20,
        save_path: str = None,
    ):
        """
        Opens a single window with all available visualizations.

        Args:
            qubo:                 QUBOInstance — always required.
            probs:                np.ndarray of QAOA output probabilities (optional).
                                  If None, the probability distribution panel is skipped.
            energies_over_time:   list of floats from the optimization loop (optional).
                                  If None, the convergence panel is skipped.
            global_optimum:       float — draws a reference line on the convergence plot (optional).
            top_k:                how many bitstrings to show in the probability panel.
            save_path:            if provided, also saves the figure to this path.
        """
        self.qubo = qubo
        self.probs = probs
        self.energies_over_time = energies_over_time
        self.global_optimum = global_optimum
        self.top_k = top_k
        self.qaoa_cfg = qaoa_cfg
        self.qubo_cfg = qubo_cfg

        has_convergence = energies_over_time is not None
        has_probs = probs is not None

        # Increase width to accommodate the Metadata Panel on the right
        num_rows = 1 + int(has_convergence or has_probs)
        fig = plt.figure(figsize=(18, 5 * num_rows))
        
        # GridSpec: 2 main columns for plots, 1 narrow column for metadata
        gs = gridspec.GridSpec(
            num_rows, 3,
            figure=fig,
            width_ratios=[1, 1, 0.4],
            hspace=0.4,
            wspace=0.3,
        )

        # --- Panel 1: Energy Landscape ---
        ax_landscape = fig.add_subplot(gs[0, 0])
        self._plot_energy_landscape(ax_landscape)

        # --- Panel 2: QUBO Matrix ---
        ax_matrix = fig.add_subplot(gs[0, 1])
        self._plot_qubo_matrix(ax_matrix)

        # --- Panel 3: Convergence ---
        if has_convergence:
            ax_conv = fig.add_subplot(gs[1, 0])
            self._plot_convergence(ax_conv)

        # --- Panel 4: Probabilities ---
        if has_probs:
            ax_probs = fig.add_subplot(gs[1, 1])
            self._plot_probability_distribution(ax_probs)

        # --- Panel 5: Metadata Info (Spans all rows) ---
        ax_info = fig.add_subplot(gs[:, 2])
        self._plot_metadata_panel(ax_info)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        #plt.show(block=False)
        plt.pause(0.1)

    def _plot_metadata_panel(self, ax):
        """Displays all parameters from QAOA and QUBO configurations."""
        ax.axis('off')
        
        info_text = []
        
        # --- QUBO Configuration ---
        info_text.append("=== QUBO CONFIG ===")
        info_text.append(f"Penalty Weight (P): {self.qubo_cfg.penalty}")
        info_text.append(f"Target Cores: {self.qubo_cfg.num_cores}")
        # Derived from the QUBO instance
        info_text.append(f"Variables (Qubits): {self.qubo.num_variables}")
        info_text.append(f"Entities (Procs): {self.qubo.num_entities}")
        info_text.append("")

        # --- QAOA Configuration ---
        info_text.append("=== QAOA CONFIG ===")
        info_text.append(f"Layers (p): {self.qaoa_cfg.layers}")
        info_text.append(f"Max Steps: {self.qaoa_cfg.steps}")
        info_text.append(f"Learning Rate: {self.qaoa_cfg.optimizer_step}")
        if self.energies_over_time:
            info_text.append(f"Actual Iterations: {len(self.energies_over_time)}")
        info_text.append("")

        # --- Process Weights (Live Data) ---
        if hasattr(self.qubo_cfg.snapshot, 'processes'):
            info_text.append("=== WORKLOAD WEIGHTS ===")
            for i, proc in enumerate(self.qubo_cfg.snapshot.processes):
                # Showing PID and CPU Weight
                info_text.append(f"PID {proc.pid}: {proc.cpu_weight:.3f}")
        
        # Render onto the Axis
        ax.text(
            0.05, 0.95, "\n".join(info_text),
            transform=ax.transAxes,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=9,
            linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.15)
        )
        
    def _plot_run_results(penalty_results):
        # --- Plotting ---
        ps = [r['p'] for r in penalty_results]
        alphas = [r['alpha'] for r in penalty_results]
        max_ps = [r['max_p'] for r in penalty_results]
        feas = [r['feasible'] for r in penalty_results]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Axis 1: Alpha (Approximation Ratio)
        color_alpha = 'tab:blue'
        ax1.set_xlabel('Penalty Weight (P)')
        ax1.set_ylabel('Approximation Ratio (Alpha)', color=color_alpha)
        ax1.plot(ps, alphas, marker='o', color=color_alpha, linewidth=2, label='Alpha')
        ax1.tick_params(axis='y', labelcolor=color_alpha)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Axis 2: Confidence (Max Probability)
        ax2 = ax1.twinx()
        color_p = 'tab:orange'
        ax2.set_ylabel('Max Output Probability (Confidence)', color=color_p)
        ax2.plot(ps, max_ps, marker='s', color=color_p, linewidth=2, label='Max Prob')
        ax2.tick_params(axis='y', labelcolor=color_p)

        # Feasibility Highlight
        for i, is_f in enumerate(feas):
            color = 'green' if is_f else 'red'
            ax1.plot(ps[i], alphas[i], marker='o', markersize=12, color=color, alpha=0.3)

        plt.title('QAOA Sensitivity to Penalty Weight (P)')
        fig.tight_layout()
        plt.savefig("results/penalty_sweep_analysis.png")
        plt.show()

    def _plot_energy_landscape(self, ax):
        num_vars = self.qubo.num_variables
        num_entities = self.qubo.num_entities
        num_cores = self.qubo.num_cores

        all_states = list(product([0, 1], repeat=num_vars))
        energies, feasibility = [], []

        for state in all_states:
            x = np.array(state)
            energy = float(x.T @ self.qubo.Q @ x)
            energies.append(energy)
            is_feasible = all(
                sum(state[i * num_cores: (i + 1) * num_cores]) == 1
                for i in range(num_entities)
            )
            feasibility.append(is_feasible)

        energies = np.array(energies)
        feasibility = np.array(feasibility)
        indices = np.arange(len(energies))

        ax.scatter(
            indices[~feasibility], energies[~feasibility],
            c="red", s=2, alpha=0.3, label="Infeasible",
        )
        ax.scatter(
            indices[feasibility], energies[feasibility],
            c="lime", s=20, edgecolors="black", linewidths=0.5,
            label="Feasible",
        )
        ax.set_title("Energy Landscape")
        ax.set_xlabel("State Index")
        ax.set_ylabel("Energy  xᵀQx")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_qubo_matrix(self, ax):
        im = ax.imshow(self.qubo.Q, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        K = self.qubo.num_cores
        for i in range(1, self.qubo.num_entities):
            ax.axhline(i * K - 0.5, color="black", linewidth=1.5)
            ax.axvline(i * K - 0.5, color="black", linewidth=1.5)

        ax.set_title("QUBO Matrix  Q")
        ax.set_xlabel("Variable index")
        ax.set_ylabel("Variable index")

    def _plot_convergence(self, ax):
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

    def _plot_probability_distribution(self, ax):
        num_vars = self.qubo.num_variables
        probs = np.array(self.probs)
        top_k = min(self.top_k, len(probs))

        print(probs)

        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_indices]

        labels, colors = [], []
        for idx in top_indices:
            bits = tuple(int(b) for b in bin(idx)[2:].zfill(num_vars))
            is_feasible = all(
                sum(bits[i * self.qubo.num_cores: (i + 1) * self.qubo.num_cores]) == 1
                for i in range(self.qubo.num_entities)
            )
            labels.append(bin(idx)[2:].zfill(num_vars))
            colors.append("lime" if is_feasible else "tomato")

        ax.bar(range(top_k), top_probs, color=colors)
        ax.set_xticks(range(top_k))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_xlabel("Bitstring")
        ax.set_ylabel("Probability")
        ax.set_title(f"Top-{top_k} Output Probabilities")

        # Legend patch
        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(color="lime", label="Feasible"),
                Patch(color="tomato", label="Infeasible"),
            ],
            fontsize=8,
        )
        ax.grid(True, alpha=0.3, axis="y")