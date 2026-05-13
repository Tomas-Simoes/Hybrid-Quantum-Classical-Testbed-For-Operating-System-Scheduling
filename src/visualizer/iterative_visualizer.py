import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import List
from data_contracts import QAOAConfig, QUBOConfig, QUBOInstance, SolverResult, Workload


class IterativeVisualizer:
    def __init__(
        self,
        solver_results: List[SolverResult],
        phi_history: List[np.ndarray],   # phi after each sub-QUBO, shape (M, K)
        workload: Workload,
        qubo_instance: QUBOInstance,      # global Q for matrix panel
        qaoa_cfg: QAOAConfig,
        qubo_cfg: QUBOConfig,
        top_k: int = 16,
    ):
        self.solver_results = solver_results
        self.phi_history    = phi_history
        self.workload       = workload
        self.qubo           = qubo_instance
        self.qaoa_cfg       = qaoa_cfg
        self.qubo_cfg       = qubo_cfg
        self.top_k          = top_k
        self.M              = len(solver_results)   # number of sub-QUBOs
        self.K              = workload.num_cores

    def panel_load_balance(self, figsize=(6, 4)) -> plt.Figure:
        """
        Bar chart of final core loads vs L_avg.
        The primary summary panel for the iterative run.
        """
        L_avg      = self.workload.total_weight / self.K
        final_phi  = self.phi_history[-1]
        cores      = [f"Core {k}" for k in range(self.K)]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(cores, final_phi, color="steelblue", label="Assigned load")
        ax.axhline(L_avg, color="lime", linestyle="--", linewidth=2,
                   label=f"L_avg = {L_avg:.4f}")

        for bar, val in zip(bars, final_phi):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

        imbalance = final_phi.max() - final_phi.min()
        ax.set_title(f"Final Core Load Balance  (imbalance = {imbalance:.4f})")
        ax.set_ylabel("Total CPU Weight")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        return fig

    def panel_phi_evolution(self, figsize=(6, 4)) -> plt.Figure:
        """
        Line plot of phi_k across sub-QUBO iterations — shows how
        loads accumulated and whether the steering term kept balance.
        """
        phi_arr = np.array(self.phi_history)   # shape (M, K)
        L_avg   = self.workload.total_weight / self.K

        fig, ax = plt.subplots(figsize=figsize)
        for k in range(self.K):
            ax.plot(range(1, self.M + 1), phi_arr[:, k],
                    marker="o", label=f"Core {k}")

        ax.axhline(L_avg, color="black", linestyle="--",
                   linewidth=1.5, label=f"L_avg = {L_avg:.4f}")
        ax.set_title("Core Load Accumulation Across Sub-QUBOs")
        ax.set_xlabel("Sub-QUBO Index")
        ax.set_ylabel("Accumulated φ_k")
        ax.set_xticks(range(1, self.M + 1))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def panel_convergence_grid(self, figsize=None) -> plt.Figure:
        """
        One convergence subplot per sub-QUBO, tiled in a row.
        """
        figsize = figsize or (5 * self.M, 3)
        fig, axes = plt.subplots(1, self.M, figsize=figsize, sharey=False)
        if self.M == 1:
            axes = [axes]

        for t, (result, ax) in enumerate(zip(self.solver_results, axes)):
            ax.plot(result.convergence_curve, color="steelblue")
            ax.set_title(f"Sub-QUBO {t+1}\n"
                         f"feasible={result.is_feasible} | "
                         f"E={result.energy:.3f}")
            ax.set_xlabel("Step")
            ax.set_ylabel("⟨C⟩" if t == 0 else "")
            ax.grid(True, alpha=0.3)

        fig.suptitle("QAOA Convergence per Sub-QUBO", fontsize=12, y=1.02)
        fig.tight_layout()
        return fig

    def panel_probabilities_grid(self, figsize=None) -> plt.Figure:
        """
        Top-k probability bar chart per sub-QUBO, tiled in a row.
        Feasible bitstrings in lime, infeasible in tomato.
        """
        from matplotlib.patches import Patch

        figsize = figsize or (5 * self.M, 4)
        fig, axes = plt.subplots(1, self.M, figsize=figsize, sharey=False)
        if self.M == 1:
            axes = [axes]

        for t, (result, ax) in enumerate(zip(self.solver_results, axes)):
            probs      = np.array(result.probs)
            num_vars   = self.qubo.num_cores * (len(probs).bit_length() - 1)
            # num_vars for this sub-QUBO = log2(len(probs))
            sub_num_vars = int(np.log2(len(probs)))
            num_entities_in_sub = sub_num_vars // self.K

            top_k       = min(self.top_k, len(probs))
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_probs   = probs[top_indices]

            labels, colors = [], []
            for idx in top_indices:
                bits     = tuple(int(b) for b in bin(idx)[2:].zfill(sub_num_vars))
                feasible = all(
                    sum(bits[i * self.K:(i + 1) * self.K]) == 1
                    for i in range(num_entities_in_sub)
                )
                labels.append(bin(idx)[2:].zfill(sub_num_vars))
                colors.append("lime" if feasible else "tomato")

            ax.bar(range(top_k), top_probs, color=colors)
            ax.set_xticks(range(top_k))
            ax.set_xticklabels(labels, rotation=90, fontsize=5)
            ax.set_title(f"Sub-QUBO {t+1} Probabilities")
            ax.set_xlabel("Bitstring")
            ax.set_ylabel("Probability" if t == 0 else "")
            ax.legend(handles=[
                Patch(color="lime",   label="Feasible"),
                Patch(color="tomato", label="Infeasible"),
            ], fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        return fig

    def panel_qubo_matrix(self, figsize=(5, 4)) -> plt.Figure:
        """Global Q matrix — same as the original visualizer."""
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(self.qubo.Q, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(1, self.qubo.num_entities):
            ax.axhline(i * self.K - 0.5, color="black", linewidth=1.5)
            ax.axvline(i * self.K - 0.5, color="black", linewidth=1.5)
        ax.set_title("Global QUBO Matrix  Q")
        ax.set_xlabel("Variable index")
        ax.set_ylabel("Variable index")
        fig.tight_layout()
        return fig

    def composite(self, save_path: str = None) -> plt.Figure:
        """
        Layout:
          Row 0: [load balance] [phi evolution] [Q matrix]
          Row 1: [convergence grid — spans all columns]
          Row 2: [probabilities grid — spans all columns]
        """
        fig = plt.figure(figsize=(18, 14))
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

        # Row 0
        self._draw_load_balance(fig.add_subplot(gs[0, 0]))
        self._draw_phi_evolution(fig.add_subplot(gs[0, 1]))
        self._draw_qubo_matrix(fig.add_subplot(gs[0, 2]))

        # Row 1 — convergence grid spanning full width
        conv_ax = fig.add_subplot(gs[1, :])
        self._draw_convergence_grid_into(conv_ax)

        # Row 2 — probability grid spanning full width
        prob_ax = fig.add_subplot(gs[2, :])
        self._draw_probabilities_grid_into(prob_ax)

        fig.suptitle(
            f"Iterative QAOA Sub-QUBO Pipeline  |  "
            f"{self.workload.num_cores} cores  |  "
            f"{len(self.workload.entities)} entities  |  "
            f"{self.M} sub-QUBOs",
            fontsize=13, y=1.01,
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    # ── private draw helpers ──────────────────────────────────────────────────

    def _draw_load_balance(self, ax):
        L_avg     = self.workload.total_weight / self.K
        final_phi = self.phi_history[-1]
        cores     = [f"Core {k}" for k in range(self.K)]
        bars = ax.bar(cores, final_phi, color="steelblue")
        ax.axhline(L_avg, color="lime", linestyle="--", linewidth=2,
                   label=f"L_avg={L_avg:.4f}")
        for bar, val in zip(bars, final_phi):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        imbalance = final_phi.max() - final_phi.min()
        ax.set_title(f"Final Load Balance\nimbalance={imbalance:.4f}")
        ax.set_ylabel("CPU Weight")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    def _draw_phi_evolution(self, ax):
        phi_arr = np.array(self.phi_history)
        L_avg   = self.workload.total_weight / self.K
        for k in range(self.K):
            ax.plot(range(1, self.M + 1), phi_arr[:, k], marker="o", label=f"Core {k}")
        ax.axhline(L_avg, color="black", linestyle="--", linewidth=1.5,
                   label=f"L_avg={L_avg:.4f}")
        ax.set_title("φ Accumulation Across Sub-QUBOs")
        ax.set_xlabel("Sub-QUBO")
        ax.set_ylabel("φ_k")
        ax.set_xticks(range(1, self.M + 1))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _draw_qubo_matrix(self, ax):
        im = ax.imshow(self.qubo.Q, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(1, self.qubo.num_entities):
            ax.axhline(i * self.K - 0.5, color="black", linewidth=1.5)
            ax.axvline(i * self.K - 0.5, color="black", linewidth=1.5)
        ax.set_title("Global Q Matrix")

    def _draw_convergence_grid_into(self, ax):
        """Tile M convergence curves as inset axes inside a single Axes."""
        ax.axis("off")
        ax.set_title("QAOA Convergence per Sub-QUBO", fontsize=10, pad=2)
        fig = ax.get_figure()
        bbox = ax.get_position()
        w = bbox.width / self.M
        for t, result in enumerate(self.solver_results):
            inset = fig.add_axes([
                bbox.x0 + t * w + 0.01,
                bbox.y0 + 0.02,
                w - 0.02,
                bbox.height - 0.06,
            ])
            inset.plot(result.convergence_curve, color="steelblue")
            inset.set_title(f"SQ{t+1} E={result.energy:.2f}\n"
                            f"{'✓' if result.is_feasible else '✗'}", fontsize=8)
            inset.grid(True, alpha=0.3)
            if t == 0:
                inset.set_ylabel("⟨C⟩", fontsize=7)

    def _draw_probabilities_grid_into(self, ax):
        from matplotlib.patches import Patch
        ax.axis("off")
        ax.set_title("Top-k Probabilities per Sub-QUBO", fontsize=10, pad=2)
        fig = ax.get_figure()
        bbox = ax.get_position()
        w = bbox.width / self.M
        for t, result in enumerate(self.solver_results):
            inset = fig.add_axes([
                bbox.x0 + t * w + 0.01,
                bbox.y0 + 0.02,
                w - 0.02,
                bbox.height - 0.06,
            ])
            probs        = np.array(result.probs)
            sub_num_vars = int(np.log2(len(probs)))
            n_entities   = sub_num_vars // self.K
            top_k        = min(self.top_k, len(probs))
            top_indices  = np.argsort(probs)[-top_k:][::-1]