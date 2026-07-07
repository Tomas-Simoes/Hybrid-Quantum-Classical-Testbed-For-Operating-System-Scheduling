from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
PHASE4_DIR = RESULTS_DIR / "phase4"
HISTORICAL_D2 = RESULTS_DIR / "sweep_20260626_153039_2c23dc71_results.jsonl"

COLORS = {
    1: "#D05A3A",
    3: "#E2A72E",
    5: "#176B87",
    10: "#24523B",
}
TIMING_COMPONENTS = (
    "workload_ms",
    "clustering_ms",
    "decomposition_ms",
    "qubo_build_ms",
    "qaoa_ms",
    "reconstruction_ms",
    "operational_validation_ms",
)


def load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    record["_source_file"] = path.name
                    records.append(record)
    return records


def get_path(data: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := finite(value)) is not None]
    return statistics.fmean(numbers) if numbers else None


def std(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := finite(value)) is not None]
    if not numbers:
        return None
    return statistics.stdev(numbers) if len(numbers) > 1 else 0.0


def rate(values: Iterable[Any]) -> float | None:
    observed = [value for value in values if value is not None]
    return sum(bool(value) for value in observed) / len(observed) if observed else None


def experiment(record: dict[str, Any]) -> str:
    return str(record.get("scenario_id", "unknown")).split(".", maxsplit=1)[0]


def sweep_value(record: dict[str, Any], name: str) -> Any:
    values = get_path(record, "config.sweep_context.values", {})
    return values.get(name) if isinstance(values, dict) else None


def direct_load_imbalance(record: dict[str, Any]) -> float | None:
    measured = finite(get_path(record, "metrics.load_imbalance"))
    if measured is not None:
        return measured

    weights = get_path(record, "config.workload.weights")
    assignments = get_path(record, "metrics.assignments")
    num_cores = get_path(record, "metrics.num_cores")
    if not weights or not isinstance(assignments, dict) or not num_cores:
        return None

    loads = [0.0] * int(num_cores)
    for index, weight in enumerate(weights):
        core = assignments.get(str(1000 + index), assignments.get(1000 + index))
        if core is not None:
            loads[int(core)] += float(weight)
    return max(loads) - min(loads)


def cluster_purity(clustering: dict[str, Any] | None) -> float | None:
    if not clustering:
        return None
    correct = 0
    total = 0
    for bundle in clustering.get("bundles", []):
        commands = bundle.get("member_commands", [])
        counts = {
            "cpu": sum(str(command).startswith("cpu_bound") for command in commands),
            "io": sum(str(command).startswith("io_bound") for command in commands),
        }
        correct += max(counts.values(), default=0)
        total += len(commands)
    return correct / total if total else None


def row_from_record(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") or {}
    validation = metrics.get("validation") or {}
    timings = metrics.get("component_timings_ms") or {}
    metadata = get_path(record, "config.metadata", {}) or {}
    clustering = get_path(metrics, "experiment_metadata.clustering")

    structure = metadata.get("structure")
    if structure is None and experiment(record) == "E3U":
        structure = "uniform"
    difficulty = metadata.get("difficulty")
    if difficulty is None:
        difficulty = metadata.get("dominant_weight", metadata.get("max_weight_difference"))

    brute_force_ms = finite(validation.get("brute_force_solve_time_ms")) or 0.0
    annealing_ms = finite(validation.get("annealing_solve_time_ms")) or 0.0
    validation_ms = finite(timings.get("validation_ms")) or 0.0
    operational_validation_ms = max(0.0, validation_ms - brute_force_ms - annealing_ms)
    energy = finite(metrics.get("energy"))
    brute_force_energy = finite(validation.get("global_energy"))
    sa_energy = finite(validation.get("annealing_energy"))
    absolute_gap = (
        None
        if energy is None or brute_force_energy is None
        else energy - brute_force_energy
    )
    strict_optimal = (
        None if absolute_gap is None else abs(absolute_gap) <= 1e-9
    )
    sa_strict_optimal = (
        None
        if sa_energy is None or brute_force_energy is None
        else abs(sa_energy - brute_force_energy) <= 1e-9
    )
    qaoa_sa_agreement = (
        None
        if energy is None or sa_energy is None
        else abs(energy - sa_energy) <= 1e-9
    )

    row = {
        "source_file": record.get("_source_file"),
        "scenario_id": record.get("scenario_id"),
        "experiment": experiment(record),
        "status": record.get("status"),
        "structure": structure,
        "instance": metadata.get("instance"),
        "difficulty": difficulty,
        "input_num_processes": get_path(record, "config.workload.num_processes"),
        "num_entities": metrics.get("num_entities"),
        "num_variables": metrics.get("num_variables"),
        "num_cores": metrics.get("num_cores"),
        "weights": json.dumps(get_path(record, "config.workload.weights")),
        "pipeline": metrics.get("pipeline"),
        "qubit_max": get_path(record, "config.decomposition.qubit_max"),
        "layers": get_path(record, "config.qaoa.layers"),
        "steps": get_path(record, "config.qaoa.steps"),
        "top_k": get_path(record, "config.qaoa.top_k"),
        "random_seed": get_path(record, "config.qaoa.random_seed"),
        "io_alpha": get_path(record, "config.decomposition.io_alpha"),
        "energy": metrics.get("energy"),
        "brute_force_energy": validation.get("global_energy"),
        "sa_energy": validation.get("annealing_energy"),
        "feasible": metrics.get("feasible"),
        "recorded_optimal": metrics.get("optimal"),
        "strict_optimal": strict_optimal,
        "optimal": strict_optimal,
        "sa_strict_optimal": sa_strict_optimal,
        "optimality_gap": metrics.get("optimality_gap"),
        "absolute_gap": absolute_gap,
        "recorded_qaoa_sa_agreement": metrics.get("sa_energy_match"),
        "qaoa_sa_agreement": qaoa_sa_agreement,
        "load_imbalance": direct_load_imbalance(record),
        "solve_time_ms": metrics.get("solve_time_ms"),
        "convergence_iterations": metrics.get("convergence_iterations_to_final_tol"),
        "max_subqubo_convergence_iterations": metrics.get(
            "max_subqubo_convergence_iterations_to_final_tol"
        ),
        "num_sub_qubos": metrics.get("num_sub_qubos"),
        "brute_force_solve_time_ms": validation.get("brute_force_solve_time_ms"),
        "annealing_solve_time_ms": validation.get("annealing_solve_time_ms"),
        "workload_ms": timings.get("workload_ms"),
        "clustering_ms": timings.get("clustering_ms"),
        "decomposition_ms": timings.get("decomposition_ms"),
        "qubo_build_ms": timings.get("qubo_build_ms"),
        "qaoa_ms": timings.get("qaoa_ms"),
        "reconstruction_ms": timings.get("reconstruction_ms"),
        "validation_ms": timings.get("validation_ms"),
        "operational_validation_ms": operational_validation_ms,
        "experimental_baselines_ms": brute_force_ms + annealing_ms,
        "cluster_num_bundles": None if not clustering else clustering.get("num_bundles"),
        "cluster_purity": cluster_purity(clustering),
        "intra_cluster_coupling_mean": get_path(
            clustering or {}, "intra_cluster_coupling.mean"
        ),
        "inter_cluster_coupling_mean": get_path(
            clustering or {}, "inter_cluster_coupling.mean"
        ),
        "cluster_bundles": json.dumps(
            [] if not clustering else clustering.get("bundles", []), sort_keys=True
        ),
    }
    row["operational_total_ms"] = sum(
        finite(row.get(component)) or 0.0 for component in TIMING_COMPONENTS
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in fields)].append(row)

    summaries = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        summary = {field: value for field, value in zip(fields, key)}
        summary.update(
            {
                "runs": len(group),
                "success_rate": rate(row.get("status") == "success" for row in group),
                "feasibility_rate": rate(row.get("feasible") for row in group),
                "optimality_rate": rate(row.get("optimal") for row in group),
                "recorded_optimality_rate": rate(
                    row.get("recorded_optimal") for row in group
                ),
                "sa_optimality_rate": rate(
                    row.get("sa_strict_optimal") for row in group
                ),
                "qaoa_sa_agreement_rate": rate(
                    row.get("qaoa_sa_agreement") for row in group
                ),
                "mean_energy": mean(row.get("energy") for row in group),
                "mean_optimality_gap": mean(row.get("optimality_gap") for row in group),
                "max_optimality_gap": max(
                    (finite(row.get("optimality_gap")) or 0.0 for row in group),
                    default=None,
                ),
                "mean_absolute_gap": mean(row.get("absolute_gap") for row in group),
                "mean_load_imbalance": mean(row.get("load_imbalance") for row in group),
                "mean_solve_time_ms": mean(row.get("solve_time_ms") for row in group),
                "std_solve_time_ms": std(row.get("solve_time_ms") for row in group),
                "mean_convergence_iterations": mean(
                    row.get("convergence_iterations") for row in group
                ),
                "mean_max_subqubo_convergence_iterations": mean(
                    row.get("max_subqubo_convergence_iterations") for row in group
                ),
            }
        )
        summaries.append(summary)
    return summaries


def e4_timing_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row["experiment"] == "E3U" and int(row["top_k"]) == 10
    ]
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[int(row["input_num_processes"])].append(row)

    summaries = []
    for num_processes, group in sorted(groups.items()):
        summary = {
            "num_processes": num_processes,
            "num_variables": int(group[0]["num_variables"]),
            "runs": len(group),
        }
        for component in TIMING_COMPONENTS:
            summary[component] = mean(row.get(component) for row in group) or 0.0
        summary["experimental_baselines_ms"] = mean(
            row.get("experimental_baselines_ms") for row in group
        ) or 0.0
        summary["operational_total_ms"] = sum(
            summary[component] for component in TIMING_COMPONENTS
        )
        summary["qaoa_share"] = (
            summary["qaoa_ms"] / summary["operational_total_ms"]
            if summary["operational_total_ms"]
            else None
        )
        summaries.append(summary)
    return summaries


def historical_d2_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        qmax = int(get_path(record, "config.decomposition.qubit_max"))
        groups[qmax].append(record)

    rows = []
    for qmax, group in sorted(groups.items()):
        rows.append(
            {
                "source": "D2_historical",
                "qubit_max": qmax,
                "pipeline": get_path(group[0], "metrics.pipeline"),
                "layers": get_path(group[0], "config.qaoa.layers"),
                "steps": get_path(group[0], "config.qaoa.steps"),
                "runs": len(group),
                "optimality_rate": rate(
                    abs(
                        float(get_path(record, "metrics.energy"))
                        - float(get_path(record, "metrics.validation.global_energy"))
                    ) <= 1e-9
                    for record in group
                ),
                "mean_optimality_gap": mean(
                    get_path(record, "metrics.optimality_gap") for record in group
                ),
                "mean_solve_time_ms": mean(
                    get_path(record, "metrics.solve_time_ms") for record in group
                ),
            }
        )
    return rows


def save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_e1(e1_summary: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, structure, title in (
        (axes[0], "dominant", "E1a: processo dominante"),
        (axes[1], "near_equal", "E1b: pesos quase iguais"),
    ):
        selected = [row for row in e1_summary if row["structure"] == structure]
        configs = sorted({(int(row["layers"]), int(row["top_k"])) for row in selected})
        for layers, top_k in configs:
            config_rows = sorted(
                [row for row in selected if int(row["layers"]) == layers and int(row["top_k"]) == top_k],
                key=lambda row: float(row["difficulty"]),
            )
            ax.plot(
                [float(row["difficulty"]) for row in config_rows],
                [100.0 * float(row["optimality_rate"]) for row in config_rows],
                marker="o",
                label=f"p={layers}, top_k={top_k}",
            )
        ax.set_title(title)
        ax.set_xlabel("Peso dominante" if structure == "dominant" else "Diferenca maxima")
        ax.set_ylabel("Taxa de otimo global (%)")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    path = output_dir / "e1_adversarial_optimality.png"
    return save(fig, path)


def plot_e1c(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    selected = [row for row in rows if row["experiment"] == "E1c"]
    groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[float(row["io_alpha"])].append(row)
    xs = sorted(groups)
    intra = [mean(row["intra_cluster_coupling_mean"] for row in groups[x]) for x in xs]
    inter = [mean(row["inter_cluster_coupling_mean"] for row in groups[x]) for x in xs]
    purity = [100.0 * (mean(row["cluster_purity"] for row in groups[x]) or 0.0) for x in xs]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(xs, intra, marker="o", label="Coupling intra-cluster", color="#176B87")
    ax.plot(xs, inter, marker="s", label="Coupling inter-cluster", color="#D05A3A")
    secondary = ax.twinx()
    secondary.plot(xs, purity, marker="^", linestyle="--", label="Pureza", color="#24523B")
    ax.set_xlabel("io_alpha")
    ax.set_ylabel("Afinidade media")
    secondary.set_ylabel("Pureza dos clusters (%)")
    secondary.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    lines = ax.lines + secondary.lines
    ax.legend(lines, [line.get_label() for line in lines], frameon=False)
    ax.set_title("E1c: separacao CPU-bound vs. I/O-bound")
    path = output_dir / "e1c_clustering_quality.png"
    return save(fig, path)


def plot_e2(e2_summary: list[dict[str, Any]], output_dir: Path) -> Path:
    layers = sorted({int(row["layers"]) for row in e2_summary})
    steps = sorted({int(row["steps"]) for row in e2_summary})
    optimality = np.zeros((len(layers), len(steps)))
    gaps = np.zeros_like(optimality)
    for row in e2_summary:
        i = layers.index(int(row["layers"]))
        j = steps.index(int(row["steps"]))
        optimality[i, j] = 100.0 * float(row["optimality_rate"])
        gaps[i, j] = 100.0 * float(row["mean_optimality_gap"] or 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, matrix, title, label, vmax in (
        (axes[0], optimality, "Taxa de otimo", "%", 100.0),
        (axes[1], gaps, "Gap relativo medio", "%", max(0.01, float(gaps.max()))),
    ):
        image = ax.imshow(matrix, origin="lower", cmap="YlGn" if label == "%" and title.startswith("Taxa") else "YlOrRd", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(steps)), steps)
        ax.set_yticks(range(len(layers)), layers)
        ax.set_xlabel("steps")
        ax.set_ylabel("p")
        ax.set_title(title)
        for i in range(len(layers)):
            for j in range(len(steps)):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=label)
    fig.suptitle("E2: recuperacao da pipeline directa de 16 variaveis")
    path = output_dir / "e2_direct_budget.png"
    return save(fig, path)


def plot_e2_tradeoff(
    e2_summary: list[dict[str, Any]], historical: list[dict[str, Any]], output_dir: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in e2_summary:
        ax.scatter(
            row["mean_solve_time_ms"],
            100.0 * float(row["mean_optimality_gap"] or 0.0),
            s=45 + 12 * int(row["layers"]),
            color="#176B87",
        )
        ax.annotate(
            f"p{row['layers']}/{row['steps']}",
            (row["mean_solve_time_ms"], 100.0 * float(row["mean_optimality_gap"] or 0.0)),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    decomposed = [row for row in historical if row["pipeline"] == "iterative"]
    ax.scatter(
        [row["mean_solve_time_ms"] for row in decomposed],
        [100.0 * float(row["mean_optimality_gap"] or 0.0) for row in decomposed],
        marker="s",
        color="#D05A3A",
        label="D2 decomposto historico",
    )
    ax.set_xlabel("Tempo QAOA medio (ms)")
    ax.set_ylabel("Gap relativo medio (%)")
    ax.set_title("E2: tradeoff tempo/qualidade vs. D2 decomposto")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    path = output_dir / "e2_tradeoff_vs_decomposition.png"
    return save(fig, path)


def plot_e3(e3_summary: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    uniform = [row for row in e3_summary if row["dataset"] == "uniform"]
    for top_k in sorted({int(row["top_k"]) for row in uniform}):
        selected = sorted(
            [row for row in uniform if int(row["top_k"]) == top_k],
            key=lambda row: int(row["input_num_processes"]),
        )
        axes[0].plot(
            [int(row["input_num_processes"]) for row in selected],
            [100.0 * float(row["optimality_rate"]) for row in selected],
            marker="o",
            color=COLORS[top_k],
            label=f"top_k={top_k}",
        )
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("Taxa de otimo global (%)")
    axes[0].set_ylim(-3, 103)
    axes[0].set_title("Workloads uniformes")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    adversarial = [row for row in e3_summary if row["dataset"] == "adversarial"]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in adversarial:
        grouped[(str(row["structure"]), int(row["top_k"]))].append(row)
    structures = ["dominant", "near_equal"]
    top_values = [1, 3, 5, 10]
    width = 0.19
    x = np.arange(len(structures))
    for offset, top_k in enumerate(top_values):
        values = [
            100.0 * (mean(row["optimality_rate"] for row in grouped[(structure, top_k)]) or 0.0)
            for structure in structures
        ]
        axes[1].bar(x + (offset - 1.5) * width, values, width, label=f"top_k={top_k}", color=COLORS[top_k])
    axes[1].set_xticks(x, ["Dominante", "Quase iguais"])
    axes[1].set_ylabel("Taxa media de otimo global (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Workloads adversariais")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("E3: generalizacao da sensibilidade a top_k")
    path = output_dir / "e3_top_k_generalization.png"
    return save(fig, path)


def plot_e4(timing_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    labels = [str(row["num_processes"]) for row in timing_rows]
    bottoms = np.zeros(len(timing_rows))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    component_labels = {
        "workload_ms": "Workload",
        "clustering_ms": "Clustering",
        "decomposition_ms": "Decomposicao",
        "qubo_build_ms": "Construcao QUBO",
        "qaoa_ms": "QAOA",
        "reconstruction_ms": "Reconstrucao",
        "operational_validation_ms": "Validacao operacional",
    }
    colors = ["#777777", "#8FA3AD", "#E2A72E", "#6A8EAE", "#176B87", "#24523B", "#D05A3A"]
    for component, color in zip(TIMING_COMPONENTS, colors):
        values = np.array([
            100.0 * row[component] / row["operational_total_ms"]
            if row["operational_total_ms"] else 0.0
            for row in timing_rows
        ])
        ax.bar(labels, values, bottom=bottoms, label=component_labels[component], color=color)
        bottoms += values
    ax.set_xlabel("Numero de processos N")
    ax.set_ylabel("Proporcao do tempo operacional (%)")
    ax.set_ylim(0, 100)
    ax.set_title("E4: decomposicao do tempo operacional (top_k=10)")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    path = output_dir / "e4_operational_timing_breakdown.png"
    return save(fig, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create Phase 4 scientific tables and plots.")
    parser.add_argument("jsonl", nargs="+", type=Path, help="Phase 4 aggregate JSONL files.")
    parser.add_argument("--historical-d2", type=Path, default=HISTORICAL_D2)
    parser.add_argument("--output-dir", type=Path, default=PHASE4_DIR / "analysis")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(path.resolve() for path in args.jsonl)
    rows = [row_from_record(record) for record in records]
    write_csv(output_dir / "phase4_runs.csv", rows)

    e1_rows = [row for row in rows if row["experiment"] in {"E1a", "E1b", "E1c"}]
    e1_summary = aggregate(
        e1_rows,
        ("experiment", "structure", "instance", "difficulty", "io_alpha", "layers", "top_k"),
    )
    write_csv(output_dir / "e1_adversarial_summary.csv", e1_summary)

    e2_rows = [row for row in rows if row["experiment"] == "E2"]
    e2_summary = aggregate(e2_rows, ("layers", "steps"))
    write_csv(output_dir / "e2_direct_budget_summary.csv", e2_summary)

    e3_input = []
    for row in rows:
        if row["experiment"] == "E3U":
            e3_input.append({**row, "dataset": "uniform"})
        elif row["experiment"] == "E3A" or (
            row["experiment"] in {"E1a", "E1b"} and int(row["layers"]) == 2
        ):
            e3_input.append({**row, "dataset": "adversarial"})
    e3_summary = aggregate(
        e3_input,
        ("dataset", "structure", "input_num_processes", "instance", "difficulty", "top_k"),
    )
    write_csv(output_dir / "e3_top_k_summary.csv", e3_summary)

    timing_summary = e4_timing_summary(rows)
    write_csv(output_dir / "e4_timing_summary.csv", timing_summary)

    historical_records = load_jsonl([args.historical_d2.resolve()])
    historical = historical_d2_summary(historical_records)
    e2_comparison = list(historical)
    e2_comparison.extend(
        {
            "source": "E2_phase4",
            "qubit_max": 16,
            "pipeline": "default",
            "layers": row["layers"],
            "steps": row["steps"],
            "runs": row["runs"],
            "optimality_rate": row["optimality_rate"],
            "mean_optimality_gap": row["mean_optimality_gap"],
            "mean_solve_time_ms": row["mean_solve_time_ms"],
        }
        for row in e2_summary
    )
    write_csv(output_dir / "e2_with_historical_d2.csv", e2_comparison)

    figures = [
        plot_e1(e1_summary, output_dir),
        plot_e1c(rows, output_dir),
        plot_e2(e2_summary, output_dir),
        plot_e2_tradeoff(e2_summary, historical, output_dir),
        plot_e3(e3_summary, output_dir),
        plot_e4(timing_summary, output_dir),
    ]

    manifest = {
        "phase": 4,
        "input_files": [str(path.resolve()) for path in args.jsonl],
        "input_records": len(records),
        "successful_records": sum(record.get("status") == "success" for record in records),
        "strict_optimality_tolerance": 1e-9,
        "recorded_optimal_records": sum(
            row.get("recorded_optimal") is True for row in rows
        ),
        "strict_optimal_records": sum(row.get("strict_optimal") is True for row in rows),
        "sa_strict_optimal_records": sum(
            row.get("sa_strict_optimal") is True for row in rows
        ),
        "tables": [
            "phase4_runs.csv",
            "e1_adversarial_summary.csv",
            "e2_direct_budget_summary.csv",
            "e2_with_historical_d2.csv",
            "e3_top_k_summary.csv",
            "e4_timing_summary.csv",
        ],
        "figures": [path.name for path in figures],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Loaded {len(records)} Phase 4 records.")
    print(f"Wrote analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
