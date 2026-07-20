from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS_DIR = Path(__file__).resolve().parent
SRC_DIR = EXPERIMENTS_DIR.parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from builder.builder_core import CoreAssignmentBuilder
from data_contracts import QUBOConfig, Workload, WorkloadEntity
from solver.brute_force_solver import BruteForceSolver
from investigative_runtime import AnnealingSolver


DEFAULT_INPUTS = (
    RESULTS_DIR / "run_20260626_152701_7a132ebd_results.jsonl",
    RESULTS_DIR / "sweep_20260626_152713_c32d7ee9_results.jsonl",
    RESULTS_DIR / "sweep_20260626_152916_694c56fc_results.jsonl",
    RESULTS_DIR / "sweep_20260626_153039_2c23dc71_results.jsonl",
    RESULTS_DIR / "sweep_20260626_163548_3b2fdff6_results.jsonl",
    RESULTS_DIR / "run_20260627_142003_9cb1a45d_results.jsonl",
)

SA_SEEDS = (101, 102, 103, 104, 105)
COLORS = {
    "qaoa": "#176B87",
    "brute_force": "#222222",
    "simulated_annealing": "#D05A3A",
    "direct": "#176B87",
    "iterative": "#D05A3A",
    "accent": "#E2A72E",
}


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
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


def experiment_code(record: dict[str, Any]) -> str:
    return str(record.get("scenario_id", "unknown")).split(".", maxsplit=1)[0]


def finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := finite_number(value)) is not None]
    return statistics.fmean(numbers) if numbers else None


def sample_std(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := finite_number(value)) is not None]
    if not numbers:
        return None
    return statistics.stdev(numbers) if len(numbers) > 1 else 0.0


def rate(values: Iterable[Any]) -> float | None:
    observed = [value for value in values if value is not None]
    if not observed:
        return None
    return sum(bool(value) for value in observed) / len(observed)


def config_value(record: dict[str, Any], table: str, name: str) -> Any:
    return get_path(record, f"config.{table}.{name}")


def sweep_value(record: dict[str, Any], name: str) -> Any:
    return get_path(record, f"config.sweep_context.values.{name}")


def raw_row(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") or {}
    validation = metrics.get("validation") or {}
    return {
        "source_file": record.get("_source_file"),
        "scenario_id": record.get("scenario_id"),
        "experiment": experiment_code(record),
        "repeat_number": record.get("repeat_number"),
        "status": record.get("status"),
        "pipeline": metrics.get("pipeline"),
        "num_entities": metrics.get("num_entities"),
        "num_cores": metrics.get("num_cores"),
        "num_variables": metrics.get("num_variables"),
        "layers": config_value(record, "qaoa", "layers"),
        "steps": config_value(record, "qaoa", "steps"),
        "learning_rate": config_value(record, "qaoa", "learning_rate"),
        "mixer_type": config_value(record, "qaoa", "mixer_type"),
        "init_strategy": config_value(record, "qaoa", "init_strategy"),
        "init_gamma": config_value(record, "qaoa", "init_gamma"),
        "init_beta": config_value(record, "qaoa", "init_beta"),
        "random_seed": config_value(record, "qaoa", "random_seed"),
        "top_k": config_value(record, "qaoa", "top_k"),
        "penalty": config_value(record, "qubo", "penalty"),
        "penalty_multiplier": sweep_value(record, "metadata.penalty_multiplier"),
        "weight_distribution": sweep_value(record, "metadata.distribution"),
        "qubit_max": config_value(record, "decomposition", "qubit_max"),
        "sorting_strategy": config_value(record, "decomposition", "sorting_strategy"),
        "energy": metrics.get("energy"),
        "brute_force_energy": validation.get("global_energy"),
        "feasible": metrics.get("feasible"),
        "optimal": metrics.get("optimal"),
        "optimality_gap": metrics.get("optimality_gap"),
        "solve_time_ms": metrics.get("solve_time_ms"),
        "max_probability": metrics.get("max_probability"),
        "convergence_iterations_to_final_tol": metrics.get(
            "convergence_iterations_to_final_tol"
        ),
        "max_subqubo_convergence_iterations_to_final_tol": metrics.get(
            "max_subqubo_convergence_iterations_to_final_tol"
        ),
        "num_sub_qubos": metrics.get("num_sub_qubos"),
        "all_sub_qubos_feasible": metrics.get("all_sub_qubos_feasible"),
        "load_imbalance": metrics.get("load_imbalance"),
        "recorded_sa_energy": metrics.get("annealing_energy"),
        "recorded_sa_gap": metrics.get("annealing_gap"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


GROUP_FIELDS = {
    "T1": ("scenario_id",),
    "B1": ("layers", "init_strategy"),
    "B1R": ("layers", "init_strategy"),
    "B2": ("steps", "learning_rate"),
    "B3": ("layers", "init_gamma", "init_beta"),
    "B4X": ("layers", "penalty", "mixer_type"),
    "B4XY": ("layers", "mixer_type"),
    "C1": ("penalty", "penalty_multiplier"),
    "C2": ("weight_distribution", "penalty_multiplier"),
    "D1": ("num_entities", "num_variables", "pipeline", "qubit_max"),
    "D2": ("qubit_max", "pipeline"),
    "D3": ("qubit_max", "pipeline"),
    "D4": ("sorting_strategy",),
    "D6": ("top_k",),
}


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        experiment = str(row["experiment"])
        fields = GROUP_FIELDS.get(experiment, ("scenario_id",))
        key = (experiment, fields, *(row.get(field) for field in fields))
        groups[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        experiment = key[0]
        fields = key[1]
        summary = {
            "experiment": experiment,
            "group": ", ".join(f"{field}={group[0].get(field)}" for field in fields),
            "runs": len(group),
            "success_rate": rate(row.get("status") == "success" for row in group),
            "feasibility_rate": rate(row.get("feasible") for row in group),
            "optimality_rate": rate(row.get("optimal") for row in group),
            "mean_energy": mean(row.get("energy") for row in group),
            "mean_brute_force_energy": mean(row.get("brute_force_energy") for row in group),
            "mean_optimality_gap": mean(row.get("optimality_gap") for row in group),
            "std_optimality_gap": sample_std(row.get("optimality_gap") for row in group),
            "mean_solve_time_ms": mean(row.get("solve_time_ms") for row in group),
            "std_solve_time_ms": sample_std(row.get("solve_time_ms") for row in group),
            "mean_convergence_iterations": mean(
                row.get("convergence_iterations_to_final_tol")
                for row in group
            ),
            "mean_max_subqubo_convergence_iterations": mean(
                row.get("max_subqubo_convergence_iterations_to_final_tol")
                for row in group
            ),
            "mean_num_sub_qubos": mean(row.get("num_sub_qubos") for row in group),
            "mean_load_imbalance": mean(row.get("load_imbalance") for row in group),
            "mean_max_probability": mean(row.get("max_probability") for row in group),
        }
        summaries.append(summary)
    return summaries


def workload_from_record(record: dict[str, Any]) -> Workload:
    workload_cfg = get_path(record, "config.workload", {})
    num_cores = int(workload_cfg.get("num_cores", 2))
    if "weights" in workload_cfg:
        weights = [float(value) for value in workload_cfg["weights"]]
    elif workload_cfg.get("weight_strategy") == "uniform_total":
        count = int(workload_cfg["num_processes"])
        total = float(workload_cfg.get("total_weight", 1.0))
        weights = [total / count] * count
    else:
        raise ValueError(f"Cannot reconstruct workload for {record.get('scenario_id')}.")

    return Workload(
        entities=[
            WorkloadEntity(
                entity_id=1000 + index,
                cpu_weight=weight,
                rss_mb=weight * 1024.0,
                label=f"proc_{index}",
            )
            for index, weight in enumerate(weights)
        ],
        num_cores=num_cores,
        snapshot_id=f"phase3-{record.get('scenario_id')}",
    )


def qubo_from_record(record: dict[str, Any]):
    workload = workload_from_record(record)
    config = QUBOConfig(
        penalty=float(config_value(record, "qubo", "penalty")),
        num_cores=workload.num_cores,
        snapshot=None,
        target_load=config_value(record, "qubo", "target_load"),
    )
    return CoreAssignmentBuilder(config).build(workload)


def d1_solver_baselines(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    representatives: dict[int, dict[str, Any]] = {}
    d1_runs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if experiment_code(record) != "D1":
            continue
        num_entities = int(get_path(record, "metrics.num_entities"))
        representatives.setdefault(num_entities, record)
        d1_runs[num_entities].append(record)

    rows: list[dict[str, Any]] = []
    for num_entities, record in sorted(representatives.items()):
        qubo = qubo_from_record(record)
        stored_optimum = float(get_path(record, "metrics.validation.global_energy"))

        brute_force = BruteForceSolver().solve(qubo)
        if not np.isclose(brute_force.energy, stored_optimum):
            raise RuntimeError(
                f"D1 N={num_entities}: recomputed brute-force energy {brute_force.energy} "
                f"does not match stored energy {stored_optimum}."
            )
        rows.append(
            {
                "num_entities": num_entities,
                "num_variables": qubo.num_variables,
                "solver": "brute_force",
                "replicate": 1,
                "energy": brute_force.energy,
                "energy_excess_vs_brute_force": 0.0,
                "feasible": brute_force.is_feasible,
                "solve_time_ms": brute_force.solve_time_ms,
            }
        )

        for seed in SA_SEEDS:
            annealing = AnnealingSolver(seed=seed).solve(qubo)
            rows.append(
                {
                    "num_entities": num_entities,
                    "num_variables": qubo.num_variables,
                    "solver": "simulated_annealing",
                    "replicate": seed,
                    "energy": annealing.energy,
                    "energy_excess_vs_brute_force": annealing.energy - brute_force.energy,
                    "feasible": annealing.is_feasible,
                    "solve_time_ms": annealing.solve_time_ms,
                }
            )

        for qaoa_index, qaoa_record in enumerate(d1_runs[num_entities], start=1):
            energy = float(get_path(qaoa_record, "metrics.energy"))
            rows.append(
                {
                    "num_entities": num_entities,
                    "num_variables": qubo.num_variables,
                    "solver": "qaoa",
                    "replicate": qaoa_index,
                    "energy": energy,
                    "energy_excess_vs_brute_force": energy - brute_force.energy,
                    "feasible": get_path(qaoa_record, "metrics.feasible"),
                    "solve_time_ms": get_path(qaoa_record, "metrics.solve_time_ms"),
                }
            )
    return rows


def grouped_records(
    records: list[dict[str, Any]], experiment: str, key_fn
) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if experiment_code(record) == experiment:
            groups[key_fn(record)].append(record)
    return dict(groups)


def save_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_time_vs_variables(records: list[dict[str, Any]], output_dir: Path) -> Path:
    groups = grouped_records(records, "D1", lambda record: int(get_path(record, "metrics.num_variables")))
    xs = sorted(groups)
    means = [mean(get_path(record, "metrics.solve_time_ms") for record in groups[x]) for x in xs]
    stds = [sample_std(get_path(record, "metrics.solve_time_ms") for record in groups[x]) for x in xs]
    pipelines = [str(get_path(groups[x][0], "metrics.pipeline")) for x in xs]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, means, color="#666666", linewidth=1.4, zorder=1)
    for pipeline in ("default", "iterative"):
        indices = [index for index, value in enumerate(pipelines) if value == pipeline]
        ax.errorbar(
            [xs[index] for index in indices],
            [means[index] for index in indices],
            yerr=[stds[index] for index in indices],
            marker="o",
            linestyle="none",
            capsize=3,
            color=COLORS["direct" if pipeline == "default" else "iterative"],
            label="Directa" if pipeline == "default" else "Iterativa",
            zorder=2,
        )
    ax.axvline(6, color="#777777", linestyle="--", linewidth=1, label="qubit_max = 6")
    ax.set_xlabel("Numero de variaveis QUBO")
    ax.set_ylabel("Tempo QAOA (ms, media +/- desvio-padrao)")
    ax.set_title("D1: tempo de resolucao vs. dimensao")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    path = output_dir / "time_vs_num_variables.png"
    save_figure(fig, path)
    return path


def grid_values(records: list[dict[str, Any]], x_name: str, y_name: str) -> tuple[list[float], list[float], np.ndarray]:
    xs = sorted({float(config_value(record, "qaoa", x_name)) for record in records})
    ys = sorted({float(config_value(record, "qaoa", y_name)) for record in records})
    matrix = np.zeros((len(ys), len(xs)))
    for y_index, y_value in enumerate(ys):
        for x_index, x_value in enumerate(xs):
            selected = [
                record
                for record in records
                if float(config_value(record, "qaoa", x_name)) == x_value
                and float(config_value(record, "qaoa", y_name)) == y_value
            ]
            matrix[y_index, x_index] = 100.0 * (mean(get_path(record, "metrics.optimality_gap") for record in selected) or 0.0)
    return xs, ys, matrix


def draw_gap_heatmap(ax, records: list[dict[str, Any]], x_name: str, y_name: str, title: str) -> None:
    xs, ys, matrix = grid_values(records, x_name, y_name)
    image = ax.imshow(matrix, origin="lower", cmap="YlOrRd", vmin=0.0, vmax=max(0.01, float(matrix.max())))
    ax.set_xticks(range(len(xs)), [f"{value:g}" for value in xs])
    ax.set_yticks(range(len(ys)), [f"{value:g}" for value in ys])
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    for row in range(len(ys)):
        for column in range(len(xs)):
            ax.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", fontsize=7)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Gap medio (%)")


def plot_qaoa_parameters(records: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for experiment, label, color in (
        ("B1", "Inicializacao fixa", COLORS["qaoa"]),
        ("B1R", "Inicializacao aleatoria", COLORS["simulated_annealing"]),
    ):
        groups = grouped_records(records, experiment, lambda record: int(config_value(record, "qaoa", "layers")))
        xs = sorted(groups)
        ys = [100.0 * (mean(get_path(record, "metrics.optimality_gap") for record in groups[x]) or 0.0) for x in xs]
        axes[0, 0].plot(xs, ys, marker="o", label=label, color=color)
    axes[0, 0].set_xlabel("Camadas QAOA (p)")
    axes[0, 0].set_ylabel("Gap medio (%)")
    axes[0, 0].set_title("B1: profundidade")
    axes[0, 0].set_ylim(-0.001, 0.01)
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)

    b2_records = [record for record in records if experiment_code(record) == "B2"]
    draw_gap_heatmap(axes[0, 1], b2_records, "steps", "learning_rate", "B2: passos e learning rate")

    b3_records = [record for record in records if experiment_code(record) == "B3"]
    for ax, layers in zip(axes[1], (2, 3)):
        selected = [record for record in b3_records if int(config_value(record, "qaoa", "layers")) == layers]
        draw_gap_heatmap(ax, selected, "init_gamma", "init_beta", f"B3: inicializacao, p={layers}")

    fig.suptitle("Optimality gap vs. parametros QAOA", fontsize=14)
    path = output_dir / "qaoa_parameters_vs_optimality_gap.png"
    save_figure(fig, path)
    return path


def plot_qubit_quality(records: list[dict[str, Any]], output_dir: Path) -> Path:
    groups = grouped_records(records, "D2", lambda record: int(config_value(record, "decomposition", "qubit_max")))
    xs = sorted(groups)
    gaps = [100.0 * (mean(get_path(record, "metrics.optimality_gap") for record in groups[x]) or 0.0) for x in xs]
    times = [mean(get_path(record, "metrics.solve_time_ms") for record in groups[x]) for x in xs]
    time_stds = [sample_std(get_path(record, "metrics.solve_time_ms") for record in groups[x]) for x in xs]
    sub_qubos = [mean(get_path(record, "metrics.num_sub_qubos") for record in groups[x]) or 0.0 for x in xs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = [COLORS["simulated_annealing"] if x == 16 else COLORS["qaoa"] for x in xs]
    axes[0].plot(xs, gaps, color="#777777", linewidth=1.2)
    axes[0].scatter(xs, gaps, color=colors, s=55, zorder=3)
    axes[0].annotate(
        "Directa: gap > 0",
        xy=(16, gaps[-1]),
        xytext=(10.8, max(gaps[-1] * 0.62, 0.02)),
        arrowprops={"arrowstyle": "->", "color": COLORS["simulated_annealing"]},
        color=COLORS["simulated_annealing"],
    )
    axes[0].set_xlabel("qubit_max")
    axes[0].set_ylabel("Optimality gap medio (%)")
    axes[0].set_title("Qualidade vs. orcamento de qubits")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(xs, times, yerr=time_stds, marker="o", capsize=3, color=COLORS["qaoa"])
    secondary = axes[1].twinx()
    secondary.step(xs, sub_qubos, where="mid", color=COLORS["accent"], label="Sub-QUBOs")
    axes[1].set_xlabel("qubit_max")
    axes[1].set_ylabel("Tempo QAOA (ms)", color=COLORS["qaoa"])
    secondary.set_ylabel("Numero medio de sub-QUBOs", color=COLORS["accent"])
    axes[1].set_title("Custo e decomposicao")
    axes[1].grid(alpha=0.25)

    fig.suptitle("D2: curva de hardware futuro sob orcamento algoritmico fixo", fontsize=14)
    path = output_dir / "quality_vs_qubit_max.png"
    save_figure(fig, path)
    return path


def plot_pipeline_comparison(records: list[dict[str, Any]], output_dir: Path) -> Path:
    groups = grouped_records(records, "D3", lambda record: str(get_path(record, "metrics.pipeline")))
    order = ["iterative", "default"]
    labels = ["Iterativa\n(qubit_max=6)", "Directa\n(qubit_max=12)"]
    gaps = [100.0 * (mean(get_path(record, "metrics.optimality_gap") for record in groups[name]) or 0.0) for name in order]
    times = [mean(get_path(record, "metrics.solve_time_ms") for record in groups[name]) for name in order]
    time_stds = [sample_std(get_path(record, "metrics.solve_time_ms") for record in groups[name]) for name in order]
    colors = [COLORS["iterative"], COLORS["direct"]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].bar(labels, gaps, color=colors, width=0.58)
    axes[0].set_ylabel("Optimality gap medio (%)")
    axes[0].set_ylim(0, max(0.01, max(gaps) * 1.2))
    axes[0].set_title("Qualidade")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].text(0.5, 0.75, "Ambas atingem o optimo global", transform=axes[0].transAxes, ha="center")

    axes[1].bar(labels, times, yerr=time_stds, capsize=4, color=colors, width=0.58)
    axes[1].set_ylabel("Tempo QAOA (ms)")
    axes[1].set_title("Tempo de resolucao")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("D3: pipeline directa vs. iterativa", fontsize=14)
    path = output_dir / "default_vs_iterative.png"
    save_figure(fig, path)
    return path


def plot_solver_comparison(baselines: list[dict[str, Any]], output_dir: Path) -> Path:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in baselines:
        grouped[(str(row["solver"]), int(row["num_variables"]))].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = {
        "qaoa": "QAOA simulado",
        "brute_force": "Brute-force",
        "simulated_annealing": "SA",
    }
    for solver in ("qaoa", "simulated_annealing", "brute_force"):
        xs = sorted(num_variables for name, num_variables in grouped if name == solver)
        excess = [mean(row["energy_excess_vs_brute_force"] for row in grouped[(solver, x)]) for x in xs]
        excess_std = [sample_std(row["energy_excess_vs_brute_force"] for row in grouped[(solver, x)]) for x in xs]
        times = [mean(row["solve_time_ms"] for row in grouped[(solver, x)]) for x in xs]
        time_std = [sample_std(row["solve_time_ms"] for row in grouped[(solver, x)]) for x in xs]
        color = COLORS[solver]
        axes[0].errorbar(xs, excess, yerr=excess_std, marker="o", capsize=3, label=labels[solver], color=color)
        axes[1].errorbar(xs, times, yerr=time_std, marker="o", capsize=3, label=labels[solver], color=color)

    axes[0].set_xlabel("Numero de variaveis QUBO")
    axes[0].set_ylabel("Excesso de energia vs. brute-force")
    axes[0].set_ylim(-1e-8, 1e-6)
    axes[0].set_title("Qualidade em D1")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].set_xlabel("Numero de variaveis QUBO")
    axes[1].set_ylabel("Tempo do solver (ms, escala log)")
    axes[1].set_yscale("log")
    axes[1].set_title("Custo computacional local")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False)
    fig.suptitle("D1: QAOA vs. brute-force vs. simulated annealing", fontsize=14)
    path = output_dir / "qaoa_vs_bruteforce_vs_sa.png"
    save_figure(fig, path)
    return path


def plot_top_k(records: list[dict[str, Any]], output_dir: Path) -> Path:
    groups = grouped_records(records, "D6", lambda record: int(config_value(record, "qaoa", "top_k")))
    xs = sorted(groups)
    gaps = [100.0 * (mean(get_path(record, "metrics.optimality_gap") for record in groups[x]) or 0.0) for x in xs]
    imbalance = [mean(get_path(record, "metrics.load_imbalance") for record in groups[x]) or 0.0 for x in xs]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, gaps, marker="o", color=COLORS["simulated_annealing"], label="Optimality gap (%)")
    secondary = ax.twinx()
    secondary.plot(xs, imbalance, marker="s", color=COLORS["qaoa"], label="Load imbalance")
    ax.set_xlabel("top_k")
    ax.set_ylabel("Optimality gap medio (%)", color=COLORS["simulated_annealing"])
    secondary.set_ylabel("Load imbalance medio", color=COLORS["qaoa"])
    ax.set_xticks(xs)
    ax.set_title("D6: sensibilidade da decomposicao a top_k")
    ax.grid(alpha=0.25)
    lines = ax.lines + secondary.lines
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right")
    path = output_dir / "top_k_sensitivity.png"
    save_figure(fig, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the curated Phase 3 tables and figures.")
    parser.add_argument("jsonl", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "analysis_20260627_phase3",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = [path.resolve() for path in args.jsonl]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing input files: {', '.join(str(path) for path in missing)}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(paths)
    raw_rows = [raw_row(record) for record in records]
    summary_rows = summarize_rows(raw_rows)
    baselines = d1_solver_baselines(records)

    write_csv(output_dir / "phase3_runs.csv", raw_rows)
    write_csv(output_dir / "phase3_summary.csv", summary_rows)
    write_csv(output_dir / "d1_solver_baselines.csv", baselines)

    figures = [
        plot_time_vs_variables(records, output_dir),
        plot_qaoa_parameters(records, output_dir),
        plot_qubit_quality(records, output_dir),
        plot_pipeline_comparison(records, output_dir),
        plot_solver_comparison(baselines, output_dir),
        plot_top_k(records, output_dir),
    ]

    manifest = {
        "phase": 3,
        "input_files": [str(path) for path in paths],
        "input_record_count": len(records),
        "successful_record_count": sum(record.get("status") == "success" for record in records),
        "sa_seeds": list(SA_SEEDS),
        "tables": ["phase3_runs.csv", "phase3_summary.csv", "d1_solver_baselines.csv"],
        "figures": [path.name for path in figures],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Loaded {len(records)} records from {len(paths)} JSONL files.")
    print(f"Wrote Phase 3 analysis to {output_dir}")
    for name in manifest["tables"] + manifest["figures"] + ["manifest.json"]:
        print(f"- {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
