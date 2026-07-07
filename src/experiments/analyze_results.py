from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scenario_runner import RESULTS_DIR


PLOT_METRICS = (
    "energy",
    "optimality_gap",
    "annealing_gap",
    "solve_time_ms",
    "load_imbalance",
    "num_sub_qubos",
    "max_probability",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_path(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True)


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") or {}
    validation = metrics.get("validation") or {}
    sweep_context = get_path(record, "config.sweep_context") or {}
    sweep_values = sweep_context.get("values") or {}

    row = {
        "run_id": record.get("run_id"),
        "scenario_file": record.get("scenario_file"),
        "scenario_id": record.get("scenario_id"),
        "scenario_name": record.get("scenario_name"),
        "status": record.get("status"),
        "repeat_number": record.get("repeat_number"),
        "repeat_count": record.get("repeat_count"),
        "duration_s": record.get("duration_s"),
    }
    for key, value in sweep_values.items():
        row[f"sweep.{key}"] = scalar(value)

    for key, value in metrics.items():
        if key != "validation":
            row[f"metrics.{key}"] = scalar(value)

    for key, value in validation.items():
        row[f"validation.{key}"] = scalar(value)

    return row


def write_csv(records: list[dict[str, Any]], csv_path: Path) -> list[dict[str, Any]]:
    rows = [flatten_record(record) for record in records]
    fieldnames = sorted({key for row in rows for key in row})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def grouped_means(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    buckets: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        x = numeric(row.get(x_key))
        y = numeric(row.get(y_key))
        if x is not None and y is not None:
            buckets[x].append(y)
    xs = sorted(buckets)
    ys = [sum(buckets[x]) / len(buckets[x]) for x in xs]
    return xs, ys


def write_plots(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_keys = sorted(key for key in {k for row in rows for k in row} if key.startswith("sweep."))
    written: list[Path] = []

    for sweep_key in sweep_keys:
        if not any(numeric(row.get(sweep_key)) is not None for row in rows):
            continue

        axis_name = sweep_key.removeprefix("sweep.")
        for metric in PLOT_METRICS:
            y_key = f"metrics.{metric}"
            xs, ys = grouped_means(rows, sweep_key, y_key)
            if len(xs) < 2:
                continue

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(xs, ys, marker="o")
            ax.set_xlabel(axis_name)
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            plot_path = output_dir / f"{axis_name}_vs_{metric}.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            written.append(plot_path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create CSV tables and simple sweep plots from experiment JSONL results."
    )
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "analysis")
    parser.add_argument("--csv-name", default="experiment_table.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records: list[dict[str, Any]] = []
    for path in args.jsonl:
        records.extend(load_jsonl(path))

    if not records:
        raise SystemExit("No records found.")

    csv_path = args.output_dir / args.csv_name
    rows = write_csv(records, csv_path)
    plots = write_plots(rows, args.output_dir)

    print(f"Wrote CSV: {csv_path}")
    if plots:
        print("Wrote plots:")
        for plot in plots:
            print(f"- {plot}")
    else:
        print("No sweep plots generated; numeric sweep axes with at least two values are required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
