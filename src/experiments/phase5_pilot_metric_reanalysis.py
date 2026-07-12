from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


DEFAULT_INPUT = Path(
    "src/experiments/results/phase5_corrected_pilot/"
    "sweep_20260707_130304_89aca6e3_results.jsonl"
)
DEFAULT_OUTPUT = Path("src/experiments/results/phase5_corrected_pilot/analysis")


def reprocess(input_path: Path) -> list[dict[str, float | int]]:
    records = [json.loads(line) for line in input_path.read_text().splitlines() if line]
    rows: list[dict[str, float | int]] = []
    for record in records:
        metrics = record["metrics"]
        processes = record["resolved_config"]["preset_snapshot"]["processes"]
        total_weight = sum(float(process["cpu_weight"]) for process in processes)
        pipeline = float(metrics["load_imbalance"])
        reference = float(metrics["baseline_load_imbalance"])
        rows.append(
            {
                "seed": int(record["config"]["workload"]["instance_seed"]),
                "qaoa_seed": int(record["config"]["qaoa"]["random_seed"]),
                "imbalance_pipeline": pipeline,
                "imbalance_ref": reference,
                "delta_imbalance": abs(pipeline - reference),
                "gap_estavel": (pipeline**2 - reference**2) / total_weight**2,
            }
        )
    return sorted(rows, key=lambda row: int(row["seed"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reprocess Phase 5 pilot metrics only.")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = reprocess(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "pilot_n10_delta_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    deltas = [float(row["delta_imbalance"]) for row in rows]
    stable = [float(row["gap_estavel"]) for row in rows]
    summary = {
        "source": str(args.input),
        "runs": len(rows),
        "delta_imbalance": {
            "mean": statistics.fmean(deltas),
            "median": statistics.median(deltas),
            "stdev": statistics.stdev(deltas),
            "min": min(deltas),
            "max": max(deltas),
            "q75": statistics.quantiles(deltas, n=4, method="inclusive")[2],
            "q90": statistics.quantiles(deltas, n=10, method="inclusive")[8],
            "q95": statistics.quantiles(deltas, n=20, method="inclusive")[18],
        },
        "gap_estavel": {
            "mean": statistics.fmean(stable),
            "min": min(stable),
            "max": max(stable),
        },
    }
    summary_path = args.output_dir / "pilot_n10_delta_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
