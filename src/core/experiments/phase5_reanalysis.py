from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

EXPERIMENTS_DIR = Path(__file__).resolve().parent
SRC_DIR = EXPERIMENTS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_contracts import Workload, WorkloadEntity
from investigative_runtime import assignment_balance_metrics


RESULTS_DIR = EXPERIMENTS_DIR / "results"
DEFAULT_INPUTS = (
    RESULTS_DIR / "sweep_20260705_142459_e5a8c292_results.jsonl",
    RESULTS_DIR / "sweep_20260705_151529_715b9a8d_results.jsonl",
)
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "phase5" / "analysis"


def load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def workload_from_record(record: dict[str, Any]) -> Workload:
    snapshot = record["resolved_config"]["preset_snapshot"]
    processes = snapshot["processes"]
    return Workload(
        entities=[
            WorkloadEntity(
                entity_id=int(process["pid"]),
                cpu_weight=float(process["cpu_weight"]),
                rss_mb=float(process["rss_mb"]),
                label=str(process["command"]),
            )
            for process in processes
        ],
        num_cores=int(snapshot["num_cores"]),
        snapshot_id=str(snapshot.get("snapshot_id") or "phase5-generated"),
    )


def reanalyse_record(record: dict[str, Any]) -> dict[str, Any]:
    config = record["config"]
    metrics = record["metrics"]
    validation = metrics["validation"]
    workload = workload_from_record(record)
    candidate = assignment_balance_metrics(workload, metrics["assignments"])
    if candidate is None:
        raise ValueError(f"Incomplete candidate assignment in {record['result_path']}")

    baseline_method = str(metrics["baseline_method"])
    baseline_assignments = (
        validation["global_assignments"]
        if baseline_method == "bruteforce"
        else validation["annealing_assignments"]
    )
    baseline = assignment_balance_metrics(workload, baseline_assignments)
    if baseline is None:
        raise ValueError(f"Incomplete baseline assignment in {record['result_path']}")

    regret = max(
        0.0,
        candidate["load_balance_objective"]
        - baseline["load_balance_objective"],
    )
    tolerance = float(validation.get("optimality_atol", 1e-9)) + float(
        validation.get("optimality_rtol", 1e-9)
    ) * abs(baseline["load_balance_objective"])
    offset_free_match = regret <= tolerance
    n = int(config["workload"]["num_processes"])
    return {
        "N": n,
        "num_variables": 2 * n,
        "num_subqubos": int(metrics["num_sub_qubos"]),
        "instance_seed": int(config["workload"]["instance_seed"]),
        "qaoa_seed": int(config["qaoa"]["random_seed"]),
        "source_run_id": record["run_id"],
        "status": record["status"],
        "baseline_method": baseline_method,
        "baseline_certified": bool(metrics["baseline_certified"]),
        "target_load": candidate["target_load"],
        "candidate_load_core_0": candidate["core_loads"][0],
        "candidate_load_core_1": candidate["core_loads"][1],
        "candidate_load_imbalance": candidate["load_imbalance"],
        "candidate_normalized_load_imbalance": candidate[
            "normalized_load_imbalance"
        ],
        "candidate_balance_objective": candidate["load_balance_objective"],
        "baseline_load_core_0": baseline["core_loads"][0],
        "baseline_load_core_1": baseline["core_loads"][1],
        "baseline_load_imbalance": baseline["load_imbalance"],
        "baseline_normalized_load_imbalance": baseline[
            "normalized_load_imbalance"
        ],
        "baseline_balance_objective": baseline["load_balance_objective"],
        "objective_regret": regret,
        "excess_normalized_load_imbalance": max(
            0.0,
            candidate["normalized_load_imbalance"]
            - baseline["normalized_load_imbalance"],
        ),
        "offset_free_tolerance": tolerance,
        "baseline_match_offset_free": offset_free_match,
        "certified_optimal_offset_free": (
            offset_free_match if metrics["baseline_certified"] else None
        ),
        "legacy_pipeline_energy": float(metrics["pipeline_energy"]),
        "legacy_baseline_energy": float(metrics["baseline_energy"]),
        "legacy_relative_gap": float(metrics["gap_relativo"]),
        "legacy_is_optimal": bool(metrics["is_optimal"]),
        "tempo_total_ms": float(metrics["tempo_total_ms"]),
        "tempo_qaoa_ms": float(metrics["tempo_qaoa_ms"]),
        "tempo_overhead_ms": float(metrics["tempo_overhead_ms"]),
        "result_path": record["result_path"],
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def median(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def maximum(rows: list[dict[str, Any]], key: str) -> float:
    return max(float(row[key]) for row in rows)


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["N"])].append(row)

    summaries = []
    for n, group in sorted(grouped.items()):
        certified = [row for row in group if row["baseline_certified"]]
        summaries.append(
            {
                "N": n,
                "num_variables": int(group[0]["num_variables"]),
                "num_subqubos": int(group[0]["num_subqubos"]),
                "runs": len(group),
                "baseline_method": group[0]["baseline_method"],
                "baseline_certified_runs": len(certified),
                "candidate_normalized_load_imbalance_mean": mean(
                    group, "candidate_normalized_load_imbalance"
                ),
                "candidate_normalized_load_imbalance_median": median(
                    group, "candidate_normalized_load_imbalance"
                ),
                "candidate_normalized_load_imbalance_max": maximum(
                    group, "candidate_normalized_load_imbalance"
                ),
                "baseline_normalized_load_imbalance_mean": mean(
                    group, "baseline_normalized_load_imbalance"
                ),
                "candidate_balance_objective_mean": mean(
                    group, "candidate_balance_objective"
                ),
                "baseline_balance_objective_mean": mean(
                    group, "baseline_balance_objective"
                ),
                "objective_regret_mean": mean(group, "objective_regret"),
                "objective_regret_median": median(group, "objective_regret"),
                "objective_regret_max": maximum(group, "objective_regret"),
                "baseline_match_offset_free_count": sum(
                    bool(row["baseline_match_offset_free"]) for row in group
                ),
                "baseline_match_offset_free_percent": 100.0
                * sum(bool(row["baseline_match_offset_free"]) for row in group)
                / len(group),
                "certified_optimal_offset_free_count": sum(
                    row["certified_optimal_offset_free"] is True for row in group
                ),
                "legacy_match_count": sum(
                    bool(row["legacy_is_optimal"]) for row in group
                ),
                "tempo_total_ms_mean": mean(group, "tempo_total_ms"),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reanalyse Phase 5 with offset-free load-balance metrics."
    )
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    records = load_jsonl(args.inputs or DEFAULT_INPUTS)
    rows = [reanalyse_record(record) for record in records]
    unique_keys = {
        (row["N"], row["instance_seed"], row["qaoa_seed"]) for row in rows
    }
    if len(rows) != 200 or len(unique_keys) != 200:
        raise ValueError(
            f"Expected 200 unique accepted runs, found {len(rows)} rows and "
            f"{len(unique_keys)} unique keys."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "phase5_offset_free_raw.csv"
    summary_path = args.output_dir / "phase5_offset_free_summary.csv"
    manifest_path = args.output_dir / "manifest.json"
    summaries = summarise(rows)
    write_csv(raw_path, rows)
    write_csv(summary_path, summaries)
    manifest_path.write_text(
        json.dumps(
            {
                "inputs": [str(path) for path in (args.inputs or DEFAULT_INPUTS)],
                "runs": len(rows),
                "unique_runs": len(unique_keys),
                "dimensions": sorted({row["N"] for row in rows}),
                "raw_csv": str(raw_path),
                "summary_csv": str(summary_path),
                "quality_definition": {
                    "load_balance_objective": "sum((load_k - mean_load)^2)",
                    "normalized_load_imbalance": "(max(load)-min(load))/mean_load",
                    "objective_regret": "max(0, candidate_objective-baseline_objective)",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
