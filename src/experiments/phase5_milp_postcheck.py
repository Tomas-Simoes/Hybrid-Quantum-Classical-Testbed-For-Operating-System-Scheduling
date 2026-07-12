from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


DEFAULT_INPUT = Path(
    "src/experiments/results/phase5_corrected_full/"
    "sweep_20260707_133507_dca506b8_results.jsonl"
)


def solve_partition(weights: list[float], scale: int, time_limit: float) -> dict:
    integer_weights = np.rint(np.asarray(weights) * scale).astype(np.int64)
    n = len(integer_weights)
    total = int(integer_weights.sum())
    matrix = np.zeros((2, n + 1))
    matrix[0, :n] = 2 * integer_weights
    matrix[0, -1] = -1
    matrix[1, :n] = -2 * integer_weights
    matrix[1, -1] = -1
    objective = np.r_[np.zeros(n), 1.0]
    start = time.perf_counter()
    result = milp(
        objective,
        integrality=np.r_[np.ones(n), 0],
        bounds=Bounds(np.zeros(n + 1), np.r_[np.ones(n), np.inf]),
        constraints=LinearConstraint(matrix, [-np.inf, -np.inf], [total, -total]),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False},
    )
    if result.x is None:
        return {
            "status": "inconclusive",
            "solver_status": int(result.status),
            "solve_seconds": time.perf_counter() - start,
        }
    chosen = np.rint(result.x[:n]).astype(np.int64)
    verified_integer_imbalance = abs(
        2 * int(integer_weights @ chosen) - total
    )
    float_load = float(np.asarray(weights) @ chosen)
    verified_float_imbalance = abs(2 * float_load - float(sum(weights)))
    reported = None if result.fun is None else float(result.fun)
    exact_certificate = bool(
        result.success
        and reported is not None
        and abs(reported - verified_integer_imbalance) < 0.5
        and getattr(result, "mip_gap", None) == 0.0
    )
    return {
        "status": "solved" if result.success else "time_limit_with_witness",
        "solver_status": int(result.status),
        "solve_seconds": time.perf_counter() - start,
        "reported_objective_units": reported,
        "verified_objective_units": verified_integer_imbalance,
        "witness_imbalance": verified_float_imbalance,
        "exact_certificate_at_scale": exact_certificate,
        "mip_gap": getattr(result, "mip_gap", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-check Phase 5 with MILP.")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--scale", type=int, default=1_000_000_000)
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input.with_name("phase5_milp_postcheck.csv")
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    rows = []
    for index, record in enumerate(records, 1):
        processes = record["resolved_config"]["preset_snapshot"]["processes"]
        weights = [float(process["cpu_weight"]) for process in processes]
        check = solve_partition(weights, args.scale, args.time_limit)
        pipeline_imbalance = float(record["metrics"]["load_imbalance"])
        witness = check.get("witness_imbalance")
        nonoptimal_witness = witness is not None and witness + 1e-9 < pipeline_imbalance
        rows.append(
            {
                "N": record["config"]["workload"]["num_processes"],
                "seed": record["config"]["workload"]["instance_seed"],
                "qaoa_seed": record["config"]["qaoa"]["random_seed"],
                "pipeline_imbalance": pipeline_imbalance,
                **check,
                "pipeline_proven_nonoptimal": nonoptimal_witness,
                "pipeline_certified_optimal": bool(
                    check.get("exact_certificate_at_scale")
                    and witness is not None
                    and abs(witness - pipeline_imbalance) <= 1e-9
                ),
            }
        )
        print(f"{index}/{len(records)} N={rows[-1]['N']} seed={rows[-1]['seed']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
