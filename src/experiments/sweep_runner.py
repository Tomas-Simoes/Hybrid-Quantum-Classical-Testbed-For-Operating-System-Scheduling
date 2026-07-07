from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import statistics
import uuid
from pathlib import Path
from typing import Any

from scenario_runner import (
    RESULTS_DIR,
    SCENARIOS_DIR,
    aggregate_output_path_for,
    cumulative_output_path_for,
    discover_scenarios,
    display_path,
    load_toml,
    run_scenario_once,
    scenario_repeat_count,
    select_scenarios,
    utc_timestamp,
)


def set_dotted_path(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid sweep path '{path}'.")

    current: dict[str, Any] = config
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set '{path}': '{part}' is not a TOML table.")
        current = child
    current[parts[-1]] = value


def get_dotted_path(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def repeat_count_for_variant(scenario: dict[str, Any]) -> int:
    tiers = scenario.get("execution", {}).get("repeat_tiers", [])
    if not tiers:
        return scenario_repeat_count(scenario)

    n = int(get_dotted_path(scenario, "workload.num_processes", 0))
    for tier in sorted(tiers, key=lambda item: int(item["max_n"])):
        if n <= int(tier["max_n"]):
            repeats = int(tier["repeats"])
            if repeats < 1:
                raise ValueError("execution.repeat_tiers repeats must be at least 1.")
            return repeats
    raise ValueError(f"No execution.repeat_tiers entry covers N={n}.")


def prepare_repeat_variant(
    variant: dict[str, Any], repeat_index: int
) -> dict[str, Any]:
    seeded = copy.deepcopy(variant)
    seeds = seeded.get("seeds", {})
    if not seeds:
        return seeded

    instance_seed = int(seeds.get("instance_seed_base", 0)) + repeat_index
    qaoa_seed = int(seeds.get("qaoa_seed_base", 10000)) + repeat_index
    set_dotted_path(seeded, "workload.instance_seed", instance_seed)
    set_dotted_path(seeded, "qaoa.random_seed", qaoa_seed)
    set_dotted_path(seeded, "validation.annealing_seed", instance_seed)
    seeded.setdefault("sweep_context", {}).setdefault("values", {}).update(
        {
            "instance_seed": instance_seed,
            "qaoa_seed": qaoa_seed,
        }
    )
    return seeded


def expand_sweep(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    sweep = scenario.get("sweep", {})
    cases = sweep.get("cases", [])
    axes = sweep.get("axes", [])
    if not cases and not axes:
        variant = copy.deepcopy(scenario)
        variant["sweep_context"] = {
            "variant_index": 1,
            "variant_count": 1,
            "values": {},
        }
        return [variant]

    axis_specs = []
    for axis in axes:
        name = axis.get("name") or axis.get("path")
        path = axis.get("path")
        values = axis.get("values")
        if not path or not values:
            raise ValueError(f"Invalid sweep axis: {axis!r}")
        axis_specs.append((name, path, values))

    combinations = (
        list(itertools.product(*(values for _, _, values in axis_specs)))
        if axis_specs
        else [()]
    )
    case_specs = cases or [{"name": None, "values": {}}]
    for case in cases:
        values = case.get("values")
        if not isinstance(values, dict) or not values:
            raise ValueError(f"Invalid sweep case: {case!r}")

    variant_count = len(case_specs) * len(combinations)
    variants: list[dict[str, Any]] = []
    for case_index, case in enumerate(case_specs, start=1):
        for combo in combinations:
            variant = copy.deepcopy(scenario)
            variant.pop("sweep", None)

            value_map = {}
            path_map = {}
            case_values = case.get("values", {})
            if cases:
                value_map["case"] = case.get("name", f"case_{case_index:03d}")
            for path, value in case_values.items():
                set_dotted_path(variant, path, value)
                value_map[str(path)] = value
                path_map[str(path)] = path

            for (name, path, _), value in zip(axis_specs, combo):
                set_dotted_path(variant, path, value)
                value_map[str(name)] = value
                path_map[str(name)] = path

            index = len(variants) + 1
            variant["sweep_context"] = {
                "variant_index": index,
                "variant_count": variant_count,
                "values": value_map,
                "paths": path_map,
            }
            variant["id"] = f"{scenario.get('id', 'sweep')}.{index}"
            variant["name"] = (
                f"{scenario.get('name', 'Sweep')} [{index}/{variant_count}]"
            )
            variants.append(variant)
    return variants


def run_variant_repeats(
    scenario_path: Path,
    variant: dict[str, Any],
    results_dir: Path,
    aggregate_path: Path,
    cumulative_path: Path,
    run_id: str,
) -> list[dict[str, Any]]:
    repeat_count = repeat_count_for_variant(variant)
    records = []
    for repeat_index in range(repeat_count):
        repeat_variant = prepare_repeat_variant(variant, repeat_index)
        records.append(
            run_scenario_once(
                scenario_path=scenario_path,
                scenario=repeat_variant,
                results_dir=results_dir,
                aggregate_path=aggregate_path,
                cumulative_path=cumulative_path,
                run_id=run_id,
                repeat_index=repeat_index,
                repeat_count=repeat_count,
            )
        )
    return records


def scalability_point_statistics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    successful = [record for record in records if record.get("status") == "success"]
    gaps = [
        float(record["metrics"]["gap_relativo"])
        for record in successful
        if record.get("metrics", {}).get("gap_relativo") is not None
    ]
    imbalances = [
        float(record["metrics"]["load_imbalance"])
        for record in successful
        if record.get("metrics", {}).get("load_imbalance") is not None
    ]
    normalized_imbalances = [
        float(record["metrics"]["normalized_load_imbalance"])
        for record in successful
        if record.get("metrics", {}).get("normalized_load_imbalance") is not None
    ]
    objective_regrets = [
        float(record["metrics"]["objective_regret"])
        for record in successful
        if record.get("metrics", {}).get("objective_regret") is not None
    ]
    durations = [
        float(record["duration_s"])
        for record in successful
        if record.get("duration_s") is not None
    ]
    return {
        "gap_mean": statistics.fmean(gaps) if gaps else None,
        "gap_median": statistics.median(gaps) if gaps else None,
        "load_imbalance_mean": statistics.fmean(imbalances) if imbalances else None,
        "normalized_load_imbalance_mean": (
            statistics.fmean(normalized_imbalances)
            if normalized_imbalances
            else None
        ),
        "objective_regret_mean": (
            statistics.fmean(objective_regrets) if objective_regrets else None
        ),
        "mean_duration": statistics.fmean(durations) if durations else None,
    }


def _reuse_compatibility_paths() -> tuple[str, ...]:
    return (
        "workload.num_cores",
        "workload.weight_strategy",
        "workload.min_weight",
        "workload.max_weight",
        "qubo.penalty",
        "qaoa.layers",
        "qaoa.steps",
        "qaoa.top_k",
        "qaoa.mixer_type",
        "qaoa.init_strategy",
        "decomposition.qubit_max",
        "decomposition.num_cores",
    )


def load_reused_scalability_records(
    scenario_path: Path,
    scenario: dict[str, Any],
    results_dir: Path,
) -> list[dict[str, Any]]:
    resume = scenario.get("sweep", {}).get("resume", {})
    source_run_ids = [str(value) for value in resume.get("source_run_ids", [])]
    completed_values = {int(value) for value in resume.get("completed_values", [])}
    if not source_run_ids and not completed_values:
        return []
    if not source_run_ids or not completed_values:
        raise ValueError(
            "sweep.resume requires both source_run_ids and completed_values."
        )

    output_dir = results_dir / f"{scenario_path.stem}_result"
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int | None, int | None]] = set()
    for source_run_id in source_run_ids:
        paths = sorted(output_dir.glob(f"*_{source_run_id}_*_result.json"))
        if not paths:
            raise ValueError(
                f"No reusable result files found for run id '{source_run_id}' in "
                f"{output_dir}."
            )
        for result_path in paths:
            record = json.loads(result_path.read_text(encoding="utf-8"))
            config = record.get("config", {})
            n_value = get_dotted_path(config, "workload.num_processes")
            if n_value is None or int(n_value) not in completed_values:
                continue
            if record.get("status") != "success":
                raise ValueError(
                    f"Cannot reuse unsuccessful pilot result {result_path.name}."
                )
            for config_path in _reuse_compatibility_paths():
                old_value = get_dotted_path(config, config_path)
                new_value = get_dotted_path(scenario, config_path)
                if old_value != new_value:
                    raise ValueError(
                        f"Cannot reuse {result_path.name}: {config_path} changed "
                        f"from {old_value!r} to {new_value!r}."
                    )
            key = (
                int(n_value),
                get_dotted_path(config, "workload.instance_seed"),
                get_dotted_path(config, "qaoa.random_seed"),
            )
            if key in seen:
                continue
            seen.add(key)
            reused = copy.deepcopy(record)
            reused["data_source"] = "reused"
            reused["source_run_id"] = source_run_id
            records.append(reused)

    found_values = {
        int(get_dotted_path(record.get("config", {}), "workload.num_processes"))
        for record in records
    }
    missing = sorted(completed_values - found_values)
    if missing:
        raise ValueError(f"Reusable results are missing completed N values: {missing}.")

    for n_value in completed_values:
        variant = copy.deepcopy(scenario)
        set_dotted_path(variant, "workload.num_processes", n_value)
        expected = repeat_count_for_variant(variant)
        actual = sum(
            int(get_dotted_path(record["config"], "workload.num_processes"))
            == n_value
            for record in records
        )
        if actual != expected:
            raise ValueError(
                f"Expected {expected} reusable runs for N={n_value}, found {actual}."
            )
        n_records = [
            record
            for record in records
            if int(get_dotted_path(record["config"], "workload.num_processes"))
            == n_value
        ]
        instance_seed_base = int(
            get_dotted_path(scenario, "seeds.instance_seed_base", 0)
        )
        qaoa_seed_base = int(get_dotted_path(scenario, "seeds.qaoa_seed_base", 10000))
        expected_instance_seeds = set(
            range(instance_seed_base, instance_seed_base + expected)
        )
        expected_qaoa_seeds = set(range(qaoa_seed_base, qaoa_seed_base + expected))
        actual_instance_seeds = {
            int(get_dotted_path(record["config"], "workload.instance_seed"))
            for record in n_records
        }
        actual_qaoa_seeds = {
            int(get_dotted_path(record["config"], "qaoa.random_seed"))
            for record in n_records
        }
        if actual_instance_seeds != expected_instance_seeds:
            raise ValueError(f"Reusable instance seeds do not match for N={n_value}.")
        if actual_qaoa_seeds != expected_qaoa_seeds:
            raise ValueError(f"Reusable QAOA seeds do not match for N={n_value}.")
    return sorted(
        records,
        key=lambda record: (
            int(get_dotted_path(record["config"], "workload.num_processes")),
            int(get_dotted_path(record["config"], "workload.instance_seed", 0)),
        ),
    )


def _tolerance_column(tolerance: float) -> str:
    return f"within_gap_{tolerance:.0e}_percent"


def write_scalability_csvs(
    scenario_path: Path,
    records: list[dict[str, Any]],
    results_dir: Path,
    run_id: str,
    tolerance_tiers: list[float] | None = None,
) -> tuple[Path, Path]:
    tolerance_tiers = tolerance_tiers or []
    output_dir = results_dir / f"{scenario_path.stem}_result"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{run_id}_raw.csv"
    summary_path = output_dir / f"{run_id}_summary.csv"
    raw_fields = [
        "N",
        "seed",
        "qaoa_seed",
        "data_source",
        "source_run_id",
        "status",
        "num_subqubos",
        "tempo_total_ms",
        "tempo_qaoa_ms",
        "tempo_overhead_ms",
        "baseline_method",
        "baseline_certified",
        "baseline_energy",
        "pipeline_energy",
        "is_optimal",
        "gap_relativo",
        "load_imbalance",
        "normalized_load_imbalance",
        "load_balance_objective",
        "baseline_load_imbalance",
        "baseline_normalized_load_imbalance",
        "baseline_load_balance_objective",
        "objective_regret",
        "excess_normalized_load_imbalance",
        "baseline_match_offset_free",
        "certified_optimal_offset_free",
        "annealing_restarts_stable",
        "annealing_restart_energy_std",
    ]
    raw_rows = []
    for record in records:
        config = record.get("config", {})
        metrics = record.get("metrics", {})
        raw_rows.append(
            {
                "N": get_dotted_path(config, "workload.num_processes"),
                "seed": get_dotted_path(config, "workload.instance_seed"),
                "qaoa_seed": get_dotted_path(config, "qaoa.random_seed"),
                "data_source": record.get("data_source", "current"),
                "source_run_id": record.get("source_run_id", record.get("run_id")),
                "status": record.get("status"),
                "num_subqubos": metrics.get("num_sub_qubos"),
                "tempo_total_ms": metrics.get("tempo_total_ms"),
                "tempo_qaoa_ms": metrics.get("tempo_qaoa_ms"),
                "tempo_overhead_ms": metrics.get("tempo_overhead_ms"),
                "baseline_method": metrics.get("baseline_method"),
                "baseline_certified": metrics.get("baseline_certified"),
                "baseline_energy": metrics.get("baseline_energy"),
                "pipeline_energy": metrics.get("pipeline_energy"),
                "is_optimal": metrics.get("is_optimal"),
                "gap_relativo": metrics.get("gap_relativo"),
                "load_imbalance": metrics.get("load_imbalance"),
                "normalized_load_imbalance": metrics.get(
                    "normalized_load_imbalance"
                ),
                "load_balance_objective": metrics.get("load_balance_objective"),
                "baseline_load_imbalance": metrics.get("baseline_load_imbalance"),
                "baseline_normalized_load_imbalance": metrics.get(
                    "baseline_normalized_load_imbalance"
                ),
                "baseline_load_balance_objective": metrics.get(
                    "baseline_load_balance_objective"
                ),
                "objective_regret": metrics.get("objective_regret"),
                "excess_normalized_load_imbalance": metrics.get(
                    "excess_normalized_load_imbalance"
                ),
                "baseline_match_offset_free": metrics.get(
                    "baseline_match_offset_free"
                ),
                "certified_optimal_offset_free": metrics.get(
                    "certified_optimal_offset_free"
                ),
                "annealing_restarts_stable": metrics.get(
                    "annealing_restarts_stable"
                ),
                "annealing_restart_energy_std": metrics.get(
                    "annealing_restart_energy_std"
                ),
            }
        )

    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    def numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) not in (None, "")]

    summary_fields = [
        "N",
        "runs",
        "successful_runs",
        "comparable_runs",
        "optimal_rate_percent",
        "gap_relativo_mean",
        "gap_relativo_median",
        "gap_relativo_std",
        "load_imbalance_mean",
        "normalized_load_imbalance_mean",
        "normalized_load_imbalance_median",
        "load_balance_objective_mean",
        "baseline_load_balance_objective_mean",
        "objective_regret_mean",
        "objective_regret_median",
        "baseline_match_offset_free_percent",
        "certified_optimal_offset_free_percent",
        "tempo_total_ms_mean",
        "tempo_total_ms_std",
        "tempo_qaoa_ms_mean",
        "tempo_overhead_ms_mean",
        "num_subqubos_mean",
        "baseline_methods",
        "uncertified_runs",
        "unstable_annealing_runs",
    ] + [_tolerance_column(tolerance) for tolerance in tolerance_tiers]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault(int(row["N"]), []).append(row)

    summary_rows = []
    for n, rows in sorted(grouped.items()):
        successful = [row for row in rows if row["status"] == "success"]
        comparable = [row for row in successful if row["is_optimal"] is not None]
        optimal_values = [bool(row["is_optimal"]) for row in comparable]
        times = numeric(successful, "tempo_total_ms")
        gaps = numeric(successful, "gap_relativo")
        imbalances = numeric(successful, "load_imbalance")
        normalized_imbalances = numeric(successful, "normalized_load_imbalance")
        objectives = numeric(successful, "load_balance_objective")
        baseline_objectives = numeric(
            successful, "baseline_load_balance_objective"
        )
        objective_regrets = numeric(successful, "objective_regret")
        baseline_matches = [
            bool(row["baseline_match_offset_free"])
            for row in successful
            if row.get("baseline_match_offset_free") is not None
        ]
        certified_matches = [
            bool(row["certified_optimal_offset_free"])
            for row in successful
            if row.get("certified_optimal_offset_free") is not None
        ]
        summary_row = {
            "N": n,
            "runs": len(rows),
            "successful_runs": len(successful),
            "comparable_runs": len(comparable),
            "optimal_rate_percent": (
                100.0 * statistics.fmean(optimal_values)
                if optimal_values
                else None
            ),
            "gap_relativo_mean": statistics.fmean(gaps) if gaps else None,
            "gap_relativo_median": statistics.median(gaps) if gaps else None,
            "gap_relativo_std": statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
            "load_imbalance_mean": (
                statistics.fmean(imbalances) if imbalances else None
            ),
            "normalized_load_imbalance_mean": (
                statistics.fmean(normalized_imbalances)
                if normalized_imbalances
                else None
            ),
            "normalized_load_imbalance_median": (
                statistics.median(normalized_imbalances)
                if normalized_imbalances
                else None
            ),
            "load_balance_objective_mean": (
                statistics.fmean(objectives) if objectives else None
            ),
            "baseline_load_balance_objective_mean": (
                statistics.fmean(baseline_objectives)
                if baseline_objectives
                else None
            ),
            "objective_regret_mean": (
                statistics.fmean(objective_regrets)
                if objective_regrets
                else None
            ),
            "objective_regret_median": (
                statistics.median(objective_regrets)
                if objective_regrets
                else None
            ),
            "baseline_match_offset_free_percent": (
                100.0 * statistics.fmean(baseline_matches)
                if baseline_matches
                else None
            ),
            "certified_optimal_offset_free_percent": (
                100.0 * statistics.fmean(certified_matches)
                if certified_matches
                else None
            ),
            "tempo_total_ms_mean": statistics.fmean(times) if times else None,
            "tempo_total_ms_std": statistics.stdev(times) if len(times) > 1 else 0.0,
            "tempo_qaoa_ms_mean": (
                statistics.fmean(numeric(successful, "tempo_qaoa_ms"))
                if numeric(successful, "tempo_qaoa_ms")
                else None
            ),
            "tempo_overhead_ms_mean": (
                statistics.fmean(numeric(successful, "tempo_overhead_ms"))
                if numeric(successful, "tempo_overhead_ms")
                else None
            ),
            "num_subqubos_mean": (
                statistics.fmean(numeric(successful, "num_subqubos"))
                if numeric(successful, "num_subqubos")
                else None
            ),
            "baseline_methods": ";".join(
                sorted({str(row["baseline_method"]) for row in comparable})
            ),
            "uncertified_runs": sum(
                row["baseline_certified"] is False for row in comparable
            ),
            "unstable_annealing_runs": sum(
                row["baseline_method"] == "simulated_annealing"
                and row["annealing_restarts_stable"] is False
                for row in comparable
            ),
        }
        for tolerance in tolerance_tiers:
            summary_row[_tolerance_column(tolerance)] = (
                100.0 * sum(gap <= tolerance for gap in gaps) / len(gaps)
                if gaps
                else None
            )
        summary_rows.append(summary_row)

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    return raw_path, summary_path


def run_adaptive_sweep_scenario(
    scenario_path: Path,
    scenario: dict[str, Any],
    results_dir: Path,
    aggregate_path: Path,
    cumulative_path: Path,
    run_id: str,
    max_variants: int | None,
    dry_run: bool,
) -> list[dict[str, Any]]:
    sweep = scenario.get("sweep", {})
    adaptive = sweep.get("adaptive", {})
    axes = sweep.get("axes", [])
    axis_name = adaptive.get("axis", "num_processes")
    matches = [
        axis
        for axis in axes
        if axis.get("name") == axis_name or axis.get("path") == axis_name
    ]
    if len(matches) != 1 or len(axes) != 1 or sweep.get("cases"):
        raise ValueError(
            "Adaptive sweeps currently require exactly one sweep axis and no sweep cases."
        )

    axis = matches[0]
    path = str(axis["path"])
    coarse_values = [int(value) for value in axis["values"]]
    if coarse_values != sorted(set(coarse_values)):
        raise ValueError("Adaptive sweep values must be unique and increasing.")
    quality_metric = str(adaptive.get("quality_metric", "gap_relativo"))
    quality_ceiling = float(
        adaptive.get(
            "quality_ceiling_mean",
            adaptive.get("gap_ceiling_mean", 0.001),
        )
    )
    monotonic_window = int(adaptive.get("monotonic_window", 3))
    monotonic_min_delta = float(adaptive.get("monotonic_min_delta", 0.0))
    refinement_points = int(adaptive.get("refinement_points", 2))
    extra_points = int(adaptive.get("extra_points_after_transition", 3))
    max_mean_seconds = float(adaptive.get("max_mean_run_seconds", 180.0))
    tolerance_tiers = [
        float(value) for value in adaptive.get("tolerance_tiers", [])
    ]
    resume_from = int(sweep.get("resume", {}).get("resume_from", coarse_values[0]))
    if monotonic_window < 2:
        raise ValueError("sweep.adaptive.monotonic_window must be at least 2.")
    if refinement_points < 0 or extra_points < 0:
        raise ValueError("Adaptive refinement and post-transition counts cannot be negative.")
    if any(value < resume_from for value in coarse_values):
        raise ValueError("Coarse sweep values must not precede sweep.resume.resume_from.")

    base_variant = copy.deepcopy(scenario)
    base_variant.pop("sweep", None)
    records: list[dict[str, Any]] = []
    reused_records = load_reused_scalability_records(
        scenario_path, scenario, results_dir
    )
    by_n: dict[int, list[dict[str, Any]]] = {}
    for record in reused_records:
        n_value = int(
            get_dotted_path(record.get("config", {}), "workload.num_processes")
        )
        by_n.setdefault(n_value, []).append(record)
    max_points = max_variants
    practical_stop = False
    executed_points = 0

    def run_n(n: int, phase: str) -> dict[str, float | None] | None:
        nonlocal practical_stop, executed_points
        if n in by_n:
            return scalability_point_statistics(by_n[n])
        if max_points is not None and executed_points >= max_points:
            return None

        variant = copy.deepcopy(base_variant)
        set_dotted_path(variant, path, n)
        index = executed_points + 1
        variant["id"] = f"{scenario.get('id', 'adaptive')}.{phase}.{n}"
        variant["name"] = f"{scenario.get('name', 'Adaptive sweep')} [N={n}; {phase}]"
        variant["sweep_context"] = {
            "variant_index": index,
            "variant_count": None,
            "adaptive_phase": phase,
            "values": {axis_name: n},
            "paths": {axis_name: path},
        }
        repeats = repeat_count_for_variant(variant)
        print(f"  N={n} ({phase}): {repeats} repeat(s)")
        if dry_run:
            by_n[n] = []
            executed_points += 1
            return None

        point_records = run_variant_repeats(
            scenario_path,
            variant,
            results_dir,
            aggregate_path,
            cumulative_path,
            run_id,
        )
        by_n[n] = point_records
        records.extend(point_records)
        executed_points += 1
        stats = scalability_point_statistics(point_records)
        quality_values = [
            float(record["metrics"][quality_metric])
            for record in point_records
            if record.get("status") == "success"
            and record.get("metrics", {}).get(quality_metric) is not None
        ]
        quality_mean = (
            statistics.fmean(quality_values) if quality_values else None
        )
        quality_text = (
            "unavailable"
            if quality_mean is None
            else f"{quality_mean:.9g}"
        )
        imbalance_text = (
            "unavailable"
            if stats["load_imbalance_mean"] is None
            else f"{stats['load_imbalance_mean']:.6f}"
        )
        time_text = (
            "unavailable"
            if stats["mean_duration"] is None
            else f"{stats['mean_duration']:.2f}s"
        )
        print(
            f"  N={n} aggregate: mean {quality_metric}={quality_text}, "
            f"mean imbalance={imbalance_text}, mean run={time_text}"
        )
        practical_stop = (
            stats["mean_duration"] is not None
            and stats["mean_duration"] > max_mean_seconds
        )
        return stats

    def quality_signal() -> tuple[str, int] | None:
        available = []
        for n_value in sorted(by_n):
            values = [
                float(record["metrics"][quality_metric])
                for record in by_n[n_value]
                if record.get("status") == "success"
                and record.get("metrics", {}).get(quality_metric) is not None
            ]
            if values:
                available.append((n_value, statistics.fmean(values)))
        if not available:
            return None
        last_n, last_quality = available[-1]
        if last_quality > quality_ceiling:
            return f"mean_{quality_metric}_ceiling", last_n
        window = available[-monotonic_window:]
        if len(window) == monotonic_window and all(
            later_quality - earlier_quality > monotonic_min_delta
            for (_, earlier_quality), (_, later_quality) in zip(window, window[1:])
        ):
            return f"monotonic_{quality_metric}_increase", last_n
        return None

    def refinement_values(low: int, high: int) -> list[int]:
        if refinement_points == 0 or high - low <= 1:
            return []
        values = {
            round(low + (high - low) * index / (refinement_points + 1))
            for index in range(1, refinement_points + 1)
        }
        return sorted(value for value in values if low < value < high)

    print(f"\n{scenario_path.stem}: adaptive N sweep ({len(coarse_values)} coarse points)")
    if reused_records:
        reused_values = sorted(by_n)
        print(
            f"  Reusing {len(reused_records)} pilot runs for N={reused_values}; "
            f"new execution starts at N={resume_from}."
        )
    trigger: tuple[str, int] | None = None
    prior_coarse = max((value for value in by_n if value < resume_from), default=None)
    for n in coarse_values:
        stats = run_n(n, "coarse")
        if dry_run:
            continue
        if stats is None or practical_stop:
            break
        trigger = quality_signal()
        if trigger is not None:
            break
        prior_coarse = n

    if dry_run:
        return []

    if not practical_stop and trigger is not None:
        trigger_reason, trigger_n = trigger
        lower_bound = max(resume_from, prior_coarse or resume_from)
        for n in refinement_values(lower_bound, trigger_n):
            if run_n(n, "refinement") is None or practical_stop:
                break
        if not practical_stop:
            post_values = [value for value in coarse_values if value > trigger_n]
            for n in post_values[:extra_points]:
                if run_n(n, "post_transition") is None or practical_stop:
                    break

    raw_path, summary_path = write_scalability_csvs(
        scenario_path,
        reused_records + records,
        results_dir,
        run_id,
        tolerance_tiers=tolerance_tiers,
    )
    print(f"Raw scalability CSV: {display_path(raw_path)}")
    print(f"Summary scalability CSV: {display_path(summary_path)}")
    unstable_annealing = sum(
        record.get("metrics", {}).get("baseline_method") == "simulated_annealing"
        and record.get("metrics", {}).get("annealing_restarts_stable") is False
        for record in reused_records + records
    )
    if unstable_annealing:
        print(
            "Baseline warning: simulated annealing restarts were unstable in "
            f"{unstable_annealing} run(s); inspect the raw CSV before attributing "
            "the gap trend to the pipeline."
        )
    if practical_stop:
        print(f"Practical stop reached: mean run time exceeded {max_mean_seconds:.1f}s.")
    elif trigger is not None:
        print(
            f"Quality transition signal at N={trigger[1]} ({trigger[0]}); "
            "refinement and post-transition points completed."
        )
    else:
        print("No quality transition found in the executed coarse grid.")
    return records


def run_sweep_scenario(
    scenario_path: Path,
    results_dir: Path,
    aggregate_path: Path,
    cumulative_path: Path,
    run_id: str,
    max_variants: int | None,
    dry_run: bool,
) -> list[dict[str, Any]]:
    scenario = load_toml(scenario_path)
    if scenario.get("sweep", {}).get("adaptive", {}).get("enabled", False):
        return run_adaptive_sweep_scenario(
            scenario_path=scenario_path,
            scenario=scenario,
            results_dir=results_dir,
            aggregate_path=aggregate_path,
            cumulative_path=cumulative_path,
            run_id=run_id,
            max_variants=max_variants,
            dry_run=dry_run,
        )

    variants = expand_sweep(scenario)
    if max_variants is not None:
        variants = variants[:max_variants]

    records: list[dict[str, Any]] = []
    print(f"\n{scenario_path.stem}: {len(variants)} sweep variant(s)")
    for variant in variants:
        context = variant.get("sweep_context", {})
        print(
            f"  variant {context.get('variant_index')}/{context.get('variant_count')}: "
            f"{context.get('values', {})}"
        )
        if dry_run:
            continue

        records.extend(
            run_variant_repeats(
                scenario_path,
                variant,
                results_dir,
                aggregate_path,
                cumulative_path,
                run_id,
            )
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expand TOML sweep.axes and run each generated scenario variant."
    )
    parser.add_argument(
        "selectors",
        nargs="*",
        help="Scenario prefixes to run. Omit with --all.",
    )
    parser.add_argument("--all", action="store_true", help="Run every scenario TOML.")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print generated variants without running.")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--scenarios-dir", type=Path, default=SCENARIOS_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    scenarios_dir = args.scenarios_dir.resolve()
    results_dir = args.results_dir.resolve()

    if args.list:
        for path in discover_scenarios(scenarios_dir):
            scenario = load_toml(path)
            axes = scenario.get("sweep", {}).get("axes", [])
            cases = scenario.get("sweep", {}).get("cases", [])
            if cases:
                suffix = f" ({len(cases)} sweep case(s))"
            elif axes:
                suffix = f" ({len(axes)} sweep axis/axes)"
            else:
                suffix = ""
            print(f"{path.stem:<45} {scenario.get('id', ''):<8} {scenario.get('name', '')}{suffix}")
        return 0

    if args.all:
        selected = discover_scenarios(scenarios_dir)
    else:
        if not args.selectors:
            parser.error("provide a scenario selector, or use --all")
        selected = select_scenarios(args.selectors, scenarios_dir)

    run_id = f"sweep_{utc_timestamp()}_{uuid.uuid4().hex[:8]}"
    aggregate_path = results_dir / f"{run_id}_results.jsonl"
    cumulative_path = results_dir / "all_results.jsonl"

    print(f"Selected {len(selected)} scenario(s).")
    print(f"Aggregate file: {display_path(aggregate_path)}")
    print(f"Readable output file: {display_path(aggregate_output_path_for(aggregate_path))}")

    records: list[dict[str, Any]] = []
    for path in selected:
        records.extend(
            run_sweep_scenario(
                scenario_path=path,
                results_dir=results_dir,
                aggregate_path=aggregate_path,
                cumulative_path=cumulative_path,
                run_id=run_id,
                max_variants=args.max_variants,
                dry_run=args.dry_run,
            )
        )

    if args.dry_run:
        print("\nDry run complete; no experiments executed.")
        return 0

    failed = [r for r in records if r.get("status") != "success"]
    print("\n=== Sweep run complete ===")
    print(f"Run id: {run_id}")
    print(f"Successful: {len(records) - len(failed)}")
    print(f"Failed: {len(failed)}")
    print(f"Aggregate file: {display_path(aggregate_path)}")
    print(f"Readable output file: {display_path(aggregate_output_path_for(aggregate_path))}")
    print(f"Cumulative file: {display_path(cumulative_path)}")
    print(f"Cumulative readable file: {display_path(cumulative_output_path_for(cumulative_path))}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
