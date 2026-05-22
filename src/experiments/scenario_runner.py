from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import time
import traceback
import uuid
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback is not expected here.
    import tomli as tomllib


EXPERIMENTS_DIR = Path(__file__).resolve().parent
SRC_DIR = EXPERIMENTS_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
SCENARIOS_DIR = EXPERIMENTS_DIR / "scenarios"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def utc_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def normalize_selector(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def scenario_match_keys(path: Path, scenario: dict[str, Any]) -> list[str]:
    scenario_id = str(scenario.get("id", ""))
    return [
        normalize_selector(path.stem),
        normalize_selector(scenario_id),
        normalize_selector(scenario_id.replace(".", "_")),
    ]


def discover_scenarios(scenarios_dir: Path) -> list[Path]:
    return sorted(scenarios_dir.glob("*.toml"))


def select_scenarios(selectors: list[str], scenarios_dir: Path) -> list[Path]:
    scenarios = discover_scenarios(scenarios_dir)
    if not selectors:
        return scenarios

    selected: list[Path] = []
    for selector in selectors:
        normalized_selector = normalize_selector(selector)
        matches = []
        for path in scenarios:
            scenario = load_toml(path)
            if any(key.startswith(normalized_selector) for key in scenario_match_keys(path, scenario)):
                matches.append(path)
        if not matches:
            raise SystemExit(f"No scenario TOML matched selector '{selector}' in {scenarios_dir}.")
        for match in matches:
            if match not in selected:
                selected.append(match)
    return selected


def jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ModuleNotFoundError:
        pass

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def parse_heuristic(name: str):
    from decomposition.subqubo_heuristics import Heuristic

    normalized = str(name).strip().upper()
    try:
        return Heuristic[normalized]
    except KeyError as exc:
        valid = ", ".join(h.name for h in Heuristic)
        raise ValueError(f"Unknown sorting_strategy '{name}'. Expected one of: {valid}") from exc


def first_sweep_value_for(config: dict[str, Any], path_name: str) -> Any | None:
    for axis in config.get("sweep", {}).get("axes", []):
        if axis.get("path") == path_name and axis.get("values"):
            return axis["values"][0]
    return None


def build_generated_weights(workload_cfg: dict[str, Any], scenario: dict[str, Any], warnings: list[str]) -> list[float]:
    n = workload_cfg.get("num_processes")
    if n is None:
        n = first_sweep_value_for(scenario, "workload.num_processes")
        if n is not None:
            warnings.append(
                "workload.num_processes was taken from the first sweep axis value; "
                "sweep expansion is intentionally not implemented."
            )
    if n is None:
        raise ValueError("Generated workload requires workload.num_processes.")

    n = int(n)
    if n <= 0:
        raise ValueError("Generated workload requires a positive workload.num_processes.")

    if "weights" in workload_cfg:
        weights = [float(w) for w in workload_cfg["weights"]]
        if len(weights) != n:
            raise ValueError("Generated workload weights length does not match num_processes.")
        return weights

    strategy = workload_cfg.get("weight_strategy", "uniform_total")
    if strategy == "uniform_total":
        total_weight = float(workload_cfg.get("total_weight", 1.0))
        return [total_weight / n for _ in range(n)]

    raise ValueError(f"Unsupported generated workload strategy '{strategy}'.")


def build_run_inputs(scenario: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any, list[str]]:
    from data_contracts import DecompositorConfig, QAOAConfig, QUBOConfig, TracerConfig
    from main import SchedulingEngine

    warnings: list[str] = []
    workload_cfg = scenario.get("workload", {})
    qaoa_cfg = scenario.get("qaoa", {})
    qubo_cfg = scenario.get("qubo", {})
    tracer_cfg = scenario.get("tracer", {})
    decomposition_cfg = scenario.get("decomposition", {})

    if "extra_couplings" in qubo_cfg:
        warnings.append(
            "qubo.extra_couplings is present in the TOML but is not supported by "
            "CoreAssignmentBuilder/QUBOConfig and will be ignored."
        )
    if scenario.get("sweep"):
        warnings.append("sweep configuration is present but is not expanded by this runner.")

    mode = workload_cfg.get("mode", "preset")
    num_cores = int(workload_cfg.get("num_cores", decomposition_cfg.get("num_cores", 2)))

    if mode == "preset":
        weights = [float(w) for w in workload_cfg.get("weights", [])]
        if not weights:
            raise ValueError("Preset workload requires workload.weights.")
        preset_snapshot = SchedulingEngine.build_preset_snapshot(weights, num_cores)
        live_mode = False
    elif mode == "generated":
        weights = build_generated_weights(workload_cfg, scenario, warnings)
        preset_snapshot = SchedulingEngine.build_preset_snapshot(weights, num_cores)
        live_mode = False
    elif mode == "live_trace":
        preset_snapshot = None
        live_mode = True
    else:
        raise ValueError(f"Unsupported workload.mode '{mode}'.")

    qaoa = QAOAConfig(
        layers=int(qaoa_cfg.get("layers", 1)),
        steps=int(qaoa_cfg.get("steps", 50)),
        learning_rate=float(qaoa_cfg.get("learning_rate", 0.05)),
        top_k=int(qaoa_cfg.get("top_k", 10)),
        mixer_type=qaoa_cfg.get("mixer_type", "xy"),
        init_gamma=float(qaoa_cfg.get("init_gamma", 0.5)),
        init_beta=float(qaoa_cfg.get("init_beta", qaoa_cfg.get("init_gamma", 0.5))),
    )
    qubo = QUBOConfig(
        penalty=float(qubo_cfg.get("penalty", 1.0)),
        num_cores=num_cores,
        snapshot=None,
        target_load=qubo_cfg.get("target_load"),
    )
    tracer = TracerConfig(
        min_rss=float(tracer_cfg.get("min_rss", 20.0)),
        min_cpu=float(tracer_cfg.get("min_cpu", 0.005)),
        cpu_interval=int(tracer_cfg.get("cpu_interval", 1)),
        num_samples=int(tracer_cfg.get("num_samples", 3)),
        live_mode=bool(tracer_cfg.get("live_mode", live_mode)),
    )
    if mode == "live_trace":
        tracer.live_mode = True

    decompositor = DecompositorConfig(
        qubit_max=int(decomposition_cfg.get("qubit_max", num_cores * 4)),
        num_cores=int(decomposition_cfg.get("num_cores", num_cores)),
        io_alpha=float(decomposition_cfg.get("io_alpha", 0.5)),
        affinity_alpha=float(decomposition_cfg.get("affinity_alpha", 0.8)),
        homogeneity_threshold=float(decomposition_cfg.get("homogeneity_threshold", 0.3)),
        zscore_threshold=float(decomposition_cfg.get("zscore_threshold", 1.5)),
        sorting_strategy=parse_heuristic(decomposition_cfg.get("sorting_strategy", "WEIGHT_DESCENDING")),
    )

    return qaoa, qubo, tracer, decompositor, preset_snapshot, warnings


def summarize_output(output: Any) -> dict[str, Any]:
    import numpy as np

    from data_contracts import IterativeSchedulingOutput, SchedulingOutput

    if isinstance(output, SchedulingOutput):
        result = output.result
        validation = output.validation
        max_probability = None
        if result.probs is not None and len(result.probs):
            max_probability = float(np.max(result.probs))
        return {
            "pipeline": "default",
            "output_type": type(output).__name__,
            "num_variables": output.qubo_instance.num_variables,
            "num_entities": output.qubo_instance.num_entities,
            "num_cores": output.qubo_instance.num_cores,
            "energy": result.energy,
            "feasible": result.is_feasible,
            "optimal": validation.get("is_optimal"),
            "optimality_gap": output.alpha,
            "solve_time_ms": result.solve_time_ms,
            "solver_backend": result.solver_backend,
            "max_probability": max_probability,
            "assignments": result.decoded_assignments,
            "validation": validation,
        }

    if isinstance(output, IterativeSchedulingOutput):
        global_result = output.global_result
        validation = output.validation or {}
        return {
            "pipeline": "iterative",
            "output_type": type(output).__name__,
            "num_variables": output.qubo_instance.num_variables,
            "num_entities": output.qubo_instance.num_entities,
            "num_cores": output.qubo_instance.num_cores,
            "energy": None if global_result is None else global_result.energy,
            "feasible": None if global_result is None else global_result.is_feasible,
            "optimal": validation.get("is_optimal"),
            "optimality_gap": output.alpha,
            "solve_time_ms": output.total_solve_time_ms,
            "num_sub_qubos": output.num_sub_qubos,
            "num_feasible_sub_qubos": output.num_feasible,
            "all_sub_qubos_feasible": output.all_feasible,
            "load_imbalance": output.load_imbalance,
            "final_phi": output.final_phi,
            "assignments": output.final_assignments,
            "validation": validation,
        }

    return {"output_type": type(output).__name__}


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(item), sort_keys=True) + "\n")


def append_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")


def aggregate_output_path_for(aggregate_path: Path) -> Path:
    name = aggregate_path.name
    if name.endswith("_results.jsonl"):
        return aggregate_path.with_name(name.removesuffix("_results.jsonl") + "_output.txt")
    return aggregate_path.with_suffix(".txt")


def cumulative_output_path_for(cumulative_path: Path) -> Path:
    return cumulative_path.with_suffix(".txt")


def format_readable_result(result_record: dict[str, Any], program_output: str) -> str:
    lines: list[str] = []
    title = f"{result_record.get('scenario_stem')} ({result_record.get('scenario_id')})"
    lines.append("=" * 88)
    lines.append(f"EXPERIMENT RESULT: {title}")
    lines.append("=" * 88)
    lines.append(f"Run id:        {result_record.get('run_id')}")
    lines.append(f"Repeat:        {result_record.get('repeat_number')}/{result_record.get('repeat_count')}")
    lines.append(f"Status:        {result_record.get('status')}")
    lines.append(f"Started UTC:   {result_record.get('started_at_utc')}")
    lines.append(f"Duration:      {result_record.get('duration_s', 0.0):.3f}s")
    lines.append(f"Scenario file: {result_record.get('scenario_file')}")
    lines.append(f"JSON result:   {result_record.get('result_path')}")
    lines.append(f"Output file:   {result_record.get('output_path')}")

    warnings = result_record.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("WARNINGS")
        lines.append("-" * 88)
        for warning in warnings:
            lines.append(f"- {warning}")

    metrics = result_record.get("metrics") or {}
    if metrics:
        lines.append("")
        lines.append("METRICS")
        lines.append("-" * 88)
        preferred_keys = [
            "pipeline",
            "output_type",
            "num_entities",
            "num_cores",
            "num_variables",
            "energy",
            "feasible",
            "optimal",
            "optimality_gap",
            "solve_time_ms",
            "solver_backend",
            "max_probability",
            "num_sub_qubos",
            "num_feasible_sub_qubos",
            "all_sub_qubos_feasible",
            "load_imbalance",
            "final_phi",
        ]
        for key in preferred_keys:
            if key in metrics:
                lines.append(f"{key}: {json.dumps(jsonable(metrics[key]), sort_keys=True)}")

        if "assignments" in metrics:
            lines.append("")
            lines.append("ASSIGNMENTS")
            lines.append("-" * 88)
            assignments = metrics["assignments"] or {}
            for entity_id, core in sorted(assignments.items(), key=lambda item: str(item[0])):
                lines.append(f"{entity_id} -> {core}")

        if "validation" in metrics:
            lines.append("")
            lines.append("VALIDATION")
            lines.append("-" * 88)
            lines.append(json.dumps(jsonable(metrics["validation"]), indent=2, sort_keys=True))

    if result_record.get("error"):
        lines.append("")
        lines.append("ERROR")
        lines.append("-" * 88)
        lines.append(json.dumps(jsonable(result_record["error"]), indent=2, sort_keys=True))

    lines.append("")
    lines.append("PROGRAM OUTPUT")
    lines.append("-" * 88)
    stripped_output = program_output.strip()
    lines.append(stripped_output if stripped_output else "(no program output captured)")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


def run_scenario_once(
    scenario_path: Path,
    scenario: dict[str, Any],
    results_dir: Path,
    aggregate_path: Path,
    cumulative_path: Path,
    run_id: str,
    repeat_index: int,
    repeat_count: int,
) -> dict[str, Any]:
    started_at = time.time()
    timestamp = utc_timestamp()
    scenario_stem = scenario_path.stem
    scenario_result_dir = results_dir / f"{scenario_stem}_result"
    scenario_result_dir.mkdir(parents=True, exist_ok=True)

    repeat_number = repeat_index + 1
    repeat_suffix = f"repeat_{repeat_number:03d}_of_{repeat_count:03d}"
    result_path = scenario_result_dir / f"{timestamp}_{run_id}_{repeat_suffix}_result.json"
    latest_path = scenario_result_dir / "latest_result.json"
    output_path = scenario_result_dir / f"{timestamp}_{run_id}_{repeat_suffix}_output.txt"
    latest_output_path = scenario_result_dir / "latest_output.txt"
    log_path = scenario_result_dir / f"{timestamp}_{run_id}_{repeat_suffix}.log"
    copied_scenario_path = scenario_result_dir / "scenario.toml"

    print(
        f"\n=== Running {scenario_stem} ({scenario.get('id', 'no-id')}) "
        f"repeat {repeat_number}/{repeat_count} ==="
    )

    log_buffer = io.StringIO()
    result_record: dict[str, Any] = {
        "run_id": run_id,
        "scenario_file": str(scenario_path.relative_to(PROJECT_ROOT)),
        "scenario_stem": scenario_stem,
        "scenario_id": scenario.get("id"),
        "scenario_name": scenario.get("name"),
        "tier": scenario.get("tier"),
        "category": scenario.get("category"),
        "repeat_index": repeat_index,
        "repeat_number": repeat_number,
        "repeat_count": repeat_count,
        "started_at_epoch": started_at,
        "started_at_utc": timestamp,
        "status": "unknown",
        "warnings": [],
        "config": scenario,
    }

    try:
        from main import SchedulingEngine

        qaoa, qubo, tracer, decompositor, preset_snapshot, warnings = build_run_inputs(scenario)
        result_record["warnings"].extend(warnings)
        result_record["resolved_config"] = {
            "qaoa": qaoa,
            "qubo": qubo,
            "tracer": tracer,
            "decomposition": decompositor,
            "preset_snapshot": preset_snapshot,
        }

        with contextlib.redirect_stdout(Tee(sys.stdout, log_buffer)):
            output = SchedulingEngine.run_job(
                qaoa_cfg=qaoa,
                qubo_cfg=qubo,
                tracer_cfg=tracer,
                decompositor_cfg=decompositor,
                preset_snapshot=preset_snapshot,
            )

        result_record["status"] = "success"
        result_record["metrics"] = summarize_output(output)
    except Exception as exc:
        result_record["status"] = "failed"
        if type(exc).__name__ == "InfeasibleSubQUBOError":
            result_record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "subqubo_index": getattr(exc, "subqubo_index", None),
                "failed_result": getattr(exc, "result", None),
                "final_assignments_before_failure": getattr(exc, "final_assignments", None),
                "phi_history_before_failure": getattr(exc, "phi_history", None),
            }
        else:
            result_record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        result_record["traceback"] = traceback.format_exc()
        print(f"FAILED {scenario_stem}: {type(exc).__name__}: {exc}")
    finally:
        ended_at = time.time()
        result_record["ended_at_epoch"] = ended_at
        result_record["duration_s"] = ended_at - started_at
        result_record["result_path"] = str(result_path.relative_to(PROJECT_ROOT))
        result_record["output_path"] = str(output_path.relative_to(PROJECT_ROOT))
        result_record["log_path"] = str(log_path.relative_to(PROJECT_ROOT))

        program_output = log_buffer.getvalue()
        log_path.write_text(program_output, encoding="utf-8")
        shutil.copy2(scenario_path, copied_scenario_path)

        serialized = json.dumps(jsonable(result_record), indent=2, sort_keys=True)
        result_path.write_text(serialized + "\n", encoding="utf-8")
        latest_path.write_text(serialized + "\n", encoding="utf-8")

        readable_result = format_readable_result(result_record, program_output)
        output_path.write_text(readable_result, encoding="utf-8")
        latest_output_path.write_text(readable_result, encoding="utf-8")

        append_jsonl(aggregate_path, result_record)
        append_jsonl(cumulative_path, result_record)
        append_text(aggregate_output_path_for(aggregate_path), readable_result)
        append_text(cumulative_output_path_for(cumulative_path), readable_result)

    print(f"Saved result: {result_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved readable output: {output_path.relative_to(PROJECT_ROOT)}")
    return result_record


def scenario_repeat_count(scenario: dict[str, Any]) -> int:
    repeats = int(scenario.get("execution", {}).get("repeats", 1))
    if repeats < 1:
        raise ValueError("execution.repeats must be greater than or equal to 1.")
    return repeats


def run_scenario(
    scenario_path: Path,
    results_dir: Path,
    aggregate_path: Path,
    cumulative_path: Path,
    run_id: str,
) -> list[dict[str, Any]]:
    scenario = load_toml(scenario_path)
    repeat_count = scenario_repeat_count(scenario)
    if repeat_count > 1:
        print(f"\n{scenario_path.stem}: configured repeats={repeat_count}")

    return [
        run_scenario_once(
            scenario_path=scenario_path,
            scenario=scenario,
            results_dir=results_dir,
            aggregate_path=aggregate_path,
            cumulative_path=cumulative_path,
            run_id=run_id,
            repeat_index=repeat_index,
            repeat_count=repeat_count,
        )
        for repeat_index in range(repeat_count)
    ]


def list_scenarios(scenarios_dir: Path) -> None:
    for path in discover_scenarios(scenarios_dir):
        scenario = load_toml(path)
        print(f"{path.stem:<45} {scenario.get('id', ''):<8} {scenario.get('name', '')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run static TOML experiment scenarios through SchedulingEngine."
    )
    parser.add_argument(
        "selectors",
        nargs="*",
        help="Scenario prefixes to run, e.g. t_1, t1, t1_1, or T1.1. Omit with --all.",
    )
    parser.add_argument("--all", action="store_true", help="Run every scenario TOML.")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit.")
    parser.add_argument("--scenarios-dir", type=Path, default=SCENARIOS_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    scenarios_dir = args.scenarios_dir.resolve()
    results_dir = args.results_dir.resolve()

    if args.list:
        list_scenarios(scenarios_dir)
        return 0

    if args.all:
        selectors: list[str] = []
    else:
        selectors = args.selectors
        if not selectors:
            parser.error("provide a scenario selector, or use --all")

    selected = select_scenarios(selectors, scenarios_dir)
    if not selected:
        raise SystemExit("No scenarios selected.")

    run_id = f"run_{utc_timestamp()}_{uuid.uuid4().hex[:8]}"
    aggregate_path = results_dir / f"{run_id}_results.jsonl"
    cumulative_path = results_dir / "all_results.jsonl"

    print(f"Selected {len(selected)} scenario(s).")
    print(f"Aggregate file: {aggregate_path.relative_to(PROJECT_ROOT)}")
    print(f"Readable output file: {aggregate_output_path_for(aggregate_path).relative_to(PROJECT_ROOT)}")

    records: list[dict[str, Any]] = []
    for path in selected:
        records.extend(run_scenario(path, results_dir, aggregate_path, cumulative_path, run_id))

    failed = [r for r in records if r.get("status") != "success"]
    print("\n=== Experiment run complete ===")
    print(f"Run id: {run_id}")
    print(f"Successful: {len(records) - len(failed)}")
    print(f"Failed: {len(failed)}")
    print(f"Aggregate file: {aggregate_path.relative_to(PROJECT_ROOT)}")
    print(f"Readable output file: {aggregate_output_path_for(aggregate_path).relative_to(PROJECT_ROOT)}")
    print(f"Cumulative file: {cumulative_path.relative_to(PROJECT_ROOT)}")
    print(f"Cumulative readable file: {cumulative_output_path_for(cumulative_path).relative_to(PROJECT_ROOT)}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
