"""Isolated runtime for research experiments.

The production scheduling engine deliberately keeps its original API and
behaviour.  Experimental controls, alternative baselines, instrumentation and
derived metrics live here and are only used by the experiment runners.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from abstract.abstract import BaseSolver
from builder.builder_core import CoreAssignmentBuilder
from data_contracts import (
    DecompositorConfig,
    IterativeSchedulingOutput,
    QAOAConfig,
    QUBOConfig,
    QUBOInstance,
    SchedulingOutput,
    SolverResult,
    SystemSnapshot,
    TracerConfig,
    Workload,
)
from decomposition.adaptive_cluster import AdaptiveCluster
from decomposition.subqubo_decomposer import SubQUBODecomposer
from pipeline.default_pipeline import DefaultPipeline
from pipeline.iterative_pipeline import IterativePipeline
from solver.brute_force_solver import BruteForceSolver
from solver.pennylane_solver import PennylaneSolver
from tracer.process_tracer import ProcessTracer
from visualizer.graph_visualizer import Visualizer
from visualizer.iterative_visualizer import IterativeVisualizer
from visualizer.snapshot_visualization import SnapshotVisualizer


@dataclass
class InvestigativeQAOAConfig(QAOAConfig):
    init_strategy: Literal["fixed", "random"] = "fixed"
    random_seed: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.init_strategy not in ("fixed", "random"):
            raise ValueError(
                f"Unsupported init_strategy '{self.init_strategy}'. "
                "Expected 'fixed' or 'random'."
            )


@dataclass(frozen=True)
class InvestigativeRunConfig:
    pipeline_mode: Literal["auto", "default", "iterative"] = "auto"
    cluster_preset_snapshot: bool = False
    enable_visualization: bool = True

    def __post_init__(self) -> None:
        if self.pipeline_mode not in ("auto", "default", "iterative"):
            raise ValueError(
                f"Unsupported pipeline mode '{self.pipeline_mode}'. "
                "Expected auto, default, or iterative."
            )


@dataclass(frozen=True)
class InvestigativeValidationConfig:
    always_run_annealing: bool = False
    annealing_seed: int | None = 42
    baseline_method: str = "legacy"
    brute_force_max_n: int = 20
    brute_force_time_box_seconds: float | None = 60.0
    annealing_sweeps: int = 1000
    annealing_restarts: int = 8
    optimality_rtol: float = 1e-9
    optimality_atol: float = 1e-9


@dataclass
class InvestigativeOutput:
    output: SchedulingOutput | IterativeSchedulingOutput
    component_timings_ms: dict[str, float] = field(default_factory=dict)
    experiment_metadata: dict[str, Any] = field(default_factory=dict)


class InvestigativePennylaneSolver(PennylaneSolver):
    """Research-only QAOA variant with seeded initialization and top-k selection."""

    def __init__(self, qaoa_cfg: InvestigativeQAOAConfig):
        super().__init__(qaoa_cfg)
        self.init_strategy = qaoa_cfg.init_strategy
        self.random_seed = qaoa_cfg.random_seed

    def _solve_on_device(self, qubo: QUBOInstance, device_name: str) -> SolverResult:
        start_time = time.perf_counter()
        num_qubits = qubo.num_variables
        num_entities = qubo.num_entities
        num_cores = qubo.num_cores
        cost_h, _ = self.matrix_to_hamiltonian(qubo.Q)

        if self.mixer_type == "x":
            mixer_h = qml.qaoa.x_mixer(range(num_qubits))

        dev = self._make_device(device_name, num_qubits)

        @qml.qnode(dev, diff_method="adjoint")
        def cost_function(params):
            self._prepare_initial_state(num_qubits, num_entities, num_cores)
            gammas, betas = params
            for layer in range(self.p):
                qml.qaoa.cost_layer(gammas[layer], cost_h)
                if self.mixer_type == "xy":
                    self._xy_mixer_layer(betas[layer], num_entities, num_cores)
                else:
                    qml.qaoa.mixer_layer(betas[layer], mixer_h)
            return qml.expval(cost_h)

        if self.init_strategy == "random":
            rng = np.random.default_rng(self.random_seed)
            initial_params = np.vstack(
                [
                    rng.uniform(0.0, 1.0, size=self.p),
                    rng.uniform(0.0, 1.0, size=self.p),
                ]
            )
        else:
            initial_params = np.array(
                [[self.init_gamma] * self.p, [self.init_beta] * self.p]
            )

        params = pnp.array(initial_params, requires_grad=True)
        optimizer = qml.AdamOptimizer(stepsize=self.learning_rate)
        energies_over_time = []
        for _ in range(self.steps):
            params, energy = optimizer.step_and_cost(cost_function, params)
            energies_over_time.append(float(energy))

        @qml.qnode(dev)
        def get_probs(params):
            self._prepare_initial_state(num_qubits, num_entities, num_cores)
            gammas, betas = params
            for layer in range(self.p):
                qml.qaoa.cost_layer(gammas[layer], cost_h)
                if self.mixer_type == "xy":
                    self._xy_mixer_layer(betas[layer], num_entities, num_cores)
                else:
                    qml.qaoa.mixer_layer(betas[layer], mixer_h)
            return qml.probs(wires=range(num_qubits))

        probs = get_probs(params)
        ranked_indices = np.argsort(np.asarray(probs))[::-1]
        candidate_indices = ranked_indices[: min(self.top_k, len(ranked_indices))]
        best_bitstring = None
        best_energy = float("inf")
        best_decoded = None
        best_feasible = False

        for state_index in candidate_indices:
            bits = np.array(
                [int(bit) for bit in bin(int(state_index))[2:].zfill(num_qubits)]
            )
            decoded, is_feasible = self.decode_assignments(bits, qubo)
            energy = float(bits.T @ qubo.Q @ bits)
            if is_feasible and (not best_feasible or energy < best_energy):
                best_bitstring = bits
                best_energy = energy
                best_decoded = decoded
                best_feasible = True

        if best_bitstring is None:
            fallback_index = int(ranked_indices[0])
            best_bitstring = np.array(
                [int(bit) for bit in bin(fallback_index)[2:].zfill(num_qubits)]
            )
            best_decoded, _ = self.decode_assignments(best_bitstring, qubo)
            best_energy = float(best_bitstring.T @ qubo.Q @ best_bitstring)

        return SolverResult(
            bitstring=pnp.array(best_bitstring),
            decoded_assignments=best_decoded,
            energy=best_energy,
            is_feasible=best_feasible,
            solver_backend=f"qaoa_pennylane_{device_name}_{self.mixer_type}_mixer",
            solve_time_ms=(time.perf_counter() - start_time) * 1000,
            solver_params={
                "p_layers": self.p,
                "opt_steps": self.steps,
                "mixer_type": self.mixer_type,
                "init_strategy": self.init_strategy,
                "init_gamma": self.init_gamma,
                "init_beta": self.init_beta,
                "random_seed": self.random_seed,
                "initial_gamma": np.asarray(initial_params[0]).tolist(),
                "initial_beta": np.asarray(initial_params[1]).tolist(),
                "final_gamma": np.asarray(params[0]).tolist(),
                "final_beta": np.asarray(params[1]).tolist(),
                "device": device_name,
                "top_k": self.top_k,
                "selection_pool": "top_k_probable_states",
            },
            probs=probs,
            convergence_curve=energies_over_time,
        )

    def decode_assignments(
        self,
        bitstring,
        qubo: QUBOInstance,
    ) -> tuple[dict[int, int], bool]:
        decoded = {}
        for entity_index in range(qubo.num_entities):
            start = entity_index * qubo.num_cores
            stop = start + qubo.num_cores
            group = np.asarray(bitstring[start:stop])
            if group.sum() != 1:
                continue
            offset = int(np.argmax(group))
            entity_id, core_id = qubo.variable_map[start + offset]
            decoded[entity_id] = core_id
        return decoded, len(decoded) == qubo.num_entities


class FeasibleBruteForceSolver(BaseSolver):
    """Exact research baseline that enumerates only one-hot assignments."""

    def __init__(
        self,
        max_entities: int = 20,
        time_box_seconds: float | None = 60.0,
    ):
        if max_entities < 1:
            raise ValueError("max_entities must be greater than or equal to 1.")
        if time_box_seconds is not None and time_box_seconds <= 0:
            raise ValueError("time_box_seconds must be positive when provided.")
        self.max_entities = max_entities
        self.time_box_seconds = time_box_seconds

    def solve(self, qubo: QUBOInstance) -> SolverResult:
        if qubo.num_entities > self.max_entities:
            raise RuntimeError(
                f"Feasible brute-force refused: {qubo.num_entities} entities "
                f"exceeds {self.max_entities} limit."
            )
        if qubo.num_variables != qubo.num_entities * qubo.num_cores:
            raise ValueError(
                "FeasibleBruteForceSolver expects a one-hot entity/core QUBO."
            )

        start = time.perf_counter()
        if qubo.num_cores == 2:
            best_x, best_energy, evaluated = self._solve_two_core_gray(qubo, start)
        else:
            best_x, best_energy, evaluated = self._solve_general(qubo, start)

        decoded, is_feasible = BruteForceSolver._decode_assignments(best_x, qubo)
        return SolverResult(
            bitstring=best_x,
            decoded_assignments=decoded,
            energy=best_energy,
            is_feasible=is_feasible,
            solver_backend="brute_force_feasible_assignments",
            solve_time_ms=(time.perf_counter() - start) * 1000,
            solver_params={
                "certified": True,
                "search_space": "one_hot_feasible_assignments",
                "assignments_evaluated": evaluated,
                "max_entities": self.max_entities,
                "time_box_seconds": self.time_box_seconds,
            },
        )

    def _solve_two_core_gray(
        self,
        qubo: QUBOInstance,
        start: float,
    ) -> tuple[np.ndarray, float, int]:
        num_entities = qubo.num_entities
        bits = np.zeros(qubo.num_variables, dtype=int)
        bits[::2] = 1
        q_col = qubo.Q @ bits
        q_row = bits @ qubo.Q
        energy = float(bits.T @ qubo.Q @ bits)
        best_bits = bits.copy()
        best_energy = energy
        previous_gray = 0
        total = 1 << num_entities

        for state in range(1, total):
            if state % 1024 == 0:
                self._check_time_box(start, state)
                energy = float(bits.T @ qubo.Q @ bits)
                q_col = qubo.Q @ bits
                q_row = bits @ qubo.Q

            gray = state ^ (state >> 1)
            changed = gray ^ previous_gray
            entity = changed.bit_length() - 1
            old_core = int(bits[entity * 2 + 1])
            old_var = entity * 2 + old_core
            new_var = entity * 2 + (1 - old_core)
            delta = float(
                (-q_col[old_var] + q_col[new_var])
                + (-q_row[old_var] + q_row[new_var])
                + qubo.Q[old_var, old_var]
                + qubo.Q[new_var, new_var]
                - qubo.Q[old_var, new_var]
                - qubo.Q[new_var, old_var]
            )
            bits[old_var] = 0
            bits[new_var] = 1
            energy += delta
            q_col += -qubo.Q[:, old_var] + qubo.Q[:, new_var]
            q_row += -qubo.Q[old_var, :] + qubo.Q[new_var, :]
            previous_gray = gray

            if energy < best_energy:
                energy = float(bits.T @ qubo.Q @ bits)
                if energy < best_energy:
                    best_energy = energy
                    best_bits = bits.copy()

        return best_bits, float(best_bits.T @ qubo.Q @ best_bits), total

    def _solve_general(
        self,
        qubo: QUBOInstance,
        start: float,
    ) -> tuple[np.ndarray, float, int]:
        best_bits = None
        best_energy = float("inf")
        evaluated = 0
        for assignment in product(
            range(qubo.num_cores),
            repeat=qubo.num_entities,
        ):
            evaluated += 1
            if evaluated % 1024 == 0:
                self._check_time_box(start, evaluated)
            bits = np.zeros(qubo.num_variables, dtype=int)
            for entity, core in enumerate(assignment):
                bits[entity * qubo.num_cores + core] = 1
            energy = float(bits.T @ qubo.Q @ bits)
            if energy < best_energy:
                best_energy = energy
                best_bits = bits
        return best_bits, best_energy, evaluated

    def _check_time_box(self, start: float, evaluated: int) -> None:
        if (
            self.time_box_seconds is not None
            and time.perf_counter() - start > self.time_box_seconds
        ):
            raise RuntimeError(
                "Feasible brute-force time box exceeded after "
                f"{evaluated} assignments ({self.time_box_seconds:.1f}s limit)."
            )


class AnnealingSolver(BaseSolver):
    """One-hot simulated annealing baseline used only by investigations."""

    def __init__(
        self,
        sweeps: int = 1000,
        restarts: int = 8,
        initial_temperature: float | None = None,
        final_temperature: float | None = None,
        seed: int | None = 42,
    ):
        if sweeps < 1:
            raise ValueError("sweeps must be greater than or equal to 1.")
        if restarts < 1:
            raise ValueError("restarts must be greater than or equal to 1.")
        self.sweeps = sweeps
        self.restarts = restarts
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.seed = seed

    def solve(self, qubo: QUBOInstance) -> SolverResult:
        if qubo.num_entities < 1:
            raise ValueError("QUBO must contain at least one entity.")
        if qubo.num_cores < 1:
            raise ValueError("QUBO must contain at least one core.")
        if qubo.num_variables != qubo.num_entities * qubo.num_cores:
            raise ValueError("AnnealingSolver expects a one-hot entity/core QUBO.")

        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        initial_temperature = self.initial_temperature
        if initial_temperature is None:
            initial_temperature = self._estimate_initial_temperature(qubo)
        final_temperature = self.final_temperature
        if final_temperature is None:
            final_temperature = max(initial_temperature * 1e-3, 1e-9)

        best_bitstring = None
        best_energy = float("inf")
        convergence_curve = []
        move_attempts = 0
        restart_best_energies = []

        for restart in range(self.restarts):
            assignments = self._initial_assignments(
                qubo.num_entities,
                qubo.num_cores,
                rng,
                restart,
            )
            bitstring = self._assignments_to_bitstring(assignments, qubo)
            energy = float(bitstring.T @ qubo.Q @ bitstring)
            restart_best_energy = energy
            q_col = qubo.Q @ bitstring
            q_row = bitstring @ qubo.Q

            if energy < best_energy:
                best_energy = energy
                best_bitstring = bitstring.copy()

            for sweep in range(self.sweeps):
                temperature = self._temperature(
                    sweep,
                    initial_temperature,
                    final_temperature,
                )
                for _ in range(qubo.num_entities):
                    if qubo.num_cores == 1:
                        break
                    entity = int(rng.integers(qubo.num_entities))
                    old_core = int(assignments[entity])
                    new_core = int(rng.integers(qubo.num_cores - 1))
                    if new_core >= old_core:
                        new_core += 1
                    old_var = entity * qubo.num_cores + old_core
                    new_var = entity * qubo.num_cores + new_core
                    delta = self._move_delta(
                        qubo.Q,
                        q_col,
                        q_row,
                        old_var,
                        new_var,
                    )
                    move_attempts += 1
                    if delta <= 0.0 or rng.random() < np.exp(
                        -delta / max(temperature, 1e-12)
                    ):
                        bitstring[old_var] = 0
                        bitstring[new_var] = 1
                        assignments[entity] = new_core
                        energy += delta
                        q_col += -qubo.Q[:, old_var] + qubo.Q[:, new_var]
                        q_row += -qubo.Q[old_var, :] + qubo.Q[new_var, :]
                        if energy < best_energy:
                            best_energy = float(energy)
                            best_bitstring = bitstring.copy()
                        if energy < restart_best_energy:
                            restart_best_energy = float(energy)
                convergence_curve.append(best_energy)
            restart_best_energies.append(float(restart_best_energy))

        best_energy = float(best_bitstring.T @ qubo.Q @ best_bitstring)
        decoded, is_feasible = self._decode_assignments(best_bitstring, qubo)
        restart_energies = np.asarray(restart_best_energies, dtype=float)
        return SolverResult(
            bitstring=best_bitstring,
            decoded_assignments=decoded,
            energy=best_energy,
            is_feasible=is_feasible,
            solver_backend="simulated_annealing_one_hot",
            solve_time_ms=(time.perf_counter() - start) * 1000,
            solver_params={
                "sweeps": self.sweeps,
                "restarts": self.restarts,
                "initial_temperature": initial_temperature,
                "final_temperature": final_temperature,
                "seed": self.seed,
                "move_attempts": move_attempts,
                "representation": "one_hot_assignment_moves",
                "restart_best_energies": restart_best_energies,
                "restart_energy_std": float(np.std(restart_energies)),
                "restart_energy_range": float(np.ptp(restart_energies)),
                "best_energy_hit_fraction": float(
                    np.isclose(
                        restart_energies,
                        best_energy,
                        rtol=1e-9,
                        atol=1e-9,
                    ).mean()
                ),
                "restarts_stable": bool(
                    np.allclose(
                        restart_energies,
                        best_energy,
                        rtol=1e-9,
                        atol=1e-9,
                    )
                ),
            },
            convergence_curve=convergence_curve,
        )

    @staticmethod
    def _estimate_initial_temperature(qubo: QUBOInstance) -> float:
        nonzero = np.abs(qubo.Q[np.abs(qubo.Q) > 1e-12])
        scale = float(np.median(nonzero)) if nonzero.size else 1.0
        return max(scale * max(qubo.num_entities, 1), 1e-6)

    def _temperature(
        self,
        sweep: int,
        initial_temperature: float,
        final_temperature: float,
    ) -> float:
        if self.sweeps == 1:
            return final_temperature
        fraction = sweep / (self.sweeps - 1)
        return initial_temperature * (
            final_temperature / initial_temperature
        ) ** fraction

    @staticmethod
    def _initial_assignments(
        num_entities: int,
        num_cores: int,
        rng: np.random.Generator,
        restart: int,
    ) -> np.ndarray:
        if restart == 0:
            return np.arange(num_entities, dtype=int) % num_cores
        return rng.integers(0, num_cores, size=num_entities, dtype=int)

    @staticmethod
    def _assignments_to_bitstring(
        assignments: np.ndarray,
        qubo: QUBOInstance,
    ) -> np.ndarray:
        bitstring = np.zeros(qubo.num_variables, dtype=int)
        for entity, core in enumerate(assignments):
            bitstring[entity * qubo.num_cores + int(core)] = 1
        return bitstring

    @staticmethod
    def _move_delta(
        matrix: np.ndarray,
        q_col: np.ndarray,
        q_row: np.ndarray,
        old_var: int,
        new_var: int,
    ) -> float:
        return float(
            (-q_col[old_var] + q_col[new_var])
            + (-q_row[old_var] + q_row[new_var])
            + matrix[old_var, old_var]
            + matrix[new_var, new_var]
            - matrix[old_var, new_var]
            - matrix[new_var, old_var]
        )

    @staticmethod
    def _decode_assignments(
        bitstring: np.ndarray,
        qubo: QUBOInstance,
    ) -> tuple[dict[int, int], bool]:
        decoded = {}
        for entity in range(qubo.num_entities):
            start = entity * qubo.num_cores
            stop = start + qubo.num_cores
            group = bitstring[start:stop]
            if group.sum() != 1:
                continue
            offset = int(np.argmax(group))
            entity_id, core_id = qubo.variable_map[start + offset]
            decoded[entity_id] = core_id
        return decoded, len(decoded) == qubo.num_entities


class InvestigativeValidator:
    """Research baseline comparison, separate from production validation."""

    def __init__(self, config: InvestigativeValidationConfig | None = None):
        self.config = config or InvestigativeValidationConfig()
        method = self.config.baseline_method.strip().lower()
        method = {
            "brute_force": "bruteforce",
            "annealing": "simulated_annealing",
            "sa": "simulated_annealing",
        }.get(method, method)
        if method not in {
            "legacy",
            "auto",
            "bruteforce",
            "simulated_annealing",
        }:
            raise ValueError(
                "baseline_method must be legacy, auto, bruteforce, "
                "or simulated_annealing."
            )
        self.baseline_method = method

    def validate(self, qubo: QUBOInstance, result: SolverResult) -> dict[str, Any]:
        candidate_energy = float(result.bitstring.T @ qubo.Q @ result.bitstring)
        errors = []
        for entity in range(qubo.num_entities):
            start = entity * qubo.num_cores
            group = result.bitstring[start : start + qubo.num_cores]
            if group.sum() != 1:
                entity_id = qubo.variable_map[start][0]
                errors.append(
                    f"Entity {entity_id} assigned to {int(group.sum())} options"
                )

        global_energy = None
        global_assignments = {}
        unconstrained_energy = None
        unconstrained_assignments = {}
        unconstrained_is_feasible = None
        brute_force_error = None
        brute_force_solve_time_ms = None
        annealing_result = None
        baseline_method_used = None
        baseline_certified = False

        should_try_legacy = self.baseline_method == "legacy"
        should_try_feasible = self.baseline_method == "bruteforce" or (
            self.baseline_method == "auto"
            and qubo.num_entities <= self.config.brute_force_max_n
        )
        if should_try_legacy:
            try:
                exact = BruteForceSolver().solve(qubo)
                global_energy = float(exact.energy)
                global_assignments = exact.decoded_assignments
                brute_force_solve_time_ms = exact.solve_time_ms
                unconstrained_energy = exact.solver_params.get("unconstrained_energy")
                unconstrained_assignments = exact.solver_params.get(
                    "unconstrained_assignments",
                    {},
                )
                unconstrained_is_feasible = exact.solver_params.get(
                    "unconstrained_is_feasible"
                )
                baseline_method_used = "bruteforce"
                baseline_certified = True
            except RuntimeError as exc:
                brute_force_error = str(exc)
        elif should_try_feasible:
            try:
                exact = FeasibleBruteForceSolver(
                    max_entities=self.config.brute_force_max_n,
                    time_box_seconds=self.config.brute_force_time_box_seconds,
                ).solve(qubo)
                global_energy = float(exact.energy)
                global_assignments = exact.decoded_assignments
                brute_force_solve_time_ms = exact.solve_time_ms
                baseline_method_used = "bruteforce"
                baseline_certified = True
            except RuntimeError as exc:
                brute_force_error = str(exc)

        needs_annealing = (
            self.baseline_method == "simulated_annealing"
            or (self.baseline_method == "auto" and global_energy is None)
            or (self.baseline_method == "legacy" and global_energy is None)
        )
        annealing_error = None
        if needs_annealing or self.config.always_run_annealing:
            try:
                annealing_result = AnnealingSolver(
                    sweeps=self.config.annealing_sweeps,
                    restarts=self.config.annealing_restarts,
                    seed=self.config.annealing_seed,
                ).solve(qubo)
            except Exception as exc:
                annealing_error = str(exc)

        annealing_energy = (
            None if annealing_result is None else float(annealing_result.energy)
        )
        annealing_params = (
            {} if annealing_result is None else annealing_result.solver_params
        )
        annealing_feasible = (
            None if annealing_result is None else annealing_result.is_feasible
        )
        # A feasible QUBO energy contains a large additive offset.  Scaling the
        # tolerance by abs(energy) therefore makes equivalence looser as N grows.
        # Energy differences cancel the offset, so use an absolute objective
        # tolerance here; offset-free load metrics are added to the run summary.
        tolerance_against_annealing = (
            self.config.optimality_atol
            if annealing_energy is not None
            else None
        )
        beats_annealing = (
            None
            if annealing_energy is None
            else not errors
            and bool(annealing_feasible)
            and candidate_energy <= annealing_energy + tolerance_against_annealing
        )

        if baseline_method_used == "bruteforce":
            baseline_energy = global_energy
        elif needs_annealing and annealing_energy is not None:
            baseline_method_used = "simulated_annealing"
            baseline_energy = annealing_energy
        else:
            baseline_energy = None

        is_optimal = None
        relative_gap = None
        if baseline_energy is not None:
            tolerance = self.config.optimality_atol
            is_optimal = not errors and candidate_energy <= baseline_energy + tolerance
            relative_gap = max(0.0, candidate_energy - baseline_energy) / max(
                abs(baseline_energy),
                self.config.optimality_atol,
                1e-15,
            )

        return {
            "valid": not errors,
            "errors": errors,
            "candidate_energy": candidate_energy,
            "candidate_assignments": result.decoded_assignments,
            "global_energy": global_energy,
            "global_assignments": global_assignments,
            "unconstrained_energy": unconstrained_energy,
            "unconstrained_assignments": unconstrained_assignments,
            "unconstrained_is_feasible": unconstrained_is_feasible,
            "is_optimal": is_optimal,
            "brute_force_error": brute_force_error,
            "brute_force_solve_time_ms": brute_force_solve_time_ms,
            "annealing_energy": annealing_energy,
            "annealing_assignments": (
                {} if annealing_result is None else annealing_result.decoded_assignments
            ),
            "annealing_is_feasible": annealing_feasible,
            "annealing_solve_time_ms": (
                None if annealing_result is None else annealing_result.solve_time_ms
            ),
            "annealing_backend": (
                None if annealing_result is None else annealing_result.solver_backend
            ),
            "annealing_gap": (
                None if annealing_energy is None else candidate_energy - annealing_energy
            ),
            "beats_annealing": beats_annealing,
            "annealing_error": annealing_error,
            "annealing_restart_best_energies": annealing_params.get(
                "restart_best_energies"
            ),
            "annealing_restart_energy_std": annealing_params.get(
                "restart_energy_std"
            ),
            "annealing_restart_energy_range": annealing_params.get(
                "restart_energy_range"
            ),
            "annealing_best_energy_hit_fraction": annealing_params.get(
                "best_energy_hit_fraction"
            ),
            "annealing_restarts_stable": annealing_params.get("restarts_stable"),
            "baseline_method_requested": self.baseline_method,
            "baseline_method": baseline_method_used,
            "baseline_certified": baseline_certified,
            "baseline_energy": baseline_energy,
            "pipeline_energy": candidate_energy,
            "relative_gap": relative_gap,
            "optimality_rtol": self.config.optimality_rtol,
            "optimality_atol": self.config.optimality_atol,
        }


class _TimedBuilder:
    def __init__(self, config: QUBOConfig, timings: dict[str, float]):
        self._builder = CoreAssignmentBuilder(config)
        self._timings = timings

    def build(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return self._builder.build(*args, **kwargs)
        finally:
            self._timings["qubo_build_ms"] += (
                time.perf_counter() - start
            ) * 1000


class _TimedSolver:
    def __init__(
        self,
        config: InvestigativeQAOAConfig,
        timings: dict[str, float],
    ):
        self._solver = InvestigativePennylaneSolver(config)
        self._timings = timings

    def solve(self, qubo: QUBOInstance) -> SolverResult:
        start = time.perf_counter()
        try:
            return self._solver.solve(qubo)
        finally:
            self._timings["qaoa_ms"] += (time.perf_counter() - start) * 1000


class _TimedValidator:
    def __init__(
        self,
        config: InvestigativeValidationConfig,
        timings: dict[str, float],
        printable_fallback: bool = False,
    ):
        self._validator = InvestigativeValidator(config)
        self._timings = timings
        self._printable_fallback = printable_fallback
        self.last_validation: dict[str, Any] | None = None

    def validate(self, qubo: QUBOInstance, result: SolverResult) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            validation = self._validator.validate(qubo, result)
            self.last_validation = validation
            if self._printable_fallback and validation["global_energy"] is None:
                printable = dict(validation)
                printable["global_energy"] = (
                    validation["baseline_energy"]
                    if validation["baseline_energy"] is not None
                    else validation["candidate_energy"]
                )
                return printable
            return validation
        finally:
            self._timings["validation_ms"] += (
                time.perf_counter() - start
            ) * 1000


class _TimedDecomposer:
    def __init__(self, timings: dict[str, float]):
        self._decomposer = SubQUBODecomposer()
        self._timings = timings

    def partition(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return self._decomposer.partition(*args, **kwargs)
        finally:
            self._timings["decomposition_ms"] += (
                time.perf_counter() - start
            ) * 1000

    def extract_subqubo(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return self._decomposer.extract_subqubo(*args, **kwargs)
        finally:
            self._timings["qubo_build_ms"] += (
                time.perf_counter() - start
            ) * 1000

    def update_phi(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return self._decomposer.update_phi(*args, **kwargs)
        finally:
            self._timings["reconstruction_ms"] += (
                time.perf_counter() - start
            ) * 1000


class InvestigativeEngine:
    """Research orchestration façade; production ``main.py`` does not import it."""

    @staticmethod
    def run_job(
        qaoa_cfg: InvestigativeQAOAConfig,
        qubo_cfg: QUBOConfig,
        tracer_cfg: TracerConfig,
        decompositor_cfg: DecompositorConfig,
        preset_snapshot: SystemSnapshot | None,
        run_cfg: InvestigativeRunConfig | None = None,
        validation_cfg: InvestigativeValidationConfig | None = None,
    ) -> InvestigativeOutput:
        run_cfg = run_cfg or InvestigativeRunConfig()
        validation_cfg = validation_cfg or InvestigativeValidationConfig()
        total_start = time.perf_counter()
        timings = {
            "workload_ms": 0.0,
            "clustering_ms": 0.0,
            "qubo_build_ms": 0.0,
            "decomposition_ms": 0.0,
            "qaoa_ms": 0.0,
            "reconstruction_ms": 0.0,
            "validation_ms": 0.0,
        }
        metadata: dict[str, Any] = {}

        if preset_snapshot is None:
            print("INITIATING LIVE SYSTEM TRACER")
            start = time.perf_counter()
            snapshot = ProcessTracer(tracer_cfg).trace()
            timings["workload_ms"] += (time.perf_counter() - start) * 1000
            snapshot.num_cores = qubo_cfg.num_cores
            SnapshotVisualizer.print_system_snapshot(snapshot)
            print(f"{'-'*40}\n")

            print("INITIATING ADAPTIVE CLUSTERING")
            clusterer = AdaptiveCluster(decompositor_cfg)
            start = time.perf_counter()
            clustered = clusterer.decompose(snapshot)
            workload = clustered.to_workload()
            timings["clustering_ms"] += (time.perf_counter() - start) * 1000
            metadata["clustering"] = cluster_diagnostics(
                snapshot,
                clustered,
                clusterer,
            )
            SnapshotVisualizer.print_clustered_snapshot(clustered)
            print(f"{'-'*40}\n")
        else:
            snapshot = preset_snapshot
            if run_cfg.cluster_preset_snapshot:
                print("INITIATING ADAPTIVE CLUSTERING FOR PRESET SNAPSHOT")
                clusterer = AdaptiveCluster(decompositor_cfg)
                start = time.perf_counter()
                clustered = clusterer.decompose(snapshot)
                workload = clustered.to_workload()
                timings["clustering_ms"] += (
                    time.perf_counter() - start
                ) * 1000
                metadata["clustering"] = cluster_diagnostics(
                    snapshot,
                    clustered,
                    clusterer,
                )
            else:
                start = time.perf_counter()
                workload = snapshot.to_workload()
                timings["workload_ms"] += (time.perf_counter() - start) * 1000

        if not workload.entities:
            fixed_phi = workload.fixed_load_per_core
            output = IterativeSchedulingOutput(
                final_assignments=dict(workload.fixed_assignments),
                solver_results=[],
                phi_history=[fixed_phi],
                used_workload=workload,
                qubo_instance=QUBOInstance(
                    Q=np.zeros((0, 0)),
                    num_variables=0,
                    variable_map={},
                    num_entities=0,
                    num_cores=workload.num_cores,
                    penalty_weight=qubo_cfg.penalty,
                    iteration_index=0,
                    source_snapshot_id=workload.snapshot_id,
                ),
                qaoa_cfg=qaoa_cfg,
                qubo_cfg=qubo_cfg,
            )
            timings["wall_clock_total_ms"] = (
                time.perf_counter() - total_start
            ) * 1000
            timings["instrumented_total_ms"] = sum(
                value
                for key, value in timings.items()
                if key not in {"instrumented_total_ms", "wall_clock_total_ms"}
            )
            return InvestigativeOutput(output, timings, metadata)

        builder = _TimedBuilder(qubo_cfg, timings)
        solver = _TimedSolver(qaoa_cfg, timings)
        qubit_count = len(workload.entities) * workload.num_cores
        use_default = run_cfg.pipeline_mode == "default" or (
            run_cfg.pipeline_mode == "auto"
            and qubit_count <= decompositor_cfg.qubit_max
        )
        if run_cfg.pipeline_mode == "default" and (
            qubit_count > decompositor_cfg.qubit_max
        ):
            raise ValueError(
                f"Forced default pipeline requires {qubit_count} qubits, "
                f"above qubit_max={decompositor_cfg.qubit_max}."
            )

        if use_default:
            validator = _TimedValidator(
                validation_cfg,
                timings,
                printable_fallback=True,
            )
            qubo, result, _ = DefaultPipeline(
                builder,
                solver,
                validator,
            ).run(
                workload=workload,
                qaoa_cfg=qaoa_cfg,
                qubo_cfg=qubo_cfg,
            )
            validation = validator.last_validation
            validation["candidate_assignments"] = dict(result.decoded_assignments)
            baseline_energy = validation.get("baseline_energy")
            alpha = (
                optimality_gap(
                    result.energy,
                    baseline_energy,
                    validation.get("unconstrained_energy"),
                )
                if baseline_energy is not None
                else float("nan")
            )
            if run_cfg.enable_visualization:
                Visualizer(
                    qubo=qubo,
                    qaoa_cfg=qaoa_cfg,
                    qubo_cfg=qubo_cfg,
                    probs=result.probs,
                    energies_over_time=result.convergence_curve,
                    global_optimum=(
                        baseline_energy
                        if baseline_energy is not None
                        else result.energy
                    ),
                )
            output = SchedulingOutput(
                result=result,
                validation=validation,
                used_snapshot=snapshot,
                alpha=alpha,
                qubo_instance=qubo,
                qaoa_cfg=qaoa_cfg,
                qubo_cfg=qubo_cfg,
            )
        else:
            validator = _TimedValidator(validation_cfg, timings)
            decomposer = _TimedDecomposer(timings)
            (
                assignments,
                solver_results,
                phi_history,
                qubo,
                global_result,
                validation,
            ) = IterativePipeline(
                builder,
                solver,
                validator,
                decomposer,
            ).run(
                workload=workload,
                qaoa_cfg=qaoa_cfg,
                qubo_cfg=qubo_cfg,
                dec_cfg=decompositor_cfg,
                filename=None,
            )
            baseline_energy = validation.get("baseline_energy")
            alpha = (
                optimality_gap(
                    global_result.energy,
                    baseline_energy,
                    validation.get("unconstrained_energy"),
                )
                if baseline_energy is not None
                else None
            )
            if run_cfg.enable_visualization:
                visualizer = IterativeVisualizer(
                    solver_results=solver_results,
                    phi_history=phi_history,
                    workload=workload,
                    qubo_instance=qubo,
                    qaoa_cfg=qaoa_cfg,
                    qubo_cfg=qubo_cfg,
                )
                figure = visualizer.composite(save_path="results/iterative_run.png")
                import matplotlib.pyplot as plt

                plt.close(figure)
            output = IterativeSchedulingOutput(
                final_assignments=assignments,
                solver_results=solver_results,
                phi_history=phi_history,
                used_workload=workload,
                qubo_instance=qubo,
                qaoa_cfg=qaoa_cfg,
                qubo_cfg=qubo_cfg,
                global_result=global_result,
                validation=validation,
                alpha=alpha,
            )

        timings["wall_clock_total_ms"] = (
            time.perf_counter() - total_start
        ) * 1000
        timings["instrumented_total_ms"] = sum(
            value
            for key, value in timings.items()
            if key not in {"instrumented_total_ms", "wall_clock_total_ms"}
        )
        return InvestigativeOutput(output, timings, metadata)


def optimality_gap(
    candidate_energy: float,
    optimal_energy: float,
    reference_energy: float | None = None,
    eps: float = 1e-12,
) -> float:
    gap = candidate_energy - optimal_energy
    if np.isclose(gap, 0.0):
        return 0.0
    if reference_energy is None:
        reference_energy = 0.0
    denominator = abs(reference_energy - optimal_energy)
    if np.isclose(denominator, 0.0):
        denominator = max(abs(candidate_energy), abs(optimal_energy), 1.0)
    return gap / (denominator + eps)


def cluster_diagnostics(
    snapshot: SystemSnapshot,
    clustered_snapshot,
    clusterer: AdaptiveCluster,
) -> dict[str, Any]:
    features = clusterer.build_feature_matrix(snapshot)
    affinity = clusterer.build_affinity_matrix(features).A
    process_by_pid = {process.pid: process for process in snapshot.processes}
    bundle_by_pid = {
        pid: bundle.bundle_id
        for bundle in clustered_snapshot.bundles
        for pid in bundle.member_pids
    }
    intra = []
    inter = []
    for left in range(len(features.pids)):
        for right in range(left + 1, len(features.pids)):
            value = float(affinity[left, right])
            target = (
                intra
                if bundle_by_pid.get(features.pids[left])
                == bundle_by_pid.get(features.pids[right])
                else inter
            )
            target.append(value)

    def summary(values: list[float]) -> dict[str, Any]:
        return {
            "count": len(values),
            "sum": float(sum(values)),
            "mean": float(np.mean(values)) if values else None,
        }

    return {
        "io_alpha": clusterer.dec_cfg.io_alpha,
        "affinity_alpha": clusterer.dec_cfg.affinity_alpha,
        "num_input_processes": len(snapshot.processes),
        "num_bundles": len(clustered_snapshot.bundles),
        "bundles": [
            {
                "bundle_id": bundle.bundle_id,
                "member_pids": list(bundle.member_pids),
                "member_commands": [
                    process_by_pid[pid].command for pid in bundle.member_pids
                ],
                "aggregate_cpu_weight": bundle.aggregate_cpu_weight,
                "aggregate_rss_mb": bundle.aggregate_rss_mb,
            }
            for bundle in clustered_snapshot.bundles
        ],
        "intra_cluster_coupling": summary(intra),
        "inter_cluster_coupling": summary(inter),
    }


def _convergence_metrics(curve: list[float] | None) -> dict[str, Any]:
    if not curve:
        return {
            "convergence_steps_recorded": 0,
            "convergence_initial_objective": None,
            "convergence_final_objective": None,
            "convergence_iterations_to_final_tol": None,
        }
    values = [float(value) for value in curve]
    final = values[-1]
    tolerance = 1e-4 * max(1.0, abs(final))
    iterations = next(
        (
            index
            for index, value in enumerate(values, start=1)
            if abs(value - final) <= tolerance
        ),
        None,
    )
    return {
        "convergence_steps_recorded": len(values),
        "convergence_initial_objective": values[0],
        "convergence_final_objective": final,
        "convergence_iterations_to_final_tol": iterations,
    }


def _probability_diagnostics(
    result: SolverResult,
    qubo: QUBOInstance,
    optimal_energy: float | None,
) -> dict[str, Any]:
    if result.probs is None or not len(result.probs):
        return {
            "feasible_probability_mass": None,
            "invalid_probability_mass": None,
            "top_state_feasible": None,
            "first_feasible_rank": None,
            "first_optimal_rank": None,
            "optimal_probability_mass": None,
            "feasible_candidates_in_top_k": None,
        }

    probs = np.asarray(result.probs, dtype=float)
    ranked = np.argsort(probs)[::-1]
    feasible = np.zeros(len(probs), dtype=bool)
    optimal = np.zeros(len(probs), dtype=bool)
    for state_index in range(len(probs)):
        bits = np.array(
            [int(bit) for bit in bin(state_index)[2:].zfill(qubo.num_variables)]
        )
        feasible[state_index] = all(
            bits[
                entity * qubo.num_cores : (entity + 1) * qubo.num_cores
            ].sum()
            == 1
            for entity in range(qubo.num_entities)
        )
        if feasible[state_index] and optimal_energy is not None:
            energy = float(bits.T @ qubo.Q @ bits)
            optimal[state_index] = bool(
                np.isclose(energy, optimal_energy, rtol=1e-9, atol=1e-9)
            )

    top_k = min(int(result.solver_params.get("top_k", 0)), len(ranked))
    return {
        "feasible_probability_mass": float(probs[feasible].sum()),
        "invalid_probability_mass": float(
            max(0.0, 1.0 - float(probs[feasible].sum()))
        ),
        "top_state_feasible": bool(feasible[ranked[0]]),
        "first_feasible_rank": next(
            (
                rank
                for rank, state_index in enumerate(ranked, start=1)
                if feasible[state_index]
            ),
            None,
        ),
        "first_optimal_rank": next(
            (
                rank
                for rank, state_index in enumerate(ranked, start=1)
                if optimal[state_index]
            ),
            None,
        ),
        "optimal_probability_mass": float(probs[optimal].sum()),
        "feasible_candidates_in_top_k": int(feasible[ranked[:top_k]].sum()),
    }


def _scalability_metrics(
    validation: dict[str, Any],
    timings: dict[str, float],
) -> dict[str, Any]:
    qaoa_ms = float(timings.get("qaoa_ms", 0.0))
    total_ms = float(timings.get("instrumented_total_ms", sum(timings.values())))
    return {
        "tempo_total_ms": total_ms,
        "tempo_qaoa_ms": qaoa_ms,
        "tempo_overhead_ms": max(0.0, total_ms - qaoa_ms),
        "baseline_method": validation.get("baseline_method"),
        "baseline_certified": validation.get("baseline_certified"),
        "baseline_energy": validation.get("baseline_energy"),
        "pipeline_energy": validation.get("pipeline_energy"),
        "is_optimal": validation.get("is_optimal"),
        "gap_relativo": validation.get("relative_gap"),
        "annealing_restarts_stable": validation.get("annealing_restarts_stable"),
        "annealing_restart_energy_std": validation.get(
            "annealing_restart_energy_std"
        ),
    }


def assignment_balance_metrics(
    workload: Workload,
    assignments: dict[int | str, int] | None,
) -> dict[str, Any] | None:
    """Evaluate scheduling quality without the QUBO's additive energy offset."""

    if assignments is None:
        return None
    normalized_assignments = {int(key): int(value) for key, value in assignments.items()}
    loads = np.asarray(workload.fixed_load_per_core, dtype=float).copy()
    for entity in workload.entities:
        core = normalized_assignments.get(entity.entity_id)
        if core is None or core < 0 or core >= workload.num_cores:
            return None
        loads[core] += float(entity.cpu_weight)

    target = float(workload.total_weight / workload.num_cores)
    imbalance = float(loads.max() - loads.min())
    objective = float(np.square(loads - target).sum())
    normalized_imbalance = (
        imbalance / target if target > np.finfo(float).eps else 0.0
    )
    return {
        "core_loads": loads.tolist(),
        "target_load": target,
        "load_balance_objective": objective,
        "load_imbalance": imbalance,
        "normalized_load_imbalance": normalized_imbalance,
    }


def _offset_free_quality_metrics(
    workload: Workload,
    candidate_assignments: dict[int | str, int] | None,
    validation: dict[str, Any],
) -> dict[str, Any]:
    candidate = assignment_balance_metrics(workload, candidate_assignments)
    baseline_method = validation.get("baseline_method")
    baseline_assignments = (
        validation.get("global_assignments")
        if baseline_method == "bruteforce"
        else validation.get("annealing_assignments")
        if baseline_method == "simulated_annealing"
        else None
    )
    baseline = assignment_balance_metrics(workload, baseline_assignments)

    result = {
        "core_loads": None,
        "target_load": None,
        "load_balance_objective": None,
        "normalized_load_imbalance": None,
        "baseline_core_loads": None,
        "baseline_load_balance_objective": None,
        "baseline_load_imbalance": None,
        "baseline_normalized_load_imbalance": None,
        "objective_regret": None,
        "excess_normalized_load_imbalance": None,
        "baseline_match_offset_free": None,
        "certified_optimal_offset_free": None,
    }
    if candidate is None:
        return result
    result.update(candidate)
    if baseline is None:
        return result

    objective_regret = max(
        0.0,
        candidate["load_balance_objective"]
        - baseline["load_balance_objective"],
    )
    tolerance = float(validation.get("optimality_atol", 1e-9)) + float(
        validation.get("optimality_rtol", 1e-9)
    ) * abs(baseline["load_balance_objective"])
    baseline_match = objective_regret <= tolerance
    result.update(
        {
            "baseline_core_loads": baseline["core_loads"],
            "baseline_load_balance_objective": baseline[
                "load_balance_objective"
            ],
            "baseline_load_imbalance": baseline["load_imbalance"],
            "baseline_normalized_load_imbalance": baseline[
                "normalized_load_imbalance"
            ],
            "objective_regret": objective_regret,
            "excess_normalized_load_imbalance": max(
                0.0,
                candidate["normalized_load_imbalance"]
                - baseline["normalized_load_imbalance"],
            ),
            "baseline_match_offset_free": baseline_match,
            "certified_optimal_offset_free": (
                baseline_match
                if validation.get("baseline_certified") is True
                else None
            ),
        }
    )
    return result


def summarize_investigation(run: InvestigativeOutput) -> dict[str, Any]:
    """Convert an investigative run into the experiment runner's metric record."""

    output = run.output
    timings = run.component_timings_ms
    metadata = run.experiment_metadata
    if isinstance(output, SchedulingOutput):
        result = output.result
        validation = output.validation
        workload = output.used_snapshot.to_workload()
        max_probability = (
            float(np.max(result.probs))
            if result.probs is not None and len(result.probs)
            else None
        )
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
            "annealing_energy": validation.get("annealing_energy"),
            "annealing_gap": validation.get("annealing_gap"),
            "beats_annealing": validation.get("beats_annealing"),
            "annealing_solve_time_ms": validation.get(
                "annealing_solve_time_ms"
            ),
            "brute_force_solve_time_ms": validation.get(
                "brute_force_solve_time_ms"
            ),
            "sa_energy_match": (
                None
                if validation.get("annealing_energy") is None
                else bool(np.isclose(result.energy, validation["annealing_energy"]))
            ),
            "component_timings_ms": timings,
            "experiment_metadata": metadata,
            "solver_params": result.solver_params,
            **_probability_diagnostics(
                result,
                output.qubo_instance,
                validation.get("global_energy"),
            ),
            **_convergence_metrics(result.convergence_curve),
            **_scalability_metrics(validation, timings),
            **_offset_free_quality_metrics(
                workload,
                result.decoded_assignments,
                validation,
            ),
            "validation": validation,
        }

    if isinstance(output, IterativeSchedulingOutput):
        result = output.global_result
        validation = output.validation or {}
        sub_curves = [
            _convergence_metrics(item.convergence_curve)
            for item in output.solver_results
        ]
        iterations = [
            item["convergence_iterations_to_final_tol"]
            for item in sub_curves
            if item["convergence_iterations_to_final_tol"] is not None
        ]
        return {
            "pipeline": "iterative",
            "output_type": type(output).__name__,
            "num_variables": output.qubo_instance.num_variables,
            "num_entities": output.qubo_instance.num_entities,
            "num_cores": output.qubo_instance.num_cores,
            "energy": None if result is None else result.energy,
            "feasible": None if result is None else result.is_feasible,
            "optimal": validation.get("is_optimal"),
            "optimality_gap": output.alpha,
            "solve_time_ms": output.total_solve_time_ms,
            "num_sub_qubos": output.num_sub_qubos,
            "num_feasible_sub_qubos": output.num_feasible,
            "all_sub_qubos_feasible": output.all_feasible,
            "load_imbalance": output.load_imbalance,
            "final_phi": output.final_phi,
            "assignments": output.final_assignments,
            "annealing_energy": validation.get("annealing_energy"),
            "annealing_gap": validation.get("annealing_gap"),
            "beats_annealing": validation.get("beats_annealing"),
            "annealing_solve_time_ms": validation.get(
                "annealing_solve_time_ms"
            ),
            "brute_force_solve_time_ms": validation.get(
                "brute_force_solve_time_ms"
            ),
            "sa_energy_match": (
                None
                if validation.get("annealing_energy") is None or result is None
                else bool(np.isclose(result.energy, validation["annealing_energy"]))
            ),
            "component_timings_ms": timings,
            "experiment_metadata": metadata,
            "solver_params": None if result is None else result.solver_params,
            "convergence_steps_recorded": sum(
                item["convergence_steps_recorded"] for item in sub_curves
            ),
            "max_subqubo_convergence_iterations_to_final_tol": (
                max(iterations) if iterations else None
            ),
            **_scalability_metrics(validation, timings),
            **_offset_free_quality_metrics(
                output.used_workload,
                output.final_assignments,
                validation,
            ),
            "validation": validation,
        }

    return {"output_type": type(output).__name__}
