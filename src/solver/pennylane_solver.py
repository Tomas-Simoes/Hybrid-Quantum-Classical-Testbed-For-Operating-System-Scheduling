from typing import Dict, Tuple
from pennylane import numpy as pnp
import numpy as np
import pennylane as qml

from abstract.abstract import BaseSolver
from data_contracts import QAOAConfig, QUBOInstance, SolverResult
import time
import warnings


class PennylaneSolver(BaseSolver):
    def __init__(self, qaoa_cfg: QAOAConfig):
        self.p = qaoa_cfg.layers
        self.steps = qaoa_cfg.steps
        self.learning_rate = qaoa_cfg.learning_rate
        self.top_k = qaoa_cfg.top_k
        self.mixer_type = qaoa_cfg.mixer_type
        self.init_gamma = qaoa_cfg.init_gamma
        self.init_beta = qaoa_cfg.init_beta

    def _candidate_indices(self, probs):
        ranked = np.argsort(np.asarray(probs))[::-1]
        return ranked[: min(self.top_k, len(ranked))]

    def _make_device(self, device_name: str, num_qubits: int):
        return qml.device(device_name, wires=num_qubits)

    def _xy_mixer_layer(self, beta: float, N: int, K: int):
        """
        XY mixer that preserves the one-hot constraint per process.
        For each process i, applies XX+YY on every pair of core qubits (j, k).
        This swaps amplitude between |01> and |10> — i.e. between core assignments —
        without ever creating |00> or |11> states, so feasibility is structurally preserved.
        """
        for i in range(N):
            for j in range(K):
                for k in range(j + 1, K):
                    wire_j = i * K + j
                    wire_k = i * K + k
                    qml.IsingXX(2 * beta, wires=[wire_j, wire_k])
                    qml.IsingYY(2 * beta, wires=[wire_j, wire_k])

    def _prepare_initial_state(self, num_qubits: int, N: int, K: int):
        """
        X mixer:  uniform superposition over all 2^n bitstrings (includes infeasible).
        XY mixer: valid one-hot state — every process assigned to core 0.
                  XY mixer then explores only feasible reassignments from here.
        """
        if self.mixer_type == "xy":
            # assign every process to core 0: flip qubit i*K+0 for each process i
            for i in range(N):
                qml.PauliX(wires=i * K)
        else:
            for i in range(num_qubits):
                qml.Hadamard(wires=i)

    def solve(self, qubo: QUBOInstance) -> SolverResult:
        try:
            return self._solve_on_device(qubo, "lightning.gpu")
        except Exception as gpu_error:
            warnings.warn(
                f"lightning.gpu failed during QAOA solve; retrying on lightning.qubit. "
                f"Original error: {gpu_error}"
            )
            return self._solve_on_device(qubo, "lightning.qubit")

    def _solve_on_device(self, qubo: QUBOInstance, device_name: str) -> SolverResult:
        start_time = time.perf_counter()
        num_qubits = qubo.num_variables
        N = qubo.num_entities
        K = qubo.num_cores

        # 1. build cost Hamiltonian
        cost_h, _ = self.matrix_to_hamiltonian(qubo.Q)

        # X mixer is built once as a Hamiltonian; XY mixer is applied inline per layer
        if self.mixer_type == "x":
            mixer_h = qml.qaoa.x_mixer(range(num_qubits))

        dev = self._make_device(device_name, num_qubits)

        @qml.qnode(dev, diff_method="adjoint")
        def cost_function(params):
            self._prepare_initial_state(num_qubits, N, K)
            gammas, betas = params
            for i in range(self.p):
                qml.qaoa.cost_layer(gammas[i], cost_h)
                if self.mixer_type == "xy":
                    self._xy_mixer_layer(betas[i], N, K)
                else:
                    qml.qaoa.mixer_layer(betas[i], mixer_h)
            return qml.expval(cost_h)

        # 2. optimization loop
        params = pnp.array([[self.init_gamma] * self.p, [self.init_beta] * self.p], requires_grad=True)
        optimizer = qml.AdamOptimizer(stepsize=self.learning_rate)

        energies_over_time = []
        for _ in range(self.steps):
            params, energy = optimizer.step_and_cost(cost_function, params)
            energies_over_time.append(float(energy))

        # 3. sample probabilities and pick the best feasible bitstring
        @qml.qnode(dev)
        def get_probs(params):
            self._prepare_initial_state(num_qubits, N, K)
            gammas, betas = params
            for i in range(self.p):
                qml.qaoa.cost_layer(gammas[i], cost_h)
                if self.mixer_type == "xy":
                    self._xy_mixer_layer(betas[i], N, K)
                else:
                    qml.qaoa.mixer_layer(betas[i], mixer_h)
            return qml.probs(wires=range(num_qubits))

        probs = get_probs(params)

        ranked_indices = np.argsort(np.asarray(probs))[::-1]
        candidate_indices = self._candidate_indices(probs)

        best_bitstring = None
        best_energy = float("inf")
        best_decoded = None
        best_feasible = False

        for idx in candidate_indices:
            bit_str = bin(int(idx))[2:].zfill(num_qubits)
            bitstring_array = np.array([int(b) for b in bit_str])

            decoded, is_feasible = self.decode_assignments(bitstring_array, qubo)
            energy = float(bitstring_array.T @ qubo.Q @ bitstring_array)

            if is_feasible:
                if not best_feasible or energy < best_energy:
                    best_bitstring = bitstring_array
                    best_energy = energy
                    best_decoded = decoded
                    best_feasible = True

        if best_bitstring is None:
            fallback_idx = int(ranked_indices[0])
            bit_str = bin(fallback_idx)[2:].zfill(num_qubits)
            best_bitstring = np.array([int(b) for b in bit_str])
            best_decoded, _ = self.decode_assignments(best_bitstring, qubo)
            best_energy = float(best_bitstring.T @ qubo.Q @ best_bitstring)

        solve_time = (time.perf_counter() - start_time) * 1000

        return SolverResult(
            bitstring=pnp.array(best_bitstring),
            decoded_assignments=best_decoded,
            energy=best_energy,
            is_feasible=best_feasible,
            solver_backend=f"qaoa_pennylane_{device_name}_{self.mixer_type}_mixer",
            solve_time_ms=solve_time,
            solver_params={
                "p_layers": self.p,
                "opt_steps": self.steps,
                "mixer_type": self.mixer_type,
                "init_gamma": self.init_gamma,
                "init_beta": self.init_beta,
                "device": device_name,
                "top_k": self.top_k,
                "selection_pool": "top_k_probable_states",
            },
            probs=probs,
            convergence_curve=energies_over_time,
        )

    def decode_assignments(self, bitstring, qubo: QUBOInstance) -> Tuple[Dict[int, int], bool]:
        decoded = {}
        active_indices = np.where(np.array(bitstring) == 1)[0]

        for idx in active_indices:
            pid, core = qubo.variable_map[idx]
            if pid in decoded:
                decoded[pid] = f"CONFLICT({decoded[pid]},{core})"
            else:
                decoded[pid] = core

        is_feasible = (
            len(decoded) == qubo.num_entities
            and all(isinstance(v, int) for v in decoded.values())
        )
        return decoded, is_feasible

    def matrix_to_hamiltonian(self, Q) -> Tuple[qml.Hamiltonian, float]:
        n = len(Q)
        linear = np.zeros(n)
        coeffs = []
        obs = []
        offset = 0.0

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    linear[i] -= Q[i, i] / 2
                    offset += Q[i, i] / 2
                else:
                    qij = Q[i, j] + Q[j, i]
                    if np.isclose(qij, 0.0):
                        continue
                    coeffs.append(qij / 4)
                    obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
                    linear[i] -= qij / 4
                    linear[j] -= qij / 4
                    offset += qij / 4

        for i in range(n):
            if not np.isclose(linear[i], 0.0):
                coeffs.append(float(linear[i]))
                obs.append(qml.PauliZ(i))

        if not np.isclose(offset, 0.0):
            coeffs.append(float(offset))
            obs.append(qml.Identity(0))

        return qml.Hamiltonian(coeffs, obs), offset
