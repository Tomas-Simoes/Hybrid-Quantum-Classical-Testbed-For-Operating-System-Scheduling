from typing import Dict, Tuple
from pennylane import numpy as pnp
import numpy as np
import pennylane as qml

from abstract.abstract import BaseSolver
from data_contracts import QAOAConfig, QUBOInstance, SolverResult
import time


class PennylaneSolver(BaseSolver):
    def __init__(self, qaoa_cfg: QAOAConfig):
        self.p = qaoa_cfg.layers
        self.steps = qaoa_cfg.steps
        self.learning_rate = qaoa_cfg.learning_rate
        self.top_k = qaoa_cfg.top_k
        self.mixer_type = qaoa_cfg.mixer_type

    def _make_device(self, num_qubits: int):
        try:
            dev = qml.device("lightning.gpu", wires=num_qubits)
            return dev
        except Exception:
            import warnings
            warnings.warn("lightning.gpu unavailable, falling back to lightning.qubit")
            return qml.device("lightning.qubit", wires=num_qubits)

    def solve(self, qubo: QUBOInstance) -> SolverResult:
        start_time = time.perf_counter()
        num_qubits = qubo.num_variables

        # 1. build Hamiltonians
        cost_h, offset = self.matrix_to_hamiltonian(qubo.Q)
        mixer_h = qml.qaoa.x_mixer(range(num_qubits))
        dev = self._make_device(num_qubits)

        @qml.qnode(dev, diff_method="adjoint")
        def cost_function(params):
            for i in range(num_qubits):
                qml.Hadamard(wires=i)
            gammas, betas = params
            for i in range(self.p):
                qml.qaoa.cost_layer(gammas[i], cost_h)
                qml.qaoa.mixer_layer(betas[i], mixer_h)
            return qml.expval(cost_h)

        # 2. optimization loop
        params = pnp.array([[0.5] * self.p, [0.5] * self.p], requires_grad=True)
        optimizer = qml.AdamOptimizer(stepsize=self.learning_rate)

        energies_over_time = []
        for _ in range(self.steps):
            params, energy = optimizer.step_and_cost(cost_function, params)
            # Subtract Ising offset so the convergence curve lives in QUBO space.
            # expval(cost_h) = <H_QUBO> + offset  →  <H_QUBO> = expval - offset
            energies_over_time.append(float(energy) - offset)

        # 3. sample probabilities and pick the best feasible bitstring
        @qml.qnode(dev)
        def get_probs(params):
            for i in range(num_qubits):
                qml.Hadamard(wires=i)
            gammas, betas = params
            for i in range(self.p):
                qml.qaoa.cost_layer(gammas[i], cost_h)
                qml.qaoa.mixer_layer(betas[i], mixer_h)
            return qml.probs(wires=range(num_qubits))

        probs = get_probs(params)

        # evaluate top_k candidates in descending probability order.
        # among feasible ones, take lowest QUBO energy.
        # if none are feasible, fall back to the highest-probability bitstring.
        k = min(self.top_k, len(probs))
        top_k_indices = pnp.argsort(probs)[-k:][::-1]  # descending by probability

        best_bitstring = None
        best_energy = float("inf")
        best_decoded = None
        best_feasible = False

        for idx in top_k_indices:
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
            else:
                # keep as fallback only if we have found nothing better yet
                if not best_feasible and best_bitstring is None:
                    best_bitstring = bitstring_array
                    best_energy = energy
                    best_decoded = decoded

        solve_time = (time.perf_counter() - start_time) * 1000

        return SolverResult(
            bitstring=pnp.array(best_bitstring),
            decoded_assignments=best_decoded,
            energy=best_energy,
            is_feasible=best_feasible,
            solver_backend="qaoa_pennylane",
            solve_time_ms=solve_time,
            solver_params={"p_layers": self.p, "opt_steps": self.steps},
            probs=probs,
            convergence_curve=energies_over_time,
        )

    def decode_assignments(self, bitstring, qubo: QUBOInstance) -> Tuple[Dict[int, int], bool]:
        decoded = {}
        active_indices = np.where(np.array(bitstring) == 1)[0]

        for idx in active_indices:
            pid, core = qubo.variable_map[idx]
            if pid in decoded:
                # mark conflict — keeps the entry but flags it as non-integer
                decoded[pid] = f"CONFLICT({decoded[pid]},{core})"
            else:
                decoded[pid] = core

        # Feasible if every process has exactly one integer core assignment
        is_feasible = (
            len(decoded) == qubo.num_entities
            and all(isinstance(v, int) for v in decoded.values())
        )
        return decoded, is_feasible

    def matrix_to_hamiltonian(self, Q) -> Tuple[qml.Hamiltonian, float]:
        """
        Converts an upper-triangular QUBO matrix Q into an Ising Hamiltonian
        using the substitution x_i = (1 - Z_i) / 2.

        Returns:
            cost_h : the PennyLane Hamiltonian  H_Ising = H_QUBO + offset
            offset  : the constant shift c such that <H_Ising> = <H_QUBO> + c

        The offset is subtracted from the convergence curve in solve() so that
        tracked energies live in the same space as the final QUBO energy
        x^T Q x computed from the bitstring.
        """
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
                elif Q[i, j] != 0:
                    coeffs.append(Q[i, j] / 4)
                    obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

                    linear[i] -= Q[i, j] / 4
                    linear[j] -= Q[i, j] / 4
                    offset += Q[i, j] / 4

        for i in range(n):
            if not np.isclose(linear[i], 0.0):
                coeffs.append(float(linear[i]))
                obs.append(qml.PauliZ(i))

        if not np.isclose(offset, 0.0):
            coeffs.append(float(offset))
            obs.append(qml.Identity(0))

        return qml.Hamiltonian(coeffs, obs), offset