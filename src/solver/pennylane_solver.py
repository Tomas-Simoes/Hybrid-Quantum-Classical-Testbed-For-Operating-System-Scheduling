from typing import Dict, Tuple
from pennylane import qaoa
from pennylane import numpy as pnp
import numpy as np
import pennylane as qml
from matplotlib import pyplot as plt
import networkx as nx

from abstract.abstract import BaseSolver
from data_contracts import QAOAConfig, QUBOInstance, SolverResult
import time

class PennylaneSolver(BaseSolver):
    def __init__(self, qaoa_cfg: QAOAConfig):
        self.p = qaoa_cfg.layers
        self.steps = qaoa_cfg.steps
        self.learning_rate= qaoa_cfg.learning_rate

    def solve(self, qubo: QUBOInstance) -> SolverResult:
        start_time = time.perf_counter()
        num_qubits = qubo.num_variables
        
        # 1. Build Hamiltonians
        cost_h = self.matrix_to_hamiltonian(qubo.Q)
        mixer_h = qml.qaoa.x_mixer(range(num_qubits))
        dev = qml.device("lightning.gpu", wires=num_qubits)

        @qml.qnode(dev, diff_method="adjoint")
        def cost_function(params):
            for i in range(num_qubits): qml.Hadamard(wires=i)
            gammas, betas = params
            for i in range(self.p):
                # GAMMA FIRST, THEN HAMILTONIAN
                qml.qaoa.cost_layer(gammas[i], cost_h) 
                qml.qaoa.mixer_layer(betas[i], mixer_h)
            return qml.expval(cost_h)

        # 2. Optimization Loop
        params = pnp.array([[0.5] * self.p, [0.5] * self.p], requires_grad=True)
        optimizer = qml.AdamOptimizer(stepsize=self.learning_rate)      

        energies_over_time = []
        for _ in range(self.steps):
            params, energy = optimizer.step_and_cost(cost_function, params)
            energies_over_time.append(float(energy))

        @qml.qnode(dev)
        def get_probs(params):
            for i in range(num_qubits): qml.Hadamard(wires=i)
            gammas, betas = params
            for i in range(self.p):
                qml.qaoa.cost_layer(gammas[i], cost_h)
                qml.qaoa.mixer_layer(betas[i], mixer_h)
            return qml.probs(wires=range(num_qubits))

        probs = get_probs(params)
        best_idx = pnp.argmax(probs)
        bit_str = bin(best_idx)[2:].zfill(num_qubits)
        bitstring_array = pnp.array([int(b) for b in bit_str])

        # 4. Decoding and Result
        decoded, is_feasible = self.decode_assignments(bitstring_array, qubo)
        
        # Calculate final QUBO energy: x^T * Q * x
        final_energy = float(bitstring_array.T @ qubo.Q @ bitstring_array)
        
        solve_time = (time.perf_counter() - start_time) * 1000

        return SolverResult(
            bitstring=bitstring_array,
            decoded_assignments=decoded,
            energy=final_energy,
            is_feasible=is_feasible,
            solver_backend="qaoa_pennylane",
            solve_time_ms=solve_time,
            solver_params={"p_layers": self.p, "opt_steps": self.steps},
            probs=probs,                          
            convergence_curve=energies_over_time 
        )

    def decode_assignments(self, bitstring, qubo: QUBOInstance) -> Tuple[Dict[int, int], bool]:
        decoded = {}
        # Find which bits are 1
        active_indices = pnp.where(bitstring == 1)[0]
        
        for idx in active_indices:
            pid, core = qubo.variable_map[idx]
            if pid in decoded:
                decoded[pid] = f"CONFLICT({decoded[pid]},{core})"
            else:
                decoded[pid] = core
            
        # Check One-Hot: Every process must be assigned to exactly one core
        # num_entities is the number of processes
        is_feasible = len(decoded) == qubo.num_entities
        return decoded, is_feasible

    def matrix_to_hamiltonian(self, Q):
        n = len(Q)
        linear = np.zeros(n) 
        coeffs = []
        obs = []
        offset = 0.0 

        for i in range(len(Q)):
            for j in range(i, len(Q)):
                if i == j:
                    linear[i] -= Q[i,i] / 2
                    offset += Q[i, i] / 2
                elif Q[i,j] != 0:
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
        
        return qml.Hamiltonian(coeffs, obs)