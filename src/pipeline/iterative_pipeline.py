import numpy as np
from builder.builder_core import CoreAssignmentBuilder
from data_contracts import DecompositorConfig, QAOAConfig, QUBOConfig, Workload
from decomposition.subqubo_decomposer import SubQUBODecomposer
from decomposition.subqubo_heuristics import Heuristic
from solver.solver_validator import SolverValidator
from solver.pennylane_solver import PennylaneSolver
import time

class IterativePipeline:
    def __init__(self, builder: CoreAssignmentBuilder, solver: PennylaneSolver, 
                 solver_validator: SolverValidator, decomposer: SubQUBODecomposer):
        self.builder = builder 
        self.solver = solver 
        self.solver_validator = solver_validator
        self.decomposer = decomposer

    def run(self, filename, workload: Workload, qaoa_cfg: QAOAConfig, qubo_cfg: QUBOConfig, dec_cfg: DecompositorConfig):
        print(f"\n--- Iterative Run Started at {time.ctime()} ---")

        print("Building Q_global...")
        start_time = time.time()
        Q_global = self.builder.build(workload)
        print(f"Q_global completed in {time.time() - start_time:.4f}s")

        print("Partitioning in Sub-QUBOs...")
        groups = self.decomposer.partition(workload, Q_global, dec_cfg)
        
        for i, sub_group in enumerate(groups):
            print(f"Sub-QUBO {i} workload: ")
            print("\n".join(f"{entity}" for entity in sub_group))
            

        return [], [], []