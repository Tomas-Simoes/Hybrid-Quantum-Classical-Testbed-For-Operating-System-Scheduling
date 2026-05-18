from typing import Dict, List, Tuple

import numpy as np
from data_contracts import DecompositorConfig, QUBOInstance, Workload, WorkloadEntity
from decomposition.subqubo_heuristics import Heuristic


class SubQUBODecomposer:
    def partition(
        self,
        workload: Workload,
        Q_global: np.ndarray,
        dec_cfg: DecompositorConfig,
    ) -> list[list[WorkloadEntity]]:
        """
        Partitions the workload into sub-groups.
        Returns a list of groups, where each group is a list of WorkloadEntity objects.
        """
        heuristic = dec_cfg.sorting_strategy

        if heuristic.is_dynamic:
            return self.partition_dynamically()

        # sorted_workload_index: list[list[int]] — each int is a direct index into
        # workload.entities (0-based, no negative indexing).
        sorted_workload_index: list[list[int]] = heuristic.apply(workload, dec_cfg.qubit_max)

        invalid_indices = [
            idx
            for sublist in sorted_workload_index
            for idx in sublist
            if idx < 0 or idx >= len(workload.entities)
        ]
        if invalid_indices:
            raise ValueError(
                "Partition heuristic returned invalid workload positions: "
                f"{invalid_indices}"
            )

        sorted_workload: list[list[WorkloadEntity]] = [
            [workload.entities[idx] for idx in sublist]
            for sublist in sorted_workload_index
        ]

        return sorted_workload
    
    def extract_subqubo(self, Q_global: np.ndarray,
        group: List[WorkloadEntity],
        workload: Workload,
        phi: np.ndarray,           # shape (K,) — accumulated load per core
        iteration_index: int,
        penalty: int):
        """
        Extract a sub-QUBO from Q_global for the given group of entities,
        applying accumulated bias propagation to the diagonal.

        Theory: Q_{jk,jk}^(t) = Q_{jk,jk}^original + 2·w_j·phi_k
        This converts severed quadratic couplings into linear biases.
        """
        K = workload.num_cores

        # Map entity_id -> original position in workload.entities
        entity_id_to_orig_idx: Dict[int, int] = {
            e.entity_id: pos for pos, e in enumerate(workload.entities)
        }

        # Flat indices in Q_global for every variable in this group
        # Order: entity0·core0, entity0·core1, ..., entity1·core0, ...
        global_flat_indices: List[int] = []
        for entity in group:
            orig_i = entity_id_to_orig_idx[entity.entity_id]
            for k in range(K):
                global_flat_indices.append(orig_i * K + k)
    
        # Extract the sub-matrix 
        ix = np.ix_(global_flat_indices, global_flat_indices)
        sub_Q = Q_global[ix].copy()

         # Apply bias propagation to the diagonal
        # For local index (local_i, k) → diagonal position local_i*K + k
        # Add 2·w_j·phi_k to account for all previously fixed processes
        for local_i, entity in enumerate(group):
            for k in range(K):
                diag_pos = local_i * K + k
                sub_Q[diag_pos, diag_pos] += 2.0 * entity.cpu_weight * phi[k]

        # Build local variable_map: local_flat_idx → (entity_id, core)
        local_variable_map: Dict[int, Tuple[int, int]] = {
            local_i * K + k: (entity.entity_id, k)
            for local_i, entity in enumerate(group)
            for k in range(K)
        }

        return QUBOInstance(
            Q=sub_Q,
            num_variables=len(group) * K,
            variable_map=local_variable_map,
            num_entities=len(group),
            num_cores=K,
            penalty_weight=penalty,
            iteration_index=iteration_index,
            source_snapshot_id=workload.snapshot_id,
        )
    
    def update_phi(
    self,
    phi: np.ndarray,
    group: List[WorkloadEntity],
    decoded_assignments: Dict[int, int],
    ) -> None:
        """
        Update accumulated core loads in-place after a sub-QUBO is solved.
        phi[k] += w_i for every entity i assigned to core k.
        """
        for entity in group:
            assigned_core = decoded_assignments.get(entity.entity_id)
            if assigned_core is not None and isinstance(assigned_core, int):
                phi[assigned_core] += entity.cpu_weight
                
    def partition_dynamically(self) -> list[list[WorkloadEntity]]:
        # TODO: implement CORE_BALANCE dynamic partitioning
        raise NotImplementedError("Dynamic CORE_BALANCE partitioning is not implemented yet.")
