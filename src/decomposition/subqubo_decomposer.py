import numpy as np
from data_contracts import DecompositorConfig, QUBOInstance, Workload, WorkloadEntity
from decomposition.subqubo_heuristics import Heuristic


class SubQUBODecomposer:
    def partition(
        self,
        workload: Workload,
        Q_global: QUBOInstance,
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

        sorted_workload: list[list[WorkloadEntity]] = [
            [workload.get_entity(idx) for idx in sublist]
            for sublist in sorted_workload_index
        ]

        return sorted_workload

    def partition_dynamically(self) -> list[list[WorkloadEntity]]:
        # TODO: implement CORE_BALANCE dynamic partitioning
        return []