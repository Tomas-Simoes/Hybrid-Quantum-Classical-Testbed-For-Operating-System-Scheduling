import math
import numpy as np
from data_contracts import DecompositorConfig, QUBOInstance, Workload
from decomposition.subqubo_heuristics import Heuristic

class SubQUBODecomposer:
    def partition(self, workload: Workload, Q_global: QUBOInstance, dec_cfg: DecompositorConfig):
        max_entities = dec_cfg.qubit_max // workload.num_cores
        heuristic = dec_cfg.sorting_strategy

        if(heuristic.is_dynamic): return self.partition_dynamically()
        else:
            sorted_workload_index = heuristic.apply(workload, dec_cfg.qubit_max)
            sorted_workload: list[Workload] = [
                [workload.get_entity(idx) for idx in sublist]
                for sublist in sorted_workload_index
            ]
        return sorted_workload
    
    # TODO
    def partition_dynamically(self): 
        return []


