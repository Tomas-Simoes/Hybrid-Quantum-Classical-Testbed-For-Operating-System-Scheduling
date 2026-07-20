from __future__ import annotations  

import numpy as np
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_contracts import Workload

def weight_descending(workload: Workload, qubit_max: int) -> list[list[int]]:
    all_indices = list(range(len(workload.entities)))
    sorted_indices = sorted(all_indices, key=lambda i: workload.entities[i].cpu_weight, reverse=True)

    max_bundles = qubit_max // workload.num_cores
    return [sorted_indices[i : i + max_bundles] for i in range(0, len(sorted_indices), max_bundles)]
    

def coupling_descending(workload: Workload, qubit_max: int) -> list[list[int]]:
    weights = [e.cpu_weight for e in workload.entities]
    max_rss = max([e.rss_mb for e in workload.entities]) or 1.0

    def get_magnitude(e):
        return e.cpu_weight + (e.rss_mb / max_rss)
    
    all_indices = list(range(len(workload.entities)))
    unassigned = sorted(all_indices, key=lambda i: get_magnitude(workload.entities[i]), reverse=True)

    groups = []
    max_bundles = qubit_max // workload.num_cores

    while unassigned:
        current_group = [unassigned.pop(0)]
        while len(current_group) < max_bundles and unassigned:
            best_candidate_idx = -1
            max_strength = -1.0
            
            for idx, candidate_idx in enumerate(unassigned):
                strength = sum(2 * weights[candidate_idx] * weights[i] for i in current_group)
                if strength > max_strength:
                    max_strength = strength
                    best_candidate_idx = idx
            
            current_group.append(unassigned.pop(best_candidate_idx))
        groups.append(current_group)

    return groups

def core_balance(groups: list[list[int]], workload: Workload, phi: np.ndarray, K: int, **kwargs) -> list[list[int]]:
    weights = [e.cpu_weight for e in workload.entities]
    L_avg = sum(weights) / K
    residual = L_avg - phi  
    target = residual[int(np.argmax(residual))]
    return sorted(groups, key=lambda g: abs(sum(weights[i] for i in g) - target))

class Heuristic(Enum):
    WEIGHT_DESCENDING = (weight_descending, False)
    COUPLING_DESCENDING = (coupling_descending, False)
    CORE_BALANCE = (core_balance, True)

    def __new__(cls, func, is_dynamic_flag):
        # Using __new__ or __init__ to handle the tuple values
        obj = object.__new__(cls)
        obj._value_ = func.__name__ # Set value to the function name string
        obj.func = func
        obj.is_dynamic_flag = is_dynamic_flag
        return obj

    def apply(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    @property
    def is_dynamic(self):
        return self.is_dynamic_flag