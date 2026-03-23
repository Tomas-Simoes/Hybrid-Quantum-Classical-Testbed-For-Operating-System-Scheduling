# File: specialized_tracers.py
from .tracer import Tracer

class MemoryTracer(Tracer):
    def get_memory_snapshots(self):
        """Simulates reading RAM usage. 
        For now, it just returns the weights as 'MB' of memory.
        """
        return [f"{w * 1024:.2f} MB" for w in self.weights]

class ProcessTracer(Tracer):
    def get_process_list(self):
        """
        Simulates reading the process table.
        Returns a list of dummy Process IDs and their priority weights.
        """
        processes = []
        for i, w in enumerate(self.weights):
            processes.append({"pid": 1000 + i, "weight": w})
        return processes