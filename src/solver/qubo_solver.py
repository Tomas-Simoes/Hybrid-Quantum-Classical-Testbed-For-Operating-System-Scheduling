import numpy as np
from itertools import product

class QUBOSolver():
    def solve(self):
        """Tries every combination to find the global minimum."""
        best_energy = float('inf')
        best_x = None
        
        for x_tuple in product([0, 1], repeat=self.size):
            x = np.array(x_tuple)
            energy = self.calculate_energy(x)
            
            if energy < best_energy:
                best_energy = energy
                best_x = x
                
        return best_x, best_energy