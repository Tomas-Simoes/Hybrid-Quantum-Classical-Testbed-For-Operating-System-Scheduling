class Tracer:
    def __init__(self, weights):
        """Initialize the tracer with a list of process weights."""
        self.weights = weights
        self.num_processes = len(weights)

