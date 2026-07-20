from abc import ABC, abstractmethod
from data_contracts import QUBOInstance, SolverResult, SystemSnapshot

class BaseSolver(ABC):
    @abstractmethod
    def solve(self, qubo: QUBOInstance) -> SolverResult:
        pass

class BaseBuilder(ABC):
    @abstractmethod
    def build(self, snapshot: SystemSnapshot) -> QUBOInstance:
        pass