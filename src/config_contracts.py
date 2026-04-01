from dataclasses import dataclass
from typing import List

from data_contracts import SystemSnapshot

@dataclass
class QUBOConfig:
    penalty: float
    num_cores: int
    snapshot: SystemSnapshot | None

@dataclass
class QAOAConfig:
    layers: int
    steps: int
    optimizer_step: float

