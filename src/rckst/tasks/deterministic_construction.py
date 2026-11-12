import torch
from torch import Tensor
from jaxtyping import Float, Int
from rckst.core.models import RCKSTSolution, RCKSTTask, RCKSTTaskConfig, RCKSTError
from .randomized_construction import RandomizedConstructionTask


class DeterministicConstructionTask(RCKSTTask):
    def __init__(self) -> None:
        super().__init__()
        self._randConstruction = RandomizedConstructionTask()

    @property
    def name(self) -> str:
        return "deterministic_construction"

    def solve(self, config: RCKSTTaskConfig) -> RCKSTSolution:
        """
        special case of randomized construction with temperature 0
        """
        config.options.params["temperature"] = 0
        return self._randConstruction.solve(config)
