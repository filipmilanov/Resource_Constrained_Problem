from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from jaxtyping import Int
from torch import Tensor

type ConfigValue = str | int | float | bool | RCKSTSolution
type TaskParams = dict[str, ConfigValue]


class RCKSTError(Exception):
    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self.message: str = message


@dataclass(slots=True, frozen=True)
class RCKSTInstance:
    name: str  # instance name without extension
    n: int  # number of nodes |V|
    E: int  # number of edges |E|
    k: int  # number of nodes to span
    m: int  # number of resource dimensions
    node_resources: Int[Tensor, "n m"]  # resource yield per node
    edge_index: Int[Tensor, "2 E"]  # endpoint pairs (u, v) per edge
    edge_weights: Int[Tensor, "E"]  # weight/cost per edge
    edge_costs: Int[Tensor, "E m"]  # resource costs per edge
    edge_lookup: Int[Tensor, "(n+1) (n+1)"] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        edge_lookup = torch.full((self.n + 1, self.n + 1), -1, dtype=torch.long, device=self.edge_index.device)
        u, v = self.edge_index
        edge_ids = torch.arange(self.edge_index.shape[1], dtype=torch.long, device=self.edge_index.device)
        edge_lookup[u, v] = edge_ids
        edge_lookup[v, u] = edge_ids
        object.__setattr__(self, "edge_lookup", edge_lookup)


@dataclass(slots=True)
class RCKSTTaskOptions:
    seed: int | None = None
    params: TaskParams = field(default_factory=dict)


@dataclass(slots=True)
class RCKSTSolution:
    instance_name: str
    edges: Int[Tensor, "2 (k-1)"]
    total_weight: int | float
    feasible: bool
    


class RCKSTNeighborhood(ABC):
    @abstractmethod
    def generate_batches(self, config: RCKSTTaskConfig, solution: RCKSTSolution, batchSize: int = 1024) -> Iterator[Int[Tensor, "B 2 (k-1)"]]:
        """Yield batches of at most batchSize candidate edge tensors."""
        
        ...

    @abstractmethod
    def sample_random(self, config: RCKSTTaskConfig, solution: RCKSTSolution) -> Int[Tensor, "2 (k-1)"] | None: ...


@dataclass(slots=True)
class RCKSTTaskConfig:
    instance: RCKSTInstance
    options: RCKSTTaskOptions = field(default_factory=RCKSTTaskOptions)
    neighborhood: list[RCKSTNeighborhood] | RCKSTNeighborhood | None = None


class RCKSTTask(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def solve(self, config: RCKSTTaskConfig) -> RCKSTSolution: ...
