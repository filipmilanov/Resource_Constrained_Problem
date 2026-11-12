import logging
import time

import pytest
import torch

from rckst.core.models import RCKSTError, RCKSTInstance, RCKSTSolution, RCKSTTaskConfig, RCKSTTaskOptions
from rckst.tasks import GRASPTask
from rckst.tasks import grasp as grasp_module


def instance() -> RCKSTInstance:
    return RCKSTInstance(
        name="test", n=3, E=3, k=2, m=1,
        node_resources=torch.ones((3, 1), dtype=torch.long),
        edge_index=torch.tensor([[1, 2, 1], [2, 3, 3]]),
        edge_weights=torch.tensor([5, 3, 7]), edge_costs=torch.zeros((3, 1), dtype=torch.long),
    )


def test_grasp_restarts_and_keeps_best_feasible(monkeypatch, caplog):
    seeds = []
    starts = []
    returned = [
        RCKSTSolution("test", torch.tensor([[1], [2]]), 1, False),
        RCKSTSolution("test", torch.tensor([[2], [3]]), 3, True),
        RCKSTSolution("test", torch.tensor([[1], [3]]), 7, True),
    ]

    class Construction:
        def solve(self, config):
            seeds.append(config.options.seed)
            return returned[len(seeds) - 1]

    class Search:
        def solve(self, config):
            starts.append(config.options.params["initialSolution"])
            assert config.options.params["mode"] == "first_improvement"
            assert config.options.params["maxSeconds"] == 30
            return starts[-1]

    monkeypatch.setattr(grasp_module, "RandomizedConstructionTask", Construction)
    monkeypatch.setattr(grasp_module, "VNDTask", Search)
    options = RCKSTTaskOptions(seed=42, params={"graspIterations": 3})
    config = RCKSTTaskConfig(instance(), options)
    with caplog.at_level(logging.INFO):
        result = GRASPTask().solve(config)
    assert seeds == [42, 43, 44]
    assert starts == returned
    assert result is returned[1]
    assert options.params == {"graspIterations": 3}
    assert len(caplog.records) == 3


def test_grasp_retries_failed_construction(monkeypatch):
    calls = []
    solution = RCKSTSolution("test", torch.tensor([[1], [2]]), 5, True)

    class Construction:
        def solve(self, config):
            calls.append(config.options.seed)
            if len(calls) == 1:
                raise RCKSTError("No solution found")
            return solution

    class Search:
        def solve(self, config):
            return config.options.params["initialSolution"]

    monkeypatch.setattr(grasp_module, "RandomizedConstructionTask", Construction)
    monkeypatch.setattr(grasp_module, "VNDTask", Search)
    result = GRASPTask().solve(RCKSTTaskConfig(instance(), RCKSTTaskOptions(seed=9, params={"graspIterations": 2})))
    assert result is solution and calls == [9, 10]


def test_grasp_real_construction_and_vnd():
    options = RCKSTTaskOptions(seed=4, params={"graspIterations": 3, "rootRetries": 1, "localSearchSeconds": 2, "maxIterations": 10})
    result = GRASPTask().solve(RCKSTTaskConfig(instance(), options))
    assert result.feasible and result.total_weight == 3
    assert result.edges.shape == (2, 1)


def test_grasp_returns_construction_if_time_expires(monkeypatch):
    solution = RCKSTSolution("test", torch.tensor([[1], [2]]), 5, True)

    class Construction:
        def solve(self, config):
            time.sleep(0.02)
            return solution

    class Search:
        def solve(self, config):
            raise AssertionError("VND must not start after the time limit")

    monkeypatch.setattr(grasp_module, "RandomizedConstructionTask", Construction)
    monkeypatch.setattr(grasp_module, "VNDTask", Search)
    config = RCKSTTaskConfig(instance(), RCKSTTaskOptions(params={"maxSeconds": 0.005}))
    assert GRASPTask().solve(config) is solution


def test_grasp_no_construction_found(monkeypatch):
    class Construction:
        def solve(self, config):
            raise RCKSTError("No solution found")

    monkeypatch.setattr(grasp_module, "RandomizedConstructionTask", Construction)
    config = RCKSTTaskConfig(instance(), RCKSTTaskOptions(params={"graspIterations": 2}))
    with pytest.raises(RCKSTError, match="could not construct"):
        GRASPTask().solve(config)


@pytest.mark.parametrize("params", [{"graspIterations": 0}, {"localSearchSeconds": 0}, {"maxSeconds": 0}])
def test_grasp_invalid_limits(params):
    with pytest.raises(RCKSTError):
        GRASPTask().solve(RCKSTTaskConfig(instance(), RCKSTTaskOptions(params=params)))
