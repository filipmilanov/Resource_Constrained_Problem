from collections.abc import Iterator

import pytest
import torch
from jaxtyping import Int

from rckst.core.models import RCKSTInstance, RCKSTNeighborhood, RCKSTSolution, RCKSTTaskConfig, RCKSTTaskOptions
from rckst.tasks.localsearch import LocalSearchTask
from rckst.tasks.vnd import VNDTask, evaluate_batch


class MockBatchNeighborhood(RCKSTNeighborhood):
    def __init__(self, batches: list[Int[torch.Tensor, "B 2 (k-1)"]]) -> None:
        self.batches = batches
        self.call_count = 0

    def generate_batches(self, config: RCKSTTaskConfig, solution: RCKSTSolution, batchSize: int = 1024) -> Iterator[Int[torch.Tensor, "B 2 (k-1)"]]:
        self.call_count += 1
        yield from self.batches

    def sample_random(self, config: RCKSTTaskConfig, solution: RCKSTSolution) -> Int[torch.Tensor, "2 (k-1)"] | None:
        self.call_count += 1
        if len(self.batches) > 0 and self.batches[0].shape[0] > 0:
            return self.batches[0][0]
        return None


def test_vnd_with_batch_neighborhood() -> None:
    inst = RCKSTInstance(
        name="test",
        n=4,
        E=3,
        k=3,
        m=1,
        node_resources=torch.tensor([[10], [10], [10], [10]], dtype=torch.int64),
        edge_index=torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.int64),
        edge_weights=torch.tensor([5, 5, 50], dtype=torch.int64),
        edge_costs=torch.tensor([[1], [1], [1]], dtype=torch.int64),
    )

    init_sol = RCKSTSolution("test", torch.tensor([[1, 2], [2, 4]], dtype=torch.int64), total_weight=55, feasible=True)
    better_candidate = torch.tensor([[[1, 2], [2, 3]]], dtype=torch.int64)  # weight = 5 + 5 = 10

    nh = MockBatchNeighborhood([better_candidate])
    config = RCKSTTaskConfig(instance=inst, options=RCKSTTaskOptions(params={"initialSolution": init_sol, "mode": "best_improvement"}), neighborhood=[nh])

    task = VNDTask()
    res = task.solve(config)

    assert res.total_weight == 10
    assert res.feasible is True
    assert nh.call_count == 2


def test_step_modes_and_feasibility() -> None:
    inst = RCKSTInstance(
        name="test", n=5, E=4, k=2, m=1,
        node_resources=torch.ones((5, 1), dtype=torch.long),
        edge_index=torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
        edge_weights=torch.tensor([10, 5, 2, 1]),
        edge_costs=torch.tensor([[0], [0], [0], [10]]),
    )
    current = RCKSTSolution("test", torch.tensor([[1], [2]]), 10, True)
    candidates = torch.tensor([[[4], [5]], [[2], [3]], [[3], [4]]])
    neighborhood = MockBatchNeighborhood([candidates])
    config = RCKSTTaskConfig(instance=inst, options=RCKSTTaskOptions(params={"batchSize": 2}))
    task = VNDTask()

    best, _, _ = task.solveBestImprovement(neighborhood, config, current)
    first, _, _ = task.solveFirstImprovement(neighborhood, config, current)
    assert best.total_weight == 2 and best.feasible
    assert first.total_weight == 5 and first.feasible

    # a feasible neighbor is an improvement even when the current infeasible solution has a lower weight
    infeasible = RCKSTSolution("test", torch.tensor([[4], [5]]), 1, False)
    nextSolution, _, _ = task.solveFirstImprovement(neighborhood, config, infeasible)
    assert nextSolution.feasible and nextSolution.total_weight == 5

    randomSolution, _, _ = task.solveRandomImprovement(neighborhood, config, current)
    assert randomSolution is current  # the sampled edge is infeasible


def test_vnd_restarts_first_neighborhood_and_respects_limit() -> None:
    inst = RCKSTInstance(
        name="test", n=4, E=3, k=2, m=1,
        node_resources=torch.ones((4, 1), dtype=torch.long),
        edge_index=torch.tensor([[1, 2, 3], [2, 3, 4]]),
        edge_weights=torch.tensor([10, 5, 2]), edge_costs=torch.zeros((3, 1), dtype=torch.long),
    )
    initial = RCKSTSolution("test", torch.tensor([[1], [2]]), 10, True)
    first = MockBatchNeighborhood([])
    second = MockBatchNeighborhood([torch.tensor([[[2], [3]]])])
    config = RCKSTTaskConfig(inst, RCKSTTaskOptions(params={"initialSolution": initial}), [first, second])
    result = VNDTask().solve(config)
    assert result.total_weight == 5
    assert first.call_count == 2 and second.call_count == 2

    config.options.params["maxIterations"] = 1
    assert VNDTask().solve(config) is initial


@pytest.mark.parametrize("mode", ["best_improvement", "first_improvement", "random_step"])
def test_vnd_empty_neighborhood(mode: str) -> None:
    inst = RCKSTInstance(
        name="test", n=2, E=1, k=2, m=1, node_resources=torch.ones((2, 1), dtype=torch.long),
        edge_index=torch.tensor([[1], [2]]), edge_weights=torch.tensor([3]), edge_costs=torch.zeros((1, 1), dtype=torch.long),
    )
    initial = RCKSTSolution("test", torch.tensor([[1], [2]]), 3, True)
    config = RCKSTTaskConfig(inst, RCKSTTaskOptions(params={"initialSolution": initial, "mode": mode}), [MockBatchNeighborhood([])])
    assert VNDTask().solve(config) is initial


def test_local_search_with_default_extension() -> None:
    inst = RCKSTInstance(
        name="test", n=4, E=3, k=3, m=1,
        node_resources=torch.ones((4, 1), dtype=torch.long),
        edge_index=torch.tensor([[1, 2, 3], [2, 3, 4]]),
        edge_weights=torch.tensor([10, 5, 1]), edge_costs=torch.zeros((3, 1), dtype=torch.long),
    )
    initial = RCKSTSolution("test", torch.tensor([[1, 2], [2, 3]]), 15, True)
    config = RCKSTTaskConfig(inst, RCKSTTaskOptions(params={"initialSolution": initial}))
    result = LocalSearchTask().solve(config)
    assert result.feasible and result.total_weight == 6


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_evaluate_batch_counts_shared_nodes_once(device: str) -> None:
    inst = RCKSTInstance(
        name="test", n=3, E=2, k=3, m=1,
        node_resources=torch.tensor([[2], [3], [4]], device=device),
        edge_index=torch.tensor([[1, 2], [2, 3]], device=device),
        edge_weights=torch.tensor([4, 5], device=device),
        edge_costs=torch.tensor([[4], [5]], device=device),
    )
    weights, feasible = evaluate_batch(inst, torch.tensor([[[1, 2], [2, 3]]], device=device))
    assert weights.tolist() == [9]
    assert feasible.tolist() == [True]


def test_vnd_uses_weight_even_when_construction_alpha_is_configured() -> None:
    inst = RCKSTInstance(
        name="test", n=3, E=2, k=2, m=1,
        node_resources=torch.tensor([[0], [0], [1000]]),
        edge_index=torch.tensor([[1, 2], [2, 3]]),
        edge_weights=torch.tensor([5, 10]), edge_costs=torch.zeros((2, 1), dtype=torch.long),
    )
    initial = RCKSTSolution("test", torch.tensor([[1], [2]]), 5, True)
    neighborhood = MockBatchNeighborhood([torch.tensor([[[2], [3]]])])
    options = RCKSTTaskOptions(params={"initialSolution": initial, "alpha": 0.7, "positiveResourceWeight": 0.05, "maxIterations": 1})
    config = RCKSTTaskConfig(inst, options, [neighborhood])
    candidate, score, feasible = VNDTask().solveFirstImprovement(neighborhood, config, initial)
    assert candidate is initial and feasible and score == 5
    assert VNDTask().solve(config).total_weight == 5
