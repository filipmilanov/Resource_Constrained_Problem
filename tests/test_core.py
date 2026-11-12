import logging
from pathlib import Path

import pytest
import torch

from rckst.core import RCKSTError, RCKSTFileManager, RCKSTInstance, RCKSTNeighborhood, RCKSTSolution, RCKSTTask, RCKSTTaskConfig, RCKSTTaskManager, RCKSTTaskOptions


def test_instance_tensors() -> None:
    inst = RCKSTInstance(
        name="test_inst",
        n=3,
        E=2,
        k=2,
        m=1,
        node_resources=torch.tensor([[2], [3], [1]], dtype=torch.int64),
        edge_index=torch.tensor([[1, 2], [2, 3]], dtype=torch.int64),
        edge_weights=torch.tensor([5, 4], dtype=torch.int64),
        edge_costs=torch.tensor([[3], [1]], dtype=torch.int64),
    )
    assert inst.name == "test_inst"
    assert inst.n == 3
    assert inst.E == 2
    assert inst.node_resources.shape == (3, 1)
    assert inst.edge_index.shape == (2, 2)
    assert inst.edge_weights.shape == (2,)
    assert inst.edge_costs.shape == (2, 1)
    assert inst.edge_lookup.shape == (4, 4)
    assert inst.edge_lookup.dtype == torch.long
    assert inst.edge_lookup.device == inst.edge_index.device
    assert inst.edge_lookup[1, 2].item() == 0
    assert inst.edge_lookup[2, 1].item() == 0
    assert inst.edge_lookup[2, 3].item() == 1
    assert inst.edge_lookup[3, 2].item() == 1
    assert inst.edge_lookup[1, 3].item() == -1
    assert torch.all(inst.edge_lookup[0] == -1)
    assert torch.all(inst.edge_lookup[:, 0] == -1)


class DummyNeighborhood(RCKSTNeighborhood):
    def generate_batches(self, config: RCKSTTaskConfig, solution: RCKSTSolution, batchSize: int = 1024):
        yield solution.edges.unsqueeze(0)

    def sample_random(self, config: RCKSTTaskConfig, solution: RCKSTSolution):
        return solution.edges


def test_task_config_optional_neighborhood() -> None:
    instance = RCKSTInstance(
        name="test_inst",
        n=2,
        E=1,
        k=2,
        m=1,
        node_resources=torch.tensor([[1], [1]], dtype=torch.int64),
        edge_index=torch.tensor([[1], [2]], dtype=torch.int64),
        edge_weights=torch.tensor([1], dtype=torch.int64),
        edge_costs=torch.tensor([[1]], dtype=torch.int64),
    )

    assert RCKSTTaskConfig(instance=instance).neighborhood is None

    neighborhood = DummyNeighborhood()
    config = RCKSTTaskConfig(instance=instance, neighborhood=neighborhood)
    dummy_sol = RCKSTSolution("test_inst", torch.tensor([[1], [2]], dtype=torch.int64), total_weight=1, feasible=True)

    assert config.neighborhood is neighborhood
    batches = list(neighborhood.generate_batches(config, dummy_sol))
    assert len(batches) == 1
    assert torch.equal(batches[0], dummy_sol.edges.unsqueeze(0))


def test_file_manager_load_and_save(tmp_path: Path) -> None:
    example_path = Path(__file__).parent.parent / "assets" / "instances" / "example_instance.txt"
    assert example_path.exists()

    instance = RCKSTFileManager.load_instance(example_path)
    assert instance.name == "example_instance"
    assert instance.n == 6
    assert instance.E == 8
    assert instance.k == 4
    assert instance.m == 1
    assert instance.node_resources.shape == (6, 1)
    assert instance.edge_index.shape == (2, 8)
    assert instance.edge_weights.shape == (8,)
    assert instance.edge_costs.shape == (8, 1)

    solution = RCKSTSolution(instance_name=instance.name, edges=torch.tensor([[2, 2, 4], [3, 4, 6]], dtype=torch.int64), total_weight=8, feasible=True)
    save_path = tmp_path / "solution.txt"
    RCKSTFileManager.save_solution(solution, save_path)

    saved_text = save_path.read_text(encoding="utf-8").strip().splitlines()
    assert saved_text[0] == "example_instance"
    assert saved_text[1] == "2 3"
    assert saved_text[2] == "2 4"
    assert saved_text[3] == "4 6"


class DummyTask(RCKSTTask):
    @property
    def name(self) -> str:
        return "dummy_task"

    def solve(self, config: RCKSTTaskConfig) -> RCKSTSolution:
        weight = int(config.options.params.get("default_weight", 5))
        if config.options.seed is not None:
            weight += config.options.seed
        return RCKSTSolution(instance_name=config.instance.name, edges=torch.tensor([[1], [2]], dtype=torch.int64), total_weight=weight, feasible=True)


def test_task_manager(tmp_path: Path) -> None:
    example_path = Path(__file__).parent.parent / "assets" / "instances" / "example_instance.txt"
    out_path = tmp_path / "task_out.txt"

    custom_logger = logging.getLogger("test_logger")
    manager = RCKSTTaskManager(tasks=[DummyTask()], logger=custom_logger)

    solution, elapsed = manager.run_task("dummy_task", instance_path=example_path, output_path=out_path, options=RCKSTTaskOptions(seed=10, params={"default_weight": 40, "mode": "fast"}))

    assert solution.instance_name == "example_instance"
    assert solution.total_weight == 50
    assert elapsed >= 0.0
    assert out_path.exists()

    with pytest.raises(RCKSTError):
        manager.run_task("unknown_task", instance_path=example_path)
