from collections import Counter

import pytest
import torch

from rckst.core.models import RCKSTInstance, RCKSTSolution, RCKSTTaskConfig
from rckst.neighborhoods import ExtensionNeighborhood


def make_case(graph_edges: list[tuple[int, int]], tree_edges: list[tuple[int, int]], n: int) -> tuple[RCKSTTaskConfig, RCKSTSolution]:
    instance = RCKSTInstance(
        name="test", n=n, E=len(graph_edges), k=len(tree_edges) + 1, m=1,
        node_resources=torch.ones((n, 1), dtype=torch.long),
        edge_index=torch.tensor(graph_edges, dtype=torch.long).T.contiguous(),
        edge_weights=torch.ones(len(graph_edges), dtype=torch.long),
        edge_costs=torch.ones((len(graph_edges), 1), dtype=torch.long),
    )
    solution = RCKSTSolution("test", torch.tensor(tree_edges, dtype=torch.long).T.contiguous(), 0, True)
    return RCKSTTaskConfig(instance), solution


def canonical(edges: torch.Tensor) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(tuple(sorted(edge)) for edge in edges.T.tolist()))


def reference(graph: list[tuple[int, int]], tree: list[tuple[int, int]]) -> Counter:
    degrees = Counter(node for edge in tree for node in edge)
    results = Counter()
    for leaf in (node for node, degree in degrees.items() if degree == 1):
        removed = next(edge for edge in tree if leaf in edge)
        remaining = [edge for edge in tree if edge != removed]
        nodes = set(degrees) - {leaf}
        for edge in graph:
            if (edge[0] in nodes) != (edge[1] in nodes) and set(edge) != set(removed):
                results[tuple(sorted(tuple(sorted(e)) for e in remaining + [edge]))] += 1
    return results


@pytest.mark.parametrize("graph,tree,n", [
    ([(2, 1), (2, 3), (3, 1), (4, 2), (3, 5), (1, 4)], [(1, 2), (3, 2)], 5),
    ([(2, 1), (1, 3), (3, 2)], [(1, 2)], 3),
    ([(1, 2), (2, 3)], [(1, 2), (2, 3)], 3),
])
def test_extension_matches_reference(graph, tree, n):
    config, solution = make_case(graph, tree, n)
    neighborhood = ExtensionNeighborhood()
    batches = list(neighborhood.generate_batches(config, solution, batchSize=2))
    assert all(0 < batch.shape[0] <= 2 and batch.shape[1:] == (2, len(tree)) for batch in batches)
    actual = Counter(canonical(candidate) for batch in batches for candidate in batch)
    assert actual == reference(graph, tree)
    sample = neighborhood.sample_random(config, solution)
    if sample is not None:
        assert sample.shape == (2, len(tree))
        assert canonical(sample) in actual


def test_invalid_sizes():
    config, solution = make_case([(1, 2), (2, 3)], [(1, 2)], 3)
    neighborhood = ExtensionNeighborhood()
    with pytest.raises(ValueError):
        list(neighborhood.generate_batches(config, solution, 0))


def test_random_sampling_does_not_generate_batches(monkeypatch):
    config, solution = make_case([(1, 2), (2, 3), (2, 4), (1, 5)], [(1, 2), (2, 3)], 5)
    neighborhood = ExtensionNeighborhood()

    def unexpected(*args, **kwargs):
        raise AssertionError("sampling must not generate batches")

    monkeypatch.setattr(neighborhood, "generate_batches", unexpected)
    samples = [neighborhood.sample_random(config, solution) for _ in range(100)]
    possible = {
        ((1, 2), (2, 4)),
        ((2, 3), (2, 4)),
        ((1, 2), (1, 5)),
    }
    assert all(sample is not None and sample.shape == (2, 2) and canonical(sample) in possible for sample in samples)


def test_random_sampling_without_neighbors():
    config, solution = make_case([(1, 2), (2, 3)], [(1, 2), (2, 3)], 3)
    assert ExtensionNeighborhood().sample_random(config, solution) is None


def test_random_sampling_can_reattach_removed_leaf():
    config, solution = make_case([(1, 2), (2, 3), (1, 3)], [(1, 2), (2, 3)], 3)
    sample = ExtensionNeighborhood().sample_random(config, solution)
    assert sample is not None
    assert sample.shape == (2, 2)
    assert (1, 3) in canonical(sample)
