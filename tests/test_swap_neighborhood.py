from collections import Counter

import networkx as nx
import pytest

from rckst.neighborhoods import SwapNeighborhood
from test_extension_neighborhood import canonical, make_case


def reference(graph, tree):
    treeEdges = {tuple(sorted(edge)) for edge in tree}
    nodes = {node for edge in tree for node in edge}
    result = Counter()
    for addition in graph:
        if set(addition) <= nodes and tuple(sorted(addition)) not in treeEdges:
            for removed in tree:
                candidate = [edge for edge in tree if edge != removed] + [addition]
                graphCandidate = nx.Graph(candidate)
                if nx.is_tree(graphCandidate):
                    result[tuple(sorted(tuple(sorted(edge)) for edge in candidate))] += 1
    return result


@pytest.mark.parametrize("graph,tree,n", [
    ([(1, 2), (2, 3), (3, 4), (1, 3), (1, 4), (4, 5)], [(1, 2), (2, 3), (3, 4)], 5),
    ([(2, 1), (3, 1), (4, 1), (2, 3), (4, 2), (3, 4)], [(1, 2), (1, 3), (1, 4)], 4),
    ([(1, 2), (2, 3)], [(1, 2), (2, 3)], 3),
])
def test_swap_batches_and_sampling(graph, tree, n):
    config, solution = make_case(graph, tree, n)
    neighborhood = SwapNeighborhood()
    batches = list(neighborhood.generate_batches(config, solution, batchSize=2))
    actual = Counter(canonical(candidate) for batch in batches for candidate in batch)
    assert actual == reference(graph, tree)
    assert all(0 < batch.shape[0] <= 2 and batch.shape[1:] == (2, len(tree)) for batch in batches)
    for _ in range(10):
        sample = neighborhood.sample_random(config, solution)
        if actual:
            assert sample is not None and sample.shape == (2, len(tree)) and canonical(sample) in actual
        else:
            assert sample is None


def test_swap_invalid_batch_size():
    config, solution = make_case([(1, 2)], [(1, 2)], 2)
    with pytest.raises(ValueError):
        list(SwapNeighborhood().generate_batches(config, solution, 0))
