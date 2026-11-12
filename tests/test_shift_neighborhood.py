from collections import Counter

import networkx as nx
import pytest

from rckst.neighborhoods import ShiftNeighborhood
from test_extension_neighborhood import canonical, make_case


def reference(graph, tree):
    treeGraph = nx.Graph()
    treeGraph.add_edges_from(tree)
    leaves = [node for node in treeGraph if treeGraph.degree[node] == 1]
    result = Counter()
    for u, v in graph:
        if u in leaves and v not in treeGraph:
            attachment = u
        elif v in leaves and u not in treeGraph:
            attachment = v
        else:
            continue
        distances = nx.single_source_shortest_path_length(treeGraph, attachment)
        furthest = max(distances[leaf] for leaf in leaves)
        for leaf in leaves:
            if distances[leaf] == furthest:
                remaining = [edge for edge in tree if leaf not in edge]
                result[tuple(sorted(tuple(sorted(edge)) for edge in remaining + [(u, v)]))] += 1
    return result


@pytest.mark.parametrize("graph,tree,n", [
    ([(1, 2), (2, 3), (3, 4), (4, 5), (2, 6)], [(1, 2), (2, 3), (3, 4)], 6),
    ([(1, 2), (1, 3), (1, 4), (2, 5), (3, 6), (1, 7)], [(1, 2), (1, 3), (1, 4)], 7),
    ([(2, 1), (1, 3), (2, 3)], [(1, 2)], 3),
    ([(1, 2), (2, 3)], [(1, 2), (2, 3)], 3),
])
def test_shift_batches_and_sampling(graph, tree, n):
    config, solution = make_case(graph, tree, n)
    neighborhood = ShiftNeighborhood()
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


def test_shift_invalid_batch_size():
    config, solution = make_case([(1, 2)], [(1, 2)], 2)
    with pytest.raises(ValueError):
        list(ShiftNeighborhood().generate_batches(config, solution, 0))
