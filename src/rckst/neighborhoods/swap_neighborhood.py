from collections.abc import Iterator

import torch
from jaxtyping import Bool, Int
from torch import Tensor

from rckst.core.models import RCKSTNeighborhood, RCKSTSolution, RCKSTTaskConfig


class SwapNeighborhood(RCKSTNeighborhood):
    # replace one edge of the tree with another edge between the same tree nodes so that the result is still a tree

    def generate_batches(self, config: RCKSTTaskConfig, solution: RCKSTSolution, batchSize: int = 1024) -> Iterator[Int[Tensor, "B 2 (k-1)"]]:
        if batchSize <= 0:
            raise ValueError("batchSize must be positive")
        inst = config.instance
        solEdges: Int[Tensor, "2 (k-1)"] = solution.edges
        if solEdges.shape[1] == 0:
            return

        # find all edges that connect two nodes of the tree but are not already part of the tree
        treeNodes, flatTreeIndices = torch.unique(solEdges.flatten(), sorted=True, return_inverse=True)
        treeEdges: Int[Tensor, "2 (k-1)"] = flatTreeIndices.reshape(2, -1)
        inTree: Bool[Tensor, "n+1"] = torch.zeros(inst.n + 1, dtype=torch.bool)
        inTree[treeNodes] = True
        solEdgeIndices: Int[Tensor, "k-1"] = inst.edge_lookup[solEdges[0], solEdges[1]]
        inSolution: Bool[Tensor, "E"] = torch.zeros(inst.E, dtype=torch.bool)
        inSolution[solEdgeIndices] = True
        u, v = inst.edge_index
        addableMask: Bool[Tensor, "E"] = inTree[u] & inTree[v] & ~inSolution
        addableEdges: Int[Tensor, "2 D"] = inst.edge_index[:, addableMask]
        if addableEdges.shape[1] == 0:
            return
        addableTreeEdges: Int[Tensor, "2 D"] = torch.searchsorted(treeNodes, addableEdges)

        # removing one tree edge splits the tree in two. calculate one side of every cut at the same time
        # start at one endpoint of each removed edge and propagate through all other tree edges
        numEdges = solEdges.shape[1]
        edgePositions: Int[Tensor, "k-1"] = torch.arange(numEdges)
        allowedEdges: Bool[Tensor, "(k-1) (k-1)"] = edgePositions.unsqueeze(0) != edgePositions.unsqueeze(1)
        inComponent: Bool[Tensor, "(k-1) k"] = torch.zeros((numEdges, treeNodes.numel()), dtype=torch.bool)
        inComponent[edgePositions, treeEdges[0]] = True
        frontier: Bool[Tensor, "(k-1) k"] = inComponent.clone()
        for _ in range(1, inst.k):
            reached: Int[Tensor, "(k-1) k"] = torch.zeros(inComponent.shape, dtype=torch.long)
            reached.scatter_add_(1, treeEdges[0].unsqueeze(0).expand(numEdges, -1), (frontier[:, treeEdges[1]] & allowedEdges).long())
            reached.scatter_add_(1, treeEdges[1].unsqueeze(0).expand(numEdges, -1), (frontier[:, treeEdges[0]] & allowedEdges).long())
            frontier = (reached > 0) & ~inComponent
            if not torch.any(frontier):
                break
            inComponent |= frontier

        # an edge can replace the removed edge if its endpoints are on different sides of the cut
        for start in range(0, addableEdges.shape[1], batchSize):
            additions: Int[Tensor, "2 D"] = addableEdges[:, start : start + batchSize]
            additionNodes: Int[Tensor, "2 D"] = addableTreeEdges[:, start : start + batchSize]
            crossesCut: Bool[Tensor, "(k-1) D"] = inComponent[:, additionNodes[0]] ^ inComponent[:, additionNodes[1]]
            removedIndices, additionIndices = torch.where(crossesCut)
            for offset in range(0, removedIndices.numel(), batchSize):
                removed: Int[Tensor, "B"] = removedIndices[offset : offset + batchSize]
                selected: Int[Tensor, "B"] = additionIndices[offset : offset + batchSize]
                keep: Bool[Tensor, "B (k-1)"] = edgePositions.unsqueeze(0) != removed.unsqueeze(1)
                kept: Int[Tensor, "B 2 (k-2)"] = solEdges.T.unsqueeze(0).expand(removed.numel(), -1, -1)[keep].reshape(removed.numel(), -1, 2).transpose(1, 2)
                yield torch.cat((kept, additions[:, selected].T.unsqueeze(-1)), dim=2)

    def sample_random(self, config: RCKSTTaskConfig, solution: RCKSTSolution) -> Int[Tensor, "2 (k-1)"] | None:
        inst = config.instance
        solEdges: Int[Tensor, "2 (k-1)"] = solution.edges
        if solEdges.shape[1] == 0:
            return None

        # choose an edge between tree nodes that is not already in the tree
        treeNodes, flatTreeIndices = torch.unique(solEdges.flatten(), sorted=True, return_inverse=True)
        treeEdges: Int[Tensor, "2 (k-1)"] = flatTreeIndices.reshape(2, -1)
        inTree: Bool[Tensor, "n+1"] = torch.zeros(inst.n + 1, dtype=torch.bool)
        inTree[treeNodes] = True
        inSolution: Bool[Tensor, "E"] = torch.zeros(inst.E, dtype=torch.bool)
        inSolution[inst.edge_lookup[solEdges[0], solEdges[1]]] = True
        u, v = inst.edge_index
        addableMask: Bool[Tensor, "E"] = inTree[u] & inTree[v] & ~inSolution
        addableEdges: Int[Tensor, "2 D"] = inst.edge_index[:, addableMask]
        if addableEdges.shape[1] == 0:
            return None
        additionIndex: Int[Tensor, "1"] = torch.randint(addableEdges.shape[1], (1,))
        addition: Int[Tensor, "2 1"] = addableEdges[:, additionIndex]
        endpoints: Int[Tensor, "2"] = torch.searchsorted(treeNodes, addition[:, 0])

        # adding this edge creates a cycle. find distances from both endpoints at the same time and select
        # an edge on the path between them, because removing any such edge breaks the cycle
        distances: Int[Tensor, "2 k"] = torch.full((2, treeNodes.numel()), -1, dtype=torch.long)
        distances[torch.arange(2), endpoints] = 0
        frontier: Bool[Tensor, "2 k"] = distances == 0
        for distance in range(1, inst.k):
            reached: Int[Tensor, "2 k"] = torch.zeros_like(distances)
            reached.scatter_add_(1, treeEdges[0].unsqueeze(0).expand(2, -1), frontier[:, treeEdges[1]].long())
            reached.scatter_add_(1, treeEdges[1].unsqueeze(0).expand(2, -1), frontier[:, treeEdges[0]].long())
            frontier = (reached > 0) & (distances < 0)
            distances[frontier] = distance
            if torch.all(distances >= 0):
                break
        pathLength: Int[Tensor, ""] = distances[0, endpoints[1]]
        a, b = treeEdges
        onPath: Bool[Tensor, "k-1"] = ((distances[0, a] + 1 + distances[1, b]) == pathLength) | ((distances[0, b] + 1 + distances[1, a]) == pathLength)
        removable: Int[Tensor, "P"] = torch.where(onPath)[0]
        removedIndex: Int[Tensor, ""] = removable[torch.randint(removable.numel(), (1,))[0]]
        keep: Bool[Tensor, "k-1"] = torch.arange(solEdges.shape[1]) != removedIndex
        return torch.cat((solEdges[:, keep], addition), dim=1)
