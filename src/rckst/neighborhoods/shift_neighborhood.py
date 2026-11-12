from collections.abc import Iterator

import torch
from jaxtyping import Bool, Int
from torch import Tensor

from rckst.core.models import RCKSTNeighborhood, RCKSTSolution, RCKSTTaskConfig


class ShiftNeighborhood(RCKSTNeighborhood):
    # add a new node at a leaf and remove a leaf that is furthest away from the new node

    def generate_batches(self, config: RCKSTTaskConfig, solution: RCKSTSolution, batchSize: int = 1024) -> Iterator[Int[Tensor, "B 2 (k-1)"]]:
        if batchSize <= 0:
            raise ValueError("batchSize must be positive")
        inst = config.instance
        solEdges: Int[Tensor, "2 (k-1)"] = solution.edges
        if solEdges.shape[1] == 0:
            return
        # find the leafs and the corresponding edge index in the solution
        edgeFlat: Int[Tensor, "2*(k-1)"] = solEdges.flatten()
        degrees: Int[Tensor, "n+1"] = torch.bincount(edgeFlat, minlength=inst.n + 1)
        leafs: Int[Tensor, "L"] = torch.where(degrees == 1)[0]
        edgeIndexInSolution: Int[Tensor, "L"] = torch.where((solEdges[0].unsqueeze(0) == leafs.unsqueeze(1)) | (solEdges[1].unsqueeze(0) == leafs.unsqueeze(1)))[1]

        # a shift can only add an edge if one endpoint is a leaf and the other is outside the tree
        u, v = inst.edge_index
        addableMask: Bool[Tensor, "E"] = ((degrees[u] == 1) & (degrees[v] == 0)) | ((degrees[v] == 1) & (degrees[u] == 0))
        addableEdges: Int[Tensor, "2 D"] = inst.edge_index[:, addableMask]
        attachmentNodes: Int[Tensor, "D"] = torch.where(degrees[addableEdges[0]] == 1, addableEdges[0], addableEdges[1])
        if addableEdges.shape[1] == 0:
            return

        # only calculate distances for leafs that have an extension. map the tree nodes to 0..k-1 so the
        # distance tensor does not contain a column for every node of the underlying graph
        treeNodes, flatTreeIndices = torch.unique(edgeFlat, sorted=True, return_inverse=True)
        treeEdges: Int[Tensor, "2 (k-1)"] = flatTreeIndices.reshape(2, -1)
        startLeafs, attachmentIndices = torch.unique(attachmentNodes, sorted=True, return_inverse=True)
        startNodes: Int[Tensor, "S"] = torch.searchsorted(treeNodes, startLeafs)
        leafNodes: Int[Tensor, "L"] = torch.searchsorted(treeNodes, leafs)
        # the largest distance is k-1, so int32 is sufficient unless the tree has more than 2^31 nodes
        distanceType = torch.int32 if inst.k <= 2**31 - 1 else torch.int64
        distances: Int[Tensor, "S k"] = torch.full((startLeafs.numel(), treeNodes.numel()), -1, dtype=distanceType)
        distances[torch.arange(startLeafs.numel()), startNodes] = 0
        frontier: Bool[Tensor, "S k"] = distances == 0
        for distance in range(1, inst.k):
            reached: Int[Tensor, "S k"] = torch.zeros_like(distances)
            reached.scatter_add_(1, treeEdges[0].unsqueeze(0).expand(startLeafs.numel(), -1), frontier[:, treeEdges[1]].to(distanceType))
            reached.scatter_add_(1, treeEdges[1].unsqueeze(0).expand(startLeafs.numel(), -1), frontier[:, treeEdges[0]].to(distanceType))
            frontier = (reached > 0) & (distances < 0)
            distances[frontier] = distance
            # check every four steps to reduce synchronization between cpu and gpu
            if distance % 4 == 0 and torch.all(distances >= 0):
                break
        leafDistances: Int[Tensor, "S L"] = distances[:, leafNodes]
        furthest: Bool[Tensor, "S L"] = leafDistances == leafDistances.max(dim=1, keepdim=True).values
        edgePositions: Int[Tensor, "k-1"] = torch.arange(solEdges.shape[1])

        # process extensions in batches and include every furthest leaf when several have the same distance
        for start in range(0, addableEdges.shape[1], batchSize):
            additions: Int[Tensor, "2 D"] = addableEdges[:, start : start + batchSize]
            batchAttachmentIndices: Int[Tensor, "D"] = attachmentIndices[start : start + batchSize]
            additionIndices, removedLeafIndices = torch.where(furthest[batchAttachmentIndices])
            for offset in range(0, additionIndices.numel(), batchSize):
                selected = additionIndices[offset : offset + batchSize]
                removedIndices = edgeIndexInSolution[removedLeafIndices[offset : offset + batchSize]]
                keep: Bool[Tensor, "B (k-1)"] = edgePositions.unsqueeze(0) != removedIndices.unsqueeze(1)
                kept: Int[Tensor, "B 2 (k-2)"] = solEdges.T.unsqueeze(0).expand(selected.numel(), -1, -1)[keep].reshape(selected.numel(), -1, 2).transpose(1, 2)
                yield torch.cat((kept, additions[:, selected].T.unsqueeze(-1)), dim=2)

    def sample_random(self, config: RCKSTTaskConfig, solution: RCKSTSolution) -> Int[Tensor, "2 (k-1)"] | None:
        inst = config.instance
        solEdges: Int[Tensor, "2 (k-1)"] = solution.edges
        if solEdges.shape[1] == 0:
            return None
        # find leafs and all edges that connect one of them to a node outside the tree
        degrees: Int[Tensor, "n+1"] = torch.bincount(solEdges.flatten(), minlength=inst.n + 1)
        leafs: Int[Tensor, "L"] = torch.where(degrees == 1)[0]
        u, v = inst.edge_index
        addableMask: Bool[Tensor, "E"] = ((degrees[u] == 1) & (degrees[v] == 0)) | ((degrees[v] == 1) & (degrees[u] == 0))
        addableEdges: Int[Tensor, "2 D"] = inst.edge_index[:, addableMask]
        if addableEdges.shape[1] == 0:
            return None

        # choose an extension and then one of its furthest leafs without generating the whole neighborhood
        additionIndex: Int[Tensor, "1"] = torch.randint(addableEdges.shape[1], (1,))
        addition: Int[Tensor, "2 1"] = addableEdges[:, additionIndex]
        attachment: Int[Tensor, ""] = torch.where(degrees[addition[0, 0]] == 1, addition[0, 0], addition[1, 0])
        # calculate distances only for the nodes in the tree
        treeNodes, flatTreeIndices = torch.unique(solEdges.flatten(), sorted=True, return_inverse=True)
        treeEdges: Int[Tensor, "2 (k-1)"] = flatTreeIndices.reshape(2, -1)
        distanceType = torch.int32 if inst.k <= 2**31 - 1 else torch.int64
        distances: Int[Tensor, "k"] = torch.full((treeNodes.numel(),), -1, dtype=distanceType)
        distances[torch.searchsorted(treeNodes, attachment)] = 0
        frontier: Bool[Tensor, "k"] = distances == 0
        for distance in range(1, inst.k):
            reached: Int[Tensor, "k"] = torch.zeros_like(distances)
            reached.scatter_add_(0, treeEdges[0], frontier[treeEdges[1]].to(distanceType))
            reached.scatter_add_(0, treeEdges[1], frontier[treeEdges[0]].to(distanceType))
            frontier = (reached > 0) & (distances < 0)
            distances[frontier] = distance
            if distance % 4 == 0 and torch.all(distances >= 0):
                break
        leafDistances: Int[Tensor, "L"] = distances[torch.searchsorted(treeNodes, leafs)]
        furthest: Int[Tensor, "F"] = leafs[leafDistances == leafDistances.max()]
        removedLeaf: Int[Tensor, ""] = furthest[torch.randint(furthest.numel(), (1,))[0]]
        keep: Bool[Tensor, "k-1"] = (solEdges[0] != removedLeaf) & (solEdges[1] != removedLeaf)
        return torch.cat((solEdges[:, keep], addableEdges[:, additionIndex]), dim=1)
