from collections.abc import Iterator

import torch
from jaxtyping import Bool, Int
from torch import Tensor

from rckst.core.models import RCKSTNeighborhood, RCKSTSolution, RCKSTTaskConfig


class ExtensionNeighborhood(RCKSTNeighborhood):

    def generate_batches(self, config: RCKSTTaskConfig, solution: RCKSTSolution, batchSize: int = 1024) -> Iterator[Int[Tensor, "B 2 (k-1)"]]:
        inst = config.instance
        solEdges: Int[Tensor, "2 (k-1)"] = solution.edges

        # find the leafs of the solution and their corresponding edges
        edgeFlat: Int[Tensor, "2*(k-1)"] = solEdges.flatten()
        edgeFlatSorted, edgeFlatSortedIndices = torch.sort(edgeFlat)
        # not leaf if a node is present in at least two edges. Therefore if we flatten and sort the vertices ids and 
        # compare to next we  get a mask where each node that is not a leaf has a one. if we have a,b,c we know if the
        # mask at position a == 1 => (b == a), we also know if b == 1 => (b == c). to check if the node is in only one edge
        # we want to check for a!=b, b!= c. contra positions of both means a!=b => a == 0 and  b!=c => b == 0 so we need
        # to check that the current position and the one before is zero this holds to the inner indices.
        # on the first is only depends if the current is 0, and at the end if the last is 0
        notLeafMask: Bool[Tensor, "(2*(k-1))-1"] = edgeFlatSorted[:-1] == edgeFlatSorted[1:]
        leafMask: Bool[Tensor, "2*(k-1)"] = torch.ones_like(edgeFlatSorted, dtype=torch.bool)
        leafMask[:-1] &= ~notLeafMask  
        leafMask[1:] &= ~notLeafMask  
        leafs: Int[Tensor, "L"] = edgeFlatSorted[leafMask]
        originalLeafIndices: Int[Tensor, "L"] = edgeFlatSortedIndices[leafMask]
        # flatten() stores the first row followed by the second row.
        edgeIndexInSolution: Int[Tensor, "L"] = originalLeafIndices.remainder(solEdges.shape[1])
        globalLeafEdgeIndices: Int[Tensor, "L"] = inst.edge_lookup[solEdges[0, edgeIndexInSolution], solEdges[1, edgeIndexInSolution]]
        # we got in leaf the node indices and in globalLeafEdgeIndices the edge indices of the leafs

        # Observation all possible extensions use edges that are asdjacent to a leaf node. Therefore we can calculate 
        # those once for the current solution and can then for each leaf remove only those that are no longer possible
        # to select because the leaf was removed
        
        # We only look at edges since those are the things we add. we choose a edge if either the start or the end of the edge
        # is in a leaf node and it connects a new node the Tree. first create a mask for each node if it is in the graph.
        # then if for all edges  
        inTree: Bool[Tensor, "n"] = torch.zeros(inst.n, dtype=torch.bool)
        inTree[edgeFlat-1] = True
        inLeaf: Bool[Tensor, "n"] = torch.zeros(inst.n, dtype=torch.bool)
        inLeaf[leafs - 1] = True
        inSrc: Bool[Tensor, "E"] = inTree[inst.edge_index[0] - 1]
        inDest: Bool[Tensor, "E"] = inTree[inst.edge_index[1] - 1]
        # a removed leaf can also be connected back to the remaining tree through a different edge
        connectsLeaf: Bool[Tensor, "E"] = inLeaf[inst.edge_index[0] - 1] | inLeaf[inst.edge_index[1] - 1]
        addableEdgesIndices: Int[Tensor, "D"] = torch.where((inSrc ^ inDest) | (inSrc & inDest & connectsLeaf))[0]
        addableEdges: Int[Tensor, "2 D"] = inst.edge_index[:, addableEdgesIndices]
        numAddable = addableEdges.shape[1]
        if numAddable == 0:
            return

        # one batch contains all extentions with L variations on the removed leafs. so one batch contains at most batchsize*D many neighbours 
        # P < Batchsize
        numPairs = leafs.numel() * numAddable
        edgePositions: Int[Tensor, "k-1"] = torch.arange(solEdges.shape[1])
        for start in range(0, numPairs, batchSize):
            # pair indices each pair has a entry and gets it's own solution. for each solution we need to know which is the leaf which is removed
            # (leafIndices) and which edge is added for it (additions) 
            pairIndices: Int[Tensor, "P"] = torch.arange(start, min(start + batchSize, numPairs))
            leafIndices: Int[Tensor, "P"] = torch.div(pairIndices, numAddable, rounding_mode="floor")
            additionIndices: Int[Tensor, "P"] = pairIndices.remainder(numAddable)
            additions: Int[Tensor, "P 2"] = addableEdges[:, additionIndices].T
            removedLeafs: Int[Tensor, "P"] = leafs[leafIndices]
            removedIndices: Int[Tensor, "P"] = edgeIndexInSolution[leafIndices]

            # need to filter the invalid extensions. Check that the added node is connected to a not removed leaf
            inRemainingSrc: Bool[Tensor, "P"] = inTree[additions[:, 0] - 1] & (additions[:, 0] != removedLeafs)
            inRemainingDest: Bool[Tensor, "P"] = inTree[additions[:, 1] - 1] & (additions[:, 1] != removedLeafs)
            addsNewNode: Bool[Tensor, "P"] = inRemainingSrc ^ inRemainingDest
            # with the mask we can select only those solutions that are valid trees 
            mask: Bool[Tensor, "P"] = addsNewNode & (addableEdgesIndices[additionIndices] != globalLeafEdgeIndices[leafIndices])
            additions = additions[mask]
            removedIndices = removedIndices[mask]
            if additions.shape[0] != 0:
                # construct the final neighbours
                keep: Bool[Tensor, "B (k-1)"] = edgePositions.unsqueeze(0) != removedIndices.unsqueeze(1)
                kept: Int[Tensor, "B 2 (k-2)"] = solEdges.T.unsqueeze(0).expand(additions.shape[0], -1, -1)[keep].reshape(additions.shape[0], -1, 2).transpose(1, 2)
                yield torch.cat((kept, additions.unsqueeze(-1)), dim=2)

    def sample_random(self, config: RCKSTTaskConfig, solution: RCKSTSolution) -> Int[Tensor, "2 (k-1)"] | None:
        inst = config.instance
        solEdges: Int[Tensor, "2 (k-1)"] = solution.edges


        # find the leaf nodes and try them in a random order until one can be extended. i
        edgeFlat: Int[Tensor, "2*(k-1)"] = solEdges.flatten()
        degrees: Int[Tensor, "n+1"] = torch.bincount(edgeFlat, minlength=inst.n + 1)
        leafs: Int[Tensor, "L"] = torch.where(degrees == 1)[0]
        for leaf in leafs[torch.randperm(leafs.numel())]:
            # remove the leaf edge and mark the nodes that remain in the tree
            leafEdgeMask: Bool[Tensor, "k-1"] = (solEdges[0] == leaf) | (solEdges[1] == leaf)
            removedEdge: Int[Tensor, "2"] = solEdges[:, leafEdgeMask][:, 0]
            inTree: Bool[Tensor, "n"] = degrees[1:] > 0
            inTree[leaf - 1] = False

            # select edges with exactly one endpoint in the remaining tree, excluding the removed edge
            u, v = inst.edge_index
            boundary: Bool[Tensor, "E"] = inTree[u - 1] ^ inTree[v - 1]
            original: Bool[Tensor, "E"] = ((u == removedEdge[0]) & (v == removedEdge[1])) | ((u == removedEdge[1]) & (v == removedEdge[0]))
            validEdges: Int[Tensor, "2 V"] = inst.edge_index[:, boundary & ~original]
            if validEdges.shape[1] > 0:
                additionIndex: Int[Tensor, "1"] = torch.randint(validEdges.shape[1], (1,))
                return torch.cat((solEdges[:, ~leafEdgeMask], validEdges[:, additionIndex]), dim=1)
        return None
