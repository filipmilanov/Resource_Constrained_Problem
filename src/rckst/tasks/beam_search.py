from typing import Callable
from rckst.core.models import RCKSTSolution, RCKSTTask, RCKSTTaskConfig, RCKSTError
from jaxtyping import Float, Int, Bool

from torch import Tensor
import torch


class BeamSearchTask(RCKSTTask):
    @property
    def name(self) -> str:
        return "beam_search"

    def solve(self, config: RCKSTTaskConfig) -> RCKSTSolution:
        """
        performs beam search implements the pseudo code from the slides "02_construction_heuristics" slide 29

        Starts with an empty solution then expands the solution by one edge in each step until a complete solution is found.
        In each step only beta many extensions are considered.

        Also by convention zeros in the solutions tensor means that it is partial

        params[beta] number of partial solutions kept in each step default 50 in jaxtyping B
        params[normalizeBeforeBlend] default is True the values for the rank calculation are normalized before blending 
        params[alpha] importance of the resource constraint. if 0 ignore the hard constraint default 0.9
        """
        inst = config.instance

        beta = config.options.params["beta"]
        normalizeBeforeRank = config.options.params.get("normalizeBeforeBlend", True)
        alpha = config.options.params.get("alpha", 0.9)

        assert type(beta) == int
        assert type(alpha) == int or type(alpha) == float
        assert type(normalizeBeforeRank) == bool

        solutions: Int[Tensor, "B 2 (k-1)"] = torch.zeros([1, 2, inst.k - 1], dtype=torch.int64)
        currentCosts: Int[Tensor, "B m"] = torch.zeros([1, inst.m], dtype=inst.node_resources.dtype)
        currentTotalWeight: Int[Tensor, "B"] = torch.zeros([1], dtype=inst.edge_weights.dtype)

        for i in range(0, inst.k - 1):
            # create all one-step extensions
            if i == 0:
                # every edge is selectable 
                selectableEdgesMask: Bool[Tensor, "B E"] = torch.ones([solutions.shape[0], inst.E], dtype=torch.bool, device=solutions.device)
            else:
                # only edges are selectable that are connected to one node in the current solution
                selectedNodes: Int[Tensor, "B selected_node_entries"] = solutions[:, :, 0:i].reshape(solutions.shape[0], -1)
                inSrc: Bool[Tensor, "B E"] = (inst.edge_index[0][None, :, None] == selectedNodes[:, None, :]).any(dim=2)
                inDest: Bool[Tensor, "B E"] = (inst.edge_index[1][None, :, None] == selectedNodes[:, None, :]).any(dim=2)
                addsNewNode: Bool[Tensor, "B E"] = inSrc != inDest
                selectableEdgesMask = (inSrc | inDest) & addsNewNode

            parentIds: Int[Tensor, "N"]
            addableEdgesIndices: Int[Tensor, "N"]
            parentIds, addableEdgesIndices = torch.nonzero(selectableEdgesMask, as_tuple=True)
            extensions: Int[Tensor, "N 2 (k-1)"] = solutions[parentIds]
            extensions[:, :, i] = inst.edge_index[:, addableEdgesIndices].T

            if extensions.shape[0] == 0:
                raise RCKSTError("not extensions found")

            addableEdges: Int[Tensor, "2 N"] = inst.edge_index[:, addableEdgesIndices]
            if i == 0:
                # add the resources of both nodes of the first edge
                newNodesResources: Int[Tensor, "N m"] = inst.node_resources[addableEdges.T - 1].sum(dim=1)
            else:
                # get the indices of the new nodes and get thier resources  
                newNodes: Int[Tensor, "N"] = torch.where(inSrc[parentIds, addableEdgesIndices], addableEdges[1], addableEdges[0])
                newNodesResources: Int[Tensor, "N m"] = inst.node_resources[newNodes - 1]

            # deta evaluation 
            newEdgeCosts: Int[Tensor, "N m"] = inst.edge_costs[addableEdgesIndices]
            resourceBalances: Int[Tensor, "N m"] = currentCosts[parentIds] + (newNodesResources - newEdgeCosts)
            edgeWeights: Int[Tensor, "N"] = inst.edge_weights[addableEdgesIndices]
            totalWeight: Int[Tensor, "N"] = currentTotalWeight[parentIds] + edgeWeights
            resourceScore: Int[Tensor, "N"] = torch.where(resourceBalances < 0, torch.pow(resourceBalances, 2), 0).sum(1)
            
            norm:Callable[[Tensor, bool], Tensor]= lambda data, doNorm: data if not doNorm else data / (data.max() + 1e-9)
            rank: Float[Tensor, "N"] = (norm(resourceScore, normalizeBeforeRank) * alpha) + (norm(totalWeight, normalizeBeforeRank) * (1 - alpha))

            if i < inst.k - 2:
                bestSols: Int[Tensor, "B"] = torch.topk(rank, min(beta, extensions.shape[0]), largest=False).indices
                solutions = extensions[bestSols]
                currentCosts = resourceBalances[bestSols]
                currentTotalWeight = totalWeight[bestSols]
            else:
                weightIndex: Int[Tensor, "N"] = torch.sort(totalWeight, descending=False).indices
                feasible: Bool[Tensor, "N"] = resourceScore[weightIndex] == 0
                bestFeasible: bool = bool(feasible.any().item())
                bestIndex: Int[Tensor, ""] = weightIndex[torch.nonzero(feasible)[0,0]] if bestFeasible else weightIndex[0]
                bestWeight: int | float = totalWeight[bestIndex].item()
                return RCKSTSolution(inst.name, extensions[bestIndex, : ,:], bestWeight, bestFeasible)

        raise RCKSTError("No solution found ")
      

