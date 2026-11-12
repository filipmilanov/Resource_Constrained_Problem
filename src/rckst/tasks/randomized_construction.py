import torch
from jaxtyping import Float, Int
from torch import Tensor

from rckst.core.models import RCKSTError, RCKSTSolution, RCKSTTask, RCKSTTaskConfig


class RandomizedConstructionTask(RCKSTTask):
    @property
    def name(self) -> str:
        return "randomized_construction"

    def solve(self, config: RCKSTTaskConfig) -> RCKSTSolution:
        """
        The procedure:
            1. samples a root node distributes softmax over the resource mean.
            2. calculates which edges can be added to the tree
            3. calculates the new balance if the edge is added
            4. calculates the resourceScore which looks at the balance and squares the axis if it is negative
                therefore got a penalty for non feasible edges
            5. alpha blends the weights of the edges and the resource score (default with normalization)
            6. samples the edge using softmax over the negative ranks and repeats that k times

        The procedure is performed rootRetries times and returns the feasible solution with the best
        score. if no feasible solution was found it returns the one with the best score. It no k
        subtree was found it throws an error


        params[alpha] importance of the resource constraint. if 0 ignore the hard constraint
        if 1 only look at the hard constraint. default is 0.9

        params[temperature] temperature for softmax sampling. default is 1.0
        params[rootRetries] how many roots are tried default is 50
        params[normalizeBeforeBlend] is True the values for the rank calculation are normalized before blending
        params[earlyStopAtTree] stops once a tree was found not necessarily feasible default False
        """

        inst = config.instance
        rootRetries = min(config.options.params.get("rootRetries", 50), inst.n)
        alpha = config.options.params.get("alpha", 0.9)
        temperature = config.options.params.get("temperature", 1.0)
        normalizeBeforeRank = config.options.params.get("normalizeBeforeBlend", True)
        earlyStopAtTree = config.options.params.get("earlyStopAtTree", False)

        if config.options.seed is not None:
            torch.manual_seed(config.options.seed)

        assert type(alpha) == int or type(alpha) == float
        assert type(temperature) == int or type(temperature) == float
        assert type(rootRetries) == int
        assert type(normalizeBeforeRank) == bool
        assert type(earlyStopAtTree) == bool

        nodeMeans = inst.node_resources.mean(dim=-1, dtype=torch.float64)
        if temperature == 0:
            _, rootTryOrder = torch.sort(nodeMeans, descending=True)
        else:
            rootProbs = torch.softmax(nodeMeans / temperature, dim=0)
            rootTryOrder = torch.multinomial(rootProbs, num_samples=rootRetries, replacement=False)

        bestSolution: RCKSTSolution | None = None
        # TODO batch the root retries
        for j in range(0, rootRetries):
            root = rootTryOrder[j] + 1
            inTree = torch.zeros(inst.n + 1, dtype=torch.bool, device=inst.edge_index.device)
            currentEdges = torch.zeros([2, inst.k - 1], dtype=torch.int64)
            currentTotalWeight = torch.zeros([1], dtype=torch.int64)
            inTree[root] = True
            currentCosts: Float[Tensor, "m"] = inst.node_resources[root - 1]

            for i in range(0, config.instance.k - 1):
                # first step is to calculate which edges can be used to add nodes which are new and are connected to the current tree
                # P is the number of Possible Edges to add
                inSrc = inTree[inst.edge_index[0, :]]
                inDest = inTree[inst.edge_index[1, :]]
                mask = inSrc ^ inDest
                addableEdgesIndices: Int[Tensor, "P"] = torch.nonzero(mask).squeeze(1)

                if addableEdgesIndices.shape[0] == 0:
                    # if no edge can be added the current root is not a possible solution therefore skip the calculations
                    break

                addableEdges: Int[Tensor, "2 P"] = inst.edge_index[:, mask]

                # calculate the resource balances if the possible edges are added
                newNodes: Int[Tensor, "P"] = torch.where(inSrc[addableEdgesIndices], addableEdges[1, :], addableEdges[0, :])
                newEdgeCosts: Float[Tensor, "P m"] = inst.edge_costs[addableEdgesIndices, :]
                newNodesResources: Float[Tensor, "P m"] = inst.node_resources[newNodes - 1, :]
                resourceBalances: Float[Tensor, "P m"] = currentCosts + (newNodesResources - newEdgeCosts)

                edgeWeights: Float[Tensor, "P m"] = inst.edge_weights[addableEdgesIndices]
                resourceScore = torch.where(resourceBalances < 0, torch.pow(resourceBalances, 2), 0).sum(1)

                norm = lambda data, doNorm: data if not doNorm else data / (data.max() + 1e-9)
                rank: Float[Tensor, "P"] = (norm(resourceScore, normalizeBeforeRank) * alpha) + (norm(edgeWeights, normalizeBeforeRank) * (1 - alpha))

                if temperature == 0:
                    selectedEdge = torch.argmin(rank, dim=0)
                else:
                    probs = torch.softmax(-rank / temperature, dim=0)
                    selectedEdge = torch.multinomial(probs, num_samples=1)[0]

                # update the tree and solution
                currentCosts = resourceBalances[selectedEdge, :]
                inTree[newNodes[selectedEdge]] = True
                currentEdges[:, i] = addableEdges[:, selectedEdge]
                currentTotalWeight = edgeWeights[selectedEdge] + currentTotalWeight

                if i == config.instance.k - 2:
                    # if done create the solution object and check if the best solution needs to be updated
                    isFeasible = resourceScore[selectedEdge].item() == 0
                    solution = RCKSTSolution(inst.name, currentEdges, currentTotalWeight.item(), isFeasible)
                    if earlyStopAtTree:
                        return solution

                    if bestSolution is None or (bestSolution.feasible == isFeasible and bestSolution.total_weight > solution.total_weight) or (not bestSolution.feasible and isFeasible):
                        bestSolution = solution

        if bestSolution is None:
            raise RCKSTError("No solution found")
        else:
            return bestSolution
