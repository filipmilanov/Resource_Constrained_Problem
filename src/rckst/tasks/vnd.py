import time

import torch
from jaxtyping import Bool, Int
from torch import Tensor

from rckst.core.models import RCKSTError, RCKSTInstance, RCKSTNeighborhood, RCKSTSolution, RCKSTTask, RCKSTTaskConfig
from rckst.tasks.deterministic_construction import DeterministicConstructionTask


def evaluate_batch(inst: RCKSTInstance, candidateEdges: Int[Tensor, "B 2 (k-1)"]) -> tuple[Int[Tensor, "B"], Bool[Tensor, "B"]]:
    """
    Evaluates a batch of candidate solutions in a fully vectorized PyTorch call.
    Returns (totalWeights, isFeasible).
    """
    B = candidateEdges.shape[0]
    u = candidateEdges[:, 0, :]
    v = candidateEdges[:, 1, :]
    edgeIds = inst.edge_lookup[u, v]

    # Total weight per candidate: (B,)
    totalWeights: Int[Tensor, "B"] = inst.edge_weights[edgeIds].sum(dim=-1)

    # Resource costs sum per candidate: (B, m)
    edgeCostsSum: Int[Tensor, "B m"] = inst.edge_costs[edgeIds].sum(dim=1)

    # sort the endpoints and only count the resources of a node once. this also works for integer tensors on cuda
    nodes: Int[Tensor, "B 2*(k-1)"] = torch.sort(candidateEdges.reshape(B, -1), dim=1).values
    uniqueNodesMask: Bool[Tensor, "B 2*(k-1)"] = torch.ones_like(nodes, dtype=torch.bool)
    uniqueNodesMask[:, 1:] = nodes[:, 1:] != nodes[:, :-1]
    nodeYieldsSum: Int[Tensor, "B m"] = (inst.node_resources[nodes - 1] * uniqueNodesMask.unsqueeze(-1)).sum(dim=1)

    resourceSlack = nodeYieldsSum - edgeCostsSum
    isFeasible: Bool[Tensor, "B"] = (resourceSlack >= 0).all(dim=-1)

    return totalWeights, isFeasible


class VNDTask(RCKSTTask):
    @property
    def name(self) -> str:
        return "vnd"
    
    def solveBestImprovement(self, neigh: RCKSTNeighborhood, config: RCKSTTaskConfig, current: RCKSTSolution, deadline: float | None = None) -> tuple[RCKSTSolution, int | float, bool]:
        batchSize = config.options.params.get("batchSize", 1024)
        assert isinstance(batchSize, int) and batchSize > 0
        bestSolution = current

        # check all neighbors in batches and keep a feasible solution over an infeasible one. if both have
        # the same feasibility keep the solution with the smaller weight
        for candidates in neigh.generate_batches(config, current, batchSize):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if candidates.shape[0] > 0:
                weights, feasible = evaluate_batch(config.instance, candidates)
                if bestSolution.feasible:
                    mask: Bool[Tensor, "B"] = feasible & (weights < bestSolution.total_weight)
                else:
                    mask: Bool[Tensor, "B"] = feasible | (weights < bestSolution.total_weight)
                if torch.any(mask):
                    # feasible candidates have priority over all infeasible candidates in this batch
                    if torch.any(mask & feasible):
                        mask &= feasible
                    indices: Int[Tensor, "C"] = torch.where(mask)[0]
                    bestIndex: Int[Tensor, ""] = indices[torch.argmin(weights[indices])]
                    bestSolution = RCKSTSolution(config.instance.name, candidates[bestIndex].clone(), weights[bestIndex].item(), bool(feasible[bestIndex]))
        return bestSolution, bestSolution.total_weight, bestSolution.feasible
        
    def solveFirstImprovement(self, neigh: RCKSTNeighborhood, config: RCKSTTaskConfig, current: RCKSTSolution, deadline: float | None = None) -> tuple[RCKSTSolution, int | float, bool]:
        batchSize = config.options.params.get("batchSize", 1024)
        assert isinstance(batchSize, int) and batchSize > 0
        # find the first improving neighbor in the order in which the batches are generated
        for candidates in neigh.generate_batches(config, current, batchSize):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if candidates.shape[0] > 0:
                weights, feasible = evaluate_batch(config.instance, candidates)
                if current.feasible:
                    mask: Bool[Tensor, "B"] = feasible & (weights < current.total_weight)
                else:
                    mask: Bool[Tensor, "B"] = feasible | (weights < current.total_weight)
                indices: Int[Tensor, "C"] = torch.where(mask)[0]
                if indices.numel() > 0:
                    index = indices[0]
                    solution = RCKSTSolution(config.instance.name, candidates[index].clone(), weights[index].item(), bool(feasible[index]))
                    return solution, solution.total_weight, solution.feasible
        return current, current.total_weight, current.feasible

    
    def solveRandomImprovement(self, neigh: RCKSTNeighborhood, config: RCKSTTaskConfig, current: RCKSTSolution) -> tuple[RCKSTSolution, int | float, bool]:
        nextS = neigh.sample_random(config, current)
        if nextS is None:
            return current, current.total_weight, current.feasible
        weights, feasible = evaluate_batch(config.instance, nextS.unsqueeze(0))
        score, isFeasible = weights[0].item(), bool(feasible[0])
        if (isFeasible and not current.feasible) or (isFeasible == current.feasible and score < current.total_weight):
            nextSolution = RCKSTSolution(config.instance.name, nextS.clone(), score, isFeasible)
            return nextSolution, score, isFeasible
        return current, current.total_weight, current.feasible
    
    
    def solve(self, config: RCKSTTaskConfig) -> RCKSTSolution:
        """
        Variable Neighborhood Descent (VND) switches in each local search step the neighborhood 
        to one of the provided in the config. 

        params[mode] step function to use ("best_improvement", "first_improvement", "random_step") default "best_improvement"
        params[maxIterations] maximum number of improvement iterations default 1000 if negative iterates for ever
        params[batchSize] chunk batch size, how many neighborhood are generated at ones default 1024
        params[initialSolution] starting solution; if not provided deterministic construction is used
        params[maxSeconds] optional time limit for the local search
        """
        mode = config.options.params.get("mode", "best_improvement")
        maxIterations = config.options.params.get("maxIterations", 1000)
        batchSize = config.options.params.get("batchSize", 1024)
        maxSeconds = config.options.params.get("maxSeconds")
        if mode not in ["best_improvement", "first_improvement", "random_step"]:
            raise RCKSTError("Unknown VND mode")
        if not isinstance(maxIterations, int) or not isinstance(batchSize, int) or batchSize <= 0:
            raise RCKSTError("maxIterations must be an integer and batchSize must be positive")
        if not isinstance(config.neighborhood, list) or not config.neighborhood:
            raise RCKSTError("VND requires a non-empty list of neighborhoods")
        if maxSeconds is not None and (not isinstance(maxSeconds, (int, float)) or not 0 < maxSeconds < float("inf")):
            raise RCKSTError("maxSeconds must be a positive finite number")

        initialSolution = config.options.params.get("initialSolution")
        if initialSolution is not None and not isinstance(initialSolution, RCKSTSolution):
            raise RCKSTError("initialSolution must be an RCKSTSolution")
        if config.options.seed is not None:
            torch.manual_seed(config.options.seed)
        deadline = time.perf_counter() + maxSeconds if maxSeconds is not None else None
        current = initialSolution if initialSolution is not None else DeterministicConstructionTask().solve(config)
        bestSolution = current

        # after an improvement start again with the first neighborhood. if no improvement was found go to
        # the next neighborhood and stop after all neighborhoods failed or maxIterations steps were made
        neighborhoodIndex = 0
        i = 0
        while neighborhoodIndex < len(config.neighborhood) and (i < maxIterations or maxIterations < 0) and (deadline is None or time.perf_counter() < deadline):
            neighborhood = config.neighborhood[neighborhoodIndex]
            if mode == "best_improvement":
                solution, score, feasible = self.solveBestImprovement(neighborhood, config, current, deadline)
            elif mode == "first_improvement":
                solution, score, feasible = self.solveFirstImprovement(neighborhood, config, current, deadline)
            else:
                solution, score, feasible = self.solveRandomImprovement(neighborhood, config, current)
            if (feasible and not current.feasible) or (feasible == current.feasible and score < current.total_weight):
                current = solution
                if (current.feasible and not bestSolution.feasible) or (current.feasible == bestSolution.feasible and current.total_weight < bestSolution.total_weight):
                    bestSolution = current
                neighborhoodIndex = 0
            else:
                neighborhoodIndex += 1
            i += 1
        return bestSolution
