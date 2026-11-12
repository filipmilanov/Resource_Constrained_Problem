import logging
import time

from rckst.core.models import RCKSTError, RCKSTNeighborhood, RCKSTSolution, RCKSTTask, RCKSTTaskConfig, RCKSTTaskOptions
from rckst.neighborhoods import ExtensionNeighborhood
from rckst.tasks.randomized_construction import RandomizedConstructionTask
from rckst.tasks.vnd import VNDTask


class GRASPTask(RCKSTTask):
    @property
    def name(self) -> str:
        return "grasp"

    def solve(self, config: RCKSTTaskConfig) -> RCKSTSolution:
        """
        Repeatedly construct a random solution and improve it with VND. The best feasible solution is
        returned, or the best infeasible solution if no feasible solution was found.

        params[graspIterations] number of constructions, default 10
        params[maxSeconds] optional total time limit; construction and a batch may finish after the limit
        params[localSearchSeconds] time limit for each VND run, default 30 seconds
        params[mode] VND step function, default first_improvement
        Other parameters are passed to randomized construction and VND.
        """
        iterations = config.options.params.get("graspIterations", 10)
        maxSeconds = config.options.params.get("maxSeconds")
        localSearchSeconds = config.options.params.get("localSearchSeconds", 30)
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
            raise RCKSTError("graspIterations must be a positive integer")
        if not isinstance(localSearchSeconds, (int, float)) or not 0 < localSearchSeconds < float("inf"):
            raise RCKSTError("localSearchSeconds must be a positive finite number")
        if maxSeconds is not None and (not isinstance(maxSeconds, (int, float)) or not 0 < maxSeconds < float("inf")):
            raise RCKSTError("maxSeconds must be a positive finite number")

        if config.neighborhood is None:
            neighborhoods: list[RCKSTNeighborhood] = [ExtensionNeighborhood()]
        elif isinstance(config.neighborhood, list):
            neighborhoods = config.neighborhood
        else:
            neighborhoods = [config.neighborhood]
        if len(neighborhoods) == 0:
            raise RCKSTError("GRASP requires at least one neighborhood")

        construction = RandomizedConstructionTask()
        vnd = VNDTask()
        logger = logging.getLogger(__name__)
        bestSolution: RCKSTSolution | None = None
        start = time.perf_counter()
        deadline = start + maxSeconds if maxSeconds is not None else None
        iteration = 0

        # construct a new tree for every restart. a different seed makes the starts reproducible without
        # constructing the same random tree in every iteration
        while iteration < iterations and (deadline is None or time.perf_counter() < deadline):
            seed = (config.options.seed + iteration) % (2**63) if config.options.seed is not None else None
            params = config.options.params.copy()
            constructionConfig = RCKSTTaskConfig(config.instance, RCKSTTaskOptions(seed=seed, params=params))
            constructionStart = time.perf_counter()
            try:
                solution = construction.solve(constructionConfig)
            except RCKSTError as error:
                if error.message != "No solution found":
                    raise
                solution = None
            constructionTime = time.perf_counter() - constructionStart

            if solution is not None:
                constructed = solution
                remaining = deadline - time.perf_counter() if deadline is not None else localSearchSeconds
                localSearchStart = time.perf_counter()
                if remaining > 0:
                    # copy the options so that the initial solution and time limit do not change the caller's config
                    vndParams = config.options.params.copy()
                    vndParams["initialSolution"] = solution
                    vndParams["maxSeconds"] = min(localSearchSeconds, remaining)
                    vndParams.setdefault("mode", "first_improvement")
                    vndConfig = RCKSTTaskConfig(config.instance, RCKSTTaskOptions(seed=seed, params=vndParams), neighborhoods)
                    solution = vnd.solve(vndConfig)
                localSearchTime = time.perf_counter() - localSearchStart

                if bestSolution is None or (solution.feasible and not bestSolution.feasible) or (solution.feasible == bestSolution.feasible and solution.total_weight < bestSolution.total_weight):
                    bestSolution = solution
                logger.info(
                    "GRASP iteration=%d seed=%s construction_weight=%s construction_feasible=%s construction_seconds=%.3f "
                    "local_weight=%s local_feasible=%s local_seconds=%.3f best_weight=%s best_feasible=%s",
                    iteration + 1, seed, constructed.total_weight, constructed.feasible, constructionTime,
                    solution.total_weight, solution.feasible, localSearchTime, bestSolution.total_weight, bestSolution.feasible,
                )
            else:
                logger.info("GRASP iteration=%d seed=%s no tree constructed construction_seconds=%.3f", iteration + 1, seed, constructionTime)
            iteration += 1

        if bestSolution is None:
            raise RCKSTError("GRASP could not construct a solution")
        return bestSolution
