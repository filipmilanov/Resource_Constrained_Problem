from rckst.core.models import RCKSTError, RCKSTSolution, RCKSTTask, RCKSTTaskConfig
from rckst.neighborhoods import ExtensionNeighborhood
from rckst.tasks.vnd import VNDTask


class LocalSearchTask(RCKSTTask):
    @property
    def name(self) -> str:
        return "local_search"

    def solve(self, config: RCKSTTaskConfig) -> RCKSTSolution:
        """
        Basic Local Search framework.

        Applies iterative improvement on a given neighborhood structure using the configured
        step function (mode="best_improvement", "first_improvement", or "random_step") until
        a local optimum or maxIterations is reached.

        params[mode] step function to use ("best_improvement", "first_improvement", "random_step") default "best_improvement"
        params[maxIterations] maximum number of improvement iterations default 1000
        params[batchSize] chunk batch size passed to neighborhood evaluations default 1024
        params[initialSolution] optional starting RCKSTSolution. If None, DeterministicConstructionTask is used.
        """
        nh = config.neighborhood if config.neighborhood is not None else ExtensionNeighborhood()
        if isinstance(nh, list):
            if len(nh) == 0:
                raise RCKSTError("Local search requires at least one neighborhood")
            neighborhoods = nh
        else:
            neighborhoods = [nh]

        vndConfig = RCKSTTaskConfig(instance=config.instance, options=config.options, neighborhood=neighborhoods)
        return VNDTask().solve(vndConfig)
