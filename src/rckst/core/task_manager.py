import logging
import time
from collections.abc import Sequence
from pathlib import Path

from rckst.core.models import RCKSTError, RCKSTInstance, RCKSTSolution, RCKSTTask, RCKSTTaskConfig, RCKSTTaskOptions
from rckst.core.rckst_file_manager import RCKSTFileManager


class RCKSTTaskManager:
    def __init__(self, tasks: Sequence[RCKSTTask], logger: logging.Logger | None = None) -> None:
        self.tasks: dict[str, RCKSTTask] = {task.name: task for task in tasks}
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

    def run_task(self, task_name: str, instance_path: Path, output_path: Path | None = None, options: RCKSTTaskOptions | None = None) -> tuple[RCKSTSolution, float]:
        if task_name not in self.tasks:
            self.logger.error("Task '%s' not found in registered tasks: %s", task_name, list(self.tasks.keys()))
            raise RCKSTError(f"Task '{task_name}' is not registered.")

        task = self.tasks[task_name]

        self.logger.info("Loading instance from '%s'", instance_path)
        load_start = time.perf_counter()
        instance: RCKSTInstance = RCKSTFileManager.load_instance(instance_path)
        self.logger.info("Loaded instance '%s' (n=%d, E=%d, k=%d, m=%d) in %.4fs", instance.name, instance.n, instance.E, instance.k, instance.m, time.perf_counter() - load_start)

        task_options = options or RCKSTTaskOptions()
        config = RCKSTTaskConfig(instance=instance, options=task_options)

        self.logger.info("Starting task '%s' on instance '%s' (seed=%s, params=%s)", task_name, instance.name, task_options.seed, task_options.params)
        solve_start = time.perf_counter()
        solution: RCKSTSolution = task.solve(config)
        solve_elapsed = time.perf_counter() - solve_start
        self.logger.info("Completed task '%s' in %.4fs (total_weight=%s)", task_name, solve_elapsed, solution.total_weight)

        if output_path is not None:
            self.logger.info("Saving solution to '%s'", output_path)
            RCKSTFileManager.save_solution(solution, output_path)
            self.logger.info("Successfully saved solution to '%s'", output_path)

        return solution, solve_elapsed
