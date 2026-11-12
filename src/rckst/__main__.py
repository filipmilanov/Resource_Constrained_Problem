import argparse
import ast
import logging
from pathlib import Path

import torch

from rckst.core.models import ConfigValue, RCKSTError, RCKSTTaskOptions, TaskParams
from rckst.core.task_manager import RCKSTTaskManager
from rckst.neighborhoods import ExtensionNeighborhood, ShiftNeighborhood, SwapNeighborhood
from rckst.tasks import BeamSearchTask, DeterministicConstructionTask, GRASPTask, LocalSearchTask, RandomizedConstructionTask, VNDTask


def _parse_val(value: str) -> ConfigValue:
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    if not isinstance(parsed, (str, int, float, bool)):
        raise ValueError("task parameter values must be strings, numbers or booleans")
    return parsed


def main() -> None:
    tasks = [BeamSearchTask(), DeterministicConstructionTask(), RandomizedConstructionTask(), LocalSearchTask(), VNDTask(), GRASPTask()]
    neighborhoods = {"extension": ExtensionNeighborhood, "shift": ShiftNeighborhood, "swap": SwapNeighborhood}
    parser = argparse.ArgumentParser(description="Run a resource-constrained k-subtree heuristic.")
    parser.add_argument("task", choices=[task.name for task in tasks])
    parser.add_argument("instance", type=Path, help="path to an instance file")
    parser.add_argument("--output", "-o", type=Path, help="path to save the solution")
    parser.add_argument("--seed", "-s", type=int, help="random seed")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="PyTorch device (default: cpu)")
    parser.add_argument("--neighborhood", choices=list(neighborhoods), nargs="+", help="neighborhoods in search order")
    parser.epilog = "Task parameters use --name value pairs, for example: --beta 50 or --graspIterations 10."
    args, unknown = parser.parse_known_args()

    if len(unknown) % 2 != 0:
        parser.error("Task parameters must be provided as --name value pairs.")
    params: TaskParams = {}
    for key, value in zip(unknown[::2], unknown[1::2], strict=True):
        if not key.startswith("--") or len(key) == 2 or value.startswith("--"):
            parser.error(f"Invalid task parameter pair: {key} {value}")
        try:
            params[key[2:]] = _parse_val(value)
        except ValueError as error:
            parser.error(str(error))

    constructionParams = {"alpha", "temperature", "rootRetries", "normalizeBeforeBlend", "earlyStopAtTree"}
    searchParams = {"mode", "maxIterations", "batchSize", "maxSeconds"}
    taskParams = {
        "beam_search": {"beta", "alpha", "normalizeBeforeBlend"},
        "deterministic_construction": constructionParams,
        "randomized_construction": constructionParams,
        "local_search": constructionParams | searchParams,
        "vnd": constructionParams | searchParams,
        "grasp": constructionParams | searchParams | {"graspIterations", "localSearchSeconds"},
    }
    invalidParams = params.keys() - taskParams[args.task]
    if invalidParams:
        parser.error(f"Unknown parameter for {args.task}: {', '.join(sorted(invalidParams))}")
    if args.task == "beam_search" and "beta" not in params:
        parser.error("beam_search requires --beta with a positive integer.")
    for name in ("beta", "rootRetries", "batchSize", "graspIterations"):
        if name in params and (type(params[name]) is not int or params[name] <= 0):
            parser.error(f"--{name} must be a positive integer.")
    if "maxIterations" in params and type(params["maxIterations"]) is not int:
        parser.error("--maxIterations must be an integer.")
    for name in ("normalizeBeforeBlend", "earlyStopAtTree"):
        if name in params and type(params[name]) is not bool:
            parser.error(f"--{name} must be true or false.")
    for name in ("alpha", "temperature", "maxSeconds", "localSearchSeconds"):
        if name in params and (type(params[name]) not in (int, float) or not 0 <= params[name] < float("inf")):
            parser.error(f"--{name} must be a nonnegative finite number.")
    for name in ("maxSeconds", "localSearchSeconds"):
        if name in params and params[name] == 0:
            parser.error(f"--{name} must be greater than zero.")
    if "alpha" in params and params["alpha"] > 1:
        parser.error("--alpha must be between 0 and 1.")
    if "mode" in params and params["mode"] not in ("best_improvement", "first_improvement", "random_step"):
        parser.error("--mode must be best_improvement, first_improvement or random_step.")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available in the current Python environment.")
    if args.neighborhood and args.task not in ("local_search", "vnd", "grasp"):
        parser.error("--neighborhood is only available for local_search, vnd and grasp.")
    if not args.instance.is_file():
        parser.error(f"Instance file not found: {args.instance}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    torch.set_default_device(args.device)
    selected = [neighborhoods[name]() for name in args.neighborhood] if args.neighborhood else None
    if args.task == "vnd" and selected is None:
        selected = [ExtensionNeighborhood()]
    options = RCKSTTaskOptions(seed=args.seed, params=params)
    try:
        solution, elapsed = RCKSTTaskManager(tasks).run_task(args.task, args.instance, args.output, options, selected)
    except (RCKSTError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"instance={solution.instance_name} weight={solution.total_weight} feasible={solution.feasible} seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
