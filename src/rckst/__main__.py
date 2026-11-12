import argparse
import ast
import logging
from pathlib import Path

from rckst.core.models import RCKSTTaskOptions, TaskParams
from rckst.core.task_manager import RCKSTTaskManager


def _parse_val(v: str):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--seed", "-s", type=int, default=None)

    args, unknown = parser.parse_known_args()
    params: TaskParams = {k.lstrip("-"): _parse_val(v) for k, v in zip(unknown[::2], unknown[1::2], strict=False)}
    options = RCKSTTaskOptions(seed=args.seed, params=params)

    RCKSTTaskManager(tasks=[]).run_task(task_name=args.task, instance_path=args.instance, output_path=args.output, options=options)


if __name__ == "__main__":
    main()
