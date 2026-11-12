import argparse
import random
import time
from pathlib import Path

import torch

from rckst.core import RCKSTFileManager, RCKSTTaskConfig, RCKSTTaskOptions
from rckst.tasks import BeamSearchTask

DEFAULT_SEED = 42
DEFAULT_INSTANCE_DIR = Path("assets/instances/train/50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample instances and report how often the current Beam Search returns a feasible solution.")
    parser.add_argument("--instances-dir", type=Path, default=DEFAULT_INSTANCE_DIR)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--beta", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    instance_paths = sorted(args.instances_dir.rglob("*.txt"))
    if not instance_paths:
        raise SystemExit(f"No instance files found in '{args.instances_dir}'.")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive.")

    sample_count = min(args.samples, len(instance_paths))
    sampled_paths = random.Random(args.seed).sample(instance_paths, sample_count)
    task = BeamSearchTask()
    feasible_count = 0
    completed_count = 0
    failed_count = 0
    total_runtime = 0.0

    print(f"seed={args.seed} samples={sample_count} beta={args.beta} alpha={args.alpha}")
    for path in sampled_paths:
        instance = RCKSTFileManager.load_instance(path)
        config = RCKSTTaskConfig(instance=instance, options=RCKSTTaskOptions(seed=args.seed, params={"beta": args.beta, "alpha": args.alpha}))

        start = time.perf_counter()
        try:
            solution = task.solve(config)
        except Exception as error:
            runtime = time.perf_counter() - start
            total_runtime += runtime
            failed_count += 1
            print(f"FAIL     {instance.name:<32} runtime={runtime:8.3f}s error={error}")
            continue

        runtime = time.perf_counter() - start
        total_runtime += runtime
        completed_count += 1
        feasible_count += int(solution.feasible)
        status = "FEASIBLE" if solution.feasible else "INFEASIBLE"
        print(f"{status:<8} {instance.name:<32} weight={solution.total_weight:<12} runtime={runtime:8.3f}s")

    feasible_rate_total = feasible_count / sample_count
    feasible_rate_completed = feasible_count / completed_count if completed_count else 0.0

    print("\nSummary")
    print(f"sampled:                 {sample_count}")
    print(f"completed:               {completed_count}")
    print(f"failed:                  {failed_count}")
    print(f"feasible:                {feasible_count}")
    print(f"feasible / sampled:      {feasible_rate_total:.2%}")
    print(f"feasible / completed:    {feasible_rate_completed:.2%}")
    print(f"total runtime:           {total_runtime:.3f}s")


if __name__ == "__main__":
    main()
