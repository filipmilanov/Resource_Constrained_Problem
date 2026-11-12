# RC-k-ST

Resource-Constrained k-node Spanning Tree heuristics — Assignment 1.

Authors: Filip Milanov, Maximilian Fischer


# Thoughts:

for exercise 1. and 2 was my original interpretation that the constructed heuristic needs to always produce a solution that
is at least feasible but not nessesarily minimal. In my tests i found out that this it self is not trival. Therefore i changed
my interpretation that we need to construct a k-subtree that is not nessesarily feasable but tries to be.




## Requirements

Python 3.12, [uv](https://docs.astral.sh/uv/)


```bash
uv sync
```

PyTorch is installed from its official CPU-only package index; CUDA and a GPU
are not required.

## Task split

| Tasks | Owner | Modules in `src/rckst/` |
|---|---|---|
| 1–7: reader/solution/checker, greedy + randomized construction, Pilot/Beam Search, neighborhoods + step functions, local search, VND, GRASP | Maximilian Fischer | `instance`, `solution`, `construction`, `beam_search`, `neighborhoods`, `local_search`, `vnd`, `grasp` |
| 8–13: metaheuristic, delta-evaluation, tuning, experiments, analysis, report | Filip Milanov | `metaheuristic`, `delta_eval`, `tuning`, `experiments`, `analysis`, `report/` |

Task 9 (delta-evaluation) plugs into the neighborhoods from task 4/5, so the interface between `neighborhoods.py` and `delta_eval.py` must be agreed on early.

## Running

```bash
uv run python -c "import rckst; print(rckst.__version__)"
```
