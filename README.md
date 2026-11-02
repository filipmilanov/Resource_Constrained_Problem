# RC-k-ST

Resource-Constrained k-node Spanning Tree heuristics — Assignment 1.

Authors: Filip Milanov, Maximilian Fischer

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
