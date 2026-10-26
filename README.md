# RC-k-ST

Resource-Constrained k-node Spanning Tree — Assignment 1 (see `milestone.md`).

Authors: Filip Milanov, Maximilian Fischer

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # mhlib, numpy, networkx, pandas, matplotlib
pip install -e .                  # makes `import rckst` work
```

## Layout

| Path | Purpose |
|---|---|
| `src/rckst/` | Python package (instance reader, heuristics, local search, metaheuristics) |
| `scripts/` | Command-line entry points (experiments, solution generation) |
| `tests/` | Tests / validator checks |
| `instances/` | Problem instances (`train/`, `test/`, `competition/`, `example_instance.txt`) |
| `results/` | Generated experiment output (CSVs, plots) — not committed except via `.gitkeep` |
| `report/` | Report source (LaTeX/Word) and figures |
| `milestone.md` | Assignment milestones/timeline |

## Task split

| Tasks | Owner | Modules in `src/rckst/` |
|---|---|---|
| 1–7: reader/solution/checker, greedy + randomized construction, Pilot/Beam Search, neighborhoods + step functions, local search, VND, GRASP | Maximilian Fischer | `instance`, `solution`, `construction`, `beam_search`, `neighborhoods`, `local_search`, `vnd`, `grasp` |
| 8–13: metaheuristic, delta-evaluation, tuning, experiments, analysis, report | Filip Milanov | `metaheuristic`, `delta_eval`, `tuning`, `experiments`, `analysis`, `report/` |

Task 9 (delta-evaluation) plugs into the neighborhoods from task 4/5, so the interface between `neighborhoods.py` and `delta_eval.py` must be agreed on early.

## Running

```bash
python -c "import rckst"
```

## Milestones

Tracked in `milestone.md` (M0–M10).
