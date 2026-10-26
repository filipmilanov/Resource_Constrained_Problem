# Assignment 1 – RC-k-ST Milestones

**Deadline:** Sunday, 29.11.2026, 23:55 (TUWEL): report + source code + solutions for at least 3 competition instances.

Full spec: [`problem/problem_description_and_general_information.pdf`](problem/problem_description_and_general_information.pdf). Key constraints from it:
- Max runtime **15 minutes per instance**, **single CPU thread** (no multithreading/multiprocessing/GPU).
- Solution file format: first line = instance filename (no path/extension), then one `u v` line per selected edge (`k − 1` lines total).
- Tune on `instances/train/`, report results on `instances/test/`, submit to the ranking server via `instances/competition/`.

---

## Requirements to install

Stack: **Python 3.10+ with `mhlib` (pip), `numpy`, `networkx`, `pandas`, `matplotlib`** (see `requirements.txt`).

| What | Why |
|---|---|
| Python (>= 3.10) + venv | main language |
| Git | version control |
| `mhlib` | ready-made LS, VND, GVNS, SA, GRASP, etc. |
| `numpy`, `networkx`, `pandas`, `matplotlib` | arrays, graphs, results, plots |
| VS Code + Python extension | editor |
| LaTeX (TeX Live / Overleaf) or Word | report |

**Task split:** tasks 1–7 → Maximilian Fischer; tasks 8–13 → Filip Milanov (see `README.md`).

`instances/train/`, `instances/test/`, `instances/competition/` and the problem description are already in this repo.

---

## Milestones

### M0 – Setup (day 1)
- [x] Read the problem description, create a git repo with the folders `src/`, `instances/`, `results/`, `report/`.
- [x] Install tools, split tasks in the group (1–7 Maximilian, 8–13 Filip).

### M1 – Problem basics (tasks 1, 9 prep)  
**Owner:** Maximilian
- [ ] Instance reader and solution representation (selected nodes + edges).
- [ ] Feasibility check (k nodes, tree, all resource constraints) and objective function.
- [ ] Solution checker/validator you can trust.

### M2 – Construction heuristics (tasks 1, 2)  
**Owner:** Maximilian
- [ ] Deterministic greedy construction.
- [ ] Randomized version (RCL with parameter α or k).

### M3 – Pilot / Beam Search (task 3)  
**Owner:** Maximilian
- [ ] Implement one of them with parameter β or γ.

### M4 – Local search framework + neighborhoods (tasks 4, 5)  
**Owner:** Maximilian (LS/neighborhoods) + Filip (delta-evaluation, task 9)
- [ ] LS with pluggable neighborhood and step function (first / best improvement, random).
- [ ] At least 3 neighborhoods (e.g. swap node, add/remove leaf, replace edge).
- [ ] Decide how to handle infeasible solutions.
- [ ] **Delta-evaluation** for each neighborhood (task 9).

### M5 – VND and GRASP (tasks 6, 7)  
**Owner:** Maximilian
- [ ] VND over your neighborhoods (order configurable).
- [ ] GRASP = randomized construction + LS (or VND).

### M6 – One metaheuristic (task 8)  
**Owner:** Filip
- [ ] Choose one: GVNS (easiest on top of VND), SA, TS or CC.

### M7 – Tuning (task 10)  
**Owner:** Filip
- [ ] Tune parameters on `instances/train/` (α, β/γ, neighborhood probabilities, shaking size / cooling / tabu length).
- [ ] Log what you tried and the effect.

### M8 – Experiments (tasks 11, 12)  
**Owner:** Filip
- [ ] Script to run all algorithms on `instances/test/`, multiple seeds each.
- [ ] Record runtime, objective, gap to best, iterations, evaluated neighbors, tightest-resource slack.
- [ ] Controlled comparisons: construction vs GRASP, construction vs Pilot/Beam, 3 neighborhoods × 2 step functions, VND order, metaheuristic parameter.
- [ ] Plots (objective over time, summary table).

### M9 – Competition submission  
**Owner:** both
- [ ] Generate solution files (filename line + `k − 1` edge lines, see spec) for at least 3 instances and check them with the validator.

### M10 – Report and hand-in (task 13)  
**Owner:** Filip
- [ ] Describe algorithms, delta-evaluation and complexity, tuning, results, conclusions.
- [ ] Answer the guiding questions of the assignment (reachability, infeasible solutions, iterations to local optimum, …).
- [ ] Clean the code, add a README, upload everything to TUWEL **before the deadline**.

---

## Suggested timeline

| Week | Milestones |
|---|---|
| 1 | M0–M2 |
| 2 | M3–M4 |
| 3 | M5–M6 |
| 4 | M7–M8 |
| 5 | M9–M10 + buffer |
