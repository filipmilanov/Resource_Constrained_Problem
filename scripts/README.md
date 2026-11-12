Command-line entry points (run from the repo root, e.g. `python scripts/run_experiments.py`).

Evaluate the feasibility rate of the current Beam Search on a reproducible random sample:

```bash
uv run python scripts/evaluate_beam_feasibility.py
```

Use `--instances-dir`, `--samples`, `--seed`, `--beta`, and `--alpha` to override the defaults.
