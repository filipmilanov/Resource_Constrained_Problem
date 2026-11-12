import subprocess
import sys
from pathlib import Path

import pytest

INSTANCE = Path(__file__).parent.parent / "assets" / "instances" / "example_instance.txt"


@pytest.mark.parametrize("task,params", [
    ("deterministic_construction", ["--rootRetries", "1"]),
    ("randomized_construction", ["--rootRetries", "1"]),
    ("beam_search", ["--beta", "5"]),
    ("local_search", ["--rootRetries", "1", "--maxIterations", "2"]),
    ("vnd", ["--rootRetries", "1", "--maxIterations", "2", "--neighborhood", "shift", "swap"]),
    ("grasp", ["--rootRetries", "1", "--maxIterations", "2", "--graspIterations", "2"]),
])
def test_cli_runs_task(task, params, tmp_path):
    output = tmp_path / "solution.txt"
    result = subprocess.run([sys.executable, "-m", "rckst", task, str(INSTANCE), "--seed", "42", "--output", str(output), *params], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "instance=example_instance" in result.stdout and "feasible=" in result.stdout
    assert output.read_text().splitlines()[0] == "example_instance"


@pytest.mark.parametrize("task,params", [
    ("beam_search", []),
    ("beam_search", ["--beta"]),
    ("beam_search", ["--beta", "0"]),
    ("beam_search", ["--beta", "5", "--batchSize", "100"]),
    ("vnd", ["--mode", "unknown"]),
    ("grasp", ["--graspIterations", "two"]),
    ("grasp", ["--earlyStopAtTree", "maybe"]),
])
def test_cli_rejects_invalid_parameters(task, params):
    result = subprocess.run([sys.executable, "-m", "rckst", task, str(INSTANCE), *params], capture_output=True, text=True)
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
