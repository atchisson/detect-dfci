import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def _run_help(script):
    return subprocess.run(
        [PYTHON, str(REPO / "scripts" / script), "--help"],
        capture_output=True, text=True, timeout=60,
    )


def test_recon_help_runs():
    r = _run_help("recon.py")
    assert r.returncode == 0, r.stderr
    assert "ModuleNotFoundError" not in r.stderr


def test_run_baseline_help_runs():
    r = _run_help("run_baseline.py")
    assert r.returncode == 0, r.stderr
    assert "ModuleNotFoundError" not in r.stderr
