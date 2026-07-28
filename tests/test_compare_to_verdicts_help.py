import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_compare_help_runs():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "compare_to_verdicts.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "ModuleNotFoundError" not in r.stderr
    assert "--detections" in r.stdout
    assert "--verdicts" in r.stdout
