import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_train_help_runs():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "train.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "ModuleNotFoundError" not in r.stderr
    assert "--device" in r.stdout
