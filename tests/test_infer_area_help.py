import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_infer_area_help_runs():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "infer_area.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "ModuleNotFoundError" not in r.stderr
    assert "--boundary" in r.stdout
    assert "--conf" in r.stdout
