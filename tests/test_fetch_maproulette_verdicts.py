import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import fetch_maproulette_verdicts as fmv  # noqa: E402
from detection_ortho.dataset import parse_verdicts  # noqa: E402


def _task(status, lon, lat, score="0.8"):
    return {
        "status": status,
        "location": {"coordinates": [lon, lat]},
        "geometries": {"features": [{"properties": {"score": score}}]},
    }


def _fake_get(challenges, tasks_by_id, fail_ids=()):
    """http_get_json factice : sert la liste des challenges et les pages de tâches."""
    def get(url, *a, **k):
        if "/project/" in url:
            return challenges
        cid = int(url.split("/challenge/")[1].split("/")[0])
        if cid in fail_ids:
            raise RuntimeError("réseau coupé")
        page = int(url.split("page=")[1])
        size = fmv.PAGE
        return tasks_by_id[cid][page * size:(page + 1) * size]
    return get


def test_iter_tasks_paginates_and_stops_on_exact_multiple(monkeypatch):
    monkeypatch.setattr(fmv, "PAGE", 2)
    tasks = [_task(1, 0.1 * i, 47.0) for i in range(4)]  # multiple exact de PAGE
    monkeypatch.setattr(fmv, "http_get_json", _fake_get([], {7: tasks}))
    assert len(list(fmv.iter_tasks(7))) == 4


def test_main_writes_one_csv_per_dept(tmp_path, monkeypatch, capsys):
    challenges = [
        {"id": 1, "name": "DECI 44"},
        {"id": 2, "name": "DECI 2A"},
        {"id": 3, "name": "Test sans rapport"},
    ]
    tasks = {
        1: [_task(1, 0.65, 47.33), _task(2, 0.66, 47.34),
            _task(5, 0.67, 47.35), _task(6, 0.68, 47.36),
            {"status": 1, "geometries": {"features": []}}],  # sans location
        2: [_task(2, 9.0, 42.0)],
    }
    monkeypatch.setattr(fmv, "http_get_json", _fake_get(challenges, tasks))
    monkeypatch.setattr(sys, "argv", [
        "fetch_maproulette_verdicts.py", "--out", str(tmp_path)])
    fmv.main()

    f44 = tmp_path / "verdicts_44.csv"
    parsed = parse_verdicts(f44.read_text(encoding="utf-8").splitlines())
    assert [v["verdict"] for v in parsed] == ["vrai", "faux"]  # 5 et 6 ignorés
    assert (tmp_path / "verdicts_2A.csv").exists()
    assert not list(tmp_path.glob("*sans*"))  # challenge non DECI ignoré
    assert len(list(tmp_path.glob("verdicts_*.csv"))) == 2
    out = capsys.readouterr().out
    assert "Test sans rapport" in out          # avertissement non-DECI
    assert "statuts: {1: 2, 2: 1, 5: 1, 6: 1}" in out
    assert "1 illisible(s)" in out


def test_main_network_failure_leaves_no_partial_csv(tmp_path, monkeypatch):
    challenges = [{"id": 1, "name": "DECI 44"}]
    monkeypatch.setattr(
        fmv, "http_get_json", _fake_get(challenges, {1: []}, fail_ids=(1,)))
    monkeypatch.setattr(sys, "argv", [
        "fetch_maproulette_verdicts.py", "--out", str(tmp_path)])
    with pytest.raises(RuntimeError):
        fmv.main()
    assert list(tmp_path.glob("verdicts_*.csv")) == []


def test_help_runs():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "fetch_maproulette_verdicts.py"),
         "--help"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "--project" in r.stdout
