import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import export_maproulette  # noqa: E402


def _pts(path, scores):
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"score": s},
         "geometry": {"type": "Point", "coordinates": [0.6 + i * 0.01, 47.3]}}
        for i, s in enumerate(scores)]}
    path.write_text(json.dumps(fc), encoding="utf-8")


def _count(out):
    return len(json.loads(out.read_text(encoding="utf-8")).get("features", []))


def test_min_score_filters(tmp_path, monkeypatch):
    inp = tmp_path / "in.geojson"
    _pts(inp, [0.9, 0.75, 0.6, 0.4, None])
    out = tmp_path / "chal.geojson"
    monkeypatch.setattr(sys, "argv", [
        "export_maproulette.py", "--input", str(inp), "--out", str(out),
        "--min-score", "0.7"])
    export_maproulette.main()
    assert _count(out) == 2  # seuls 0.9 et 0.75 passent


def test_no_filter_keeps_all(tmp_path, monkeypatch):
    inp = tmp_path / "in.geojson"
    _pts(inp, [0.9, 0.4, None])
    out = tmp_path / "chal.geojson"
    monkeypatch.setattr(sys, "argv", [
        "export_maproulette.py", "--input", str(inp), "--out", str(out)])
    export_maproulette.main()
    assert _count(out) == 3  # défaut min-score=0 -> tout gardé
