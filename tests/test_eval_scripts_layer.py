import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import eval_points  # noqa: E402
import sweep_threshold  # noqa: E402

L26 = "ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026"


def _csv(tmp_path):
    p = tmp_path / "v.csv"
    p.write_text("index,lat,lon,score,verdict\n1,47.33,0.65,0.9,vrai\n2,47.34,0.66,0.5,faux\n",
                 encoding="utf-8")
    return p


class _FakeModel:
    def __init__(self, *a, **k):
        pass

    def predict(self, img, **k):
        return [types.SimpleNamespace(boxes=None)]


def _patch(monkeypatch, module, seen, read):
    def fake_prefetch(points, cache, layers, workers=12, zoom=19, window_px=640):
        seen["layers"] = list(layers)
        seen["n"] = len(points)
        seen["workers"] = workers
        return []

    def fake_assemble(lon, lat, zoom, window, cache, layer=None, **k):
        read.append(layer)
        return np.zeros((640, 640, 3), np.uint8), 0.0, 0.0

    monkeypatch.setattr(module, "YOLO", _FakeModel)
    monkeypatch.setattr(module, "prefetch_points", fake_prefetch)
    monkeypatch.setattr(module, "assemble_window", fake_assemble)
    monkeypatch.setattr(module, "result_to_boxes", lambda b: [])
    monkeypatch.setattr(module, "boxes_to_points", lambda *a, **k: [])


@pytest.mark.parametrize("module,name", [(sweep_threshold, "sweep_threshold.py"),
                                         (eval_points, "eval_points.py")])
def test_scripts_prefetch_then_read_the_requested_layer(module, name, tmp_path, monkeypatch):
    seen, read = {}, []
    _patch(monkeypatch, module, seen, read)
    monkeypatch.setattr(sys, "argv", [name, "--weights", "w.pt", "--verdicts",
                                      str(_csv(tmp_path)), "--layer", L26,
                                      "--workers", "5", "--cache", str(tmp_path / "c")])
    module.main()
    assert seen == {"layers": [L26], "n": 2, "workers": 5}
    assert read == [L26, L26]


@pytest.mark.parametrize("module,name", [(sweep_threshold, "sweep_threshold.py"),
                                         (eval_points, "eval_points.py")])
def test_scripts_default_layer_is_the_standard_one(module, name, tmp_path, monkeypatch):
    seen, read = {}, []
    _patch(monkeypatch, module, seen, read)
    monkeypatch.setattr(sys, "argv", [name, "--weights", "w.pt", "--verdicts",
                                      str(_csv(tmp_path)), "--cache", str(tmp_path / "c")])
    module.main()
    assert seen["layers"] == ["ORTHOIMAGERY.ORTHOPHOTOS"]
    assert read == ["ORTHOIMAGERY.ORTHOPHOTOS"] * 2


@pytest.mark.parametrize("name", ["sweep_threshold.py", "eval_points.py"])
def test_scripts_reject_nir_with_another_layer(name):
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / name), "--weights", "w.pt",
         "--verdicts", "v.csv", "--nir", "--layer", L26],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 2
    assert "incompatible" in r.stderr


@pytest.mark.parametrize("name", ["sweep_threshold.py", "eval_points.py"])
def test_scripts_help_mentions_layer(name):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / name), "--help"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "--layer" in r.stdout and "--workers" in r.stdout
