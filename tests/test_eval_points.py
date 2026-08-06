# tests/test_eval_points.py
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import eval_points  # noqa: E402
from detection_ortho.dataset import window_tiles  # noqa: E402
from detection_ortho.tiles import LAYER_IRC  # noqa: E402


def _seed(cache, lon, lat, tag, value):
    cache.mkdir(parents=True, exist_ok=True)
    tiles, _, _ = window_tiles(lon, lat, 19, 640)
    suffix = f"_{tag}" if tag else ""
    for x, y in tiles:
        cv2.imwrite(str(cache / f"19_{x}_{y}{suffix}.jpg"),
                    np.full((256, 256, 3), value, np.uint8))


class _Box:
    def __init__(self, cx, cy, score):
        self.xywh = [[cx, cy, 20.0, 20.0]]
        self.conf = [score]


class _Res:
    def __init__(self, boxes):
        self.boxes = boxes


def test_nir_eval_composes_and_tallies(tmp_path, monkeypatch, capsys):
    lon, lat = 0.65, 47.33
    cache = tmp_path / "cache"
    _seed(cache, lon, lat, "", 100)                                  # RVB
    _seed(cache, lon, lat, LAYER_IRC.rsplit(".", 1)[-1].lower(), 200)  # IRC

    verdicts = tmp_path / "v.csv"
    verdicts.write_text(f"index,lat,lon,score,verdict\n1,{lat},{lon},0.9,faux\n",
                        encoding="utf-8")

    seen = []

    class FakeYOLO:
        def __init__(self, weights):
            pass

        def predict(self, img, conf=0.25, device="cpu", verbose=False):
            seen.append(img)
            return [_Res([_Box(320.0, 320.0, 0.9)])]  # tire au centre de la fenêtre

    monkeypatch.setattr(eval_points, "YOLO", FakeYOLO)
    monkeypatch.setattr(sys, "argv", [
        "eval_points.py", "--weights", "x.pt", "--verdicts", str(verdicts),
        "--conf", "0.55", "--nir", "--cache", str(cache)])
    eval_points.main()

    # --nir a composé : le bleu de l'image vue par le modèle vient de l'IRC (200)
    assert seen and int(seen[0][:, :, 0].mean()) > 150
    # le point faux est encore détecté (le stub tire) -> 0 supprimé sur 1
    out = capsys.readouterr().out
    assert "Faux positifs supprimés        : 0/1" in out


def test_rgb_path_no_nir_composition(tmp_path, monkeypatch, capsys):
    lon, lat = 0.65, 47.33
    cache = tmp_path / "cache"
    _seed(cache, lon, lat, "", 100)  # RVB seulement, pas d'IRC

    verdicts = tmp_path / "v.csv"
    verdicts.write_text(f"index,lat,lon,score,verdict\n1,{lat},{lon},0.9,faux\n",
                        encoding="utf-8")

    seen = []

    class FakeYOLO:
        def __init__(self, weights):
            pass

        def predict(self, img, conf=0.25, device="cpu", verbose=False):
            seen.append(img)
            return [_Res([_Box(320.0, 320.0, 0.9)])]  # tire au centre de la fenêtre

    monkeypatch.setattr(eval_points, "YOLO", FakeYOLO)
    monkeypatch.setattr(sys, "argv", [
        "eval_points.py", "--weights", "x.pt", "--verdicts", str(verdicts),
        "--conf", "0.55", "--cache", str(cache)])
    eval_points.main()

    # sans --nir : pas de composition, le bleu reste celui de la tuile RVB (100)
    assert seen and int(seen[0][:, :, 0].mean()) < 150
    out = capsys.readouterr().out
    assert "Faux positifs supprimés        : 0/1" in out


def test_silent_model_suppresses(tmp_path, monkeypatch, capsys):
    lon, lat = 0.65, 47.33
    cache = tmp_path / "cache"
    _seed(cache, lon, lat, "", 100)  # RVB seulement

    verdicts = tmp_path / "v.csv"
    verdicts.write_text(f"index,lat,lon,score,verdict\n1,{lat},{lon},0.9,faux\n",
                        encoding="utf-8")

    class FakeYOLO:
        def __init__(self, weights):
            pass

        def predict(self, img, conf=0.25, device="cpu", verbose=False):
            return [_Res([])]  # le modèle ne tire pas

    monkeypatch.setattr(eval_points, "YOLO", FakeYOLO)
    monkeypatch.setattr(sys, "argv", [
        "eval_points.py", "--weights", "x.pt", "--verdicts", str(verdicts),
        "--conf", "0.55", "--cache", str(cache)])
    eval_points.main()

    # le faux n'est plus détecté -> supprimé
    out = capsys.readouterr().out
    assert "Faux positifs supprimés        : 1/1" in out
