# tests/test_sweep_threshold.py
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import sweep_threshold  # noqa: E402
from detection_ortho.dataset import window_tiles  # noqa: E402


def _seed(cache, lon, lat, value):
    cache.mkdir(parents=True, exist_ok=True)
    tiles, _, _ = window_tiles(lon, lat, 19, 640)
    for x, y in tiles:
        cv2.imwrite(str(cache / f"19_{x}_{y}.jpg"),
                    np.full((256, 256, 3), value, np.uint8))


class _Box:
    def __init__(self, cx, cy, score):
        self.xywh = [[cx, cy, 20.0, 20.0]]
        self.conf = [score]


class _Res:
    def __init__(self, boxes):
        self.boxes = boxes


def test_sweep_prints_table(tmp_path, monkeypatch, capsys):
    # un vrai (score 0.9) et un faux (score 0.4)
    vrai = (0.65, 47.33)
    faux = (0.70, 47.40)
    cache = tmp_path / "cache"
    _seed(cache, *vrai, 100)
    _seed(cache, *faux, 120)

    verdicts = tmp_path / "v.csv"
    verdicts.write_text(
        "index,lat,lon,score,verdict\n"
        f"1,{vrai[1]},{vrai[0]},0.9,vrai\n"
        f"2,{faux[1]},{faux[0]},0.4,faux\n",
        encoding="utf-8")

    score_by_lon = {round(vrai[0], 4): 0.9, round(faux[0], 4): 0.4}

    class FakeYOLO:
        def __init__(self, weights):
            pass

        def predict(self, img, conf=0.05, device="cpu", verbose=False):
            # score choisi selon la moyenne des pixels (100=vrai, 120=faux)
            s = 0.9 if int(img.mean()) < 110 else 0.4
            return [_Res([_Box(320.0, 320.0, s)])]

    monkeypatch.setattr(sweep_threshold, "YOLO", FakeYOLO)
    monkeypatch.setattr(sys, "argv", [
        "sweep_threshold.py", "--weights", "x.pt", "--verdicts", str(verdicts),
        "--conf-min", "0.1", "--conf-max", "0.9", "--step", "0.4",
        "--cache", str(cache)])
    sweep_threshold.main()

    out = capsys.readouterr().out
    # à seuil 0.5 : le vrai (0.9) tire, le faux (0.4) non -> précision 1.0, rappel 1.0
    assert "0.50" in out or "0.5" in out
    assert "précision" in out.lower() or "precision" in out.lower()
