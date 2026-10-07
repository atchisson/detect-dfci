import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import build_dataset  # noqa: E402
from detection_ortho.dataset import window_tiles  # noqa: E402
from detection_ortho.tiles import LAYER, layer_tag  # noqa: E402

L26 = "ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026"


def _seed(cache, lon, lat, layer, value):
    cache.mkdir(parents=True, exist_ok=True)
    tiles, _, _ = window_tiles(lon, lat, 19, 640)
    for x, y in tiles:
        cv2.imwrite(str(cache / f"19_{x}_{y}{layer_tag(layer)}.jpg"),
                    np.full((256, 256, 3), value, np.uint8))


def test_layers_one_chip_per_available_layer_same_partition(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(build_dataset, "fetch_features_geom", lambda *a, **k: [])
    out = tmp_path / "ds"
    cache = out / "tiles_cache"
    a, b = (0.65, 47.33), (0.66, 47.34)
    _seed(cache, *a, LAYER, 128)
    _seed(cache, *a, L26, 120)
    _seed(cache, *b, LAYER, 128)
    _seed(cache, *b, L26, 255)           # fenêtre blanche en 2026 pour B
    verdicts = tmp_path / "v.csv"
    verdicts.write_text(
        "index,lat,lon,score,verdict\n"
        f"1,{a[1]},{a[0]},0.9,vrai\n"
        f"2,{b[1]},{b[0]},0.9,vrai\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "build_dataset.py", "--bbox", "0.6", "47.3", "0.7", "47.4",
        "--negatives", "0", "--max-pools", "0", "--layers", LAYER, L26,
        "--verdicts", str(verdicts), "--out", str(out)])
    build_dataset.main()

    images = {p.stem: p.parent.name for p in (out / "images").rglob("*.jpg")}
    assert set(images) == {"revpos_0000", "revpos_0001", "revpos_0000__rvb-express-2026"}
    assert images["revpos_0000"] == images["revpos_0000__rvb-express-2026"]
    labels = {p.stem for p in (out / "labels").rglob("*.txt")}
    assert labels == set(images)
    summary = capsys.readouterr().out
    assert "1 fenêtre(s) vide(s)" in summary


def test_layers_incompatible_with_nir(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_dataset.py", "--bbox", "0", "0", "1", "1",
                                      "--nir", "--layers", LAYER, L26])
    with pytest.raises(SystemExit) as exc:
        build_dataset.main()
    assert exc.value.code == 2
