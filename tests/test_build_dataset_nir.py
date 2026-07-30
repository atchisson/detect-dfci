import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import build_dataset  # noqa: E402
from detection_ortho.dataset import window_tiles  # noqa: E402
from detection_ortho.tiles import LAYER_IRC  # noqa: E402


def _seed(cache, lon, lat, layer_tag, value):
    cache.mkdir(parents=True, exist_ok=True)
    tiles, _, _ = window_tiles(lon, lat, 19, 640)
    suffix = f"_{layer_tag}" if layer_tag else ""
    for x, y in tiles:
        cv2.imwrite(str(cache / f"19_{x}_{y}{suffix}.jpg"),
                    np.full((256, 256, 3), value, np.uint8))


def test_nir_chip_blue_from_irc(tmp_path, monkeypatch):
    monkeypatch.setattr(build_dataset, "fetch_features_geom", lambda *a, **k: [])
    out = tmp_path / "ds"
    cache = out / "tiles_cache"
    lon, lat = 0.65, 47.33
    # tuiles RVB (gris 100) et IRC (rouge=200) pré-semées
    _seed(cache, lon, lat, "", 100)
    irc_tag = LAYER_IRC.rsplit(".", 1)[-1].lower()   # "irc"
    _seed(cache, lon, lat, irc_tag, 200)
    verdicts = tmp_path / "v.csv"
    verdicts.write_text(f"index,lat,lon,score,verdict\n1,{lat},{lon},0.9,vrai\n",
                        encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "build_dataset.py", "--bbox", "0.6", "47.3", "0.7", "47.4",
        "--negatives", "0", "--max-pools", "0", "--nir",
        "--verdicts", str(verdicts), "--out", str(out)])
    build_dataset.main()

    imgs = list((out / "images").rglob("revpos_*.jpg"))
    assert len(imgs) == 1
    chip = cv2.imread(str(imgs[0]))
    # canal bleu du chip = NIR (tuile IRC=200), pas la valeur RVB (100)
    assert int(chip[:, :, 0].mean()) > 150
