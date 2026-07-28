import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import build_dataset  # noqa: E402
from detection_ortho.dataset import window_tiles  # noqa: E402


def _seed_tiles(cache, lon, lat):
    cache.mkdir(parents=True, exist_ok=True)
    tiles, _, _ = window_tiles(lon, lat, 19, 640)
    for x, y in tiles:
        cv2.imwrite(str(cache / f"19_{x}_{y}.jpg"),
                    np.full((256, 256, 3), 128, np.uint8))


def test_verdicts_add_hardneg_and_revpos(tmp_path, monkeypatch):
    # Pas de citernes/piscines OSM : on isole l'effet des verdicts.
    monkeypatch.setattr(build_dataset, "fetch_features_geom", lambda *a, **k: [])
    out = tmp_path / "ds"
    cache = out / "tiles_cache"
    faux_lon, faux_lat = 0.65, 47.33
    vrai_lon, vrai_lat = 0.66, 47.34
    _seed_tiles(cache, faux_lon, faux_lat)
    _seed_tiles(cache, vrai_lon, vrai_lat)
    verdicts = tmp_path / "v.csv"
    verdicts.write_text(
        "index,lat,lon,score,verdict\n"
        f"1,{faux_lat},{faux_lon},0.5,faux\n"
        f"2,{vrai_lat},{vrai_lon},0.9,vrai\n",
        encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "build_dataset.py", "--bbox", "0.6", "47.3", "0.7", "47.4",
        "--negatives", "0", "--max-pools", "0",
        "--verdicts", str(verdicts), "--out", str(out)])
    build_dataset.main()

    # un chip hardneg_* (label vide) et un chip revpos_* (label non vide)
    labels = list((out / "labels").rglob("*.txt"))
    hard = [p for p in labels if p.stem.startswith("hardneg")]
    rev = [p for p in labels if p.stem.startswith("revpos")]
    assert len(hard) == 1 and hard[0].read_text().strip() == ""
    assert len(rev) == 1 and rev[0].read_text().strip().startswith("0 ")
