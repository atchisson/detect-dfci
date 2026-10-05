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


def test_multiple_verdict_files_with_dedup(tmp_path, monkeypatch, capsys):
    osm_lon, osm_lat = 0.65, 47.33

    def fake_fetch(selectors, *a, **k):
        if selectors[0][0] == "emergency":  # une citerne OSM (nœud)
            return [{"type": "node", "lon": osm_lon, "lat": osm_lat, "tags": {}}]
        return []  # pas de piscines
    monkeypatch.setattr(build_dataset, "fetch_features_geom", fake_fetch)

    out = tmp_path / "ds"
    cache = out / "tiles_cache"
    far_lon, far_lat = 0.66, 47.34
    faux_lon, faux_lat = 0.67, 47.35
    for lon, lat in ((osm_lon, osm_lat), (far_lon, far_lat), (faux_lon, faux_lat)):
        _seed_tiles(cache, lon, lat)

    a = tmp_path / "a.csv"  # vrai sur la citerne OSM (~3 m) -> doublon
    a.write_text("index,lat,lon,score,verdict\n"
                 f"1,{osm_lat},{osm_lon + 0.00004},0.9,vrai\n", encoding="utf-8")
    b = tmp_path / "b.csv"  # vrai lointain + faux
    b.write_text("index,lat,lon,score,verdict\n"
                 f"1,{far_lat},{far_lon},0.9,vrai\n"
                 f"2,{faux_lat},{faux_lon},0.5,faux\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "build_dataset.py", "--bbox", "0.6", "47.3", "0.7", "47.4",
        "--negatives", "0", "--max-pools", "0",
        "--verdicts", str(a), str(b), "--out", str(out)])
    build_dataset.main()

    names = [p.stem for p in (out / "labels").rglob("*.txt")]
    assert sum(n.startswith("citerne") for n in names) == 1
    assert sum(n.startswith("revpos") for n in names) == 1   # le doublon est écarté
    assert sum(n.startswith("hardneg") for n in names) == 1
    assert "1 doublon" in capsys.readouterr().out


def test_holdout_drops_records_near_held_out_points(tmp_path, monkeypatch, capsys):
    near_lon, near_lat = 0.65, 47.33   # nœud OSM proche d'un point mis de côté
    far_lon, far_lat = 0.68, 47.36     # nœud OSM lointain

    def fake_fetch(selectors, *a, **k):
        if selectors[0][0] == "emergency":
            return [{"type": "node", "lon": near_lon, "lat": near_lat, "tags": {}},
                    {"type": "node", "lon": far_lon, "lat": far_lat, "tags": {}}]
        return []
    monkeypatch.setattr(build_dataset, "fetch_features_geom", fake_fetch)

    out = tmp_path / "ds"
    for lon, lat in ((near_lon, near_lat), (far_lon, far_lat)):
        _seed_tiles(out / "tiles_cache", lon, lat)

    held = tmp_path / "held.csv"  # ~30 m de la citerne OSM proche
    held.write_text("index,lat,lon,score,verdict\n"
                    f"1,{near_lat},{near_lon + 0.0004},0.9,vrai\n",
                    encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "build_dataset.py", "--bbox", "0.6", "47.3", "0.7", "47.4",
        "--negatives", "0", "--max-pools", "0",
        "--holdout", str(held), "--out", str(out)])
    build_dataset.main()

    names = [p.stem for p in (out / "labels").rglob("*.txt")]
    assert sum(n.startswith("citerne") for n in names) == 1
    assert ("Holdout : 1 enregistrement(s) écarté(s) à moins de 100 m de "
            "1 point(s)") in capsys.readouterr().out
