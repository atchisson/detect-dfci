"""Test d'intégration offline de scripts/build_dataset.py::main().

Aucun appel réseau : fetch_features_geom est monkeypatché (une fausse
citerne, aucune piscine) et le cache de tuiles est pré-rempli avec des
tuiles unies avant l'exécution de main().
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import build_dataset  # noqa: E402  (chargé après l'ajout au sys.path)
from detection_ortho.dataset import fixed_box_geo, window_tiles  # noqa: E402

LON, LAT = 0.653, 47.33
ZOOM, WINDOW = 19, 640


def _fake_way(lon: float, lat: float, size_m: float) -> dict:
    w, s, e, n = fixed_box_geo(lon, lat, size_m)
    geometry = [
        {"lon": w, "lat": s},
        {"lon": e, "lat": s},
        {"lon": e, "lat": n},
        {"lon": w, "lat": n},
    ]
    return {"type": "way", "tags": {"emergency": "water_tank"}, "geometry": geometry}


def _fake_fetch_features_geom(selectors, west, south, east, north, session=None):
    """Remplace l'appel Overpass : 1 citerne au premier appel, rien ensuite."""
    if selectors == [("emergency", "water_tank")]:
        return [_fake_way(LON, LAT, 13.0)]
    return []


def _seed_tile_cache(cache_dir: Path) -> None:
    tiles, _, _ = window_tiles(LON, LAT, ZOOM, WINDOW)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for (x, y) in tiles:
        img = np.full((256, 256, 3), 120, np.uint8)
        cv2.imwrite(str(cache_dir / f"{ZOOM}_{x}_{y}.jpg"), img)


def test_build_dataset_main_end_to_end_offline(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _seed_tile_cache(out / "tiles_cache")

    monkeypatch.setattr(build_dataset, "fetch_features_geom", _fake_fetch_features_geom)
    monkeypatch.setattr(
        sys, "argv",
        [
            "build_dataset.py",
            "--bbox", "0.05", "46.72", "1.06", "47.72",
            "--negatives", "0",
            "--out", str(out),
        ],
    )

    build_dataset.main()

    data_yaml = out / "data.yaml"
    assert data_yaml.exists()

    # Au moins une image écrite.
    images = list(out.glob("images/**/*.jpg"))
    assert len(images) >= 1

    # La citerne doit avoir produit un label YOLO non vide (classe 0), quel
    # que soit le split (train/val/test) où elle est tombée.
    label_files = list(out.glob("labels/**/*.txt"))
    assert len(label_files) >= 1
    non_empty = [
        p for p in label_files
        if p.read_text(encoding="utf-8").strip().startswith("0 ")
    ]
    assert non_empty, "aucun label non vide commençant par '0 ' trouvé"
