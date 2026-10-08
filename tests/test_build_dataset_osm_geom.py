import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import build_dataset  # noqa: E402
from detection_ortho.dataset import window_tiles  # noqa: E402

M = 111320.0
A = (0.65, 47.33)      # a un polygone OSM de 24 m x 8 m
B = (0.66, 47.34)      # aucun polygone OSM


def _seed(cache, lon, lat):
    cache.mkdir(parents=True, exist_ok=True)
    tiles, _, _ = window_tiles(lon, lat, 19, 640)
    for x, y in tiles:
        cv2.imwrite(str(cache / f"19_{x}_{y}.jpg"), np.full((256, 256, 3), 128, np.uint8))


def _way(lon, lat, w_m, h_m):
    dlat = h_m / 2 / M
    dlon = w_m / 2 / (M * math.cos(math.radians(lat)))
    pts = [(lon - dlon, lat - dlat), (lon + dlon, lat - dlat), (lon + dlon, lat + dlat),
           (lon - dlon, lat + dlat), (lon - dlon, lat - dlat)]
    return {"type": "way", "tags": {}, "geometry": [{"lon": x, "lat": y} for x, y in pts]}


def _label(out, stem):
    p = next((out / "labels").rglob(f"{stem}.txt"))
    _cls, _cx, _cy, w, h = p.read_text().split()
    return float(w), float(h)


def _run(tmp_path, monkeypatch, osm_geom):
    tmp_path.mkdir(parents=True, exist_ok=True)
    calls = []

    def fake_fetch(selectors, w, s, e, n):
        calls.append((selectors, w, s, e, n))
        if selectors[0][0] == "leisure":
            return []
        if (w, s, e, n) == (0.6, 47.3, 0.7, 47.4):     # l'emprise --bbox : aucun positif OSM
            return []
        return [_way(A[0], A[1], 24, 8)]                 # requête --osm-geom du fichier de verdicts

    monkeypatch.setattr(build_dataset, "fetch_features_geom", fake_fetch)
    out = tmp_path / "ds"
    for pt in (A, B, (B[0], B[1] + 0.01)):       # les trois points créent une imagette : tuiles préremplies
        _seed(out / "tiles_cache", *pt)
    verdicts = tmp_path / "v.csv"
    verdicts.write_text("index,lat,lon,score,verdict\n"
                        f"1,{A[1]},{A[0]},0.9,vrai\n"
                        f"2,{B[1]},{B[0]},0.9,vrai\n"
                        f"3,{B[1] + 0.01},{B[0]},0.4,faux\n", encoding="utf-8")
    argv = ["build_dataset.py", "--bbox", "0.6", "47.3", "0.7", "47.4", "--negatives", "0",
            "--max-pools", "0", "--verdicts", str(verdicts), "--out", str(out)]
    if osm_geom:
        argv.append("--osm-geom")
    monkeypatch.setattr(sys, "argv", argv)
    build_dataset.main()
    return out, calls


def test_osm_geom_uses_polygon_box_and_falls_back(tmp_path, monkeypatch, capsys):
    out, calls = _run(tmp_path, monkeypatch, osm_geom=True)
    wa, ha = _label(out, "revpos_0000")     # polygone 24 m x 8 m (~0,2026 m/px, fenêtre 640 px)
    assert abs(wa - 24 / 0.2026 / 640) < 0.02 and abs(ha - 8 / 0.2026 / 640) < 0.02
    wb, hb = _label(out, "revpos_0001")     # repli : carré de 13 m
    assert abs(wb - 13 / 0.2026 / 640) < 0.02 and abs(hb - 13 / 0.2026 / 640) < 0.02
    summary = capsys.readouterr().out
    assert "1 polygone(s)" in summary and "1 repli(s)" in summary
    # une seule requête OSM supplémentaire (un fichier de verdicts) hors --bbox et piscines
    extra = [c for c in calls if c[0][0][0] == "emergency" and (c[1], c[2], c[3], c[4]) != (0.6, 47.3, 0.7, 47.4)]
    assert len(extra) == 1


def test_qa_polygones_mosaic_only_with_osm_geom(tmp_path, monkeypatch):
    out_on, _ = _run(tmp_path / "on", monkeypatch, osm_geom=True)
    assert (out_on / "qa_polygones.png").exists()
    out_off, _ = _run(tmp_path / "off", monkeypatch, osm_geom=False)
    assert not (out_off / "qa_polygones.png").exists()


def _run_radius(tmp_path, monkeypatch, extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    def fake_fetch(selectors, w, s, e, n):
        if selectors[0][0] == "leisure" or (w, s, e, n) == (0.6, 47.3, 0.7, 47.4):
            return []
        # polygone 24 m x 8 m centré à 7 m à l'est du point (le point est dedans)
        return [_way(A[0] + 7 / (M * math.cos(math.radians(A[1]))), A[1], 24, 8)]

    monkeypatch.setattr(build_dataset, "fetch_features_geom", fake_fetch)
    out = tmp_path / "ds"
    _seed(out / "tiles_cache", *A)
    verdicts = tmp_path / "v.csv"
    verdicts.write_text("index,lat,lon,score,verdict\n"
                        f"1,{A[1]},{A[0]},0.9,vrai\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "build_dataset.py", "--bbox", "0.6", "47.3", "0.7", "47.4", "--negatives", "0",
        "--max-pools", "0", "--verdicts", str(verdicts), "--out", str(out),
        "--osm-geom", *extra])
    build_dataset.main()
    return out


def test_osm_geom_m_is_passed_through(tmp_path, monkeypatch):
    out = _run_radius(tmp_path / "a", monkeypatch, [])
    w, h = _label(out, "revpos_0000")
    assert abs(w - 24 / 0.2026 / 640) < 0.02 and abs(h - 8 / 0.2026 / 640) < 0.02
    out = _run_radius(tmp_path / "b", monkeypatch, ["--osm-geom-m", "5"])
    w, h = _label(out, "revpos_0000")
    assert abs(w - 13 / 0.2026 / 640) < 0.02 and abs(h - 13 / 0.2026 / 640) < 0.02


def test_default_is_unchanged_without_osm_geom(tmp_path, monkeypatch):
    out, calls = _run(tmp_path, monkeypatch, osm_geom=False)
    for stem in ("revpos_0000", "revpos_0001"):
        w, h = _label(out, stem)
        assert abs(w - 13 / 0.2026 / 640) < 0.02 and abs(h - 13 / 0.2026 / 640) < 0.02
    assert len(calls) == 2           # positifs de --bbox + piscines, rien d'autre


def test_no_overpass_call_for_a_file_without_vrai(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(selectors, w, s, e, n):
        calls.append((w, s, e, n))
        return []

    monkeypatch.setattr(build_dataset, "fetch_features_geom", fake_fetch)
    n_poly, n_fb = build_dataset.attach_osm_boxes(
        [{"lon": 0.65, "lat": 47.33, "verdict": "faux"}], 15.0)
    assert (n_poly, n_fb) == (0, 0) and calls == []
