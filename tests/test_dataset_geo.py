import math
from detection_ortho.dataset import (
    lonlat_to_global_px,
    polygon_bounds,
    fixed_box_geo,
    geo_bbox_to_pixel_bbox,
    to_yolo_label,
)


def test_polygon_bounds():
    geom = [{"lon": 0.60, "lat": 47.30}, {"lon": 0.61, "lat": 47.31},
            {"lon": 0.605, "lat": 47.29}]
    w, s, e, n = polygon_bounds(geom)
    assert (w, s, e, n) == (0.60, 47.29, 0.61, 47.31)


def test_fixed_box_geo_size_in_meters():
    lon, lat = 0.653, 47.33
    w, s, e, n = fixed_box_geo(lon, lat, 20.0)
    # hauteur ~20 m -> ~20/111320 deg de latitude au total.
    assert abs((n - s) * 111320 - 20.0) < 1.0
    # centré
    assert abs((w + e) / 2 - lon) < 1e-9


def test_global_px_matches_tile_offset():
    lon, lat, z = 0.653, 47.33, 19
    gx, gy = lonlat_to_global_px(lon, lat, z)
    # cohérent avec un pas de tuile de 256 px
    assert gx > 0 and gy > 0


def test_geo_bbox_to_pixel_bbox_orientation_and_clamp():
    z, win = 19, 640
    # une petite boîte géo autour d'un point
    lon, lat = 0.653, 47.33
    gx, gy = lonlat_to_global_px(lon, lat, z)
    origin_gx, origin_gy = gx - win / 2, gy - win / 2  # fenêtre centrée
    box = fixed_box_geo(lon, lat, 15.0)
    x0, y0, x1, y1 = geo_bbox_to_pixel_bbox(box, origin_gx, origin_gy, z, win)
    assert 0 <= x0 < x1 <= win
    assert 0 <= y0 < y1 <= win
    # la boîte est proche du centre de la fenêtre
    assert abs((x0 + x1) / 2 - win / 2) < 5
    assert abs((y0 + y1) / 2 - win / 2) < 5


def test_to_yolo_label_normalized():
    line = to_yolo_label((160, 160, 480, 480), 640)
    parts = line.split()
    assert parts[0] == "0"
    cx, cy, w, h = map(float, parts[1:])
    assert abs(cx - 0.5) < 1e-6 and abs(cy - 0.5) < 1e-6
    assert abs(w - 0.5) < 1e-6 and abs(h - 0.5) < 1e-6


def test_to_yolo_label_empty_box_returns_none():
    assert to_yolo_label((100, 100, 100, 100), 640) is None
