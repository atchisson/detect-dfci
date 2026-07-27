"""Génération d'un dataset YOLO de citernes depuis OSM + ortho IGN.

Maths pures de boîtes/fenêtres (testables sans I/O) puis assemblage des
imagettes depuis les tuiles et écriture au format YOLO.
"""
from __future__ import annotations

import math

from detection_ortho.tiles import lonlat_to_pixel

_M_PER_DEG_LAT = 111320.0


def lonlat_to_global_px(
    lon: float, lat: float, zoom: int, tile_size: int = 256
) -> tuple[float, float]:
    """Pixel absolu (global) dans la grille slippy au zoom donné."""
    x, y, px, py = lonlat_to_pixel(lon, lat, zoom, tile_size)
    return x * tile_size + px, y * tile_size + py


def polygon_bounds(geometry: list[dict]) -> tuple[float, float, float, float]:
    """BBox géo (west, south, east, north) d'une liste de sommets {lon,lat}."""
    lons = [p["lon"] for p in geometry]
    lats = [p["lat"] for p in geometry]
    return min(lons), min(lats), max(lons), max(lats)


def fixed_box_geo(
    lon: float, lat: float, size_m: float
) -> tuple[float, float, float, float]:
    """BBox géo carrée de côté size_m centrée sur (lon, lat)."""
    half = size_m / 2.0
    dlat = half / _M_PER_DEG_LAT
    dlon = half / (_M_PER_DEG_LAT * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def geo_bbox_to_pixel_bbox(
    bbox_geo: tuple[float, float, float, float],
    origin_gx: float, origin_gy: float, zoom: int, window_px: int,
    tile_size: int = 256,
) -> tuple[float, float, float, float]:
    """Convertit une bbox géo en bbox pixel (x0,y0,x1,y1) dans la fenêtre.

    origin_gx/origin_gy : pixel absolu du coin haut-gauche de la fenêtre.
    Le nord (lat max) correspond au haut de l'image (y plus petit).
    """
    w, s, e, n = bbox_geo
    gx0, gy0 = lonlat_to_global_px(w, n, zoom, tile_size)  # haut-gauche
    gx1, gy1 = lonlat_to_global_px(e, s, zoom, tile_size)  # bas-droite
    x0 = max(0.0, min(window_px, gx0 - origin_gx))
    y0 = max(0.0, min(window_px, gy0 - origin_gy))
    x1 = max(0.0, min(window_px, gx1 - origin_gx))
    y1 = max(0.0, min(window_px, gy1 - origin_gy))
    return x0, y0, x1, y1


def to_yolo_label(
    px_bbox: tuple[float, float, float, float], window_px: int, cls: int = 0
) -> str | None:
    """Ligne YOLO `cls cx cy w h` normalisée, ou None si la boîte est vide."""
    x0, y0, x1, y1 = px_bbox
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return None
    cx = (x0 + x1) / 2 / window_px
    cy = (y0 + y1) / 2 / window_px
    return f"{cls} {cx:.6f} {cy:.6f} {w / window_px:.6f} {h / window_px:.6f}"
