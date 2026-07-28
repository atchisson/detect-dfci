"""Inférence YOLO à l'échelle d'une emprise : fenêtrage glissant sur polygone,
conversions pixel<->géo. Logique pure séparée de l'I/O modèle/réseau.
"""
from __future__ import annotations

from shapely.geometry import LineString, Point
from shapely.ops import linemerge, polygonize, unary_union

from detection_ortho.dataset import lonlat_to_global_px, global_px_to_lonlat


def ways_to_polygon(ways: list[list[dict]]):
    """Assemble les ways (contours) en un polygone shapely (le plus grand)."""
    lines = [
        LineString([(p["lon"], p["lat"]) for p in way])
        for way in ways if len(way) >= 2
    ]
    merged = linemerge(unary_union(lines))
    polys = list(polygonize(merged))
    if not polys:
        raise ValueError("Aucun polygone assemblable depuis les ways fournis.")
    return max(polys, key=lambda p: p.area)


def windows_over_polygon(
    polygon, zoom: int, window_px: int, overlap: float,
    tile_size: int = 256,
) -> list[tuple[float, float]]:
    """Centres (lon, lat) des fenêtres couvrant le polygone (centre à l'intérieur).

    Grille de pas window_px*(1-overlap) en pixels sur la bbox du polygone.
    """
    west, south, east, north = polygon.bounds  # (minx, miny, maxx, maxy)
    gx0, gy0 = lonlat_to_global_px(west, north, zoom, tile_size)  # coin NO
    gx1, gy1 = lonlat_to_global_px(east, south, zoom, tile_size)  # coin SE
    step = max(1, int(window_px * (1.0 - overlap)))
    centers: list[tuple[float, float]] = []
    gy = gy0
    while gy <= gy1:
        gx = gx0
        while gx <= gx1:
            lon, lat = global_px_to_lonlat(gx, gy, zoom, tile_size)
            if polygon.contains(Point(lon, lat)):
                centers.append((lon, lat))
            gx += step
        gy += step
    return centers
