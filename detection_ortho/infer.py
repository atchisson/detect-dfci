"""Inférence YOLO à l'échelle d'une emprise : fenêtrage glissant sur polygone,
conversions pixel<->géo. Logique pure séparée de l'I/O modèle/réseau.
"""
from __future__ import annotations

from shapely.geometry import LineString, Point
from shapely.ops import linemerge, polygonize, unary_union
from shapely.prepared import prep

from detection_ortho.dataset import lonlat_to_global_px, global_px_to_lonlat
from detection_ortho.geo import haversine_m


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
    prepared = prep(polygon)
    gy = gy0
    while gy <= gy1:
        gx = gx0
        while gx <= gx1:
            lon, lat = global_px_to_lonlat(gx, gy, zoom, tile_size)
            if prepared.contains(Point(lon, lat)):
                centers.append((lon, lat))
            gx += step
        gy += step
    return centers


def result_to_boxes(boxes) -> list[tuple[float, float, float]]:
    """Extrait (cx, cy, score) de chaque boîte d'un résultat Ultralytics.

    `boxes` = itérable d'objets exposant .xywh (Nx4, [cx,cy,w,h]) et .conf.
    """
    out = []
    for b in boxes:
        out.append((float(b.xywh[0][0]), float(b.xywh[0][1]), float(b.conf[0])))
    return out


def boxes_to_points(
    boxes: list[tuple[float, float, float]],
    origin_gx: float, origin_gy: float, zoom: int,
) -> list[dict]:
    """Convertit des centres de boîtes (px dans la fenêtre) + score en points géo.

    origin_gx/origin_gy : pixel absolu du coin haut-gauche de la fenêtre.
    """
    points: list[dict] = []
    for cx, cy, score in boxes:
        lon, lat = global_px_to_lonlat(origin_gx + cx, origin_gy + cy, zoom)
        points.append({"lon": lon, "lat": lat, "score": float(score)})
    return points


def is_detected_near(det_points, lon: float, lat: float, radius_m: float) -> bool:
    """Vrai si une détection tombe à <= radius_m du centre (lon, lat)."""
    return any(
        haversine_m(lon, lat, p["lon"], p["lat"]) <= radius_m
        for p in det_points
    )
