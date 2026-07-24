"""Distances géographiques et dédoublonnage de points par proximité."""
from __future__ import annotations

import math

_EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Distance en mètres entre deux points (lon, lat) en degrés."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def dedup_points(points: list[dict], radius_m: float) -> list[dict]:
    """Fusionne les points à moins de radius_m, gardant le meilleur score.

    Approche gloutonne : on traite les points par score décroissant ; chaque
    point non encore absorbé devient un représentant qui absorbe ses voisins.
    Chaque point doit avoir les clés lon, lat, score.
    """
    remaining = sorted(points, key=lambda p: p["score"], reverse=True)
    kept: list[dict] = []
    for p in remaining:
        if any(
            haversine_m(p["lon"], p["lat"], k["lon"], k["lat"]) <= radius_m
            for k in kept
        ):
            continue
        kept.append(p)
    return kept
