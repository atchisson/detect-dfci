"""Appariement spatial entre détections et citernes OSM connues."""
from __future__ import annotations

from detection_ortho.geo import haversine_m


def match_detections(
    detections: list[dict], osm_points: list[dict], radius_m: float
) -> dict:
    """Classe détections et points OSM en trois catégories.

    - matched        : détections appariées à un point OSM (< radius_m).
    - detected_only  : détections sans OSM proche -> candidats MapRoulette.
    - osm_only        : points OSM non détectés -> faux négatifs / disparus.

    Chaque point OSM ne peut être apparié qu'une fois (au plus proche voisin).
    Les détections sont traitées par score décroissant pour la stabilité.
    """
    used_osm: set[int] = set()
    matched: list[dict] = []
    detected_only: list[dict] = []

    ordered = sorted(
        detections, key=lambda d: d.get("score", 0.0), reverse=True
    )
    for det in ordered:
        best_i, best_d = None, radius_m
        for i, osm in enumerate(osm_points):
            if i in used_osm:
                continue
            dist = haversine_m(det["lon"], det["lat"], osm["lon"], osm["lat"])
            if dist <= best_d:
                best_i, best_d = i, dist
        if best_i is None:
            detected_only.append(det)
        else:
            used_osm.add(best_i)
            matched.append({"detection": det, "osm": osm_points[best_i]})

    osm_only = [osm for i, osm in enumerate(osm_points) if i not in used_osm]
    return {"matched": matched, "detected_only": detected_only, "osm_only": osm_only}
