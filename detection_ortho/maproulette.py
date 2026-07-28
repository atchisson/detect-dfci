"""Génération d'un fichier de tâches MapRoulette (GeoJSON).

IMPORTANT : ce module GÉNÈRE UN FICHIER uniquement. Aucun appel à l'API
MapRoulette, aucune publication, aucun envoi réseau. L'utilisateur importe
lui-même le fichier dans l'interface MapRoulette.
"""
from __future__ import annotations


def to_maproulette_tasks(points: list[dict], instruction: str) -> dict:
    """FeatureCollection GeoJSON : une tâche (Point) par candidat."""
    features = []
    for p in points:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
            "properties": {
                "instruction": instruction,
                "score": p.get("score"),
            },
        })
    return {"type": "FeatureCollection", "features": features}
