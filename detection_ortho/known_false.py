"""Faux positifs déjà rejetés : ne pas les remontrer lors d'une nouvelle passe."""
from __future__ import annotations

from pathlib import Path

from detection_ortho.dataset import near_any, parse_verdicts


def load_known_false(paths) -> list[tuple[float, float]]:
    """Points (lon, lat) des verdicts « faux » des CSV de revue.

    Lève FileNotFoundError si un fichier manque : à appeler au démarrage d'un
    long run, pour échouer tout de suite plutôt qu'après des heures de calcul.
    """
    points: list[tuple[float, float]] = []
    for path in paths:
        for v in parse_verdicts(Path(path).read_text(encoding="utf-8").splitlines()):
            if v["verdict"] == "faux":
                points.append((v["lon"], v["lat"]))
    return points


def suppress_known_false(points, false_points, radius_m: float = 25.0):
    """Sépare les candidats (dicts avec `lon`, `lat`) proches d'un faux connu.

    Retourne (conservés, écartés). Un candidat est écarté s'il est à `radius_m`
    ou moins d'un point de `false_points`.
    """
    points = list(points)
    if not points or not false_points:
        return points, []
    flags = near_any([(p["lon"], p["lat"]) for p in points], false_points, radius_m)
    kept = [p for p, f in zip(points, flags) if not f]
    dropped = [p for p, f in zip(points, flags) if f]
    return kept, dropped
