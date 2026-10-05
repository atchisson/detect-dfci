"""Tâches MapRoulette : génération d'un fichier d'import et lecture de verdicts.

IMPORTANT : ce module est PUR (aucun accès réseau). `to_maproulette_tasks`
GÉNÈRE UN FICHIER que l'utilisateur importe lui-même dans MapRoulette ; les
fonctions de verdict transforment des tâches DÉJÀ téléchargées (la récupération,
en lecture seule, vit dans scripts/fetch_maproulette_verdicts.py). Aucun envoi
vers MapRoulette ni OSM.
"""
from __future__ import annotations

import re

# Statuts de tâche MapRoulette : 1 = fixed (vraie citerne ajoutée à OSM),
# 2 = false positive / « not an issue ». Les autres sont ignorés.
STATUS_VERDICT = {1: "vrai", 2: "faux"}

_DEPT_RE = re.compile(r"\bDECI\s+(\w+)\s*$", re.IGNORECASE)


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


def dept_from_challenge_name(name: str | None) -> str | None:
    """Code département d'un challenge nommé « DECI <code> », sinon None."""
    m = _DEPT_RE.search(name or "")
    return m.group(1).upper() if m else None


def task_to_row(task: dict) -> dict | None:
    """Tâche MapRoulette → {lon, lat, score, verdict}, ou None si à ignorer."""
    verdict = STATUS_VERDICT.get(task.get("status"))
    if verdict is None:
        return None
    try:
        lon, lat = (float(v) for v in task["location"]["coordinates"][:2])
    except (KeyError, TypeError, ValueError):
        return None
    try:
        score = float(task["geometries"]["features"][0]["properties"]["score"])
    except (KeyError, IndexError, TypeError, ValueError):
        score = 0.0
    return {"lon": lon, "lat": lat, "score": score, "verdict": verdict}


def rows_to_csv(rows: list[dict]) -> str:
    """CSV `index,lat,lon,score,verdict` (index dès 1), lisible par parse_verdicts."""
    lines = ["index,lat,lon,score,verdict"]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i},{r['lat']:.9f},{r['lon']:.9f},{r['score']:.4f},{r['verdict']}")
    return "\n".join(lines) + "\n"
