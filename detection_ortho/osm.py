"""Récupération des citernes connues via l'API Overpass (OSM)."""
from __future__ import annotations

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Certaines instances Overpass (derrière un WAF) renvoient 406 sur le
# User-Agent par défaut de python-requests. On s'identifie explicitement
# avec l'URL du dépôt, comme le recommande l'étiquette Overpass.
USER_AGENT = "detect-dfci/0.1 (+https://github.com/atchisson/detect-dfci)"

# (clé, valeur) des tags OSM candidats pour les citernes / réserves incendie.
# À affiner au Jalon 0 selon ce qui est réellement présent sur la zone.
CITERNE_TAGS: list[tuple[str, str]] = [
    ("emergency", "water_tank"),
    ("man_made", "water_tank"),
    ("emergency", "fire_water_pond"),
]


def build_overpass_query(
    west: float, south: float, east: float, north: float
) -> str:
    """Requête Overpass QL récupérant nodes et ways citernes dans la bbox."""
    bbox = f"{south},{west},{north},{east}"  # Overpass attend s,w,n,e
    clauses = []
    for key, value in CITERNE_TAGS:
        clauses.append(f'  node["{key}"="{value}"]({bbox});')
        clauses.append(f'  way["{key}"="{value}"]({bbox});')
    body = "\n".join(clauses)
    return f"[out:json][timeout:60];\n(\n{body}\n);\nout center;"


def parse_overpass_response(data: dict) -> list[dict]:
    """Transforme la réponse Overpass en points {lon, lat, tags}.

    Les ways sont réduits à leur centre (`out center`). Les éléments sans
    position exploitable sont ignorés.
    """
    points: list[dict] = []
    for el in data.get("elements", []):
        if el.get("type") == "node":
            lon, lat = el.get("lon"), el.get("lat")
        else:  # way / relation avec center
            center = el.get("center") or {}
            lon, lat = center.get("lon"), center.get("lat")
        if lon is None or lat is None:
            continue
        points.append({"lon": lon, "lat": lat, "tags": el.get("tags", {})})
    return points


def fetch_citernes(
    west: float, south: float, east: float, north: float, session=None
) -> list[dict]:
    """Interroge Overpass et retourne les citernes de la bbox."""
    sess = session or requests.Session()
    query = build_overpass_query(west, south, east, north)
    resp = sess.post(
        OVERPASS_URL,
        data=query,
        headers={"User-Agent": USER_AGENT},
        timeout=90,
    )
    resp.raise_for_status()
    return parse_overpass_response(resp.json())
