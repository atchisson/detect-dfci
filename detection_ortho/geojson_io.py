"""Lecture/écriture de points au format GeoJSON."""
from __future__ import annotations

import json
from pathlib import Path


def points_to_geojson(points: list[dict], extra_props: dict | None = None) -> dict:
    """FeatureCollection à partir de points {lon, lat, ...autres->properties}."""
    features = []
    for p in points:
        props = {k: v for k, v in p.items() if k not in ("lon", "lat")}
        if extra_props:
            props.update(extra_props)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def write_geojson(fc: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")
