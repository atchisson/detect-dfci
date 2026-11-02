"""Liste des objets OSM (citernes) que le modèle n'a pas retrouvés, en challenge de vérification.

Rassemble les `osm_only.geojson` des départements traités :
  - `inference_<insee>/` : double détection (ortho 2026 + habituelle), objet non
    retrouvé par aucune des deux couches ;
  - `inference_rattrapage/<insee>/` : première campagne, une seule couche (2026).
Écrit un challenge MapRoulette (une tâche par objet OSM, à vérifier : existe-t-il
encore, est-il bien placé et bien tagué ?) et un CSV de la même liste.

Usage:
    python scripts/osm_non_retrouves.py --out out/osm_non_retrouves
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_ortho.geo import dedup_points

INSTRUCTION = ("Cet objet OSM n'a pas été retrouvé par la détection sur l'ortho IGN. "
               "Vérifiez sur l'imagerie qu'il existe toujours, qu'il est bien placé et "
               "bien étiqueté (citerne souple, réservoir, bassin…) ; corrigez ou "
               "supprimez-le sinon.")
CLES_TAGS = ("emergency", "man_made", "content", "tank:type", "water_tank:type",
             "building", "name", "capacity", "volume")


def resume_tags(tags: dict) -> str:
    return ", ".join(f"{k}={tags[k]}" for k in CLES_TAGS if k in tags)


def collecter(racine: Path) -> list[dict]:
    """Un point par objet OSM non retrouvé, avec son département et les couches testées."""
    sources = [(d, "2026+standard") for d in sorted(racine.glob("inference_??"))]
    sources += [(d, "2026") for d in sorted((racine / "inference_rattrapage").glob("??"))]
    pts = []
    for d, couches in sources:
        f = d / "osm_only.geojson"
        if not f.exists():
            continue
        insee = d.name.removeprefix("inference_")
        for feat in json.loads(f.read_text(encoding="utf-8"))["features"]:
            lon, lat = feat["geometry"]["coordinates"][:2]
            tags = feat["properties"].get("tags", {})
            pts.append({"lon": lon, "lat": lat, "dept": insee, "couches": couches,
                        "tags": resume_tags(tags), "score": 0.0})
    return dedup_points(pts, 3.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--racine", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("out/osm_non_retrouves"),
                    help="préfixe de sortie (.geojson et .csv)")
    a = ap.parse_args()
    pts = collecter(a.racine)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
         "properties": {"instruction": INSTRUCTION, "dept": p["dept"],
                        "couches": p["couches"], "osm_tags": p["tags"]}} for p in pts]}
    a.out.with_suffix(".geojson").write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    with a.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dept", "lon", "lat", "couches", "tags"])
        for p in pts:
            w.writerow([p["dept"], p["lon"], p["lat"], p["couches"], p["tags"]])
    par: dict[str, int] = {}
    for p in pts:
        par[p["dept"]] = par.get(p["dept"], 0) + 1
    print(f"{len(pts)} objet(s) OSM non retrouvé(s) : "
          + ", ".join(f"{k}:{v}" for k, v in sorted(par.items())))


if __name__ == "__main__":
    main()
