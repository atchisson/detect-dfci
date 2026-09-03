"""Jalon 3 — Génère le fichier de tâches MapRoulette depuis un GeoJSON de points.

Usage:
    python scripts/export_maproulette.py --input inference_out/detected_only.geojson \
        --out inference_out/maproulette_challenge.geojson

Génération de FICHIER uniquement — aucun upload MapRoulette.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_ortho.geojson_io import write_geojson
from detection_ortho.maproulette import to_maproulette_tasks

INSTRUCTION = ("Une citerne semble présente ici sur l'ortho IGN. "
               "Vérifiez et ajoutez-la à OSM si confirmé.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="GeoJSON de points candidats (detected_only.geojson)")
    ap.add_argument("--out", type=Path, default=Path("maproulette_challenge.geojson"))
    ap.add_argument("--instruction", type=str, default=INSTRUCTION)
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="ne garder que les points de score >= ce seuil "
                         "(qualité du challenge ; ex. 0.7)")
    args = ap.parse_args()

    fc = json.loads(args.input.read_text(encoding="utf-8"))
    points, skipped = [], 0
    for feat in fc.get("features", []):
        coords = feat["geometry"]["coordinates"]
        lon, lat = coords[0], coords[1]
        score = feat.get("properties", {}).get("score")
        if args.min_score > 0.0 and (score is None or score < args.min_score):
            skipped += 1
            continue
        points.append({"lon": lon, "lat": lat, "score": score})
    write_geojson(to_maproulette_tasks(points, args.instruction), args.out)
    print(f"{len(points)} tâche(s) écrites dans {args.out} "
          f"(seuil {args.min_score}, {skipped} écartée(s)). "
          f"À importer manuellement dans MapRoulette.")


if __name__ == "__main__":
    main()
