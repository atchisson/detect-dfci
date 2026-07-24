"""Jalon 1 — Baseline OpenCV sur une bbox + comparaison OSM.

Usage:
    python scripts/run_baseline.py --bbox 6.14 43.41 6.16 43.43 --zoom 19 --out baseline_out

Parcourt les tuiles de la bbox, applique la détection couleur/forme, reprojette
les détections en lon/lat, dédoublonne, compare aux citernes OSM, et écrit les
GeoJSON (matched / detected_only / osm_only) plus un résumé chiffré.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from detection_ortho.osm import fetch_citernes
from detection_ortho.tiles import download_tile, tiles_in_bbox, pixel_to_lonlat
from detection_ortho.baseline_cv import detect_in_image, default_params
from detection_ortho.geo import dedup_points
from detection_ortho.compare import match_detections
from detection_ortho.geojson_io import points_to_geojson, write_geojson


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--radius", type=float, default=25.0,
                    help="rayon d'appariement OSM en mètres")
    ap.add_argument("--out", type=Path, default=Path("baseline_out"))
    args = ap.parse_args()

    west, south, east, north = args.bbox
    cache = args.out / "tiles_cache"
    params = default_params()

    tiles = tiles_in_bbox(west, south, east, north, args.zoom)
    print(f"{len(tiles)} tuile(s) à traiter...")

    detections: list[dict] = []
    for x, y in tiles:
        tile_path = download_tile(x, y, args.zoom, cache)
        img = cv2.imread(str(tile_path))
        if img is None:
            continue
        for d in detect_in_image(img, params):
            lon, lat = pixel_to_lonlat(x, y, args.zoom, d["px"], d["py"])
            detections.append({"lon": lon, "lat": lat, "score": d["score"]})

    detections = dedup_points(detections, radius_m=args.radius)
    print(f"{len(detections)} détection(s) après dédoublonnage.")

    osm = fetch_citernes(west, south, east, north)
    print(f"{len(osm)} citerne(s) OSM sur la zone.")

    res = match_detections(detections, osm, radius_m=args.radius)

    write_geojson(points_to_geojson([m["detection"] for m in res["matched"]]),
                  args.out / "matched.geojson")
    write_geojson(points_to_geojson(res["detected_only"]),
                  args.out / "detected_only.geojson")
    write_geojson(points_to_geojson(res["osm_only"]),
                  args.out / "osm_only.geojson")

    n_match = len(res["matched"])
    n_osm = len(osm)
    recall = n_match / n_osm if n_osm else float("nan")
    print("\n=== Résumé baseline ===")
    print(f"  Appariées (détectées ∩ OSM) : {n_match}")
    print(f"  Nouvelles (détectées \\ OSM) : {len(res['detected_only'])}")
    print(f"  Manquées (OSM \\ détectées)  : {len(res['osm_only'])}")
    print(f"  Rappel approximatif          : {recall:.0%}" if n_osm else
          "  Rappel : n/a (aucune citerne OSM)")
    print(f"\nGeoJSON écrits dans : {args.out}")


if __name__ == "__main__":
    main()
