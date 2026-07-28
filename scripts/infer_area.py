"""Jalon 3 — Inférence YOLO sur une emprise + comparaison OSM + MapRoulette.

Usage:
    python scripts/infer_area.py --boundary "Tours Métropole Val de Loire" \
        --weights runs/citernes/weights/best.pt --conf 0.4 --out inference_out

Emprise = polygone administratif OSM. Écrit detections/matched/detected_only/
osm_only .geojson + maproulette_challenge.geojson + un résumé chiffré.
Aucun upload MapRoulette : génération de fichiers uniquement.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import requests
from shapely.geometry import Point

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **k):
        return it

from detection_ortho.osm import fetch_relation_ways, fetch_citernes
from detection_ortho.infer import ways_to_polygon, windows_over_polygon, boxes_to_points
from detection_ortho.dataset import assemble_window, window_tiles
from detection_ortho.tiles import download_tile
from detection_ortho.geo import dedup_points
from detection_ortho.compare import match_detections
from detection_ortho.geojson_io import points_to_geojson, write_geojson
from detection_ortho.maproulette import to_maproulette_tasks

ZOOM = 19
WINDOW = 640
INSTRUCTION = ("Une citerne semble présente ici sur l'ortho IGN. "
               "Vérifiez et ajoutez-la à OSM si confirmé.")


def fetch_retry(fn, *a, tries=5, pause=6.0):
    last = None
    for i in range(tries):
        try:
            return fn(*a)
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  Overpass retry {i + 1}/{tries} ({exc})", file=sys.stderr)
            time.sleep(pause)
    raise last


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", type=str, required=True,
                    help="nom de la relation administrative OSM")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--radius", type=float, default=25.0,
                    help="rayon d'appariement OSM (m)")
    ap.add_argument("--dedup", type=float, default=10.0,
                    help="rayon de dédoublonnage des détections (m)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=Path, default=Path("inference_out"))
    args = ap.parse_args()

    cache = args.out / "tiles_cache"

    # --- A. Emprise ---
    print(f"Récupération de l'emprise « {args.boundary} »...")
    ways = fetch_retry(fetch_relation_ways, args.boundary)
    polygon = ways_to_polygon(ways)
    west, south, east, north = polygon.bounds
    print(f"Emprise: bbox=({west:.4f},{south:.4f},{east:.4f},{north:.4f})")

    centers = windows_over_polygon(polygon, ZOOM, WINDOW, args.overlap)
    print(f"{len(centers)} fenêtre(s) d'inférence.")

    # --- Pré-téléchargement parallèle des tuiles ---
    needed = set()
    for lon, lat in centers:
        tiles, _, _ = window_tiles(lon, lat, ZOOM, WINDOW)
        needed.update(tiles)
    print(f"{len(needed)} tuile(s) à récupérer (parallèle x{args.workers})...")
    session = requests.Session()

    def _dl(xy):
        try:
            download_tile(xy[0], xy[1], ZOOM, cache, session=session)
        except Exception as exc:  # noqa: BLE001
            return exc
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_dl, xy) for xy in needed]
        for _ in tqdm(as_completed(futs), total=len(futs),
                      desc="Récupération des tuiles", unit="tuile"):
            pass

    # --- B/C. Inférence + post-traitement ---
    from ultralytics import YOLO
    model = YOLO(args.weights)
    detections: list[dict] = []
    for lon, lat in tqdm(centers, desc="Inférence", unit="fenêtre"):
        try:
            img, ogx, ogy = assemble_window(lon, lat, ZOOM, WINDOW, cache)
        except Exception as exc:  # noqa: BLE001
            print(f"  fenêtre ({lon:.5f},{lat:.5f}) échec ({exc})", file=sys.stderr)
            continue
        res = model.predict(img, conf=args.conf, device=args.device, verbose=False)[0]
        boxes = []
        for b in res.boxes:
            cx, cy = float(b.xywh[0][0]), float(b.xywh[0][1])
            boxes.append((cx, cy, float(b.conf[0])))
        detections.extend(boxes_to_points(boxes, ogx, ogy, ZOOM))

    detections = dedup_points(detections, radius_m=args.dedup)
    print(f"{len(detections)} détection(s) après dédoublonnage.")
    write_geojson(points_to_geojson(detections), args.out / "detections.geojson")

    # --- D. Citernes OSM de la zone (filtrées au polygone) ---
    osm = fetch_retry(fetch_citernes, west, south, east, north)
    osm = [o for o in osm if polygon.contains(Point(o["lon"], o["lat"]))]
    print(f"{len(osm)} citerne(s) OSM dans l'emprise.")

    # --- E. Comparaison ---
    res = match_detections(detections, osm, radius_m=args.radius)
    write_geojson(points_to_geojson([m["detection"] for m in res["matched"]]),
                  args.out / "matched.geojson")
    write_geojson(points_to_geojson(res["detected_only"]),
                  args.out / "detected_only.geojson")
    write_geojson(points_to_geojson(res["osm_only"]),
                  args.out / "osm_only.geojson")

    # --- F. MapRoulette (fichier uniquement) ---
    write_geojson(to_maproulette_tasks(res["detected_only"], INSTRUCTION),
                  args.out / "maproulette_challenge.geojson")

    n_match = len(res["matched"])
    rappel = n_match / len(osm) if osm else float("nan")
    print("\n=== Résumé inférence ===")
    print(f"  Détections (après dédup)      : {len(detections)}")
    print(f"  Confirmées (∩ OSM)            : {n_match}")
    print(f"  Candidates (∉ OSM) -> MapRoul.: {len(res['detected_only'])}")
    print(f"  Manquées (OSM non détectées)  : {len(res['osm_only'])}")
    if osm:
        print(f"  Rappel réel (∩OSM / OSM)      : {rappel:.0%}")
    print(f"\nLivrables dans {args.out}. "
          f"Chargez maproulette_challenge.geojson manuellement dans MapRoulette.")


if __name__ == "__main__":
    main()
