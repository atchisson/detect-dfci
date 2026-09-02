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

import requests
from shapely.geometry import Point

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detection_ortho.osm import fetch_relation_ways, fetch_citernes
from detection_ortho.infer import (
    ways_to_polygon, windows_over_polygon, boxes_to_points, result_to_boxes,
)
from detection_ortho.dataset import assemble_window, window_tiles
from detection_ortho.local_ortho import open_ortho, read_window
from detection_ortho.tiles import download_tile
from detection_ortho.geo import dedup_points
from detection_ortho.compare import match_detections
from detection_ortho.geojson_io import points_to_geojson, write_geojson
from detection_ortho.maproulette import to_maproulette_tasks


def progress(iterable, total, label, min_interval=20.0, status=None):
    """Affiche l'avancement avec temps écoulé + ETA sur stdout (fiable partout,
    ex. PowerShell, contrairement aux barres tqdm sur stderr).

    Imprime au plus une ligne toutes `min_interval` secondes (et la dernière),
    pour rester lisible sur un run long de plusieurs centaines de milliers de
    fenêtres (l'ancienne version tous les 2 % restait muette trop longtemps).

    `status` : callable optionnel renvoyant un texte à ajouter en fin de ligne
    (ex. le nombre de détections en direct)."""
    total = int(total or 0)
    t0 = time.perf_counter()
    last = t0
    for i, item in enumerate(iterable, 1):
        now = time.perf_counter()
        if now - last >= min_interval or i == total:
            el = now - t0
            rate = i / el if el > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            pct = f" ({i * 100 // total}%)" if total else ""
            extra = f" — {status()}" if status else ""
            print(f"  {label}: {i}/{total}{pct} — écoulé {el / 60:.1f} min, "
                  f"ETA {eta / 60:.1f} min{extra}", flush=True)
            last = now
        yield item

ZOOM = 19
WINDOW = 640
INSTRUCTION = ("Une citerne semble présente ici sur l'ortho IGN. "
               "Vérifiez et ajoutez-la à OSM si confirmé.")


def fetch_retry(fn, *a, tries=5, pause=6.0, **kw):
    last = None
    for i in range(tries):
        try:
            return fn(*a, **kw)
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
    ap.add_argument("--ortho", type=str, default=None,
                    help="chemin BD ORTHO locale (raster/VRT/dossier de dalles) ; "
                         "si fourni, lecture locale au lieu du WMTS")
    args = ap.parse_args()

    cache = args.out / "tiles_cache"
    session = requests.Session()

    # --- A. Emprise ---
    print(f"Récupération de l'emprise « {args.boundary} »...")
    ways = fetch_retry(fetch_relation_ways, args.boundary, session)
    if not ways:
        sys.exit(f"Aucune relation OSM « {args.boundary} » trouvée — "
                  f"vérifiez le nom exact.")
    polygon = ways_to_polygon(ways)
    west, south, east, north = polygon.bounds
    print(f"Emprise: bbox=({west:.4f},{south:.4f},{east:.4f},{north:.4f})")

    centers = windows_over_polygon(polygon, ZOOM, WINDOW, args.overlap)
    print(f"{len(centers)} fenêtre(s) d'inférence.")

    # --- Pré-téléchargement parallèle des tuiles (uniquement en mode WMTS) ---
    ortho_vrt = None
    if args.ortho:
        ortho_vrt = open_ortho(args.ortho, zoom=ZOOM)
        print(f"Ortho locale : {args.ortho} (lecture rasterio, pas de WMTS).")
    else:
        needed = set()
        for lon, lat in centers:
            tiles, _, _ = window_tiles(lon, lat, ZOOM, WINDOW)
            needed.update(tiles)
        print(f"{len(needed)} tuile(s) à récupérer (parallèle x{args.workers})...")

        def _dl(xy):
            try:
                download_tile(xy[0], xy[1], ZOOM, cache, session=session)
            except Exception as exc:  # noqa: BLE001
                return exc
            return None

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_dl, xy) for xy in needed]
            n_fail = 0
            for fut in progress(as_completed(futs), len(futs),
                                "Récupération des tuiles"):
                if fut.result() is not None:
                    n_fail += 1
            if n_fail:
                print(f"  {n_fail} tuile(s) en échec au pré-téléchargement "
                      f"(réessayées à l'assemblage).", file=sys.stderr)

    # --- B/C. Inférence + post-traitement ---
    from ultralytics import YOLO
    model = YOLO(args.weights)
    detections: list[dict] = []
    live_path = args.out / "detections_live.geojson"  # aperçu au fil de l'eau
    last_flush = time.perf_counter()
    try:
        for lon, lat in progress(centers, len(centers), "Inférence",
                                  status=lambda: f"{len(detections)} détection(s)"):
            try:
                if ortho_vrt is not None:
                    img, ogx, ogy = read_window(ortho_vrt, lon, lat, ZOOM, WINDOW)
                else:
                    img, ogx, ogy = assemble_window(lon, lat, ZOOM, WINDOW, cache)
            except Exception as exc:  # noqa: BLE001
                print(f"  fenêtre ({lon:.5f},{lat:.5f}) échec ({exc})", file=sys.stderr)
                continue
            res = model.predict(img, conf=args.conf, device=args.device, verbose=False)[0]
            boxes = result_to_boxes(res.boxes)
            detections.extend(boxes_to_points(boxes, ogx, ogy, ZOOM))
            now = time.perf_counter()
            if now - last_flush >= 30.0:  # flush périodique pour la carte live
                write_geojson(points_to_geojson(detections), live_path)
                last_flush = now
    finally:
        if ortho_vrt is not None:
            ortho_vrt.close()

    detections = dedup_points(detections, radius_m=args.dedup)
    print(f"{len(detections)} détection(s) après dédoublonnage.")
    write_geojson(points_to_geojson(detections), args.out / "detections.geojson")
    live_path.unlink(missing_ok=True)  # aperçu remplacé par les livrables finaux

    # --- D. Citernes OSM de la zone (filtrées au polygone) ---
    osm = fetch_retry(fetch_citernes, west, south, east, north, session)
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

    # Overlay d'inspection (best-effort, ne bloque jamais le run).
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        exterior = getattr(polygon, "exterior", None)
        if exterior is None and hasattr(polygon, "geoms"):
            exterior = getattr(polygon.geoms[0], "exterior", None)
        if exterior is not None:
            xs, ys = exterior.xy
            ax.plot(xs, ys, color="black", linewidth=0.8, label="emprise")
        else:
            bw, bs, be, bn = polygon.bounds
            ax.plot([bw, be, be, bw, bw], [bs, bs, bn, bn, bs],
                    color="black", linewidth=0.8, label="emprise")

        def _scatter(items, **kw):
            if items:
                ax.scatter([p["lon"] for p in items], [p["lat"] for p in items],
                           s=8, **kw)

        _scatter([m["detection"] for m in res["matched"]], color="green", label="∩ OSM")
        _scatter(res["detected_only"], color="blue", label="candidats (∉ OSM)")
        _scatter(res["osm_only"], color="red", label="OSM non détectées")
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title("Détections vs OSM")
        fig.savefig(args.out / "overlay.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Overlay: {args.out / 'overlay.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  overlay non généré ({exc})", file=sys.stderr)

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
