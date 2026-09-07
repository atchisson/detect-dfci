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

from detection_ortho.osm import fetch_boundary_relations, fetch_citernes
from detection_ortho.infer import (
    ways_to_polygon, windows_over_polygon, boxes_to_points, result_to_boxes,
)
from detection_ortho.dataset import assemble_window, window_tiles
from detection_ortho.local_ortho import open_ortho, read_window
from detection_ortho.tiles import download_tile
from detection_ortho import checkpoint
from detection_ortho.tilecache import (
    DEFAULT_TILE_BYTES, max_tiles_for_budget, next_chunk, purge_cache,
)
from detection_ortho.geo import dedup_points
from detection_ortho.compare import match_detections
from detection_ortho.geojson_io import points_to_geojson, write_geojson
from detection_ortho.maproulette import to_maproulette_tasks


def progress(iterable, total, label, min_interval=20.0, status=None, offset=0):
    """Affiche l'avancement avec temps écoulé + ETA sur stdout (fiable partout,
    ex. PowerShell, contrairement aux barres tqdm sur stderr).

    Imprime au plus une ligne toutes `min_interval` secondes (et la dernière),
    pour rester lisible sur un run long de plusieurs centaines de milliers de
    fenêtres (l'ancienne version tous les 2 % restait muette trop longtemps).

    `status` : callable optionnel renvoyant un texte à ajouter en fin de ligne
    (ex. le nombre de détections en direct).

    `offset` : fenêtres déjà traitées lors d'un run précédent (reprise). Le
    compteur affiché est absolu, mais le débit et l'ETA ne comptent que le
    travail de la session en cours."""
    total = int(total or 0)
    t0 = time.perf_counter()
    last = t0
    for i, item in enumerate(iterable, 1):
        now = time.perf_counter()
        fait = i + offset
        if now - last >= min_interval or fait == total:
            el = now - t0
            rate = i / el if el > 0 else 0
            eta = (total - fait) / rate if rate > 0 else 0
            pct = f" ({fait * 100 // total}%)" if total else ""
            extra = f" — {status()}" if status else ""
            print(f"  {label}: {fait}/{total}{pct} — écoulé {el / 60:.1f} min, "
                  f"ETA {eta / 60:.1f} min{extra}", flush=True)
            last = now
        yield item

ZOOM = 19
WINDOW = 640
INSTRUCTION = ("Une citerne semble présente ici sur l'ortho IGN. "
               "Vérifiez et ajoutez-la à OSM si confirmé.")


def download_ok(xy, cache, session):
    """Télécharge une tuile ; retourne None si OK, l'exception sinon."""
    try:
        download_tile(xy[0], xy[1], ZOOM, cache, session=session)
    except Exception as exc:  # noqa: BLE001
        return exc
    return None


def stream_windows(centers, cache, budget_bytes, pool, session, start=0):
    """Égrène les fenêtres en gardant le cache de tuiles sous `budget_bytes`.

    Découpe `centers` en tranches (cf. `tilecache`) : on attend les tuiles de la
    tranche courante, on lance en fond celles de la suivante, on égrène les
    fenêtres, puis on purge tout ce qui ne sert plus. Le disque ne contient donc
    jamais plus de deux tranches, et le téléchargement se recouvre avec
    l'inférence au lieu de la précéder.

    `start` : index de la première fenêtre à traiter (reprise après une pause).
    """
    tile_bytes = DEFAULT_TILE_BYTES
    max_tiles = max_tiles_for_budget(budget_bytes, tile_bytes)
    chunk, tiles, i = next_chunk(centers, start, ZOOM, WINDOW, max_tiles)
    futs = [pool.submit(download_ok, xy, cache, session) for xy in tiles]
    while chunk:
        n_fail = sum(1 for f in futs if f.result() is not None)
        if n_fail:
            print(f"  {n_fail} tuile(s) en échec (réessayées à l'assemblage).",
                  file=sys.stderr)
        # Tranche suivante lancée en fond pendant l'inférence de la courante.
        nxt, nxt_tiles, i = next_chunk(centers, i, ZOOM, WINDOW, max_tiles)
        futs = [pool.submit(download_ok, xy, cache, session)
                for xy in nxt_tiles - tiles]

        yield from chunk

        deleted, kept, kept_bytes = purge_cache(cache, nxt_tiles, ZOOM)
        if kept:  # la taille observée remplace l'estimation initiale
            tile_bytes = kept_bytes / kept
            max_tiles = max_tiles_for_budget(budget_bytes, tile_bytes)
        print(f"  cache: {kept} tuile(s) / {kept_bytes / 1e9:.2f} Go "
              f"({deleted} purgée(s))", flush=True)
        chunk, tiles = nxt, nxt_tiles


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
    ap.add_argument("--insee", type=str, default=None,
                    help="code ref:INSEE pour lever une ambiguïté de nom "
                         "(ex. 36 pour le département de l'Indre)")
    ap.add_argument("--admin-level", type=str, default=None,
                    help="niveau administratif OSM (6 = département, 8 = commune)")
    ap.add_argument("--restart", action="store_true",
                    help="ignorer un point de reprise existant et repartir de zéro")
    ap.add_argument("--checkpoint-every", type=float, default=120.0,
                    help="intervalle d'enregistrement du point de reprise (s)")
    ap.add_argument("--cache-gb", type=float, default=10.0,
                    help="plafond disque du cache de tuiles WMTS, en Go : les "
                         "tuiles sont téléchargées par tranches puis purgées "
                         "(0 = tout pré-télécharger, cache non borné)")
    args = ap.parse_args()

    cache = args.out / "tiles_cache"
    session = requests.Session()

    # --- A. Emprise ---
    print(f"Récupération de l'emprise « {args.boundary} »...")
    rels = fetch_retry(fetch_boundary_relations, args.boundary, session,
                       args.admin_level, args.insee)
    if not rels:
        sys.exit(f"Aucune relation OSM « {args.boundary} » trouvée — "
                  f"vérifiez le nom exact.")
    if len(rels) > 1:
        # Fusionner des homonymes donnerait une emprise aberrante (« Indre » est
        # à la fois le département 36 et une commune de Loire-Atlantique).
        lignes = "\n".join(
            f"    relation {r['id']} — admin_level "
            f"{r['tags'].get('admin_level', '?')}, ref:INSEE "
            f"{r['tags'].get('ref:INSEE', '?')}, boundary "
            f"{r['tags'].get('boundary', '?')}"
            for r in rels)
        sys.exit(
            f"{len(rels)} frontières OSM portent le nom « {args.boundary} » :\n"
            f"{lignes}\n"
            f"  Les fusionner produirait une emprise aberrante. Précisez avec "
            f"--insee <code> (le plus sûr) ou --admin-level <niveau>.")
    ways = rels[0]["ways"]
    tags = rels[0]["tags"]
    print(f"  relation {rels[0]['id']} — admin_level "
          f"{tags.get('admin_level', '?')}, ref:INSEE {tags.get('ref:INSEE', '?')}")
    polygon = ways_to_polygon(ways)
    west, south, east, north = polygon.bounds
    print(f"Emprise: bbox=({west:.4f},{south:.4f},{east:.4f},{north:.4f})")

    centers = windows_over_polygon(polygon, ZOOM, WINDOW, args.overlap)
    print(f"{len(centers)} fenêtre(s) d'inférence.")

    # --- A bis. Point de reprise ---
    empreinte = checkpoint.fingerprint(centers, {
        "boundary": args.boundary, "conf": args.conf, "overlap": args.overlap,
        "zoom": ZOOM, "window": WINDOW, "weights": Path(args.weights).name,
        "ortho": args.ortho or "",
        "insee": args.insee or "", "admin_level": args.admin_level or "",
    })
    start = 0
    detections: list[dict] = []
    if args.restart:
        checkpoint.clear(args.out)
    else:
        ckpt = checkpoint.load(args.out)
        if ckpt is None:
            pass
        elif ckpt["fingerprint"] != empreinte:
            sys.exit(
                f"Point de reprise incompatible dans {args.out} : les paramètres "
                f"ou l'emprise OSM ont changé depuis. Relancez avec --restart "
                f"pour repartir de zéro (le travail déjà fait sera perdu).")
        elif ckpt["done"] >= len(centers):
            print("Point de reprise complet : toutes les fenêtres ont déjà été "
                  "inférées, on passe directement au post-traitement.")
            start, detections = len(centers), ckpt["detections"]
        else:
            start, detections = ckpt["done"], ckpt["detections"]
            print(f"Reprise à la fenêtre {start}/{len(centers)} "
                  f"({start * 100 // len(centers)}%) avec {len(detections)} "
                  f"détection(s) déjà trouvée(s).")
    checkpoint.clear_stop(args.out)  # un STOP résiduel arrêterait le run aussitôt

    # --- Approvisionnement en imagerie ---
    ortho_vrt = None
    tile_pool = None
    windows = centers[start:]  # itérable des fenêtres restant à inférer
    if args.ortho:
        ortho_vrt = open_ortho(args.ortho, zoom=ZOOM)
        print(f"Ortho locale : {args.ortho} (lecture rasterio, pas de WMTS).")
    elif args.cache_gb > 0:
        # Streaming : téléchargement par tranches + purge, cache disque borné.
        tile_pool = ThreadPoolExecutor(max_workers=args.workers)
        windows = stream_windows(centers, cache, args.cache_gb * 1e9,
                                 tile_pool, session, start=start)
        print(f"WMTS en streaming : cache plafonné à {args.cache_gb:g} Go "
              f"(téléchargement par tranches x{args.workers}, purge au fil de "
              f"l'eau).")
    else:
        # Pré-téléchargement intégral (cache non borné, réutilisable).
        needed = set()
        for lon, lat in centers:
            tiles, _, _ = window_tiles(lon, lat, ZOOM, WINDOW)
            needed.update(tiles)
        print(f"{len(needed)} tuile(s) à récupérer (parallèle x{args.workers})...")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(download_ok, xy, cache, session) for xy in needed]
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
    live_path = args.out / "detections_live.geojson"  # aperçu au fil de l'eau
    last_flush = time.perf_counter()
    last_ckpt = last_flush
    done = start
    interrompu = False
    try:
        for lon, lat in progress(windows, len(centers), "Inférence", offset=start,
                                  status=lambda: f"{len(detections)} détection(s)"):
            # Demande de pause : testée une fenêtre sur 64 (un stat par fenêtre
            # coûterait pour rien), avant de compter la fenêtre comme traitée.
            if done % 64 == 0 and checkpoint.stop_requested(args.out):
                interrompu = True
                break
            done += 1  # compté même en cas d'échec, sinon l'index se décale
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
            if now - last_ckpt >= args.checkpoint_every:
                checkpoint.save(args.out, empreinte, done, detections)
                last_ckpt = now
    except KeyboardInterrupt:
        interrompu = True
        print("\nInterruption clavier reçue.", file=sys.stderr)
    finally:
        if ortho_vrt is not None:
            ortho_vrt.close()
        if tile_pool is not None:
            tile_pool.shutdown(cancel_futures=True)

    if interrompu:
        checkpoint.save(args.out, empreinte, done, detections)
        write_geojson(points_to_geojson(detections), live_path)
        checkpoint.clear_stop(args.out)
        pct = done * 100 // len(centers) if centers else 0
        print("\n=== Pause ===")
        print(f"  {done}/{len(centers)} fenêtre(s) traitées ({pct} %), "
              f"{len(detections)} détection(s) brutes conservées.")
        print(f"  Point de reprise : {args.out / checkpoint.NAME}")
        print("  Relancez la MÊME commande pour reprendre où le run s'est arrêté "
              "(--restart pour repartir de zéro).")
        return

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
    checkpoint.clear(args.out)  # run terminé : plus rien à reprendre
    print(f"\nLivrables dans {args.out}. "
          f"Chargez maproulette_challenge.geojson manuellement dans MapRoulette.")


if __name__ == "__main__":
    main()
