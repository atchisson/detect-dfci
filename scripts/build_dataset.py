# scripts/build_dataset.py
"""Jalon 2 — Génère le dataset YOLO de citernes depuis OSM + ortho IGN.

Usage:
    python scripts/build_dataset.py --bbox 0.05 46.72 1.06 47.72 \
        --negatives 120 --out dataset

Positifs : emergency=water_tank (boîtes dérivées des polygones OSM).
Négatifs difficiles : piscines (leisure=swimming_pool) + tuiles de fond
aléatoires. Écrit images/labels train/val/test + data.yaml + une mosaïque QA.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import requests

from detection_ortho.osm import fetch_features_geom
from detection_ortho.dataset import (
    element_to_box, assemble_window, geo_bbox_to_pixel_bbox, to_yolo_label,
    write_chip, split_indices, spatial_split_indices, write_data_yaml,
    window_tiles, fixed_box_geo, DEFAULT_BOX_M, parse_verdicts, compose_rgn,
    dedup_verdicts, near_any, window_is_blank,
)
from detection_ortho.tiles import LAYER, LAYER_IRC, layer_tag, prefetch_tiles, tile_cache_path


def progress(iterable, total, label):
    """Avancement en texte clair sur stdout (fiable partout, ex. PowerShell)."""
    total = int(total or 0)
    step = max(1, total // 50) if total else 1000
    for i, item in enumerate(iterable, 1):
        if i % step == 0 or i == total:
            pct = f" ({i * 100 // total}%)" if total else ""
            print(f"  {label}: {i}/{total}{pct}", flush=True)
        yield item

ZOOM = 19
WINDOW = 640


def fetch_retry(selectors, w, s, e, n, tries=5, pause=6.0):
    last = None
    for i in range(tries):
        try:
            return fetch_features_geom(selectors, w, s, e, n)
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
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    ap.add_argument("--negatives", type=int, default=120,
                    help="nombre de tuiles de fond aléatoires")
    ap.add_argument("--max-pools", type=int, default=300,
                    help="nombre max de piscines (négatifs durs) échantillonnées")
    ap.add_argument("--workers", type=int, default=12,
                    help="téléchargements de tuiles en parallèle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", type=int, nargs="*", default=[],
                    help="indices de positifs à écarter (intrus repérés au QA)")
    ap.add_argument("--verdicts", type=Path, nargs="+", default=None,
                    help="CSV de revue (un ou plusieurs) : faux -> négatifs "
                         "durs, vrai -> positifs")
    ap.add_argument("--dedup-m", type=float, default=15.0,
                    help="rayon (m) sous lequel un vrai doublonne un positif "
                         "déjà présent et est écarté")
    ap.add_argument("--holdout", type=Path, nargs="+", default=None,
                    help="CSV de verdicts mis de côté (test) : tout enregistrement "
                         "à moins de --holdout-m d'un de leurs points est écarté "
                         "(évite les fuites via OSM ou le --bbox)")
    ap.add_argument("--holdout-m", type=float, default=100.0,
                    help="rayon (m) de l'écart autour des points --holdout")
    ap.add_argument("--nir", action="store_true",
                    help="imagettes [R,G,NIR] (bleu remplacé par le NIR de l'IRC)")
    ap.add_argument("--layers", nargs="+", default=[LAYER], metavar="COUCHE",
                    help="couches WMTS à rendre : chaque enregistrement donne une "
                         "imagette par couche disponible (défaut : ortho habituelle)")
    ap.add_argument("--spatial-split", action="store_true",
                    help="split géographique par cellule (au lieu d'aléatoire)")
    ap.add_argument("--cell-deg", type=float, default=0.05,
                    help="taille de cellule du split spatial, en degrés")
    ap.add_argument("--out", type=Path, default=Path("dataset"))
    args = ap.parse_args()
    args.layers = list(dict.fromkeys(args.layers))  # sans doublons, ordre gardé
    if args.nir and list(args.layers) != [LAYER]:
        ap.error("--nir est incompatible avec --layers")

    west, south, east, north = args.bbox
    cache = args.out / "tiles_cache"
    imgs = args.out / "images"
    lbls = args.out / "labels"

    # --- Positifs ---
    print("Récupération des citernes (emergency=water_tank, out geom)...")
    pos_el = fetch_retry([("emergency", "water_tank")], west, south, east, north)
    boxes = [element_to_box(el) for el in pos_el]
    boxes = [b for i, b in enumerate(boxes) if i not in set(args.exclude)]
    print(f"{len(boxes)} citerne(s) retenue(s).")

    rng = random.Random(args.seed)

    # --- Négatifs : piscines (échantillonnées pour rester du même ordre que les positifs) ---
    print("Récupération des piscines (leisure=swimming_pool)...")
    pool_el = fetch_retry([("leisure", "swimming_pool")], west, south, east, north)
    pools = [element_to_box(el) for el in pool_el]
    n_pools_found = len(pools)
    if n_pools_found > args.max_pools:
        pools = rng.sample(pools, args.max_pools)
    print(f"{n_pools_found} piscine(s) trouvée(s), {len(pools)} retenue(s) "
          f"(--max-pools={args.max_pools}).")

    records = []  # (name, lon, lat, bbox_geo|None)
    for i, b in enumerate(boxes):
        records.append((f"citerne_{i:04d}", b["lon"], b["lat"], b["bbox_geo"]))
    for i, p in enumerate(pools):
        records.append((f"pool_{i:04d}", p["lon"], p["lat"], None))

    # --- Négatifs : tuiles de fond aléatoires ---
    for i in range(args.negatives):
        lon = rng.uniform(west, east)
        lat = rng.uniform(south, north)
        records.append((f"bg_{i:04d}", lon, lat, None))

    # --- Chips issus de la revue (hard-negative mining) ---
    if args.verdicts:
        vs = []
        for path in args.verdicts:
            vs += parse_verdicts(path.read_text(encoding="utf-8").splitlines())
        vs, n_dup = dedup_verdicts(
            vs, [(b["lon"], b["lat"]) for b in boxes], args.dedup_m)
        n_hard = n_rev = 0
        for v in vs:
            if v["verdict"] == "faux":
                records.append((f"hardneg_{n_hard:04d}", v["lon"], v["lat"], None))
                n_hard += 1
            else:  # vrai
                bbox = fixed_box_geo(v["lon"], v["lat"], DEFAULT_BOX_M)
                records.append((f"revpos_{n_rev:04d}", v["lon"], v["lat"], bbox))
                n_rev += 1
        print(f"Verdicts ingérés : {n_hard} négatif(s) dur(s), {n_rev} positif(s), "
              f"{n_dup} doublon(s) écarté(s).")

    # --- Mise de côté : aucun enregistrement près des points de test ---
    if args.holdout:
        held = []
        for path in args.holdout:
            held += [(v["lon"], v["lat"]) for v in
                     parse_verdicts(path.read_text(encoding="utf-8").splitlines())]
        flags = near_any([(r[1], r[2]) for r in records], held, args.holdout_m)
        n_drop = sum(flags)
        records = [r for r, f in zip(records, flags) if not f]
        print(f"Holdout : {n_drop} enregistrement(s) écarté(s) à moins de "
              f"{args.holdout_m:g} m de {len(held)} point(s) mis de côté.")

    # --- Récupération des images : préchargement parallèle des tuiles (dédupliquées), par couche ---
    needed = set()
    for _name, lon, lat, _bbox in records:
        tiles, _, _ = window_tiles(lon, lat, ZOOM, WINDOW)
        needed.update(tiles)

    def _tile_progress(i, n):
        if i % max(1, n // 50) == 0 or i == n:
            print(f"  Récupération des tuiles: {i}/{n}", flush=True)

    # Session partagée = keep-alive (évite un handshake TCP/TLS par tuile).
    session = requests.Session()
    for n_layer, layer in enumerate(list(args.layers) + ([LAYER_IRC] if args.nir else [])):
        print(f"{len(needed)} tuile(s) à récupérer, couche {layer} "
              f"(parallèle x{args.workers})...")
        # Première couche : un 404 peut être passager, on réessaie. Couches
        # suivantes : un 404 signifie « pas d'imagerie », une seule tentative.
        errors = prefetch_tiles(needed, cache, layer=layer, zoom=ZOOM,
                                workers=args.workers, session=session,
                                on_progress=_tile_progress,
                                tries=3 if n_layer == 0 else 1)
        if errors:
            print(f"  {len(errors)} tuile(s) en échec (réessayées à l'assemblage).",
                  file=sys.stderr)

    # Repart d'un dataset propre (évite les orphelins d'un run précédent) ;
    # le cache de tuiles (dossier séparé) est préservé.
    shutil.rmtree(imgs, ignore_errors=True)
    shutil.rmtree(lbls, ignore_errors=True)

    # --- Split et génération des chips (assemblées depuis le cache) ---
    if args.spatial_split:
        split = spatial_split_indices(
            [(lon, lat) for _n, lon, lat, _b in records],
            cell_deg=args.cell_deg, seed=args.seed)
    else:
        split = split_indices(len(records), seed=args.seed)

    # Warn if any partition is empty while records exist
    for part in ("train", "val", "test"):
        if not split[part] and records:
            print(f"  ⚠ lot '{part}' vide (split trop grossier ? augmente la zone "
                  f"ou baisse --cell-deg)", file=sys.stderr)

    where = {}
    for part, idxs in split.items():
        for i in idxs:
            where[i] = part

    qa_crops = []
    written = {layer: 0 for layer in args.layers}
    blank = {layer: 0 for layer in args.layers}
    failed = {layer: 0 for layer in args.layers}
    absent = {layer: 0 for layer in args.layers}
    for i, (name, lon, lat, bbox_geo) in enumerate(
        progress(records, len(records), "Génération des chips")
    ):
        part = where[i]  # un enregistrement = une seule partie, quelle que soit la couche
        for k, layer in enumerate(args.layers):
            if k == 0:
                chip_name = name
            else:
                chip_name = f"{name}__{layer_tag(layer).lstrip('_') or 'standard'}"
                # Tuile manquante après le préchargement = pas d'imagerie :
                # fenêtre « absente », sans réseau ni nouvelle tentative.
                wtiles, _, _ = window_tiles(lon, lat, ZOOM, WINDOW)
                if any(not tile_cache_path(tx, ty, ZOOM, cache, layer).exists()
                       for tx, ty in wtiles):
                    absent[layer] += 1
                    continue
            try:
                win_img, ogx, ogy = assemble_window(lon, lat, ZOOM, WINDOW, cache,
                                                    layer=layer)
                if args.nir:
                    irc_img, _, _ = assemble_window(
                        lon, lat, ZOOM, WINDOW, cache, layer=LAYER_IRC)
                    win_img = compose_rgn(win_img, irc_img)
            except Exception as exc:  # noqa: BLE001
                failed[layer] += 1
                print(f"  {chip_name}: échec fenêtre ({exc})", file=sys.stderr)
                continue
            if window_is_blank(win_img):
                blank[layer] += 1
                continue
            labels = []
            if bbox_geo is not None:
                px = geo_bbox_to_pixel_bbox(bbox_geo, ogx, ogy, ZOOM, WINDOW)
                line = to_yolo_label(px, WINDOW)
                if line:
                    labels.append(line)
                    if k == 0 and name.startswith("citerne") and len(qa_crops) < 48:
                        x0, y0, x1, y1 = (int(v) for v in px)
                        vis = win_img.copy()
                        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
                        qa_crops.append(cv2.resize(vis, (128, 128)))
            write_chip(win_img, labels, imgs / part, lbls / part, chip_name)
            written[layer] += 1
    for layer in args.layers:
        print(f"Couche {layer} : {written[layer]} imagette(s), "
              f"{blank[layer]} fenêtre(s) vide(s), {absent[layer]} absente(s), "
              f"{failed[layer]} échec(s).")

    write_data_yaml(args.out, args.out / "data.yaml")

    # --- Mosaïque QA ---
    if qa_crops:
        cols = 8
        rows = (len(qa_crops) + cols - 1) // cols
        montage = np.full((rows * 128, cols * 128, 3), 50, np.uint8)
        for i, c in enumerate(qa_crops):
            r, cc = divmod(i, cols)
            montage[r * 128:(r + 1) * 128, cc * 128:(cc + 1) * 128] = c
        cv2.imwrite(str(args.out / "qa_positives.png"), montage)

    print(f"\nDataset écrit dans {args.out} (data.yaml + images/labels).")
    print(f"QA positifs : {args.out / 'qa_positives.png'} — vérifiez et relancez "
          f"avec --exclude <indices> pour retirer les intrus.")


if __name__ == "__main__":
    main()
