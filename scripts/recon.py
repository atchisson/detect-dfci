"""Jalon 0 — Reconnaissance des données.

Usage:
    python scripts/recon.py --bbox 6.14 43.41 6.16 43.43 --zoom 19 --out recon_out

Compte les citernes OSM de la bbox, télécharge leur tuile ortho, marque leur
position, sauvegarde les imagettes et un décompte des tags. Objectif : REGARDER
les données avant tout code de détection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from detection_ortho.osm import fetch_citernes
from detection_ortho.tiles import (
    download_tile,
    lonlat_to_pixel,
    save_tile_with_marker,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--out", type=Path, default=Path("recon_out"))
    args = ap.parse_args()

    west, south, east, north = args.bbox
    cache = args.out / "tiles_cache"
    images = args.out / "images"

    print(f"Requête OSM sur bbox {args.bbox}...")
    citernes = fetch_citernes(west, south, east, north)
    print(f"{len(citernes)} citerne(s) trouvée(s) dans OSM.")

    tag_counter: Counter = Counter()
    for i, c in enumerate(citernes):
        for k, v in c["tags"].items():
            tag_counter[f"{k}={v}"] += 1
        x, y, px, py = lonlat_to_pixel(c["lon"], c["lat"], args.zoom)
        tile_path = download_tile(x, y, args.zoom, cache)
        out_img = images / f"citerne_{i:03d}.png"
        save_tile_with_marker(tile_path, px, py, out_img)

    print("\nHistogramme des tags :")
    for tag, n in tag_counter.most_common():
        print(f"  {n:4d}  {tag}")
    print(f"\nImagettes marquées écrites dans : {images}")


if __name__ == "__main__":
    main()
