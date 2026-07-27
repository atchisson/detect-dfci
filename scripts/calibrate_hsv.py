"""Calibre les seuils HSV de la baseline sur des citernes OSM connues.

Idée : les points OSM sont la vérité terrain. On télécharge la tuile de chaque
citerne, on échantillonne la couleur dans une petite fenêtre centrée sur le
point, et on agrège pour proposer des bornes HSV (hsv_low/hsv_high) à passer à
`detect_in_image`.

Usage:
    python scripts/calibrate_hsv.py --bbox 0.05 46.72 1.06 47.72 --limit 60 --flexible
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from detection_ortho.osm import fetch_citernes
from detection_ortho.tiles import download_tile, lonlat_to_pixel


def fetch_with_retry(west, south, east, north, tries=4, pause=5.0):
    """Overpass renvoie parfois 406/504 ; on réessaie avec une pause."""
    last = None
    for i in range(tries):
        try:
            return fetch_citernes(west, south, east, north)
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  tentative {i + 1}/{tries} échouée ({exc}); pause {pause}s...",
                  file=sys.stderr)
            time.sleep(pause)
    raise last


def sample_median_hsv(tile_path: Path, px: float, py: float, win: int = 5):
    """HSV médian d'une fenêtre win×win centrée sur (px, py). None si hors image."""
    img = cv2.imread(str(tile_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    x, y = int(round(px)), int(round(py))
    r = win // 2
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return np.median(hsv, axis=0)  # (H, S, V)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--limit", type=int, default=60,
                    help="nombre max de citernes échantillonnées")
    ap.add_argument("--win", type=int, default=5,
                    help="taille (px) de la fenêtre d'échantillonnage centrée")
    ap.add_argument("--flexible", action="store_true",
                    help="ne garder que les citernes taguées water_tank:type=flexible")
    ap.add_argument("--out", type=Path, default=Path("out/calib"))
    args = ap.parse_args()

    west, south, east, north = args.bbox
    cache = args.out / "tiles_cache"

    print(f"Requête OSM sur bbox {args.bbox} ...")
    citernes = fetch_with_retry(west, south, east, north)
    if args.flexible:
        citernes = [c for c in citernes
                    if c["tags"].get("water_tank:type") == "flexible"]
    print(f"{len(citernes)} citerne(s) retenue(s)"
          f"{' (flexibles)' if args.flexible else ''}.")

    citernes = citernes[:args.limit]
    print(f"Échantillonnage de {len(citernes)} citerne(s) (fenêtre {args.win}px)...\n")

    medians = []
    for i, c in enumerate(citernes):
        x, y, px, py = lonlat_to_pixel(c["lon"], c["lat"], args.zoom)
        try:
            tile = download_tile(x, y, args.zoom, cache)
        except Exception as exc:  # noqa: BLE001
            print(f"  citerne {i}: échec tuile ({exc})", file=sys.stderr)
            continue
        med = sample_median_hsv(tile, px, py, args.win)
        if med is not None:
            medians.append(med)

    if not medians:
        print("Aucun échantillon exploitable.")
        return

    arr = np.array(medians)  # (N, 3) H,S,V
    print(f"=== {len(arr)} échantillons couleur (HSV médian par citerne) ===")
    labels = ["H (0-179)", "S (0-255)", "V (0-255)"]
    for j, lab in enumerate(labels):
        col = arr[:, j]
        print(f"  {lab:12s} min={col.min():3.0f}  p10={np.percentile(col,10):3.0f}"
              f"  médiane={np.median(col):3.0f}  p90={np.percentile(col,90):3.0f}"
              f"  max={col.max():3.0f}")

    # Bornes suggérées : p10..p90 par canal, légèrement élargies.
    low = np.percentile(arr, 10, axis=0)
    high = np.percentile(arr, 90, axis=0)
    margin = np.array([5, 30, 30])
    hsv_low = np.clip(low - margin, [0, 0, 0], [179, 255, 255]).astype(int)
    hsv_high = np.clip(high + margin, [0, 0, 0], [179, 255, 255]).astype(int)
    print("\n=== Bornes HSV suggérées (p10-p90 ± marge) ===")
    print(f"  --hsv-low  {hsv_low[0]} {hsv_low[1]} {hsv_low[2]}")
    print(f"  --hsv-high {hsv_high[0]} {hsv_high[1]} {hsv_high[2]}")


if __name__ == "__main__":
    main()
