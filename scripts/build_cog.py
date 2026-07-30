"""Pré-reprojette l'ortho en GeoTIFF tuilé EPSG:3857 aligné sur la grille z19.

Reprojette **dalle par dalle** : chaque dalle JP2 est décodée UNE seule fois puis
warpée dans la sortie (via rasterio.warp.reproject). Bien plus rapide que de
lire bloc par bloc à travers un WarpedVRT (qui re-décode). Aucun outil GDAL
externe requis. Ensuite read_window lit ce raster déjà en 3857 et tuilé -> vite.

Usage:
    python scripts/build_cog.py --src <dossier de dalles ou 1 raster> --out ortho37_3857.tif
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.windows import Window

_R = 6378137.0


def _mpp(zoom: int, tile_size: int = 256) -> float:
    return (2 * math.pi * _R) / (tile_size * (2 ** zoom))


def _dalle_list(src: str) -> list[str]:
    p = Path(src)
    if p.is_dir():
        files = sorted(glob.glob(str(p / "**" / "*.jp2"), recursive=True)
                       + glob.glob(str(p / "**" / "*.tif"), recursive=True))
        if not files:
            raise SystemExit(f"Aucune dalle .jp2/.tif sous {p}")
        return files
    return [src]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True,
                    help="dossier de dalles (jp2/tif) ou un raster unique")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compress", choices=["jpeg", "deflate"], default="jpeg")
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--blocksize", type=int, default=512)
    args = ap.parse_args()

    mpp = _mpp(args.zoom)
    files = _dalle_list(args.src)
    print(f"{len(files)} dalle(s).")

    # 1. Emprise globale (3857) -> plage de pixels globaux z19 + métadonnées.
    minx = miny = 1e18
    maxx = maxy = -1e18
    nbands = None
    for f in files:
        with rasterio.open(f) as d:
            nbands = nbands or d.count
            b = transform_bounds(d.crs, "EPSG:3857", *d.bounds, densify_pts=21)
        minx, miny = min(minx, b[0]), min(miny, b[1])
        maxx, maxy = max(maxx, b[2]), max(maxy, b[3])

    gx0 = int(math.floor((minx + math.pi * _R) / mpp))
    gy0 = int(math.floor((math.pi * _R - maxy) / mpp))  # maxy -> haut -> gy petit
    gx1 = int(math.ceil((maxx + math.pi * _R) / mpp))
    gy1 = int(math.ceil((math.pi * _R - miny) / mpp))
    W, H = gx1 - gx0, gy1 - gy0
    out_transform = Affine(mpp, 0.0, -math.pi * _R + gx0 * mpp,
                           0.0, -mpp, math.pi * _R - gy0 * mpp)

    profile = dict(driver="GTiff", width=W, height=H, count=nbands,
                   dtype="uint8", crs="EPSG:3857", transform=out_transform,
                   tiled=True, blockxsize=args.blocksize,
                   blockysize=args.blocksize, BIGTIFF="YES")
    if args.compress == "jpeg" and nbands == 3:
        profile.update(compress="JPEG", photometric="YCBCR", jpeg_quality=85)
    else:
        if args.compress == "jpeg":
            print("  (compress jpeg ignoré : != 3 bandes -> deflate)")
        profile.update(compress="DEFLATE")

    print(f"Sortie {W}x{H} px, {nbands} bandes, compress={profile['compress']}")

    t0 = time.perf_counter()
    with rasterio.open(args.out, "w", **profile) as dst:
        for i, f in enumerate(files, 1):
            with rasterio.open(f) as d:
                db = transform_bounds(d.crs, "EPSG:3857", *d.bounds, densify_pts=21)
                c0 = int(math.floor((db[0] + math.pi * _R) / mpp)) - gx0
                c1 = int(math.ceil((db[2] + math.pi * _R) / mpp)) - gx0
                r0 = int(math.floor((math.pi * _R - db[3]) / mpp)) - gy0
                r1 = int(math.ceil((math.pi * _R - db[1]) / mpp)) - gy0
                wpx, hpx = c1 - c0, r1 - r0
                win_transform = Affine(
                    mpp, 0.0, -math.pi * _R + (gx0 + c0) * mpp,
                    0.0, -mpp, math.pi * _R - (gy0 + r0) * mpp)
                src_arr = d.read()  # décode la dalle UNE fois
                dest = np.zeros((d.count, hpx, wpx), np.uint8)
                for b in range(d.count):
                    reproject(
                        source=src_arr[b], destination=dest[b],
                        src_transform=d.transform, src_crs=d.crs,
                        dst_transform=win_transform, dst_crs="EPSG:3857",
                        resampling=Resampling.bilinear)
                dst.write(dest, window=Window(c0, r0, wpx, hpx))
            el = time.perf_counter() - t0
            eta = el / i * (len(files) - i)
            print(f"  dalle {i}/{len(files)} ({i * 100 // len(files)}%) "
                  f"— écoulé {el / 60:.1f} min, ETA {eta / 60:.1f} min", flush=True)

    print(f"COG écrit : {args.out}. Passez-le à infer_area.py via --ortho.")


if __name__ == "__main__":
    main()
