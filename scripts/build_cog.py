# scripts/build_cog.py
"""Pré-reprojette l'ortho en GeoTIFF tuilé EPSG:3857 aligné sur la grille z19.

Fait UNE fois le décodage JP2 + la reprojection (via le WarpedVRT de
local_ortho), en recopiant bloc par bloc. Ensuite read_window lit ce raster
déjà en 3857 et tuilé -> lectures rapides. Aucun outil GDAL externe requis.

Usage:
    python scripts/build_cog.py --src ortho37.vrt --out ortho37_3857.tif
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rasterio
from rasterio import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from detection_ortho.local_ortho import open_ortho

_R = 6378137.0


def _mpp(zoom: int, tile_size: int = 256) -> float:
    return (2 * math.pi * _R) / (tile_size * (2 ** zoom))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True,
                    help="ortho source (VRT/raster, ex. ortho37.vrt)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compress", choices=["jpeg", "deflate"], default="jpeg")
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--blocksize", type=int, default=512)
    args = ap.parse_args()

    mpp = _mpp(args.zoom)

    # 1. Bornes source -> EPSG:3857 -> plage de pixels globaux z19.
    with rasterio.open(args.src) as s:
        nbands = s.count
        minx, miny, maxx, maxy = transform_bounds(
            s.crs, "EPSG:3857", *s.bounds, densify_pts=21)
    gx0 = int(math.floor((minx + math.pi * _R) / mpp))
    gx1 = int(math.ceil((maxx + math.pi * _R) / mpp))
    gy0 = int(math.floor((math.pi * _R - maxy) / mpp))   # maxy -> haut -> gy petit
    gy1 = int(math.ceil((math.pi * _R - miny) / mpp))
    W, H = gx1 - gx0, gy1 - gy0
    transform = Affine(mpp, 0.0, -math.pi * _R + gx0 * mpp,
                       0.0, -mpp, math.pi * _R - gy0 * mpp)

    profile = dict(driver="GTiff", width=W, height=H, count=nbands,
                   dtype="uint8", crs="EPSG:3857", transform=transform,
                   tiled=True, blockxsize=args.blocksize,
                   blockysize=args.blocksize, BIGTIFF="YES")
    if args.compress == "jpeg" and nbands == 3:
        profile.update(compress="JPEG", photometric="YCBCR", jpeg_quality=85)
    else:
        profile.update(compress="DEFLATE")

    print(f"Sortie {W}x{H} px, {nbands} bandes, compress={profile['compress']}")
    n_blocks = ((H + args.blocksize - 1) // args.blocksize) * \
               ((W + args.blocksize - 1) // args.blocksize)
    step = max(1, n_blocks // 50)

    vrt = open_ortho(args.src, zoom=args.zoom)
    try:
        with rasterio.open(args.out, "w", **profile) as dst:
            done = 0
            for r0 in range(0, H, args.blocksize):
                for c0 in range(0, W, args.blocksize):
                    w = min(args.blocksize, W - c0)
                    h = min(args.blocksize, H - r0)
                    data = vrt.read(
                        indexes=list(range(1, nbands + 1)),
                        window=Window(gx0 + c0, gy0 + r0, w, h))
                    dst.write(data, window=Window(c0, r0, w, h))
                    done += 1
                    if done % step == 0 or done == n_blocks:
                        print(f"  build_cog: {done}/{n_blocks} "
                              f"({done * 100 // n_blocks}%)", flush=True)
    finally:
        vrt.close()

    print(f"COG écrit : {args.out}. Passez-le à infer_area.py via --ortho.")


if __name__ == "__main__":
    main()
