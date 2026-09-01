"""Reprojection dalle-par-dalle vers EPSG:3857 calée sur la grille z19, + VRT.

Chaque dalle (BD ORTHO, EPSG:2154) est reprojetée dans son PROPRE GeoTIFF 3857
aligné sur la grille tuile z19 mondiale, ce qui permet de paralléliser (aucune
écriture concurrente dans un même fichier) puis d'assembler une VRT légère que
`local_ortho.read_window` lit vite (le WarpedVRT devient quasi-identité).

C'est le remplaçant robuste de `build_cog.py --src <VRT>` : ne JAMAIS reprojeter
une mosaïque entière d'un coup (un `d.read()` du département = des centaines de
Go en RAM → sortie noire).
"""
from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.warp import reproject, transform_bounds, Resampling

_R = 6378137.0

# Un seul `d.read()` au-delà de ce nombre d'octets est presque sûrement une
# mosaïque/VRT passée par erreur (une dalle BD ORTHO 25000² x3 ≈ 1,9 Go).
MAX_SOURCE_BYTES = 8_000_000_000


def zoom_mpp(zoom: int, tile_size: int = 256) -> float:
    """Mètres par pixel de la grille tuile Web Mercator au niveau `zoom`."""
    return (2 * math.pi * _R) / (tile_size * (2 ** zoom))


def world_grid_window(west: float, south: float, east: float, north: float,
                      mpp: float):
    """Fenêtre (col0, row0, wpx, hpx) + transform pour une emprise 3857, calée
    sur la grille tuile z mondiale (origine -πR / +πR, pas `mpp`)."""
    c0 = int(math.floor((west + math.pi * _R) / mpp))
    r0 = int(math.floor((math.pi * _R - north) / mpp))
    c1 = int(math.ceil((east + math.pi * _R) / mpp))
    r1 = int(math.ceil((math.pi * _R - south) / mpp))
    wpx, hpx = c1 - c0, r1 - r0
    transform = Affine(mpp, 0.0, -math.pi * _R + c0 * mpp,
                       0.0, -mpp, math.pi * _R - r0 * mpp)
    return c0, r0, wpx, hpx, transform


def source_read_bytes(width: int, height: int, count: int) -> int:
    """Octets qu'un `d.read()` complet allouerait (uint8)."""
    return int(width) * int(height) * int(count)


def reproject_dalle(dalle_path, out_path, zoom: int = 19,
                    compress: str = "jpeg") -> Path:
    """Reprojette UNE dalle vers un GeoTIFF EPSG:3857 aligné grille z, tuilé.

    Décode la dalle une seule fois. Refuse une source trop grosse (mosaïque/VRT
    passée par erreur) plutôt que de saturer la RAM.
    """
    out_path = Path(out_path)
    mpp = zoom_mpp(zoom)
    with rasterio.open(dalle_path) as d:
        if source_read_bytes(d.width, d.height, d.count) > MAX_SOURCE_BYTES:
            raise ValueError(
                f"Source trop grande pour un reproject en un bloc "
                f"({d.width}x{d.height}x{d.count}). Passez des dalles "
                f"individuelles, pas une mosaïque/VRT.")
        db = transform_bounds(d.crs, "EPSG:3857", *d.bounds, densify_pts=21)
        c0, r0, wpx, hpx, transform = world_grid_window(db[0], db[1], db[2], db[3], mpp)
        src = d.read()
        dest = np.zeros((d.count, hpx, wpx), np.uint8)
        for b in range(d.count):
            reproject(source=src[b], destination=dest[b],
                      src_transform=d.transform, src_crs=d.crs,
                      dst_transform=transform, dst_crs="EPSG:3857",
                      resampling=Resampling.bilinear)
        profile = dict(driver="GTiff", width=wpx, height=hpx, count=d.count,
                       dtype="uint8", crs="EPSG:3857", transform=transform,
                       tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")
        if compress == "jpeg" and d.count == 3:
            profile.update(compress="JPEG", photometric="YCBCR", jpeg_quality=85)
        else:
            profile.update(compress="DEFLATE")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as o:
            o.write(dest)
    return out_path


def write_vrt(tiles, out_vrt) -> Path:
    """Écrit une VRT mosaïque par-dessus des tuiles de MÊME CRS et résolution,
    non recouvrantes (le cas des tuiles reprojetées ci-dessus)."""
    tiles = [str(t) for t in tiles]
    if not tiles:
        raise ValueError("Aucune tuile à assembler en VRT.")
    with rasterio.open(tiles[0]) as d0:
        res = d0.res[0]
        crs_wkt = d0.crs.to_wkt()
        nbands = d0.count
    minx = miny = 1e18
    maxx = maxy = -1e18
    meta = []  # (path, left, top, width, height)
    for t in tiles:
        with rasterio.open(t) as d:
            b = d.bounds
            meta.append((t, b.left, b.top, d.width, d.height))
            minx, maxx = min(minx, b.left), max(maxx, b.right)
            miny, maxy = min(miny, b.bottom), max(maxy, b.top)
    W = round((maxx - minx) / res)
    H = round((maxy - miny) / res)
    lines = [
        f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}">',
        f'  <SRS>{escape(crs_wkt)}</SRS>',
        f'  <GeoTransform>{minx}, {res}, 0, {maxy}, 0, {-res}</GeoTransform>',
    ]
    for band in range(1, nbands + 1):
        lines.append(f'  <VRTRasterBand dataType="Byte" band="{band}">')
        for path, left, top, w, h in meta:
            xoff = round((left - minx) / res)
            yoff = round((maxy - top) / res)
            src = escape(Path(path).resolve().as_posix())
            lines.append(
                f'    <SimpleSource>'
                f'<SourceFilename relativeToVRT="0">{src}</SourceFilename>'
                f'<SourceBand>{band}</SourceBand>'
                f'<SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>'
                f'<DstRect xOff="{xoff}" yOff="{yoff}" xSize="{w}" ySize="{h}"/>'
                f'</SimpleSource>')
        lines.append('  </VRTRasterBand>')
    lines.append('</VRTDataset>')
    out_vrt = Path(out_vrt)
    out_vrt.write_text("\n".join(lines), encoding="utf-8")
    return out_vrt
