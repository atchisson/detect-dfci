# detection_ortho/local_ortho.py
"""Lecture locale de la BD ORTHO via rasterio, reprojetée en EPSG:3857 sur la
grille pixel WMTS (Web Mercator) du zoom — pour coller à l'entraînement.

read_window a le MÊME contrat de retour que dataset.assemble_window :
(image BGR uint8 window_px×window_px, origin_gx, origin_gy).
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from detection_ortho.dataset import lonlat_to_global_px

_R = 6378137.0  # rayon Web Mercator


def _mpp(zoom: int, tile_size: int = 256) -> float:
    """Mètres par pixel de la grille Web Mercator au zoom donné."""
    return (2 * math.pi * _R) / (tile_size * (2 ** zoom))


def _source_path(path) -> str:
    """Retourne un chemin ouvrable par rasterio : fichier tel quel, ou VRT
    construit depuis un dossier de dalles (nécessite osgeo.gdal)."""
    p = Path(path)
    if p.is_dir():
        dalles = sorted(glob.glob(str(p / "*.jp2")) + glob.glob(str(p / "*.tif")))
        if not dalles:
            raise FileNotFoundError(f"Aucune dalle .jp2/.tif dans {p}")
        try:
            from osgeo import gdal
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Dossier de dalles fourni mais osgeo/GDAL indisponible. "
                "Construisez un VRT : `gdalbuildvrt ortho.vrt *.jp2` "
                "(ou QGIS > Raster virtuel) et passez ortho.vrt."
            ) from exc
        vrt_path = str(p / "_mosaic.vrt")
        gdal.BuildVRT(vrt_path, dalles)
        return vrt_path
    return str(p)


def open_ortho(path, zoom: int = 19, tile_size: int = 256) -> WarpedVRT:
    """Ouvre l'ortho reprojetée en EPSG:3857 sur la grille pixel du zoom.

    La transform est alignée sur la grille WMTS PM : le pixel (col, row) du VRT
    correspond au pixel global (gx, gy) du zoom. À fermer par l'appelant.
    """
    mpp = _mpp(zoom, tile_size)
    n = tile_size * (2 ** zoom)  # dimension monde en pixels (virtuel)
    transform = Affine(mpp, 0.0, -math.pi * _R, 0.0, -mpp, math.pi * _R)
    src = rasterio.open(_source_path(path))
    vrt = WarpedVRT(
        src, crs="EPSG:3857", transform=transform, width=n, height=n,
        resampling=Resampling.bilinear,
    )
    # WarpedVRT.close() ne ferme pas le dataset source qu'il enveloppe : sans
    # ce correctif, le fichier/handle source reste ouvert (fuite) même après
    # que l'appelant a fermé le VRT comme le docstring le lui promet.
    _orig_close = vrt.close

    def _close_all(*args, **kwargs):
        try:
            _orig_close(*args, **kwargs)
        finally:
            if not src.closed:
                src.close()

    vrt.close = _close_all
    return vrt


def read_window(
    vrt: WarpedVRT, lon: float, lat: float, zoom: int, window_px: int,
    tile_size: int = 256,
) -> tuple[np.ndarray, float, float]:
    """Lit la fenêtre window_px centrée sur (lon, lat). Retourne (BGR, ogx, ogy)."""
    gx, gy = lonlat_to_global_px(lon, lat, zoom, tile_size)
    origin_gx = gx - window_px / 2.0
    origin_gy = gy - window_px / 2.0
    win = Window(int(round(origin_gx)), int(round(origin_gy)), window_px, window_px)
    # WarpedVRT n'autorise pas les lectures boundless (rasterio lève une
    # ValueError). Comme le VRT est dimensionné sur la grille pixel du monde
    # entier au zoom donné (voir open_ortho), la fenêtre est toujours dans
    # les bornes du VRT ; les zones hors de l'emprise réelle de la source
    # sont déjà remplies à 0 par le warp (pas de donnée source = nodata).
    arr = vrt.read(indexes=[1, 2, 3], window=win)  # (3, window_px, window_px), RGB
    img = np.transpose(arr, (1, 2, 0))[:, :, ::-1]  # RGB -> BGR
    return np.ascontiguousarray(img, dtype=np.uint8), float(origin_gx), float(origin_gy)
