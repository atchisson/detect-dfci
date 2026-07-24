"""Maths de tuiles WMTS IGN et conversions pixel <-> géographique.

TileMatrixSet PM = Web Mercator (EPSG:3857), tuiles 256x256, identique au
schéma slippy-map standard : TILEMATRIX=zoom, TILECOL=x, TILEROW=y.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import mercantile
import requests
from pyproj import Transformer

WMTS_BASE = "https://data.geopf.fr/wmts"
LAYER = "ORTHOIMAGERY.ORTHOPHOTOS"

# Transformateurs Web Mercator (EPSG:3857) <-> WGS84 (EPSG:4326).
_TO_MERC = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_TO_WGS = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def tile_for_lonlat(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    t = mercantile.tile(lon, lat, zoom)
    return t.x, t.y


def tile_url(x: int, y: int, zoom: int) -> str:
    return (
        f"{WMTS_BASE}?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
        f"&LAYER={LAYER}&STYLE=normal&TILEMATRIXSET=PM"
        f"&TILEMATRIX={zoom}&TILEROW={y}&TILECOL={x}&FORMAT=image/jpeg"
    )


def pixel_to_lonlat(
    x: int, y: int, zoom: int, px: float, py: float, tile_size: int = 256
) -> tuple[float, float]:
    """Coordonnées géo du pixel (px, py) dans la tuile (x, y) au zoom donné.

    La projection étant linéaire en mercator sur l'étendue d'une tuile, on
    interpole en mètres mercator puis on reprojette en WGS84.
    """
    b = mercantile.xy_bounds(mercantile.Tile(x, y, zoom))  # bornes en mercator
    fx = px / tile_size
    fy = py / tile_size
    mx = b.left + (b.right - b.left) * fx
    my = b.top + (b.bottom - b.top) * fy  # top->bottom quand py augmente
    lon, lat = _TO_WGS.transform(mx, my)
    return lon, lat


def lonlat_to_pixel(
    lon: float, lat: float, zoom: int, tile_size: int = 256
) -> tuple[int, int, float, float]:
    """Tuile contenant (lon, lat) + position pixel dans cette tuile."""
    x, y = tile_for_lonlat(lon, lat, zoom)
    b = mercantile.xy_bounds(mercantile.Tile(x, y, zoom))
    mx, my = _TO_MERC.transform(lon, lat)
    px = (mx - b.left) / (b.right - b.left) * tile_size
    py = (b.top - my) / (b.top - b.bottom) * tile_size
    return x, y, px, py


def tiles_in_bbox(
    west: float, south: float, east: float, north: float, zoom: int
) -> list[tuple[int, int]]:
    """Toutes les tuiles (x, y) couvrant la bbox géographique au zoom donné."""
    return [
        (t.x, t.y)
        for t in mercantile.tiles(west, south, east, north, zooms=zoom)
    ]


def download_tile(
    x: int, y: int, zoom: int, cache_dir: Path, session=None
) -> Path:
    """Télécharge la tuile (x, y, zoom) si absente du cache, retourne son chemin."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{zoom}_{x}_{y}.jpg"
    if path.exists():
        return path
    sess = session or requests.Session()
    resp = sess.get(tile_url(x, y, zoom), timeout=30)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def save_tile_with_marker(
    tile_path: Path, px: float, py: float, out_path: Path
) -> None:
    """Dessine une croix rouge au pixel (px, py) sur la tuile et sauvegarde."""
    img = cv2.imread(str(tile_path))
    if img is None:
        raise FileNotFoundError(tile_path)
    x, y = int(round(px)), int(round(py))
    color = (0, 0, 255)  # BGR rouge
    cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_CROSS,
                   markerSize=20, thickness=2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
