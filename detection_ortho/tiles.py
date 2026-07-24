"""Maths de tuiles WMTS IGN et conversions pixel <-> géographique.

TileMatrixSet PM = Web Mercator (EPSG:3857), tuiles 256x256, identique au
schéma slippy-map standard : TILEMATRIX=zoom, TILECOL=x, TILEROW=y.
"""
from __future__ import annotations

import mercantile
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
