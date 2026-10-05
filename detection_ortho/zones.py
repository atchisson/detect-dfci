"""Zones interdites à la prise de vue aérienne (ZIPTV) et à la captation
aérienne de données (ZICAD).

L'ortho IGN floute ces sites : toute détection y est un faux positif. On écarte
donc les fenêtres d'inférence qui touchent ces zones, avant téléchargement.

Les fichiers sont téléchargés à l'exécution et gardés en cache (non versionnés :
la licence de ZICAD n'est pas précisée).
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from detection_ortho.dataset import global_px_to_lonlat, lonlat_to_global_px

ZIPTV_URL = ("https://static.data.gouv.fr/resources/"
             "zones-interdites-a-la-prise-de-vue-aerienne/20181007-134434/"
             "2017-10.geojson")
ZICAD_URL = ("https://data.geopf.fr/annexes/ressources/documentation/"
             "Arrete_ZICAD_10-2024.kml")
ZIPTV_FILE = "ziptv.geojson"
ZICAD_FILE = "zicad.kml"


def _polygons(geom) -> list[Polygon]:
    """Polygones contenus dans une géométrie (points et lignes ignorés)."""
    if isinstance(geom, Polygon):
        return [geom]
    return [p for g in getattr(geom, "geoms", []) for p in _polygons(g)]


def parse_geojson_zones(data: dict) -> list[Polygon]:
    out: list[Polygon] = []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if geom:
            out.extend(_polygons(shape(geom)))
    return out


def _local(tag: str) -> str:
    """Nom de balise sans espace de noms XML."""
    return tag.rsplit("}", 1)[-1]


def _parse_coords(text: str) -> list[tuple[float, float]]:
    pts = []
    for tok in text.split():
        parts = tok.split(",")  # lon,lat[,alt] — sans espace dans un triplet
        pts.append((float(parts[0]), float(parts[1])))
    return pts


def parse_kml_zones(data: bytes | str) -> list[Polygon]:
    """Anneaux extérieurs de tous les <Polygon> d'un KML (trous ignorés : on
    préfère écarter trop que pas assez)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    root = ET.fromstring(data)
    out: list[Polygon] = []
    for poly in root.iter():
        if _local(poly.tag) != "Polygon":
            continue
        for outer in poly:
            if _local(outer.tag) != "outerBoundaryIs":
                continue
            for el in outer.iter():
                if _local(el.tag) == "coordinates":
                    pts = _parse_coords(el.text or "")
                    if len(pts) >= 3:
                        out.append(Polygon(pts))
    return out


def window_footprint(lon: float, lat: float, zoom: int, window_px: int) -> Polygon:
    """Emprise WGS84 d'une fenêtre window_px centrée sur (lon, lat).

    Même géométrie que windows_over_polygon / window_tiles : carré en pixels
    Web Mercator, donc rectangle aligné sur lon/lat.
    """
    gx, gy = lonlat_to_global_px(lon, lat, zoom)
    half = window_px / 2
    west, north = global_px_to_lonlat(gx - half, gy - half, zoom)
    east, south = global_px_to_lonlat(gx + half, gy + half, zoom)
    return box(west, south, east, north)


def filter_windows(centers, zones, zoom: int, window_px: int):
    """(centres gardés, nombre écartés) : écarte les fenêtres qui touchent `zones`."""
    if zones.is_empty or not centers:
        return list(centers), 0
    prepared = prep(zones)
    gardees = [c for c in centers
               if not prepared.intersects(window_footprint(c[0], c[1], zoom, window_px))]
    return gardees, len(centers) - len(gardees)


def _fetch(url: str, path: Path, refresh: bool, session) -> bytes:
    """Contenu du fichier : cache si présent (sauf refresh), sinon téléchargement."""
    if path.exists() and not refresh:
        return path.read_bytes()
    try:
        resp = (session or requests).get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        if path.exists():
            print(f"  Téléchargement impossible ({exc}) : cache conservé "
                  f"{path}.", file=sys.stderr)
            return path.read_bytes()
        raise RuntimeError(
            f"Impossible de télécharger les zones interdites ({url}) et aucun "
            f"cache dans {path.parent} : {exc}. Relancez avec "
            f"--no-skip-restricted-zones pour inférer sans filtre.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return resp.content


def load_zones(cache_dir: Path, refresh: bool = False, session=None):
    """Union ZIPTV + ZICAD (géométrie shapely WGS84)."""
    cache_dir = Path(cache_dir)
    ziptv = _fetch(ZIPTV_URL, cache_dir / ZIPTV_FILE, refresh, session)
    zicad = _fetch(ZICAD_URL, cache_dir / ZICAD_FILE, refresh, session)
    import json
    polys = parse_geojson_zones(json.loads(ziptv)) + parse_kml_zones(zicad)
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    return unary_union(polys)


def apply_zone_filter(centers, cache_dir: Path, zoom: int, window_px: int,
                      refresh: bool = False, session=None):
    """Charge les zones puis filtre les centres : (gardés, nombre écartés)."""
    if not centers:
        return [], 0
    zones = load_zones(cache_dir, refresh=refresh, session=session)
    return filter_windows(centers, zones, zoom, window_px)
