"""Zones interdites à la prise de vue aérienne (ZIPTV) et à la captation
aérienne de données (ZICAD).

L'ortho IGN floute ces sites : toute détection y est un faux positif. On écarte
donc les fenêtres d'inférence qui touchent ces zones, avant téléchargement.

Les fichiers sont téléchargés à l'exécution et gardés en cache (non versionnés :
la licence de ZICAD n'est pas précisée).
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.validation import make_valid

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


def _parse_geojson_bytes(data: bytes) -> list[Polygon]:
    return parse_geojson_zones(json.loads(data))


def _parse_checked(nom: str, data: bytes, parse) -> list[Polygon]:
    """Polygones de `data` ; ValueError si illisible ou sans aucun polygone."""
    try:
        polys = parse(data)
    except (ValueError, ET.ParseError) as exc:  # JSONDecodeError est un ValueError
        raise ValueError(f"contenu {nom} illisible ({exc})") from exc
    if not polys:
        raise ValueError(
            f"la source {nom} ne contient aucun polygone : inspectez le fichier "
            f"source (format changé ?)")
    return polys


def _load_source(nom: str, url: str, path: Path, refresh: bool, session,
                 parse) -> list[Polygon]:
    """Polygones d'une source : cache si présent (sauf refresh), sinon
    téléchargement. Le contenu n'est écrit en cache qu'après avoir été validé."""
    if path.exists() and not refresh:
        try:
            return _parse_checked(nom, path.read_bytes(), parse)
        except ValueError as exc:
            raise RuntimeError(
                f"Cache {nom} inutilisable ({path}) : {exc}. Relancez avec "
                f"--refresh-zones pour le retélécharger.") from exc
    try:
        resp = (session or requests).get(url, timeout=60)
        resp.raise_for_status()
        polys = _parse_checked(nom, resp.content, parse)
    except (requests.RequestException, ValueError) as exc:
        if path.exists():
            try:
                polys = _parse_checked(nom, path.read_bytes(), parse)
            except ValueError as exc2:
                raise RuntimeError(
                    f"Cache {nom} inutilisable ({path}) : {exc2}. Relancez avec "
                    f"--refresh-zones.") from exc2
            print(f"  Téléchargement {nom} inutilisable ({exc}) : cache conservé "
                  f"{path}.", file=sys.stderr)
            return polys
        raise RuntimeError(
            f"Impossible d'obtenir les zones interdites {nom} ({url}) et aucun "
            f"cache dans {path.parent} : {exc}. Relancez avec "
            f"--no-skip-restricted-zones pour inférer sans filtre.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return polys


def load_zones(cache_dir: Path, refresh: bool = False, session=None):
    """Union ZIPTV + ZICAD (géométrie shapely WGS84)."""
    cache_dir = Path(cache_dir)
    ziptv = _load_source("ZIPTV", ZIPTV_URL, cache_dir / ZIPTV_FILE, refresh,
                         session, _parse_geojson_bytes)
    zicad = _load_source("ZICAD", ZICAD_URL, cache_dir / ZICAD_FILE, refresh,
                         session, parse_kml_zones)
    print(f"  ZIPTV : {len(ziptv)} polygone(s), ZICAD : {len(zicad)} polygone(s).",
          file=sys.stderr)
    polys = [p if p.is_valid else make_valid(p) for p in ziptv + zicad]
    return unary_union(polys)


def apply_zone_filter(centers, cache_dir: Path, zoom: int, window_px: int,
                      refresh: bool = False, session=None):
    """Charge les zones puis filtre les centres : (gardés, nombre écartés)."""
    if not centers:
        return [], 0
    zones = load_zones(cache_dir, refresh=refresh, session=session)
    return filter_windows(centers, zones, zoom, window_px)
