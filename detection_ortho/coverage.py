"""Couverture d'une couche WMTS : où y a-t-il de l'imagerie ?

Sonde des tuiles de faible zoom EN MÉMOIRE (aucune écriture disque) pour écarter,
avant une inférence départementale, les fenêtres où la couche n'a pas de données
(404 ou tuile blanche). Une réponse indéterminée (erreur réseau, contenu illisible)
ne fait jamais écarter de fenêtre : mieux vaut inférer une fenêtre de trop que
d'en perdre une.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import numpy as np
import requests

from detection_ortho.dataset import window_is_blank
from detection_ortho.tiles import USER_AGENT, tile_for_lonlat, tile_url

# Zoom de sonde : une tuile couvre environ 3 km à nos latitudes, soit quelques
# centaines de tuiles pour un département.
PROBE_ZOOM = 13


def fetch_probe_tile(x, y, zoom, layer, session=None, tries=3, pause=1.0):
    """Octets JPEG de la tuile ; None si la couche n'en a pas (HTTP 404).

    Réessaie `tries` fois sur les autres erreurs, puis lève la dernière.
    """
    sess = session or requests.Session()
    last = None
    for attempt in range(max(1, tries)):
        try:
            resp = sess.get(tile_url(x, y, zoom, layer),
                            headers={"User-Agent": USER_AGENT}, timeout=30)
            if getattr(resp, "status_code", 200) == 404:
                return None
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 < max(1, tries):
                time.sleep(pause * (attempt + 1))
    raise last


def probe_tiles(tiles, layer, workers: int = 12, fetch=None) -> dict:
    """{(x, y): True (données) | False (vide) | None (indéterminé)}, en parallèle."""
    tiles = list(tiles)
    if not tiles:
        return {}
    fetch = fetch or partial(fetch_probe_tile, session=requests.Session())

    def one(xy):
        try:
            content = fetch(xy[0], xy[1], PROBE_ZOOM, layer)
        except Exception:  # noqa: BLE001
            return xy, None
        if content is None:
            return xy, False
        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return xy, None
        return xy, not window_is_blank(img)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return dict(pool.map(one, tiles))


def filter_windows_by_coverage(centers, layer, workers: int = 12, fetch=None):
    """Écarte les fenêtres dont la tuile de sonde n'a pas de données.

    Retourne (fenêtres conservées, nombre écartées, nombre de tuiles de sonde
    indéterminées). Une tuile indéterminée garde ses fenêtres.
    """
    keys = [tile_for_lonlat(lon, lat, PROBE_ZOOM) for lon, lat in centers]
    status = probe_tiles(set(keys), layer, workers, fetch)
    kept = [c for c, k in zip(centers, keys) if status[k] is not False]
    n_unknown = sum(1 for v in status.values() if v is None)
    return kept, len(centers) - len(kept), n_unknown
