"""Couverture d'une couche WMTS : où y a-t-il de l'imagerie ?

Sonde des tuiles de faible zoom EN MÉMOIRE (aucune écriture disque) pour écarter,
avant une inférence départementale, les fenêtres où la couche n'a pas de données
(404 ou tuile blanche). Une réponse indéterminée (erreur réseau, contenu illisible)
ne fait jamais écarter de fenêtre : mieux vaut inférer une fenêtre de trop que
d'en perdre une.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import requests

from detection_ortho.dataset import window_is_blank
from detection_ortho.tiles import USER_AGENT, tile_for_lonlat, tile_url

# Zoom de sonde : une tuile couvre environ 3 km à nos latitudes, soit quelques
# centaines de tuiles pour un département.
PROBE_ZOOM = 13


def fetch_probe_tile(x, y, zoom, layer, session=None, tries=3, pause=1.0,
                     not_found_tries=2):
    """Octets JPEG de la tuile ; None si la couche n'en a pas (HTTP 404).

    Un 404 est confirmé : il faut `not_found_tries` réponses 404 d'affilée (des 404
    transitoires existent, et un faux « vide » écarterait des fenêtres à tort).
    Réessaie `tries` fois sur les autres erreurs, puis lève la dernière.
    """
    sess = session or requests.Session()
    last = None
    n404 = 0
    n = max(1, tries, not_found_tries)
    for attempt in range(n):
        try:
            resp = sess.get(tile_url(x, y, zoom, layer),
                            headers={"User-Agent": USER_AGENT}, timeout=30)
            if getattr(resp, "status_code", 200) == 404:
                n404 += 1
                if n404 >= max(1, not_found_tries):
                    return None
                time.sleep(pause)
                continue
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 < n:
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


def probe_tile_count(centers) -> int:
    """Nombre de tuiles de sonde distinctes où tombent ces centres."""
    return len({tile_for_lonlat(lon, lat, PROBE_ZOOM) for lon, lat in centers})


def _load_probe_cache(path, layer) -> dict:
    """Statuts mémorisés {(x, y): True|False|None} ; {} si absent/autre couche/illisible."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("layer") != layer or data.get("zoom") != PROBE_ZOOM:
            return {}
        return {(int(x), int(y)): s for x, y, s in data["tiles"]}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _save_probe_cache(path, layer, status) -> None:
    """Écriture atomique (fichier .tmp puis os.replace) ; aucune tuile, juste du JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"layer": layer, "zoom": PROBE_ZOOM,
            "tiles": [[x, y, s] for (x, y), s in sorted(status.items())]}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def filter_windows_by_coverage(centers, layer, workers: int = 12, fetch=None,
                               cache_path=None):
    """Écarte les fenêtres dont la tuile de sonde n'a pas de données.

    Retourne (fenêtres conservées, nombre écartées, nombre de tuiles de sonde
    indéterminées). Une tuile indéterminée garde ses fenêtres.

    `cache_path` : fichier JSON où mémoriser les statuts sondés, pour qu'une reprise
    retrouve exactement la même grille de fenêtres (seules les tuiles absentes du
    fichier sont sondées).
    """
    keys = [tile_for_lonlat(lon, lat, PROBE_ZOOM) for lon, lat in centers]
    wanted = set(keys)
    stored = _load_probe_cache(cache_path, layer) if cache_path else {}
    status = {k: stored[k] for k in wanted if k in stored}
    fresh = probe_tiles(wanted - set(status), layer, workers, fetch)
    status.update(fresh)
    if cache_path and fresh:
        _save_probe_cache(cache_path, layer, {**stored, **status})
    kept = [c for c, k in zip(centers, keys) if status[k] is not False]
    n_unknown = sum(1 for v in status.values() if v is None)
    return kept, len(centers) - len(kept), n_unknown
