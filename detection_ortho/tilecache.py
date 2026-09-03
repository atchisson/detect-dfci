"""Cache de tuiles WMTS borné : découpe des fenêtres en tranches + purge.

À l'échelle départementale, pré-télécharger toutes les tuiles avant d'inférer
demande ~21 Go (49) et bloque plusieurs heures. On découpe plutôt la liste des
fenêtres en **tranches** contiguës : on télécharge les tuiles d'une tranche, on
l'infère, puis on supprime les tuiles qui ne servent plus. Comme
`windows_over_polygon` parcourt la grille ligne par ligne, une tranche est une
bande géographique et les tuiles se recyclent vite.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from detection_ortho.dataset import window_tiles

# Taille moyenne d'une tuile ortho IGN JPEG 256x256 (mesurée sur le 49 : ~7,8 ko).
# Sert d'amorce au dimensionnement des tranches ; la valeur observée prend le
# relais dès la première purge.
DEFAULT_TILE_BYTES = 8_000

# Nombre de fenêtres max par tranche : borne la durée d'un cycle
# (téléchargement + inférence + purge) même si le budget disque est large.
MAX_WINDOWS_PER_CHUNK = 5_000

_TILE_RE = re.compile(r"^(\d+)_(-?\d+)_(-?\d+)(_[a-z]+)?\.jpg$")


def max_tiles_for_budget(budget_bytes: float, tile_bytes: float) -> int:
    """Nombre de tuiles par tranche tenant dans `budget_bytes`.

    Deux tranches cohabitent sur le disque (celle en cours d'inférence et celle
    pré-téléchargée en parallèle), d'où la division par 2.
    """
    tile_bytes = max(1.0, float(tile_bytes))
    return max(1, int(budget_bytes / (2 * tile_bytes)))


def next_chunk(
    centers: list[tuple[float, float]], start: int, zoom: int, window_px: int,
    max_tiles: int, max_windows: int = MAX_WINDOWS_PER_CHUNK,
    tile_size: int = 256,
) -> tuple[list[tuple[float, float]], set[tuple[int, int]], int]:
    """Tranche de fenêtres à partir de `start`, bornée en tuiles et en fenêtres.

    Retourne (fenêtres de la tranche, tuiles (x, y) qu'elles couvrent, index de
    reprise). Tranche vide (et index inchangé) quand `start` est en fin de liste.
    """
    tiles: set[tuple[int, int]] = set()
    end = start
    while (end < len(centers) and len(tiles) < max_tiles
           and end - start < max_windows):
        lon, lat = centers[end]
        t, _, _ = window_tiles(lon, lat, zoom, window_px, tile_size)
        tiles.update(t)
        end += 1
    return centers[start:end], tiles, end


def purge_cache(
    cache_dir, keep: set[tuple[int, int]], zoom: int,
) -> tuple[int, int, int]:
    """Supprime du cache les tuiles du zoom donné absentes de `keep`.

    Les fichiers ne suivant pas le motif `<zoom>_<x>_<y>.jpg` (ou d'un autre
    zoom) sont laissés intacts. Retourne (supprimées, restantes, octets restants).
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return 0, 0, 0
    deleted = kept = kept_bytes = 0
    with os.scandir(cache_dir) as it:
        for entry in it:
            m = _TILE_RE.match(entry.name)
            if m is None or int(m.group(1)) != zoom:
                continue
            xy = (int(m.group(2)), int(m.group(3)))
            if xy in keep:
                kept += 1
                kept_bytes += entry.stat().st_size
                continue
            try:
                os.remove(entry.path)
            except OSError:  # fichier verrouillé/déjà supprimé : on réessaiera
                kept += 1
                kept_bytes += entry.stat().st_size
                continue
            deleted += 1
    return deleted, kept, kept_bytes
