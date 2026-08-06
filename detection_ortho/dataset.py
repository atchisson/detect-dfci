"""Génération d'un dataset YOLO de citernes depuis OSM + ortho IGN.

Maths pures de boîtes/fenêtres (testables sans I/O) puis assemblage des
imagettes depuis les tuiles et écriture au format YOLO.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from detection_ortho.tiles import lonlat_to_pixel, download_tile, LAYER

_M_PER_DEG_LAT = 111320.0


def lonlat_to_global_px(
    lon: float, lat: float, zoom: int, tile_size: int = 256
) -> tuple[float, float]:
    """Pixel absolu (global) dans la grille slippy au zoom donné."""
    x, y, px, py = lonlat_to_pixel(lon, lat, zoom, tile_size)
    return x * tile_size + px, y * tile_size + py


def polygon_bounds(geometry: list[dict]) -> tuple[float, float, float, float]:
    """BBox géo (west, south, east, north) d'une liste de sommets {lon,lat}."""
    lons = [p["lon"] for p in geometry]
    lats = [p["lat"] for p in geometry]
    return min(lons), min(lats), max(lons), max(lats)


def fixed_box_geo(
    lon: float, lat: float, size_m: float
) -> tuple[float, float, float, float]:
    """BBox géo carrée de côté size_m centrée sur (lon, lat)."""
    half = size_m / 2.0
    dlat = half / _M_PER_DEG_LAT
    dlon = half / (_M_PER_DEG_LAT * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def geo_bbox_to_pixel_bbox(
    bbox_geo: tuple[float, float, float, float],
    origin_gx: float, origin_gy: float, zoom: int, window_px: int,
    tile_size: int = 256,
) -> tuple[float, float, float, float]:
    """Convertit une bbox géo en bbox pixel (x0,y0,x1,y1) dans la fenêtre.

    origin_gx/origin_gy : pixel absolu du coin haut-gauche de la fenêtre.
    Le nord (lat max) correspond au haut de l'image (y plus petit).
    """
    w, s, e, n = bbox_geo
    gx0, gy0 = lonlat_to_global_px(w, n, zoom, tile_size)  # haut-gauche
    gx1, gy1 = lonlat_to_global_px(e, s, zoom, tile_size)  # bas-droite
    x0 = max(0.0, min(window_px, gx0 - origin_gx))
    y0 = max(0.0, min(window_px, gy0 - origin_gy))
    x1 = max(0.0, min(window_px, gx1 - origin_gx))
    y1 = max(0.0, min(window_px, gy1 - origin_gy))
    return x0, y0, x1, y1


def to_yolo_label(
    px_bbox: tuple[float, float, float, float], window_px: int, cls: int = 0
) -> str | None:
    """Ligne YOLO `cls cx cy w h` normalisée, ou None si la boîte est vide."""
    x0, y0, x1, y1 = px_bbox
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return None
    cx = (x0 + x1) / 2 / window_px
    cy = (y0 + y1) / 2 / window_px
    return f"{cls} {cx:.6f} {cy:.6f} {w / window_px:.6f} {h / window_px:.6f}"


def window_tiles(
    center_lon: float, center_lat: float, zoom: int, window_px: int,
    tile_size: int = 256,
) -> tuple[list[tuple[int, int]], float, float]:
    """Tuiles couvrant une fenêtre window_px centrée sur le point.

    Retourne (liste de (x,y), origin_gx, origin_gy) où origin est le pixel
    absolu du coin haut-gauche de la fenêtre.
    """
    gx, gy = lonlat_to_global_px(center_lon, center_lat, zoom, tile_size)
    origin_gx = gx - window_px / 2
    origin_gy = gy - window_px / 2
    tx_min = int(origin_gx // tile_size)
    ty_min = int(origin_gy // tile_size)
    tx_max = int((origin_gx + window_px - 1) // tile_size)
    ty_max = int((origin_gy + window_px - 1) // tile_size)
    tiles = [(x, y)
             for x in range(tx_min, tx_max + 1)
             for y in range(ty_min, ty_max + 1)]
    return tiles, origin_gx, origin_gy


def assemble_window(
    center_lon: float, center_lat: float, zoom: int, window_px: int,
    cache_dir, session=None, tile_size: int = 256, layer=LAYER,
) -> tuple[np.ndarray, float, float]:
    """Assemble une mosaïque de tuiles et en extrait la fenêtre centrée."""
    tiles, origin_gx, origin_gy = window_tiles(
        center_lon, center_lat, zoom, window_px, tile_size)
    xs = [t[0] for t in tiles]
    ys = [t[1] for t in tiles]
    tx_min, ty_min = min(xs), min(ys)
    mosaic_h = (max(ys) - ty_min + 1) * tile_size
    mosaic_w = (max(xs) - tx_min + 1) * tile_size
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), np.uint8)
    for (x, y) in tiles:
        path = download_tile(x, y, zoom, cache_dir, session=session, layer=layer)
        tile = cv2.imread(str(path))
        if tile is None:
            print(f"  tuile illisible: {path}", file=sys.stderr)
            continue
        oy = (y - ty_min) * tile_size
        ox = (x - tx_min) * tile_size
        mosaic[oy:oy + tile_size, ox:ox + tile_size] = tile
    # coin haut-gauche de la fenêtre dans le repère mosaïque
    cx = int(round(origin_gx - tx_min * tile_size))
    cy = int(round(origin_gy - ty_min * tile_size))
    # borne pour garantir une fenêtre exactement window_px x window_px même
    # si l'arrondi place la fenêtre à cheval sur le bord de la mosaïque.
    cx = max(0, min(cx, mosaic_w - window_px))
    cy = max(0, min(cy, mosaic_h - window_px))
    window = mosaic[cy:cy + window_px, cx:cx + window_px]
    assert window.shape[:2] == (window_px, window_px), (
        f"fenêtre {window.shape[:2]} != ({window_px}, {window_px})")
    return window, origin_gx, origin_gy


def write_chip(image, label_lines, images_dir, labels_dir, name: str) -> None:
    """Écrit l'imagette .jpg et le label .txt (liste vide = négatif)."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(images_dir / f"{name}.jpg"), image)
    (labels_dir / f"{name}.txt").write_text(
        "\n".join(label_lines), encoding="utf-8")


DEFAULT_BOX_M = 13.0  # médiane observée des citernes dpt 37


def element_to_box(element: dict, default_box_m: float = DEFAULT_BOX_M) -> dict:
    """Boîte géo d'un élément OSM : polygone (way) ou boîte fixe (node)."""
    if element["type"] == "way":
        bbox = polygon_bounds(element["geometry"])
        lon = (bbox[0] + bbox[2]) / 2
        lat = (bbox[1] + bbox[3]) / 2
    else:
        lon, lat = element["lon"], element["lat"]
        bbox = fixed_box_geo(lon, lat, default_box_m)
    return {"lon": lon, "lat": lat, "bbox_geo": bbox, "tags": element.get("tags", {})}


def split_indices(
    n: int, seed: int = 0, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> dict:
    """Partition déterministe des indices 0..n-1 en train/val/test."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": sorted(idx[:n_train]),
        "val": sorted(idx[n_train:n_train + n_val]),
        "test": sorted(idx[n_train + n_val:]),
    }


def spatial_split_indices(
    points, cell_deg: float = 0.05, seed: int = 0,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict:
    """Partition spatiale : les points d'une même cellule de grille restent groupés.

    `points` = liste de (lon, lat) alignée sur les indices 0..n-1. Les cellules
    (floor(lon/cell_deg), floor(lat/cell_deg)) sont mélangées de façon
    déterministe puis affectées à train→val→test par cellules entières jusqu'à
    approcher `ratios` (comptés en nombre de points).
    """
    cells: dict = {}
    for i, (lon, lat) in enumerate(points):
        key = (math.floor(lon / cell_deg), math.floor(lat / cell_deg))
        cells.setdefault(key, []).append(i)
    keys = sorted(cells)
    random.Random(seed).shuffle(keys)
    total = len(points)
    t_train, t_val = total * ratios[0], total * ratios[1]
    train, val, test = [], [], []
    count = 0
    for k in keys:
        members = cells[k]
        if count < t_train:
            train += members
        elif count < t_train + t_val:
            val += members
        else:
            test += members
        count += len(members)
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def write_data_yaml(root, path) -> None:
    """Écrit un data.yaml Ultralytics (1 classe) pointant vers root."""
    root = Path(root).resolve()
    content = (
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: citerne\n"
    )
    Path(path).write_text(content, encoding="utf-8")


def global_px_to_lonlat(
    gx: float, gy: float, zoom: int, tile_size: int = 256
) -> tuple[float, float]:
    """Inverse de lonlat_to_global_px : pixel absolu -> (lon, lat)."""
    from detection_ortho.tiles import pixel_to_lonlat
    x, px = divmod(gx, tile_size)
    y, py = divmod(gy, tile_size)
    return pixel_to_lonlat(int(x), int(y), zoom, px, py, tile_size)


def parse_verdicts(lines: list[str]) -> list[dict]:
    """Parse les lignes d'un CSV de revue `index,lat,lon,score,verdict`.

    Ne conserve que les verdicts `vrai`/`faux` ; ignore l'en-tête, `skip`,
    `non_revu` et les lignes malformées. Retourne {lon, lat, verdict}.
    """
    out: list[dict] = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        _idx, lat, lon, _score, verdict = parts[:5]
        if verdict not in ("vrai", "faux"):
            continue
        try:
            out.append({"lon": float(lon), "lat": float(lat), "verdict": verdict})
        except ValueError:
            continue
    return out


def compose_rgn(rgb_bgr, irc_bgr):
    """Image BGR où le bleu est remplacé par le NIR (= canal rouge de l'IRC).

    Donne un proxy 3 canaux [R, G, NIR] pour tester l'apport du NIR sans
    plomberie 4-canaux.
    """
    out = rgb_bgr.copy()
    out[:, :, 0] = irc_bgr[:, :, 2]
    return out
