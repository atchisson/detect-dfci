# Jalon 3 — Inférence + OSM + MapRoulette : Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inférer le détecteur YOLO sur Tours Métropole (emprise = polygone OSM), comparer à OSM, et générer un fichier de challenge MapRoulette des citernes candidates — sans aucun upload automatique.

**Architecture :** Réutilise l'existant (`tiles`, `dataset.assemble_window`, `geo.dedup_points`, `compare.match_detections`, `geojson_io`, `osm.fetch_citernes`, téléchargement parallèle du Jalon 2). Ajoute la logique pure d'inférence (fenêtrage sur polygone, conversions pixel↔géo) + un module MapRoulette, et deux scripts d'orchestration.

**Tech Stack :** Python 3.12, ultralytics, shapely, opencv-python, numpy, requests, pytest.

## Global Constraints

- Python **3.12** via le venv : interpréteur **`.venv/Scripts/python`** (bare `python` = 3.14). Ne PAS utiliser `python` nu.
- Code sous `detection_ortho/`, scripts sous `scripts/`, tests sous `tests/`.
- Code réseau/modèle isolé ; **aucun test ne fait d'appel réseau réel ni ne charge de poids** (fixtures / fonctions pures).
- Coordonnées ordre **(lon, lat)** partout. Zoom **19**, tuiles **256**, fenêtre **640** px. Seuil de confiance défaut **0,4**.
- **Aucun upload/publication MapRoulette** : le code écrit UNIQUEMENT des fichiers. Aucun appel à une API MapRoulette.
- User-Agent déjà en place (`osm.USER_AGENT`, `tiles.USER_AGENT`). Retry Overpass dans les scripts.
- Les étapes **[RÉSEAU/MODÈLE — DIFFÉRÉ]** ne sont PAS exécutées par l'implémenteur : écrire le code, vérifier la syntaxe, committer, signaler l'étape différée.
- Commits fréquents, format `type: description`. `git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit` si l'identité n'est pas configurée.

---

## File Structure

- `detection_ortho/dataset.py` — **modifié** : ajout `global_px_to_lonlat` (inverse de `lonlat_to_global_px`).
- `detection_ortho/osm.py` — **modifié** : ajout `build_boundary_query`, `parse_relation_ways`, `fetch_relation_ways`.
- `detection_ortho/infer.py` — **nouveau** : `ways_to_polygon`, `windows_over_polygon`, `boxes_to_points` (purs).
- `detection_ortho/maproulette.py` — **nouveau** : `to_maproulette_tasks` (pur).
- `scripts/infer_area.py` — **nouveau** : orchestration inférence → OSM → livrables.
- `scripts/export_maproulette.py` — **nouveau** : `detected_only.geojson` → fichier de tâches.
- `README.md` — **modifié** : section Jalon 3.
- `tests/` — un fichier par module de logique pure.

---

## Task 1: Conversion pixel global → géographique

**Files:**
- Modify: `detection_ortho/dataset.py`
- Test: `tests/test_global_px_to_lonlat.py`

**Interfaces:**
- Consumes: `tiles.pixel_to_lonlat`.
- Produces: `global_px_to_lonlat(gx: float, gy: float, zoom: int, tile_size: int = 256) -> tuple[float, float]` — inverse de `lonlat_to_global_px`.

- [ ] **Step 1: Écrire le test (aller-retour)**

```python
# tests/test_global_px_to_lonlat.py
from detection_ortho.dataset import lonlat_to_global_px, global_px_to_lonlat


def test_roundtrip_global_px():
    lon, lat, z = 0.6531, 47.3305, 19
    gx, gy = lonlat_to_global_px(lon, lat, z)
    lon2, lat2 = global_px_to_lonlat(gx, gy, z)
    assert abs(lon - lon2) < 1e-6
    assert abs(lat - lat2) < 1e-6


def test_known_tile_origin():
    # Le pixel global (x*256, y*256) doit retomber sur le coin NO de la tuile.
    import mercantile
    from detection_ortho.tiles import pixel_to_lonlat
    x, y, z = 264000, 180000, 19
    lon, lat = global_px_to_lonlat(x * 256, y * 256, z)
    lon0, lat0 = pixel_to_lonlat(x, y, z, 0, 0)
    assert abs(lon - lon0) < 1e-9 and abs(lat - lat0) < 1e-9
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_global_px_to_lonlat.py -q`
Expected: FAIL avec `ImportError` sur `global_px_to_lonlat`.

- [ ] **Step 3: Ajouter la fonction à `dataset.py`**

Ajouter à la fin de `detection_ortho/dataset.py` :

```python
def global_px_to_lonlat(
    gx: float, gy: float, zoom: int, tile_size: int = 256
) -> tuple[float, float]:
    """Inverse de lonlat_to_global_px : pixel absolu -> (lon, lat)."""
    from detection_ortho.tiles import pixel_to_lonlat
    x, px = divmod(gx, tile_size)
    y, py = divmod(gy, tile_size)
    return pixel_to_lonlat(int(x), int(y), zoom, px, py, tile_size)
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_global_px_to_lonlat.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/dataset.py tests/test_global_px_to_lonlat.py
git commit -m "feat: global_px_to_lonlat (inverse de lonlat_to_global_px)"
```

---

## Task 2: Récupération de la géométrie d'une relation OSM

**Files:**
- Modify: `detection_ortho/osm.py`
- Test: `tests/test_osm_boundary.py`

**Interfaces:**
- Consumes: `requests`, `OVERPASS_URL`, `USER_AGENT` (déjà dans `osm.py`).
- Produces:
  - `build_boundary_query(name: str) -> str`.
  - `parse_relation_ways(data: dict) -> list[list[dict]]` → pour chaque way membre, sa liste de sommets `{lon,lat}`.
  - `fetch_relation_ways(name: str, session=None) -> list[list[dict]]` (I/O).

- [ ] **Step 1: Écrire les tests (requête + parsing), réseau mocké**

```python
# tests/test_osm_boundary.py
from detection_ortho.osm import (
    build_boundary_query, parse_relation_ways, fetch_relation_ways,
)


def test_boundary_query_targets_relation_with_geom():
    q = build_boundary_query("Tours Métropole Val de Loire")
    assert 'relation' in q
    assert '"name"="Tours Métropole Val de Loire"' in q
    assert '"boundary"="administrative"' in q
    assert "out geom;" in q


def test_parse_relation_ways_extracts_member_geometries():
    data = {"elements": [{
        "type": "relation",
        "members": [
            {"type": "way", "role": "outer",
             "geometry": [{"lon": 0.0, "lat": 0.0}, {"lon": 1.0, "lat": 0.0}]},
            {"type": "way", "role": "outer",
             "geometry": [{"lon": 1.0, "lat": 0.0}, {"lon": 0.0, "lat": 0.0}]},
            {"type": "node", "role": "admin_centre", "lon": 0.5, "lat": 0.5},
        ],
    }]}
    ways = parse_relation_ways(data)
    assert len(ways) == 2
    assert ways[0][0] == {"lon": 0.0, "lat": 0.0}


def test_fetch_relation_ways_uses_session():
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"elements": [{"type": "relation", "members": [
                {"type": "way", "geometry": [{"lon": 0.0, "lat": 0.0}]},
            ]}]}

    class FakeSession:
        def post(self, url, data, headers=None, timeout=180):
            return FakeResp()

    ways = fetch_relation_ways("X", session=FakeSession())
    assert len(ways) == 1
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_osm_boundary.py -q`
Expected: FAIL avec `ImportError`.

- [ ] **Step 3: Étendre `osm.py`**

Ajouter à la fin de `detection_ortho/osm.py` :

```python
def build_boundary_query(name: str) -> str:
    """Requête Overpass : relation administrative `name` avec géométrie."""
    return (
        f'[out:json][timeout:180];\n'
        f'relation["name"="{name}"]["boundary"="administrative"];\n'
        f'out geom;'
    )


def parse_relation_ways(data: dict) -> list[list[dict]]:
    """Liste des géométries (sommets {lon,lat}) des ways membres des relations."""
    ways: list[list[dict]] = []
    for el in data.get("elements", []):
        if el.get("type") != "relation":
            continue
        for m in el.get("members", []):
            if m.get("type") == "way" and m.get("geometry"):
                ways.append(m["geometry"])
    return ways


def fetch_relation_ways(name: str, session=None) -> list[list[dict]]:
    """Récupère les ways membres de la relation administrative `name`."""
    sess = session or requests.Session()
    resp = sess.post(
        OVERPASS_URL,
        data=build_boundary_query(name),
        headers={"User-Agent": USER_AGENT},
        timeout=180,
    )
    resp.raise_for_status()
    return parse_relation_ways(resp.json())
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_osm_boundary.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/osm.py tests/test_osm_boundary.py
git commit -m "feat: récupération de la géométrie d'une relation admin OSM"
```

---

## Task 3: Polygone d'emprise et fenêtrage glissant

**Files:**
- Create: `detection_ortho/infer.py`
- Test: `tests/test_infer_windows.py`

**Interfaces:**
- Consumes: `shapely`, `dataset.lonlat_to_global_px`, `dataset.global_px_to_lonlat` (Task 1).
- Produces:
  - `ways_to_polygon(ways: list[list[dict]])` → un `shapely.geometry.Polygon`/`MultiPolygon` (le plus grand assemblé).
  - `windows_over_polygon(polygon, zoom: int, window_px: int, overlap: float) -> list[tuple[float, float]]` → centres `(lon, lat)` des fenêtres dont le centre est **dans** le polygone.

- [ ] **Step 1: Écrire les tests**

```python
# tests/test_infer_windows.py
from shapely.geometry import Point
from detection_ortho.infer import ways_to_polygon, windows_over_polygon


def _square_ways():
    # Carré [0,0]-[0.02,0.02] en deux ways (deux moitiés du contour).
    return [
        [{"lon": 0.0, "lat": 0.0}, {"lon": 0.02, "lat": 0.0}, {"lon": 0.02, "lat": 0.02}],
        [{"lon": 0.02, "lat": 0.02}, {"lon": 0.0, "lat": 0.02}, {"lon": 0.0, "lat": 0.0}],
    ]


def test_ways_to_polygon_builds_square():
    poly = ways_to_polygon(_square_ways())
    assert poly.area > 0
    assert poly.contains(Point(0.01, 0.01))       # centre dedans
    assert not poly.contains(Point(0.05, 0.05))    # loin dehors


def test_windows_cover_polygon_and_stay_inside():
    poly = ways_to_polygon(_square_ways())
    centers = windows_over_polygon(poly, zoom=19, window_px=640, overlap=0.2)
    assert len(centers) > 0
    # tous les centres sont dans le polygone
    for lon, lat in centers:
        assert poly.contains(Point(lon, lat))


def test_windows_empty_for_tiny_polygon_far_away():
    # Un polygone minuscule peut ne contenir aucun centre de grille : au moins
    # la fonction ne plante pas et retourne une liste.
    poly = ways_to_polygon(_square_ways())
    centers = windows_over_polygon(poly, zoom=19, window_px=640, overlap=0.0)
    assert isinstance(centers, list)
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_infer_windows.py -q`
Expected: FAIL avec `ImportError` sur `detection_ortho.infer`.

- [ ] **Step 3: Créer `infer.py` (partie polygone + fenêtrage)**

```python
# detection_ortho/infer.py
"""Inférence YOLO à l'échelle d'une emprise : fenêtrage glissant sur polygone,
conversions pixel<->géo. Logique pure séparée de l'I/O modèle/réseau.
"""
from __future__ import annotations

from shapely.geometry import LineString, Point
from shapely.ops import linemerge, polygonize, unary_union

from detection_ortho.dataset import lonlat_to_global_px, global_px_to_lonlat


def ways_to_polygon(ways: list[list[dict]]):
    """Assemble les ways (contours) en un polygone shapely (le plus grand)."""
    lines = [
        LineString([(p["lon"], p["lat"]) for p in way])
        for way in ways if len(way) >= 2
    ]
    merged = linemerge(unary_union(lines))
    polys = list(polygonize(merged))
    if not polys:
        raise ValueError("Aucun polygone assemblable depuis les ways fournis.")
    return max(polys, key=lambda p: p.area)


def windows_over_polygon(
    polygon, zoom: int, window_px: int, overlap: float,
    tile_size: int = 256,
) -> list[tuple[float, float]]:
    """Centres (lon, lat) des fenêtres couvrant le polygone (centre à l'intérieur).

    Grille de pas window_px*(1-overlap) en pixels sur la bbox du polygone.
    """
    west, south, east, north = polygon.bounds  # (minx, miny, maxx, maxy)
    gx0, gy0 = lonlat_to_global_px(west, north, zoom, tile_size)  # coin NO
    gx1, gy1 = lonlat_to_global_px(east, south, zoom, tile_size)  # coin SE
    step = max(1, int(window_px * (1.0 - overlap)))
    centers: list[tuple[float, float]] = []
    gy = gy0
    while gy <= gy1:
        gx = gx0
        while gx <= gx1:
            lon, lat = global_px_to_lonlat(gx, gy, zoom, tile_size)
            if polygon.contains(Point(lon, lat)):
                centers.append((lon, lat))
            gx += step
        gy += step
    return centers
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_infer_windows.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/infer.py tests/test_infer_windows.py
git commit -m "feat: polygone d'emprise + fenêtrage glissant sur polygone"
```

---

## Task 4: Conversion des boîtes détectées en points géographiques

**Files:**
- Modify: `detection_ortho/infer.py`
- Test: `tests/test_boxes_to_points.py`

**Interfaces:**
- Consumes: `dataset.global_px_to_lonlat` (Task 1).
- Produces: `boxes_to_points(boxes: list[tuple[float,float,float]], origin_gx: float, origin_gy: float, zoom: int) -> list[dict]` — chaque `(cx, cy, score)` (centre pixel dans la fenêtre + score) → `{"lon","lat","score"}`.

- [ ] **Step 1: Écrire le test**

```python
# tests/test_boxes_to_points.py
from detection_ortho.dataset import lonlat_to_global_px
from detection_ortho.infer import boxes_to_points


def test_box_center_maps_back_to_lonlat():
    lon, lat, z, win = 0.6531, 47.3305, 19, 640
    gx, gy = lonlat_to_global_px(lon, lat, z)
    origin_gx, origin_gy = gx - win / 2, gy - win / 2  # fenêtre centrée
    # une boîte au centre de la fenêtre (cx=cy=320) doit retomber sur (lon,lat)
    pts = boxes_to_points([(win / 2, win / 2, 0.9)], origin_gx, origin_gy, z)
    assert len(pts) == 1
    assert abs(pts[0]["lon"] - lon) < 1e-5
    assert abs(pts[0]["lat"] - lat) < 1e-5
    assert pts[0]["score"] == 0.9


def test_empty_boxes():
    assert boxes_to_points([], 0, 0, 19) == []
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_boxes_to_points.py -q`
Expected: FAIL avec `ImportError` sur `boxes_to_points`.

- [ ] **Step 3: Ajouter `boxes_to_points` à `infer.py`**

Ajouter à la fin de `detection_ortho/infer.py` :

```python
def boxes_to_points(
    boxes: list[tuple[float, float, float]],
    origin_gx: float, origin_gy: float, zoom: int,
) -> list[dict]:
    """Convertit des centres de boîtes (px dans la fenêtre) + score en points géo.

    origin_gx/origin_gy : pixel absolu du coin haut-gauche de la fenêtre.
    """
    points: list[dict] = []
    for cx, cy, score in boxes:
        lon, lat = global_px_to_lonlat(origin_gx + cx, origin_gy + cy, zoom)
        points.append({"lon": lon, "lat": lat, "score": float(score)})
    return points
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_boxes_to_points.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/infer.py tests/test_boxes_to_points.py
git commit -m "feat: conversion des boîtes détectées en points géographiques"
```

---

## Task 5: Export MapRoulette (génération de fichier)

**Files:**
- Create: `detection_ortho/maproulette.py`
- Test: `tests/test_maproulette.py`

**Interfaces:**
- Consumes: rien (pur).
- Produces: `to_maproulette_tasks(points: list[dict], instruction: str) -> dict` — FeatureCollection GeoJSON, une Feature Point par point, propriétés `{"instruction", "score"}`.

> **Contrainte : aucun appel réseau, aucune API MapRoulette. Génération de structure/fichier uniquement.**

- [ ] **Step 1: Écrire le test**

```python
# tests/test_maproulette.py
from detection_ortho.maproulette import to_maproulette_tasks


def test_tasks_structure():
    pts = [{"lon": 0.65, "lat": 47.33, "score": 0.82}]
    fc = to_maproulette_tasks(pts, "Vérifiez cette citerne.")
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    feat = fc["features"][0]
    assert feat["geometry"] == {"type": "Point", "coordinates": [0.65, 47.33]}
    assert feat["properties"]["instruction"] == "Vérifiez cette citerne."
    assert feat["properties"]["score"] == 0.82


def test_empty_points():
    fc = to_maproulette_tasks([], "x")
    assert fc["features"] == []
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_maproulette.py -q`
Expected: FAIL avec `ImportError` sur `detection_ortho.maproulette`.

- [ ] **Step 3: Créer `maproulette.py`**

```python
# detection_ortho/maproulette.py
"""Génération d'un fichier de tâches MapRoulette (GeoJSON).

IMPORTANT : ce module GÉNÈRE UN FICHIER uniquement. Aucun appel à l'API
MapRoulette, aucune publication, aucun envoi réseau. L'utilisateur importe
lui-même le fichier dans l'interface MapRoulette.
"""
from __future__ import annotations


def to_maproulette_tasks(points: list[dict], instruction: str) -> dict:
    """FeatureCollection GeoJSON : une tâche (Point) par candidat."""
    features = []
    for p in points:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
            "properties": {
                "instruction": instruction,
                "score": p.get("score"),
            },
        })
    return {"type": "FeatureCollection", "features": features}
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_maproulette.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/maproulette.py tests/test_maproulette.py
git commit -m "feat: génération du fichier de tâches MapRoulette (fichier seul)"
```

---

## Task 6: Script d'inférence sur emprise (orchestration)

**Files:**
- Create: `scripts/infer_area.py`
- Test: `tests/test_infer_area_help.py`

**Interfaces:**
- Consumes: tout ce qui précède + `tiles.download_tile`, `dataset.assemble_window`, `dataset.window_tiles`, `geo.dedup_points`, `compare.match_detections`, `geojson_io`, `osm.fetch_citernes` / `fetch_relation_ways`, `maproulette`, `ultralytics.YOLO`.
- Produces: `scripts/infer_area.py` — emprise → détections → comparaison OSM → livrables. La YOLO est importée **paresseusement** dans `main()`.

- [ ] **Step 1: Écrire le test smoke (--help, sans charger le modèle)**

```python
# tests/test_infer_area_help.py
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_infer_area_help_runs():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "infer_area.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "ModuleNotFoundError" not in r.stderr
    assert "--boundary" in r.stdout
    assert "--conf" in r.stdout
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_infer_area_help.py -q`
Expected: FAIL (script inexistant).

- [ ] **Step 3: Écrire `scripts/infer_area.py`**

```python
# scripts/infer_area.py
"""Jalon 3 — Inférence YOLO sur une emprise + comparaison OSM + MapRoulette.

Usage:
    python scripts/infer_area.py --boundary "Tours Métropole Val de Loire" \
        --weights runs/citernes/weights/best.pt --conf 0.4 --out inference_out

Emprise = polygone administratif OSM. Écrit detections/matched/detected_only/
osm_only .geojson + maproulette_challenge.geojson + un résumé chiffré.
Aucun upload MapRoulette : génération de fichiers uniquement.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import requests
from shapely.geometry import Point

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **k):
        return it

from detection_ortho.osm import fetch_relation_ways, fetch_citernes
from detection_ortho.infer import ways_to_polygon, windows_over_polygon, boxes_to_points
from detection_ortho.dataset import assemble_window, window_tiles
from detection_ortho.tiles import download_tile
from detection_ortho.geo import dedup_points
from detection_ortho.compare import match_detections
from detection_ortho.geojson_io import points_to_geojson, write_geojson
from detection_ortho.maproulette import to_maproulette_tasks

ZOOM = 19
WINDOW = 640
INSTRUCTION = ("Une citerne semble présente ici sur l'ortho IGN. "
               "Vérifiez et ajoutez-la à OSM si confirmé.")


def fetch_retry(fn, *a, tries=5, pause=6.0):
    last = None
    for i in range(tries):
        try:
            return fn(*a)
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  Overpass retry {i + 1}/{tries} ({exc})", file=sys.stderr)
            time.sleep(pause)
    raise last


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", type=str, required=True,
                    help="nom de la relation administrative OSM")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--radius", type=float, default=25.0,
                    help="rayon d'appariement OSM (m)")
    ap.add_argument("--dedup", type=float, default=10.0,
                    help="rayon de dédoublonnage des détections (m)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=Path, default=Path("inference_out"))
    args = ap.parse_args()

    cache = args.out / "tiles_cache"

    # --- A. Emprise ---
    print(f"Récupération de l'emprise « {args.boundary} »...")
    ways = fetch_retry(fetch_relation_ways, args.boundary)
    polygon = ways_to_polygon(ways)
    west, south, east, north = polygon.bounds
    print(f"Emprise: bbox=({west:.4f},{south:.4f},{east:.4f},{north:.4f})")

    centers = windows_over_polygon(polygon, ZOOM, WINDOW, args.overlap)
    print(f"{len(centers)} fenêtre(s) d'inférence.")

    # --- Pré-téléchargement parallèle des tuiles ---
    needed = set()
    for lon, lat in centers:
        tiles, _, _ = window_tiles(lon, lat, ZOOM, WINDOW)
        needed.update(tiles)
    print(f"{len(needed)} tuile(s) à récupérer (parallèle x{args.workers})...")
    session = requests.Session()

    def _dl(xy):
        try:
            download_tile(xy[0], xy[1], ZOOM, cache, session=session)
        except Exception as exc:  # noqa: BLE001
            return exc
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_dl, xy) for xy in needed]
        for _ in tqdm(as_completed(futs), total=len(futs),
                      desc="Récupération des tuiles", unit="tuile"):
            pass

    # --- B/C. Inférence + post-traitement ---
    from ultralytics import YOLO
    model = YOLO(args.weights)
    detections: list[dict] = []
    for lon, lat in tqdm(centers, desc="Inférence", unit="fenêtre"):
        try:
            img, ogx, ogy = assemble_window(lon, lat, ZOOM, WINDOW, cache)
        except Exception as exc:  # noqa: BLE001
            print(f"  fenêtre ({lon:.5f},{lat:.5f}) échec ({exc})", file=sys.stderr)
            continue
        res = model.predict(img, conf=args.conf, device=args.device, verbose=False)[0]
        boxes = []
        for b in res.boxes:
            cx, cy = float(b.xywh[0][0]), float(b.xywh[0][1])
            boxes.append((cx, cy, float(b.conf[0])))
        detections.extend(boxes_to_points(boxes, ogx, ogy, ZOOM))

    detections = dedup_points(detections, radius_m=args.dedup)
    print(f"{len(detections)} détection(s) après dédoublonnage.")
    write_geojson(points_to_geojson(detections), args.out / "detections.geojson")

    # --- D. Citernes OSM de la zone (filtrées au polygone) ---
    osm = fetch_retry(fetch_citernes, west, south, east, north)
    osm = [o for o in osm if polygon.contains(Point(o["lon"], o["lat"]))]
    print(f"{len(osm)} citerne(s) OSM dans l'emprise.")

    # --- E. Comparaison ---
    res = match_detections(detections, osm, radius_m=args.radius)
    write_geojson(points_to_geojson([m["detection"] for m in res["matched"]]),
                  args.out / "matched.geojson")
    write_geojson(points_to_geojson(res["detected_only"]),
                  args.out / "detected_only.geojson")
    write_geojson(points_to_geojson(res["osm_only"]),
                  args.out / "osm_only.geojson")

    # --- F. MapRoulette (fichier uniquement) ---
    write_geojson(to_maproulette_tasks(res["detected_only"], INSTRUCTION),
                  args.out / "maproulette_challenge.geojson")

    n_match = len(res["matched"])
    rappel = n_match / len(osm) if osm else float("nan")
    print("\n=== Résumé inférence ===")
    print(f"  Détections (après dédup)      : {len(detections)}")
    print(f"  Confirmées (∩ OSM)            : {n_match}")
    print(f"  Candidates (∉ OSM) -> MapRoul.: {len(res['detected_only'])}")
    print(f"  Manquées (OSM non détectées)  : {len(res['osm_only'])}")
    if osm:
        print(f"  Rappel réel (∩OSM / OSM)      : {rappel:.0%}")
    print(f"\nLivrables dans {args.out}. "
          f"Chargez maproulette_challenge.geojson manuellement dans MapRoulette.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_infer_area_help.py -q`
Expected: 1 passed.

- [ ] **Step 5: Vérifier la syntaxe [RÉSEAU/MODÈLE — DIFFÉRÉ]**

Run: `.venv\Scripts\python -c "import ast; ast.parse(open('scripts/infer_area.py').read())"`
Expected: aucune erreur. **Ne PAS exécuter le script** (réseau IGN/OSM + chargement du modèle + longue inférence) — c'est une exécution manuelle ultérieure.

- [ ] **Step 6: Lancer la suite complète**

Run: `.venv\Scripts\python -m pytest -q`
Expected: tous les tests passent.

- [ ] **Step 7: Commit**

```bash
git add scripts/infer_area.py tests/test_infer_area_help.py
git commit -m "feat: script d'inférence sur emprise -> OSM -> MapRoulette (Jalon 3)"
```

---

## Task 7: Script d'export MapRoulette autonome + doc

**Files:**
- Create: `scripts/export_maproulette.py`
- Modify: `README.md`
- Test: `tests/test_export_maproulette_help.py`

**Interfaces:**
- Consumes: `geojson_io`, `maproulette`.
- Produces: `scripts/export_maproulette.py` — `detected_only.geojson` → `maproulette_challenge.geojson` (utile pour régénérer le challenge sans relancer l'inférence).

- [ ] **Step 1: Écrire le test smoke (--help)**

```python
# tests/test_export_maproulette_help.py
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_export_help_runs():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "export_maproulette.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "ModuleNotFoundError" not in r.stderr
    assert "--input" in r.stdout
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_export_maproulette_help.py -q`
Expected: FAIL (script inexistant).

- [ ] **Step 3: Écrire `scripts/export_maproulette.py`**

```python
# scripts/export_maproulette.py
"""Jalon 3 — Génère le fichier de tâches MapRoulette depuis un GeoJSON de points.

Usage:
    python scripts/export_maproulette.py --input inference_out/detected_only.geojson \
        --out inference_out/maproulette_challenge.geojson

Génération de FICHIER uniquement — aucun upload MapRoulette.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_ortho.geojson_io import write_geojson
from detection_ortho.maproulette import to_maproulette_tasks

INSTRUCTION = ("Une citerne semble présente ici sur l'ortho IGN. "
               "Vérifiez et ajoutez-la à OSM si confirmé.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="GeoJSON de points candidats (detected_only.geojson)")
    ap.add_argument("--out", type=Path, default=Path("maproulette_challenge.geojson"))
    ap.add_argument("--instruction", type=str, default=INSTRUCTION)
    args = ap.parse_args()

    fc = json.loads(args.input.read_text(encoding="utf-8"))
    points = []
    for feat in fc.get("features", []):
        lon, lat = feat["geometry"]["coordinates"]
        points.append({"lon": lon, "lat": lat,
                       "score": feat.get("properties", {}).get("score")})
    write_geojson(to_maproulette_tasks(points, args.instruction), args.out)
    print(f"{len(points)} tâche(s) écrites dans {args.out}. "
          f"À importer manuellement dans MapRoulette.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_export_maproulette_help.py -q`
Expected: 1 passed.

- [ ] **Step 5: Ajouter la section Jalon 3 au `README.md`**

Ajouter à la fin de `README.md` :

```markdown
## Jalon 3 — Inférence sur une zone + MapRoulette

Inférer sur une emprise (relation OSM), comparer à OSM, générer le challenge :

    python scripts/infer_area.py --boundary "Tours Métropole Val de Loire" \
        --weights runs/detect/runs/citernes/weights/best.pt --conf 0.4 --out inference_out

Livrables dans `inference_out/` : `detections.geojson`, `matched/detected_only/
osm_only.geojson`, et `maproulette_challenge.geojson`.

Regénérer seulement le fichier MapRoulette depuis les candidats :

    python scripts/export_maproulette.py --input inference_out/detected_only.geojson \
        --out inference_out/maproulette_challenge.geojson

**Aucun upload automatique** : importez `maproulette_challenge.geojson`
vous-même dans l'interface MapRoulette.
```

- [ ] **Step 6: Lancer la suite complète**

Run: `.venv\Scripts\python -m pytest -q`
Expected: tous les tests passent.

- [ ] **Step 7: Commit**

```bash
git add scripts/export_maproulette.py README.md tests/test_export_maproulette_help.py
git commit -m "feat: export MapRoulette autonome + doc Jalon 3"
```

---

## Self-Review (fait à la rédaction)

- **Couverture du spec :** emprise polygone OSM (Task 2 fetch + Task 3 assemblage) ✓ ; fenêtrage glissant sur polygone (Task 3) ✓ ; inférence + conversion géo (Task 1 `global_px_to_lonlat`, Task 4 `boxes_to_points`, Task 6 orchestration) ✓ ; dédoublonnage (réutilise `dedup_points`, Task 6) ✓ ; comparaison OSM 3 catégories (réutilise `match_detections`, Task 6) ✓ ; export MapRoulette fichier-seul (Task 5 + Task 6 + Task 7) ✓ ; livrables GeoJSON + résumé (Task 6) ✓ ; **aucun upload** (Task 5 docstring + Task 6/7 prints, aucun appel API) ✓ ; seuil 0,4 par défaut (Task 6 `--conf`) ✓.
- **Placeholders :** aucun « TBD ». Les tags exacts de la relation admin sont dans `build_boundary_query` (name + boundary=administrative) ; un repli bbox est mentionné au spec si la relation est introuvable (décision d'exécution, pas un trou de code).
- **Cohérence des types :** `fetch_relation_ways -> list[list[dict]]` consommé par `ways_to_polygon -> Polygon` consommé par `windows_over_polygon -> list[(lon,lat)]` ; `assemble_window -> (img, ogx, ogy)` dont `ogx,ogy` alimentent `boxes_to_points(..., ogx, ogy, ...)` ; `boxes_to_points -> [{lon,lat,score}]` → `dedup_points` → `match_detections` → `to_maproulette_tasks`. `global_px_to_lonlat` est bien l'inverse de `lonlat_to_global_px`. Convention (lon, lat), zoom 19, fenêtre 640 respectées.
