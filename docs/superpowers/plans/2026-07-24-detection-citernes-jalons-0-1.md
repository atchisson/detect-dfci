# Détection de citernes — Plan d'implémentation (Jalons 0 & 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Récupérer les citernes connues d'OSM et l'ortho IGN sur un petit secteur, les visualiser (Jalon 0), puis produire une baseline de détection OpenCV comparée à OSM (Jalon 1).

**Architecture :** Pipeline géospatial en modules découplés. La logique pure (maths de tuiles, parsing OSM, distances, seuillage couleur, appariement) est séparée du réseau (WMTS, Overpass) pour être testable sans I/O. Chaque module produit un artefact fichier inspectable.

**Tech Stack :** Python 3.11+, requests, mercantile, pyproj, shapely, numpy, opencv-python, pytest.

## Global Constraints

- Python **3.11+**, tout le code dans un package `detection_ortho/`, les tests dans `tests/`.
- Le code réseau (WMTS, Overpass) est **isolé** dans des fonctions dédiées ; **aucun test ne fait d'appel réseau réel** — les tests portent sur la logique pure avec des fixtures.
- Ortho IGN : couche WMTS **`ORTHOIMAGERY.ORTHOPHOTOS`**, TileMatrixSet **`PM`** (Web Mercator, EPSG:3857, tuiles 256×256), qui suit exactement le schéma de tuiles « slippy map » standard (`TILEMATRIX`=zoom, `TILECOL`=x, `TILEROW`=y).
- Zoom de travail par défaut : **19** (≈ 30 cm/pixel).
- Une seule classe d'objet : `citerne`. Une détection = un **point** (lon, lat) + score.
- Coordonnées : ordre **(lon, lat)** partout dans le code (convention GeoJSON/shapely). Les bbox sont `(west, south, east, north)` en degrés.
- Commits fréquents, un par tâche minimum. Format : `type: description`.

---

## File Structure

- `detection_ortho/__init__.py` — marque le package.
- `detection_ortho/tiles.py` — maths de tuiles + conversions pixel↔géo (pur) et téléchargement/cache WMTS (I/O).
- `detection_ortho/osm.py` — construction requête Overpass + parsing (pur) et fetch (I/O).
- `detection_ortho/geo.py` — distances géographiques et dédoublonnage par proximité (pur).
- `detection_ortho/baseline_cv.py` — détection couleur/forme OpenCV sur une image (pur).
- `detection_ortho/compare.py` — appariement spatial détections↔OSM (pur).
- `detection_ortho/viz.py` — superposition détections/OSM sur tuiles (I/O image).
- `scripts/recon.py` — Jalon 0 : orchestration reconnaissance des données.
- `scripts/run_baseline.py` — Jalon 1 : orchestration baseline + comparaison.
- `tests/` — un fichier de test par module de logique pure.
- `pyproject.toml`, `requirements.txt`, `README.md`.

---

## Task 1: Scaffolding du projet

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `detection_ortho/__init__.py`, `tests/__init__.py`, `README.md`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: rien.
- Produces: le package importable `detection_ortho` et un `pytest` fonctionnel.

- [ ] **Step 1: Créer `requirements.txt`**

```
requests>=2.31
mercantile>=1.2
pyproj>=3.6
shapely>=2.0
numpy>=1.26
opencv-python>=4.9
pytest>=8.0
```

- [ ] **Step 2: Créer `pyproject.toml`**

```toml
[project]
name = "detection_ortho"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Créer les fichiers de package vides**

`detection_ortho/__init__.py` :

```python
"""Détection de citernes souples sur ortho IGN."""
```

`tests/__init__.py` : fichier vide.

`README.md` :

```markdown
# detection_ortho

Détection de citernes souples de secours sur l'ortho IGN, comparaison OSM,
export MapRoulette. Voir `docs/superpowers/specs/`.

## Installation

    python -m venv .venv
    .venv\Scripts\activate   # Windows
    pip install -r requirements.txt

## Tests

    pytest
```

- [ ] **Step 4: Écrire le test de smoke**

```python
# tests/test_smoke.py
import detection_ortho


def test_package_importable():
    assert detection_ortho.__doc__ is not None
```

- [ ] **Step 5: Installer et lancer les tests**

Run: `python -m venv .venv && .venv\Scripts\python -m pip install -r requirements.txt && .venv\Scripts\python -m pytest -q`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: scaffolding du projet detection_ortho"
```

---

## Task 2: Maths de tuiles et conversions pixel↔géo

**Files:**
- Create: `detection_ortho/tiles.py`
- Test: `tests/test_tiles.py`

**Interfaces:**
- Consumes: `mercantile`, `pyproj`.
- Produces:
  - `tile_for_lonlat(lon: float, lat: float, zoom: int) -> tuple[int, int]` → `(x, y)`.
  - `tile_url(x: int, y: int, zoom: int) -> str` → URL WMTS IGN.
  - `pixel_to_lonlat(x: int, y: int, zoom: int, px: float, py: float, tile_size: int = 256) -> tuple[float, float]` → `(lon, lat)` du pixel `(px, py)` dans la tuile `(x, y)`.
  - `lonlat_to_pixel(lon: float, lat: float, zoom: int, tile_size: int = 256) -> tuple[int, int, float, float]` → `(x, y, px, py)` : tuile contenant le point + position pixel dans cette tuile.

- [ ] **Step 1: Écrire les tests (aller-retour et URL)**

```python
# tests/test_tiles.py
import mercantile
from detection_ortho.tiles import (
    tile_for_lonlat,
    tile_url,
    pixel_to_lonlat,
    lonlat_to_pixel,
)


def test_tile_for_lonlat_matches_mercantile():
    lon, lat, z = 6.15, 43.42, 19  # secteur Var
    x, y = tile_for_lonlat(lon, lat, z)
    expected = mercantile.tile(lon, lat, z)
    assert (x, y) == (expected.x, expected.y)


def test_tile_url_contains_layer_and_indices():
    url = tile_url(42, 43, 19)
    assert "ORTHOIMAGERY.ORTHOPHOTOS" in url
    assert "TILEMATRIXSET=PM" in url
    assert "TILEMATRIX=19" in url
    assert "TILECOL=42" in url
    assert "TILEROW=43" in url


def test_pixel_lonlat_roundtrip():
    lon, lat, z = 6.15, 43.42, 19
    x, y, px, py = lonlat_to_pixel(lon, lat, z)
    lon2, lat2 = pixel_to_lonlat(x, y, z, px, py)
    assert abs(lon - lon2) < 1e-5
    assert abs(lat - lat2) < 1e-5


def test_pixel_center_is_inside_tile_bounds():
    lon, lat, z = 6.15, 43.42, 19
    x, y, px, py = lonlat_to_pixel(lon, lat, z)
    assert 0 <= px < 256
    assert 0 <= py < 256
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_tiles.py -q`
Expected: FAIL avec `ModuleNotFoundError` / `ImportError` sur `detection_ortho.tiles`.

- [ ] **Step 3: Implémenter `tiles.py` (partie pure)**

```python
# detection_ortho/tiles.py
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
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_tiles.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/tiles.py tests/test_tiles.py
git commit -m "feat: maths de tuiles WMTS IGN et conversions pixel<->géo"
```

---

## Task 3: Téléchargement et cache des tuiles ortho

**Files:**
- Modify: `detection_ortho/tiles.py`
- Test: `tests/test_tiles_download.py`

**Interfaces:**
- Consumes: `tile_url` (Task 2), `requests`, `mercantile`.
- Produces:
  - `download_tile(x: int, y: int, zoom: int, cache_dir: Path, session=None) -> Path` → chemin du fichier `.jpg` en cache (télécharge si absent).
  - `tiles_in_bbox(west: float, south: float, east: float, north: float, zoom: int) -> list[tuple[int, int]]` → liste des `(x, y)` couvrant la bbox.

- [ ] **Step 1: Écrire les tests (cache + tuilage bbox), réseau mocké**

```python
# tests/test_tiles_download.py
from pathlib import Path
from detection_ortho.tiles import download_tile, tiles_in_bbox


class FakeResp:
    content = b"\xff\xd8\xff\xe0FAKEJPEG"  # entête JPEG bidon

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self):
        self.calls = 0

    def get(self, url, timeout=30):
        self.calls += 1
        return FakeResp()


def test_download_tile_writes_and_caches(tmp_path):
    sess = FakeSession()
    p1 = download_tile(42, 43, 19, tmp_path, session=sess)
    assert p1.exists()
    assert p1.read_bytes() == FakeResp.content
    # Deuxième appel : servi depuis le cache, pas de nouvel appel réseau.
    p2 = download_tile(42, 43, 19, tmp_path, session=sess)
    assert p2 == p1
    assert sess.calls == 1


def test_tiles_in_bbox_covers_area():
    # Petite bbox : au moins une tuile, toutes distinctes.
    tiles = tiles_in_bbox(6.14, 43.41, 6.16, 43.43, 17)
    assert len(tiles) >= 1
    assert len(set(tiles)) == len(tiles)
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_tiles_download.py -q`
Expected: FAIL avec `ImportError` sur `download_tile` / `tiles_in_bbox`.

- [ ] **Step 3: Ajouter la partie I/O à `tiles.py`**

Ajouter en haut du fichier :

```python
from pathlib import Path

import requests
```

Ajouter à la fin du fichier :

```python
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
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_tiles_download.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/tiles.py tests/test_tiles_download.py
git commit -m "feat: téléchargement et cache des tuiles ortho IGN"
```

---

## Task 4: Récupération des citernes OSM (Overpass)

**Files:**
- Create: `detection_ortho/osm.py`
- Test: `tests/test_osm.py`

**Interfaces:**
- Consumes: `requests`.
- Produces:
  - `build_overpass_query(west, south, east, north) -> str`.
  - `parse_overpass_response(data: dict) -> list[dict]` → liste de `{"lon": float, "lat": float, "tags": dict}`.
  - `fetch_citernes(west, south, east, north, session=None) -> list[dict]` (I/O).
  - Constante `CITERNE_TAGS: list[tuple[str, str]]` (clé, valeur) des tags recherchés.

- [ ] **Step 1: Écrire les tests (requête + parsing), réseau mocké**

```python
# tests/test_osm.py
from detection_ortho.osm import (
    build_overpass_query,
    parse_overpass_response,
    fetch_citernes,
    CITERNE_TAGS,
)


def test_query_contains_bbox_and_tags():
    q = build_overpass_query(6.14, 43.41, 6.16, 43.43)
    assert "[out:json]" in q
    assert "43.41,6.14,43.43,6.16" in q  # ordre Overpass : s,w,n,e
    assert '"emergency"="water_tank"' in q
    assert "out center;" in q


def test_parse_node_and_way_center():
    data = {
        "elements": [
            {"type": "node", "lon": 6.1, "lat": 43.4, "tags": {"emergency": "water_tank"}},
            {"type": "way", "center": {"lon": 6.2, "lat": 43.5}, "tags": {"man_made": "water_tank"}},
            {"type": "way", "tags": {"man_made": "water_tank"}},  # sans center -> ignoré
        ]
    }
    pts = parse_overpass_response(data)
    assert len(pts) == 2
    assert pts[0] == {"lon": 6.1, "lat": 43.4, "tags": {"emergency": "water_tank"}}
    assert pts[1]["lon"] == 6.2 and pts[1]["lat"] == 43.5


def test_fetch_citernes_uses_session(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"elements": [
                {"type": "node", "lon": 6.1, "lat": 43.4, "tags": {"emergency": "water_tank"}},
            ]}

    class FakeSession:
        def post(self, url, data, timeout=90):
            return FakeResp()

    pts = fetch_citernes(6.14, 43.41, 6.16, 43.43, session=FakeSession())
    assert len(pts) == 1
    assert CITERNE_TAGS  # non vide
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_osm.py -q`
Expected: FAIL avec `ImportError` sur `detection_ortho.osm`.

- [ ] **Step 3: Implémenter `osm.py`**

```python
# detection_ortho/osm.py
"""Récupération des citernes connues via l'API Overpass (OSM)."""
from __future__ import annotations

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# (clé, valeur) des tags OSM candidats pour les citernes / réserves incendie.
# À affiner au Jalon 0 selon ce qui est réellement présent sur la zone.
CITERNE_TAGS: list[tuple[str, str]] = [
    ("emergency", "water_tank"),
    ("man_made", "water_tank"),
    ("emergency", "fire_water_pond"),
]


def build_overpass_query(
    west: float, south: float, east: float, north: float
) -> str:
    """Requête Overpass QL récupérant nodes et ways citernes dans la bbox."""
    bbox = f"{south},{west},{north},{east}"  # Overpass attend s,w,n,e
    clauses = []
    for key, value in CITERNE_TAGS:
        clauses.append(f'  node["{key}"="{value}"]({bbox});')
        clauses.append(f'  way["{key}"="{value}"]({bbox});')
    body = "\n".join(clauses)
    return f"[out:json][timeout:60];\n(\n{body}\n);\nout center;"


def parse_overpass_response(data: dict) -> list[dict]:
    """Transforme la réponse Overpass en points {lon, lat, tags}.

    Les ways sont réduits à leur centre (`out center`). Les éléments sans
    position exploitable sont ignorés.
    """
    points: list[dict] = []
    for el in data.get("elements", []):
        if el.get("type") == "node":
            lon, lat = el.get("lon"), el.get("lat")
        else:  # way / relation avec center
            center = el.get("center") or {}
            lon, lat = center.get("lon"), center.get("lat")
        if lon is None or lat is None:
            continue
        points.append({"lon": lon, "lat": lat, "tags": el.get("tags", {})})
    return points


def fetch_citernes(
    west: float, south: float, east: float, north: float, session=None
) -> list[dict]:
    """Interroge Overpass et retourne les citernes de la bbox."""
    sess = session or requests.Session()
    query = build_overpass_query(west, south, east, north)
    resp = sess.post(OVERPASS_URL, data=query, timeout=90)
    resp.raise_for_status()
    return parse_overpass_response(resp.json())
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_osm.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/osm.py tests/test_osm.py
git commit -m "feat: récupération des citernes OSM via Overpass"
```

---

## Task 5: Script de reconnaissance (Jalon 0)

**Files:**
- Create: `scripts/recon.py`
- Modify: `detection_ortho/tiles.py` (ajout `save_tile_with_marker`)
- Test: `tests/test_marker.py`

**Interfaces:**
- Consumes: `fetch_citernes` (Task 4), `lonlat_to_pixel` + `download_tile` (Tasks 2–3), `cv2`.
- Produces:
  - `save_tile_with_marker(tile_path: Path, px: float, py: float, out_path: Path) -> None` : dessine une croix au pixel `(px, py)` sur la tuile et sauvegarde.
  - Un script `scripts/recon.py` exécutable : compte les citernes OSM d'une bbox, télécharge leur tuile, marque leur position, sauvegarde les images et un histogramme des tags.

- [ ] **Step 1: Écrire le test du marqueur**

```python
# tests/test_marker.py
import cv2
import numpy as np
from detection_ortho.tiles import save_tile_with_marker


def test_marker_modifies_image(tmp_path):
    # Tuile grise unie.
    src = tmp_path / "tile.jpg"
    cv2.imwrite(str(src), np.full((256, 256, 3), 128, dtype=np.uint8))
    out = tmp_path / "out.png"
    save_tile_with_marker(src, 128.0, 128.0, out)
    assert out.exists()
    img = cv2.imread(str(out))
    # Au centre, la croix a modifié des pixels (plus uniformément gris).
    center = img[120:136, 120:136]
    assert center.min() != center.max()
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_marker.py -q`
Expected: FAIL avec `ImportError` sur `save_tile_with_marker`.

- [ ] **Step 3: Ajouter `save_tile_with_marker` à `tiles.py`**

Ajouter l'import en haut du fichier :

```python
import cv2
```

Ajouter à la fin du fichier :

```python
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
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_marker.py -q`
Expected: 1 passed.

- [ ] **Step 5: Écrire le script `scripts/recon.py`**

```python
# scripts/recon.py
"""Jalon 0 — Reconnaissance des données.

Usage:
    python scripts/recon.py --bbox 6.14 43.41 6.16 43.43 --zoom 19 --out recon_out

Compte les citernes OSM de la bbox, télécharge leur tuile ortho, marque leur
position, sauvegarde les imagettes et un décompte des tags. Objectif : REGARDER
les données avant tout code de détection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from detection_ortho.osm import fetch_citernes
from detection_ortho.tiles import (
    download_tile,
    lonlat_to_pixel,
    save_tile_with_marker,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--out", type=Path, default=Path("recon_out"))
    args = ap.parse_args()

    west, south, east, north = args.bbox
    cache = args.out / "tiles_cache"
    images = args.out / "images"

    print(f"Requête OSM sur bbox {args.bbox}...")
    citernes = fetch_citernes(west, south, east, north)
    print(f"{len(citernes)} citerne(s) trouvée(s) dans OSM.")

    tag_counter: Counter = Counter()
    for i, c in enumerate(citernes):
        for k, v in c["tags"].items():
            tag_counter[f"{k}={v}"] += 1
        x, y, px, py = lonlat_to_pixel(c["lon"], c["lat"], args.zoom)
        tile_path = download_tile(x, y, args.zoom, cache)
        out_img = images / f"citerne_{i:03d}.png"
        save_tile_with_marker(tile_path, px, py, out_img)

    print("\nHistogramme des tags :")
    for tag, n in tag_counter.most_common():
        print(f"  {n:4d}  {tag}")
    print(f"\nImagettes marquées écrites dans : {images}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Vérification manuelle sur un vrai secteur (réseau réel)**

> Choisissez une bbox contenant des citernes connues (repérez-en une sur openstreetmap.org, notez une petite bbox autour). Cette étape fait de vrais appels réseau — c'est le livrable du Jalon 0.

Run: `.venv\Scripts\python scripts/recon.py --bbox 6.14 43.41 6.16 43.43 --zoom 19 --out recon_out`
Expected : un décompte de citernes, un histogramme de tags, et des PNG dans `recon_out/images/` où la croix rouge tombe sur une citerne visible. **Ouvrez quelques images et vérifiez à l'œil que les citernes sont identifiables.** Notez les tags réellement présents (pour affiner `CITERNE_TAGS` si besoin).

- [ ] **Step 7: Commit**

```bash
git add scripts/recon.py detection_ortho/tiles.py tests/test_marker.py
git commit -m "feat: script de reconnaissance des données (Jalon 0)"
```

---

## Task 6: Distances géographiques et dédoublonnage

**Files:**
- Create: `detection_ortho/geo.py`
- Test: `tests/test_geo.py`

**Interfaces:**
- Consumes: rien (maths pures).
- Produces:
  - `haversine_m(lon1, lat1, lon2, lat2) -> float` → distance en mètres.
  - `dedup_points(points: list[dict], radius_m: float) -> list[dict]` : fusionne les points `{"lon","lat","score"}` distants de moins de `radius_m`, gardant le meilleur score.

- [ ] **Step 1: Écrire les tests**

```python
# tests/test_geo.py
from detection_ortho.geo import haversine_m, dedup_points


def test_haversine_known_distance():
    # ~1 degré de latitude ≈ 111 km.
    d = haversine_m(6.0, 43.0, 6.0, 44.0)
    assert 110_000 < d < 112_000


def test_haversine_zero():
    assert haversine_m(6.0, 43.0, 6.0, 43.0) == 0.0


def test_dedup_merges_close_points_keeps_best_score():
    pts = [
        {"lon": 6.0000, "lat": 43.0000, "score": 0.7},
        {"lon": 6.00002, "lat": 43.00001, "score": 0.9},  # ~2 m -> doublon
        {"lon": 6.0100, "lat": 43.0000, "score": 0.5},    # ~800 m -> distinct
    ]
    out = dedup_points(pts, radius_m=20)
    assert len(out) == 2
    kept = max(out, key=lambda p: p["score"])
    assert kept["score"] == 0.9


def test_dedup_empty():
    assert dedup_points([], radius_m=20) == []
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_geo.py -q`
Expected: FAIL avec `ImportError` sur `detection_ortho.geo`.

- [ ] **Step 3: Implémenter `geo.py`**

```python
# detection_ortho/geo.py
"""Distances géographiques et dédoublonnage de points par proximité."""
from __future__ import annotations

import math

_EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Distance en mètres entre deux points (lon, lat) en degrés."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def dedup_points(points: list[dict], radius_m: float) -> list[dict]:
    """Fusionne les points à moins de radius_m, gardant le meilleur score.

    Approche gloutonne : on traite les points par score décroissant ; chaque
    point non encore absorbé devient un représentant qui absorbe ses voisins.
    Chaque point doit avoir les clés lon, lat, score.
    """
    remaining = sorted(points, key=lambda p: p["score"], reverse=True)
    kept: list[dict] = []
    for p in remaining:
        if any(
            haversine_m(p["lon"], p["lat"], k["lon"], k["lat"]) <= radius_m
            for k in kept
        ):
            continue
        kept.append(p)
    return kept
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_geo.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/geo.py tests/test_geo.py
git commit -m "feat: distances géographiques et dédoublonnage par proximité"
```

---

## Task 7: Baseline de détection OpenCV

**Files:**
- Create: `detection_ortho/baseline_cv.py`
- Test: `tests/test_baseline_cv.py`

**Interfaces:**
- Consumes: `cv2`, `numpy`.
- Produces:
  - `DetectionParams` (dataclass) : `hsv_low`, `hsv_high` (tuples BGR→HSV), `min_area`, `max_area`, `min_aspect`, `max_aspect`.
  - `default_params() -> DetectionParams` (valeurs de départ, à affiner au Jalon 1).
  - `detect_in_image(img_bgr: np.ndarray, params: DetectionParams) -> list[dict]` → liste de `{"px": float, "py": float, "area": float, "score": float}` (centroïdes des objets retenus). `score` = 1.0 (baseline sans confiance apprise).

- [ ] **Step 1: Écrire les tests avec des images synthétiques**

```python
# tests/test_baseline_cv.py
import cv2
import numpy as np
from detection_ortho.baseline_cv import (
    detect_in_image,
    default_params,
    DetectionParams,
)


def _canvas():
    # Fond vert « végétation » sombre.
    return np.full((256, 256, 3), (40, 90, 40), dtype=np.uint8)  # BGR


def test_detects_bright_green_rectangle():
    img = _canvas()
    # Citerne factice : rectangle vert vif bien distinct.
    cv2.rectangle(img, (100, 110), (140, 150), (60, 200, 60), thickness=-1)
    params = DetectionParams(
        hsv_low=(40, 120, 80), hsv_high=(80, 255, 255),
        min_area=200, max_area=20000, min_aspect=0.3, max_aspect=3.0,
    )
    dets = detect_in_image(img, params)
    assert len(dets) == 1
    d = dets[0]
    assert 100 <= d["px"] <= 140
    assert 110 <= d["py"] <= 150


def test_ignores_too_small_blob():
    img = _canvas()
    cv2.rectangle(img, (10, 10), (14, 14), (60, 200, 60), thickness=-1)  # minuscule
    params = DetectionParams(
        hsv_low=(40, 120, 80), hsv_high=(80, 255, 255),
        min_area=200, max_area=20000, min_aspect=0.3, max_aspect=3.0,
    )
    assert detect_in_image(img, params) == []


def test_default_params_returns_dataclass():
    p = default_params()
    assert isinstance(p, DetectionParams)
    assert p.min_area < p.max_area
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_baseline_cv.py -q`
Expected: FAIL avec `ImportError` sur `detection_ortho.baseline_cv`.

- [ ] **Step 3: Implémenter `baseline_cv.py`**

```python
# detection_ortho/baseline_cv.py
"""Baseline de détection sans deep learning : seuillage HSV + filtrage forme.

Fragile par nature (ombres, bâches, toits de même teinte) : sert de référence
pédagogique et de premier point de comparaison OSM, pas de solution finale.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectionParams:
    hsv_low: tuple[int, int, int]
    hsv_high: tuple[int, int, int]
    min_area: float
    max_area: float
    min_aspect: float
    max_aspect: float


def default_params() -> DetectionParams:
    """Valeurs de départ, à affiner en observant les vraies imagettes (Jalon 1)."""
    return DetectionParams(
        hsv_low=(35, 80, 60),
        hsv_high=(85, 255, 255),
        min_area=300,
        max_area=30000,
        min_aspect=0.25,
        max_aspect=4.0,
    )


def detect_in_image(img_bgr: np.ndarray, params: DetectionParams) -> list[dict]:
    """Détecte les objets correspondant aux critères couleur/forme.

    Retourne un centroïde par objet retenu : {px, py, area, score}.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(params.hsv_low), np.array(params.hsv_high))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets: list[dict] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params.min_area or area > params.max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h else 0
        if aspect < params.min_aspect or aspect > params.max_aspect:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        dets.append({"px": cx, "py": cy, "area": area, "score": 1.0})
    return dets
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_baseline_cv.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/baseline_cv.py tests/test_baseline_cv.py
git commit -m "feat: baseline de détection OpenCV (couleur + forme)"
```

---

## Task 8: Appariement spatial détections ↔ OSM

**Files:**
- Create: `detection_ortho/compare.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: `haversine_m` (Task 6).
- Produces:
  - `match_detections(detections: list[dict], osm_points: list[dict], radius_m: float) -> dict` → `{"matched": [...], "detected_only": [...], "osm_only": [...]}` où chaque détection/point porte au moins `lon`, `lat`. Un appariement lie une détection à au plus un point OSM (le plus proche dans le rayon).

- [ ] **Step 1: Écrire les tests**

```python
# tests/test_compare.py
from detection_ortho.compare import match_detections


def test_three_categories():
    dets = [
        {"lon": 6.0000, "lat": 43.0000, "score": 0.9},   # colle à osm A
        {"lon": 6.0500, "lat": 43.0000, "score": 0.8},   # nouveau (detected_only)
    ]
    osm = [
        {"lon": 6.00001, "lat": 43.00000, "tags": {"emergency": "water_tank"}},  # A
        {"lon": 6.2000, "lat": 43.0000, "tags": {"man_made": "water_tank"}},     # osm_only
    ]
    res = match_detections(dets, osm, radius_m=25)
    assert len(res["matched"]) == 1
    assert len(res["detected_only"]) == 1
    assert len(res["osm_only"]) == 1
    assert res["detected_only"][0]["lon"] == 6.05


def test_one_osm_point_matched_once():
    # Deux détections proches du même point OSM : une seule s'apparie.
    dets = [
        {"lon": 6.00000, "lat": 43.0, "score": 0.9},
        {"lon": 6.00003, "lat": 43.0, "score": 0.8},
    ]
    osm = [{"lon": 6.00001, "lat": 43.0, "tags": {}}]
    res = match_detections(dets, osm, radius_m=25)
    assert len(res["matched"]) == 1
    assert len(res["osm_only"]) == 0
    assert len(res["detected_only"]) == 1


def test_empty_inputs():
    res = match_detections([], [], radius_m=25)
    assert res == {"matched": [], "detected_only": [], "osm_only": []}
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_compare.py -q`
Expected: FAIL avec `ImportError` sur `detection_ortho.compare`.

- [ ] **Step 3: Implémenter `compare.py`**

```python
# detection_ortho/compare.py
"""Appariement spatial entre détections et citernes OSM connues."""
from __future__ import annotations

from detection_ortho.geo import haversine_m


def match_detections(
    detections: list[dict], osm_points: list[dict], radius_m: float
) -> dict:
    """Classe détections et points OSM en trois catégories.

    - matched        : détections appariées à un point OSM (< radius_m).
    - detected_only  : détections sans OSM proche -> candidats MapRoulette.
    - osm_only        : points OSM non détectés -> faux négatifs / disparus.

    Chaque point OSM ne peut être apparié qu'une fois (au plus proche voisin).
    Les détections sont traitées par score décroissant pour la stabilité.
    """
    used_osm: set[int] = set()
    matched: list[dict] = []
    detected_only: list[dict] = []

    ordered = sorted(
        detections, key=lambda d: d.get("score", 0.0), reverse=True
    )
    for det in ordered:
        best_i, best_d = None, radius_m
        for i, osm in enumerate(osm_points):
            if i in used_osm:
                continue
            dist = haversine_m(det["lon"], det["lat"], osm["lon"], osm["lat"])
            if dist <= best_d:
                best_i, best_d = i, dist
        if best_i is None:
            detected_only.append(det)
        else:
            used_osm.add(best_i)
            matched.append({"detection": det, "osm": osm_points[best_i]})

    osm_only = [osm for i, osm in enumerate(osm_points) if i not in used_osm]
    return {"matched": matched, "detected_only": detected_only, "osm_only": osm_only}
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_compare.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/compare.py tests/test_compare.py
git commit -m "feat: appariement spatial détections<->OSM en trois catégories"
```

---

## Task 9: Export GeoJSON et script baseline (Jalon 1)

**Files:**
- Create: `detection_ortho/geojson_io.py`, `scripts/run_baseline.py`
- Test: `tests/test_geojson_io.py`

**Interfaces:**
- Consumes: tout ce qui précède (`tiles`, `osm`, `baseline_cv`, `geo`, `compare`).
- Produces:
  - `points_to_geojson(points: list[dict], extra_props=None) -> dict` : FeatureCollection GeoJSON à partir de points `{lon, lat, ...}`.
  - `write_geojson(fc: dict, path: Path) -> None`.
  - Un script `scripts/run_baseline.py` : parcourt les tuiles d'une bbox, applique la baseline, reprojette en lon/lat, dédoublonne, compare à OSM, et écrit les GeoJSON de résultat + un résumé chiffré.

- [ ] **Step 1: Écrire le test de l'export GeoJSON**

```python
# tests/test_geojson_io.py
import json
from detection_ortho.geojson_io import points_to_geojson, write_geojson


def test_points_to_geojson_structure():
    pts = [{"lon": 6.1, "lat": 43.4, "score": 0.9}]
    fc = points_to_geojson(pts)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    feat = fc["features"][0]
    assert feat["geometry"] == {"type": "Point", "coordinates": [6.1, 43.4]}
    assert feat["properties"]["score"] == 0.9


def test_write_and_reload(tmp_path):
    fc = points_to_geojson([{"lon": 6.1, "lat": 43.4}])
    p = tmp_path / "out.geojson"
    write_geojson(fc, p)
    reloaded = json.loads(p.read_text(encoding="utf-8"))
    assert reloaded["features"][0]["geometry"]["coordinates"] == [6.1, 43.4]
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_geojson_io.py -q`
Expected: FAIL avec `ImportError` sur `detection_ortho.geojson_io`.

- [ ] **Step 3: Implémenter `geojson_io.py`**

```python
# detection_ortho/geojson_io.py
"""Lecture/écriture de points au format GeoJSON."""
from __future__ import annotations

import json
from pathlib import Path


def points_to_geojson(points: list[dict], extra_props: dict | None = None) -> dict:
    """FeatureCollection à partir de points {lon, lat, ...autres->properties}."""
    features = []
    for p in points:
        props = {k: v for k, v in p.items() if k not in ("lon", "lat")}
        if extra_props:
            props.update(extra_props)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def write_geojson(fc: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_geojson_io.py -q`
Expected: 2 passed.

- [ ] **Step 5: Écrire le script `scripts/run_baseline.py`**

```python
# scripts/run_baseline.py
"""Jalon 1 — Baseline OpenCV sur une bbox + comparaison OSM.

Usage:
    python scripts/run_baseline.py --bbox 6.14 43.41 6.16 43.43 --zoom 19 --out baseline_out

Parcourt les tuiles de la bbox, applique la détection couleur/forme, reprojette
les détections en lon/lat, dédoublonne, compare aux citernes OSM, et écrit les
GeoJSON (matched / detected_only / osm_only) plus un résumé chiffré.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from detection_ortho.osm import fetch_citernes
from detection_ortho.tiles import download_tile, tiles_in_bbox, pixel_to_lonlat
from detection_ortho.baseline_cv import detect_in_image, default_params
from detection_ortho.geo import dedup_points
from detection_ortho.compare import match_detections
from detection_ortho.geojson_io import points_to_geojson, write_geojson


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--radius", type=float, default=25.0,
                    help="rayon d'appariement OSM en mètres")
    ap.add_argument("--out", type=Path, default=Path("baseline_out"))
    args = ap.parse_args()

    west, south, east, north = args.bbox
    cache = args.out / "tiles_cache"
    params = default_params()

    tiles = tiles_in_bbox(west, south, east, north, args.zoom)
    print(f"{len(tiles)} tuile(s) à traiter...")

    detections: list[dict] = []
    for x, y in tiles:
        tile_path = download_tile(x, y, args.zoom, cache)
        img = cv2.imread(str(tile_path))
        if img is None:
            continue
        for d in detect_in_image(img, params):
            lon, lat = pixel_to_lonlat(x, y, args.zoom, d["px"], d["py"])
            detections.append({"lon": lon, "lat": lat, "score": d["score"]})

    detections = dedup_points(detections, radius_m=args.radius)
    print(f"{len(detections)} détection(s) après dédoublonnage.")

    osm = fetch_citernes(west, south, east, north)
    print(f"{len(osm)} citerne(s) OSM sur la zone.")

    res = match_detections(detections, osm, radius_m=args.radius)

    write_geojson(points_to_geojson([m["detection"] for m in res["matched"]]),
                  args.out / "matched.geojson")
    write_geojson(points_to_geojson(res["detected_only"]),
                  args.out / "detected_only.geojson")
    write_geojson(points_to_geojson(res["osm_only"]),
                  args.out / "osm_only.geojson")

    n_match = len(res["matched"])
    n_osm = len(osm)
    recall = n_match / n_osm if n_osm else float("nan")
    print("\n=== Résumé baseline ===")
    print(f"  Appariées (détectées ∩ OSM) : {n_match}")
    print(f"  Nouvelles (détectées \\ OSM) : {len(res['detected_only'])}")
    print(f"  Manquées (OSM \\ détectées)  : {len(res['osm_only'])}")
    print(f"  Rappel approximatif          : {recall:.0%}" if n_osm else
          "  Rappel : n/a (aucune citerne OSM)")
    print(f"\nGeoJSON écrits dans : {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Vérification manuelle sur le vrai secteur (réseau réel)**

Run: `.venv\Scripts\python scripts/run_baseline.py --bbox 6.14 43.41 6.16 43.43 --zoom 19 --out baseline_out`
Expected : un résumé chiffré (appariées / nouvelles / manquées / rappel) et trois fichiers GeoJSON dans `baseline_out/`. Ouvrez `detected_only.geojson` et `osm_only.geojson` dans un visualiseur (geojson.io, QGIS) pour juger la baseline. **C'est le livrable du Jalon 1** : il donne la référence chiffrée avant de passer au détecteur YOLO.

- [ ] **Step 7: Lancer toute la suite de tests**

Run: `.venv\Scripts\python -m pytest -q`
Expected: tous les tests passent (smoke, tiles, tiles_download, osm, marker, geo, baseline_cv, compare, geojson_io).

- [ ] **Step 8: Commit**

```bash
git add detection_ortho/geojson_io.py scripts/run_baseline.py tests/test_geojson_io.py
git commit -m "feat: export GeoJSON et script baseline+comparaison (Jalon 1)"
```

---

## Task 10: Conteneurisation Docker

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `docker-compose.yml`
- Modify: `README.md` (section Docker)

**Interfaces:**
- Consumes: `requirements.txt`, le package `detection_ortho/`, `scripts/`, `tests/`.
- Produces: une image `detection_ortho:latest` qui exécute `pytest` par défaut et
  permet de lancer les scripts `recon.py` / `run_baseline.py` dans un
  environnement reproductible (Python 3.12).

> **Note d'exécution :** cette tâche doit être faite **en dernier** (après la
> Task 9), car l'image copie tout le code et lance la suite de tests complète.
> Base **`python:3.12-slim`** (pas 3.14). opencv-python nécessite les libs
> système `libgl1` et `libglib2.0-0`.

- [ ] **Step 1: Écrire le `.dockerignore`**

```
.venv/
__pycache__/
*.pyc
.git/
.superpowers/
tiles_cache/
recon_out/
baseline_out/
*.pt
datasets/
runs/
```

- [ ] **Step 2: Écrire le `Dockerfile`**

```dockerfile
FROM python:3.12-slim

# Dépendances système requises par opencv-python (libGL.so.1, glib).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Couche dédiée aux dépendances Python (cache de build).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code du projet.
COPY detection_ortho ./detection_ortho
COPY scripts ./scripts
COPY tests ./tests
COPY pyproject.toml .

ENV PYTHONUNBUFFERED=1

# Par défaut : lance la suite de tests (vérifie que l'image est saine).
CMD ["python", "-m", "pytest", "-q"]
```

- [ ] **Step 3: Écrire le `docker-compose.yml`**

```yaml
services:
  app:
    build: .
    image: detection_ortho:latest
    # Sorties persistées côté hôte : passer --out /app/out/... aux scripts.
    volumes:
      - ./out:/app/out
```

- [ ] **Step 4: Ajouter la section Docker au `README.md`**

Ajouter à la fin de `README.md` :

```markdown
## Docker (environnement reproductible)

Construire l'image (lance aussi les tests) :

    docker build -t detection_ortho .
    docker run --rm detection_ortho          # exécute pytest

Lancer un script en persistant les sorties dans ./out :

    docker run --rm -v "$PWD/out:/app/out" detection_ortho \
        python scripts/recon.py --bbox 6.14 43.41 6.16 43.43 --zoom 19 --out /app/out/recon_out

Ou via docker compose :

    docker compose run --rm app python scripts/run_baseline.py \
        --bbox 6.14 43.41 6.16 43.43 --out /app/out/baseline_out
```

- [ ] **Step 5: Construire l'image et vérifier les tests dans le conteneur**

Run: `docker build -t detection_ortho . && docker run --rm detection_ortho`
Expected : build réussi, puis la suite `pytest` passe **à l'intérieur du conteneur**
(mêmes tests que Task 9 Step 7). Si le build échoue par absence de réseau dans
l'environnement d'exécution, reporter DONE_WITH_CONCERNS en indiquant que le
build n'a pas pu être vérifié ici (les fichiers restent corrects).

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.yml README.md
git commit -m "feat: conteneurisation Docker (image reproductible + pytest)"
```

---

## Self-Review (fait à la rédaction)

- **Couverture du spec :** Jalon 0 (acquisition ortho §Task 2–3, labels OSM §Task 4, visualisation §Task 5) ✓ ; Jalon 1 (baseline OpenCV §Task 7, comparaison OSM §Task 8, export §Task 9, dédoublonnage §Task 6) ✓. Les Jalons 2–4 (YOLO, échelle, MapRoulette) sont hors de ce plan par décision — un second plan les couvrira après examen des vraies données.
- **Placeholders :** aucun « TBD/TODO » dans le code. Les mentions « à affiner au Jalon 0/1 » concernent des paramètres (tags OSM, seuils HSV) dont l'ajustement EST le but de ces jalons — ce sont des valeurs de départ réelles et fonctionnelles, pas des trous.
- **Cohérence des types :** `lonlat_to_pixel` retourne `(x, y, px, py)` et est consommé tel quel dans `recon.py` et `run_baseline.py` ; `detect_in_image` retourne des dicts `{px, py, area, score}` consommés en `pixel_to_lonlat(x, y, zoom, d["px"], d["py"])` ; `match_detections` retourne `{matched, detected_only, osm_only}` consommé fidèlement dans `run_baseline.py`. Convention (lon, lat) respectée partout.
