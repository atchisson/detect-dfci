# Échelle départementale — BD ORTHO locale + GPU : Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lire l'ortho en local (BD ORTHO via rasterio/WarpedVRT reprojeté en EPSG:3857, pixel-cohérent avec l'entraînement) et permettre l'inférence départementale sur GPU P620, en réutilisant tout le pipeline du Jalon 3.

**Architecture :** Nouveau module `local_ortho.py` (lecture fenêtrée locale, même contrat de retour que `assemble_window`). `infer_area.py` gagne `--ortho` (lecture locale au lieu du WMTS). `windows_over_polygon` passe en géométrie préparée. Le reste (dedup, comparaison OSM, MapRoulette) est inchangé.

**Tech Stack :** Python 3.12, rasterio, shapely, ultralytics, numpy, pyproj, pytest.

## Global Constraints

- Python **3.12** via `.venv/Scripts/python` (bare `python` = 3.14 ; ne pas l'utiliser).
- Code sous `detection_ortho/`, scripts sous `scripts/`, tests sous `tests/`.
- **Aucun test ne fait d'appel réseau ni ne lit de vraie BD ORTHO** : les tests raster utilisent un **petit GeoTIFF synthétique** créé avec rasterio.
- Coordonnées **(lon, lat)**. Fenêtre **640 px**, zoom **19**, tuiles 256.
- **Cohérence entraînement** : la lecture locale doit produire la **même fenêtre géographique** que le chemin WMTS — reprojection EPSG:3857, grille pixel z19 identique (`WarpedVRT`), sortie **BGR uint8**, et retour **`(image, origin_gx, origin_gy)`** identique à `dataset.assemble_window` (pour que `boxes_to_points` marche sans changement).
- Étapes **[MODÈLE/DONNÉES — DIFFÉRÉ]** non exécutées par l'implémenteur (téléchargement BD ORTHO, run GPU) : syntaxe vérifiée, run manuel.
- Commits fréquents `type: description` ; `git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit` si identité absente.

---

## File Structure

- `requirements.txt` — **modifié** : ajout `rasterio`.
- `detection_ortho/local_ortho.py` — **nouveau** : `open_ortho`, `read_window`.
- `detection_ortho/infer.py` — **modifié** : `windows_over_polygon` avec `prep()`.
- `scripts/infer_area.py` — **modifié** : option `--ortho`.
- `README.md` — **modifié** : install CUDA + workflow départemental (dont build VRT).
- `tests/` — test `read_window` (raster synthétique), tests fenêtrage inchangés.

---

## Task 1: Dépendance rasterio

**Files:**
- Modify: `requirements.txt`
- Test: `tests/test_rasterio_dep.py`

**Interfaces:**
- Produces: `rasterio` importable.

- [ ] **Step 1: Ajouter la dépendance**

Ajouter à la fin de `requirements.txt` :

```
rasterio>=1.3
```

- [ ] **Step 2: Écrire le test d'import**

```python
# tests/test_rasterio_dep.py
def test_rasterio_importable():
    import rasterio
    from rasterio.vrt import WarpedVRT
    assert rasterio is not None and WarpedVRT is not None
```

- [ ] **Step 3: Installer et lancer le test**

Run: `.venv\Scripts\python -m pip install -r requirements.txt && .venv\Scripts\python -m pytest tests/test_rasterio_dep.py -q`
Expected: 1 passed. (Si l'install échoue faute de réseau, reporter BLOCKED avec l'erreur pip exacte.)

- [ ] **Step 4: Commit**

```bash
git add requirements.txt tests/test_rasterio_dep.py
git commit -m "chore: ajout de la dépendance rasterio"
```

---

## Task 2: Fenêtrage accéléré (géométrie préparée)

**Files:**
- Modify: `detection_ortho/infer.py`
- Test: `tests/test_infer_windows.py` (existant — doit continuer à passer)

**Interfaces:**
- `windows_over_polygon` : comportement **inchangé**, mais utilise `shapely.prepared.prep` pour accélérer les tests `contains` à grande échelle.

- [ ] **Step 1: Modifier `windows_over_polygon` dans `infer.py`**

Ajouter l'import en haut de `detection_ortho/infer.py` :

```python
from shapely.prepared import prep
```

Dans `windows_over_polygon`, préparer le polygone une fois et tester avec :

```python
    prepared = prep(polygon)
    ...
            if prepared.contains(Point(lon, lat)):
                centers.append((lon, lat))
```

(Remplacer l'appel `polygon.contains(Point(lon, lat))` par `prepared.contains(...)`. Le reste de la fonction est inchangé.)

- [ ] **Step 2: Lancer les tests de fenêtrage existants**

Run: `.venv\Scripts\python -m pytest tests/test_infer_windows.py -q`
Expected: tous passent (comportement identique).

- [ ] **Step 3: Commit**

```bash
git add detection_ortho/infer.py
git commit -m "perf: windows_over_polygon avec géométrie préparée (prep)"
```

---

## Task 3: Lecture locale de l'ortho (WarpedVRT)

**Files:**
- Create: `detection_ortho/local_ortho.py`
- Test: `tests/test_local_ortho.py`

**Interfaces:**
- Consumes: `rasterio`, `affine`, `detection_ortho.dataset.lonlat_to_global_px`, `math`, `numpy`.
- Produces:
  - `open_ortho(path, zoom: int = 19, tile_size: int = 256) -> rasterio.vrt.WarpedVRT` : ouvre un raster/VRT (ou construit un VRT depuis un dossier de dalles si `osgeo` dispo) et le reprojette en EPSG:3857 sur la **grille pixel du zoom** (transform alignée sur la grille WMTS PM). À fermer par l'appelant.
  - `read_window(vrt, lon, lat, zoom, window_px, tile_size=256) -> tuple[np.ndarray, float, float]` : lit la fenêtre centrée sur `(lon, lat)` → `(image BGR uint8 (window_px, window_px, 3), origin_gx, origin_gy)`, **même contrat que `assemble_window`**.

- [ ] **Step 1: Écrire le test (raster synthétique EPSG:3857)**

```python
# tests/test_local_ortho.py
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer

from detection_ortho.local_ortho import open_ortho, read_window
from detection_ortho.dataset import lonlat_to_global_px


def _make_ortho(path, lon, lat, color_rgb=(10, 200, 60), size=2000, res=0.29):
    """GeoTIFF EPSG:3857 uni, centré sur (lon,lat), couvrant ~size*res mètres."""
    mx, my = Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
    transform = from_origin(mx - size / 2 * res, my + size / 2 * res, res, res)
    data = np.zeros((3, size, size), np.uint8)
    for b in range(3):
        data[b] = color_rgb[b]
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=3,
        dtype="uint8", crs="EPSG:3857", transform=transform,
    ) as dst:
        dst.write(data)


def test_read_window_shape_origin_and_bgr(tmp_path):
    lon, lat, z, win = 0.65, 47.33, 19, 640
    tif = tmp_path / "ortho.tif"
    _make_ortho(tif, lon, lat, color_rgb=(10, 200, 60))
    vrt = open_ortho(tif, zoom=z)
    try:
        img, ogx, ogy = read_window(vrt, lon, lat, z, win)
    finally:
        vrt.close()
    # taille et type
    assert img.shape == (win, win, 3) and img.dtype == np.uint8
    # origine cohérente avec le chemin WMTS
    gx, gy = lonlat_to_global_px(lon, lat, z)
    assert abs(ogx - (gx - win / 2)) < 1e-6
    assert abs(ogy - (gy - win / 2)) < 1e-6
    # pixel central = couleur du raster, convertie RGB(10,200,60) -> BGR(60,200,10)
    b, g, r = img[win // 2, win // 2]
    assert (int(b), int(g), int(r)) == (60, 200, 10)


def test_read_window_outside_data_is_black(tmp_path):
    # Fenêtre loin du raster -> lecture boundless remplie de 0 (pas de crash).
    tif = tmp_path / "ortho.tif"
    _make_ortho(tif, 0.65, 47.33)
    vrt = open_ortho(tif, zoom=19)
    try:
        img, _, _ = read_window(vrt, 2.0, 48.5, 19, 640)  # ailleurs
    finally:
        vrt.close()
    assert img.shape == (640, 640, 3)
    assert int(img.sum()) == 0
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_local_ortho.py -q`
Expected: FAIL avec `ImportError` sur `detection_ortho.local_ortho`.

- [ ] **Step 3: Créer `local_ortho.py`**

```python
# detection_ortho/local_ortho.py
"""Lecture locale de la BD ORTHO via rasterio, reprojetée en EPSG:3857 sur la
grille pixel WMTS (Web Mercator) du zoom — pour coller à l'entraînement.

read_window a le MÊME contrat de retour que dataset.assemble_window :
(image BGR uint8 window_px×window_px, origin_gx, origin_gy).
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from detection_ortho.dataset import lonlat_to_global_px

_R = 6378137.0  # rayon Web Mercator


def _mpp(zoom: int, tile_size: int = 256) -> float:
    """Mètres par pixel de la grille Web Mercator au zoom donné."""
    return (2 * math.pi * _R) / (tile_size * (2 ** zoom))


def _source_path(path) -> str:
    """Retourne un chemin ouvrable par rasterio : fichier tel quel, ou VRT
    construit depuis un dossier de dalles (nécessite osgeo.gdal)."""
    p = Path(path)
    if p.is_dir():
        dalles = sorted(glob.glob(str(p / "*.jp2")) + glob.glob(str(p / "*.tif")))
        if not dalles:
            raise FileNotFoundError(f"Aucune dalle .jp2/.tif dans {p}")
        try:
            from osgeo import gdal
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Dossier de dalles fourni mais osgeo/GDAL indisponible. "
                "Construisez un VRT : `gdalbuildvrt ortho.vrt *.jp2` "
                "(ou QGIS > Raster virtuel) et passez ortho.vrt."
            ) from exc
        vrt_path = str(p / "_mosaic.vrt")
        gdal.BuildVRT(vrt_path, dalles)
        return vrt_path
    return str(p)


def open_ortho(path, zoom: int = 19, tile_size: int = 256) -> WarpedVRT:
    """Ouvre l'ortho reprojetée en EPSG:3857 sur la grille pixel du zoom.

    La transform est alignée sur la grille WMTS PM : le pixel (col, row) du VRT
    correspond au pixel global (gx, gy) du zoom. À fermer par l'appelant.
    """
    mpp = _mpp(zoom, tile_size)
    n = tile_size * (2 ** zoom)  # dimension monde en pixels (virtuel)
    transform = Affine(mpp, 0.0, -math.pi * _R, 0.0, -mpp, math.pi * _R)
    src = rasterio.open(_source_path(path))
    return WarpedVRT(
        src, crs="EPSG:3857", transform=transform, width=n, height=n,
        resampling=Resampling.bilinear,
    )


def read_window(
    vrt: WarpedVRT, lon: float, lat: float, zoom: int, window_px: int,
    tile_size: int = 256,
) -> tuple[np.ndarray, float, float]:
    """Lit la fenêtre window_px centrée sur (lon, lat). Retourne (BGR, ogx, ogy)."""
    gx, gy = lonlat_to_global_px(lon, lat, zoom, tile_size)
    origin_gx = gx - window_px / 2.0
    origin_gy = gy - window_px / 2.0
    win = Window(int(round(origin_gx)), int(round(origin_gy)), window_px, window_px)
    arr = vrt.read(
        indexes=[1, 2, 3], window=win, boundless=True, fill_value=0,
    )  # (3, window_px, window_px), ordre RGB
    img = np.transpose(arr, (1, 2, 0))[:, :, ::-1]  # RGB -> BGR
    return np.ascontiguousarray(img, dtype=np.uint8), float(origin_gx), float(origin_gy)
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_local_ortho.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/local_ortho.py tests/test_local_ortho.py
git commit -m "feat: lecture locale de l'ortho (rasterio WarpedVRT EPSG:3857)"
```

---

## Task 4: Option `--ortho` dans infer_area

**Files:**
- Modify: `scripts/infer_area.py`
- Test: `tests/test_infer_area_help.py` (existant — doit continuer à passer et exposer `--ortho`)

**Interfaces:**
- Consumes: `local_ortho.open_ortho`, `local_ortho.read_window`.
- Produces: option **`--ortho <chemin>`** : si fournie, l'inférence lit via `read_window` (et **saute** le pré-téléchargement des tuiles WMTS) ; sinon, comportement WMTS inchangé.

- [ ] **Step 1: Mettre à jour le test smoke pour exiger `--ortho`**

Dans `tests/test_infer_area_help.py`, ajouter une assertion :

```python
    assert "--ortho" in r.stdout
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_infer_area_help.py -q`
Expected: FAIL (option `--ortho` absente du help).

- [ ] **Step 3: Étendre `scripts/infer_area.py`**

Ajouter l'import (avec les autres imports `detection_ortho`) :

```python
from detection_ortho.local_ortho import open_ortho, read_window
```

Ajouter l'argument (près des autres `ap.add_argument`) :

```python
    ap.add_argument("--ortho", type=str, default=None,
                    help="chemin BD ORTHO locale (raster/VRT/dossier de dalles) ; "
                         "si fourni, lecture locale au lieu du WMTS")
```

Remplacer le bloc « pré-téléchargement des tuiles + boucle d'inférence » par une
logique qui bascule selon `--ortho`. En mode local, on ouvre le VRT une fois, on
saute le pré-téléchargement, et `read_window` remplace `assemble_window` :

```python
    ortho_vrt = None
    if args.ortho:
        ortho_vrt = open_ortho(args.ortho, zoom=ZOOM)
        print(f"Ortho locale : {args.ortho} (lecture rasterio, pas de WMTS).")
    else:
        # ... pré-téléchargement WMTS parallèle existant (inchangé) ...

    from ultralytics import YOLO
    model = YOLO(args.weights)
    detections = []
    try:
        for lon, lat in progress(centers, len(centers), "Inférence"):
            try:
                if ortho_vrt is not None:
                    img, ogx, ogy = read_window(ortho_vrt, lon, lat, ZOOM, WINDOW)
                else:
                    img, ogx, ogy = assemble_window(lon, lat, ZOOM, WINDOW, cache)
            except Exception as exc:  # noqa: BLE001
                print(f"  fenêtre ({lon:.5f},{lat:.5f}) échec ({exc})", file=sys.stderr)
                continue
            res = model.predict(img, conf=args.conf, device=args.device, verbose=False)[0]
            detections.extend(boxes_to_points(result_to_boxes(res.boxes), ogx, ogy, ZOOM))
    finally:
        if ortho_vrt is not None:
            ortho_vrt.close()
```

> L'implémenteur adapte l'insertion au code réel de `infer_area.py` (le bloc
> pré-téléchargement WMTS reste utilisé quand `--ortho` n'est pas fourni). La
> boucle d'inférence choisit `read_window` ou `assemble_window` selon le mode.

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_infer_area_help.py -q`
Expected: 1 passed (`--ortho` présent).

- [ ] **Step 5: Vérifier la syntaxe [MODÈLE/DONNÉES — DIFFÉRÉ]**

Run: `.venv\Scripts\python -c "import ast; ast.parse(open('scripts/infer_area.py').read())"`
Expected: aucune erreur. **Ne pas exécuter le script** (modèle + BD ORTHO).

- [ ] **Step 6: Lancer la suite complète**

Run: `.venv\Scripts\python -m pytest -q`
Expected: tous les tests passent.

- [ ] **Step 7: Commit**

```bash
git add scripts/infer_area.py tests/test_infer_area_help.py
git commit -m "feat: infer_area --ortho (lecture locale BD ORTHO au lieu du WMTS)"
```

---

## Task 5: Documentation (CUDA + workflow départemental)

**Files:**
- Modify: `README.md`

**Interfaces:** doc uniquement.

- [ ] **Step 1: Ajouter la section « Échelle départementale » au `README.md`**

Ajouter à la fin de `README.md` :

```markdown
## Échelle départementale (BD ORTHO locale + GPU)

### 1. Récupérer la BD ORTHO du département
Télécharger la **BD ORTHO 20 cm RVB Lambert-93** du département depuis
cartes.gouv.fr (jeu IGNF_BD-ORTHO), décompresser les dalles `.jp2` dans un
dossier. Construire un mosaïque virtuelle (une fois) :

    gdalbuildvrt ortho37.vrt chemin/vers/dalles/*.jp2

(ou QGIS → Raster → Divers → Construire un raster virtuel).

### 2. (Optionnel) GPU : installer PyTorch CUDA pour la Quadro P620
Par défaut le venv est en CPU. Pour utiliser la P620 :

    .venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    .venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"   # True attendu

### 3. Inférer sur le département (lecture locale, GPU)

    python scripts/infer_area.py --boundary "Indre-et-Loire" \
        --weights runs/citernes/weights/best.pt --ortho ortho37.vrt \
        --conf 0.55 --device 0 --out inference_dept37

Repli CPU : `--device cpu`. Livrables identiques au Jalon 3
(detections/detected_only/... + maproulette_challenge.geojson + overlay.png),
en lecture 100 % locale.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: workflow échelle départementale (BD ORTHO locale + GPU)"
```

---

## Self-Review (fait à la rédaction)

- **Couverture du spec :** dépendance rasterio (Task 1) ✓ ; `read_window` WarpedVRT EPSG:3857 pixel-cohérent + contrat `(img,ogx,ogy)` = `assemble_window` (Task 3) ✓ ; `--ortho` bascule lecture locale / saute le pré-téléchargement (Task 4) ✓ ; `prep()` pour le fenêtrage (Task 2) ✓ ; install CUDA P620 + `--device 0` + emprise « Indre-et-Loire » + build VRT documentés (Task 5) ✓ ; GPU/BD ORTHO = runs manuels différés ✓.
- **Placeholders :** aucun. Le run réel (BD ORTHO + GPU) est une étape manuelle explicite. La branche dossier→VRT via osgeo a un repli documenté (message clair si osgeo absent).
- **Cohérence des types :** `read_window -> (np.ndarray BGR, ogx, ogy)` identique à `assemble_window` → consommé par `boxes_to_points(..., ogx, ogy, ZOOM)` sans changement ; `open_ortho -> WarpedVRT` ouvert une fois, fermé en `finally`. Convention (lon,lat), zoom 19, fenêtre 640 respectées. `windows_over_polygon` garde sa signature (seule l'implémentation `prep()` change).
```
