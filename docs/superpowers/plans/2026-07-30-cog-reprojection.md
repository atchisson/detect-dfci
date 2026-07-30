# Pré-reprojection COG : Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter `scripts/build_cog.py` qui reprojette une fois l'ortho en GeoTIFF tuilé EPSG:3857 aligné z19 (via le WarpedVRT existant), pour que `read_window` (inchangé) lise vite. Documenter le workflow.

**Architecture :** Un seul nouveau script réutilisant `local_ortho.open_ortho`. Écriture rasterio d'un GeoTIFF tuilé/compressé, recopie bloc par bloc depuis le WarpedVRT z19. `read_window`/`infer_area` ne changent pas.

**Tech Stack :** Python 3.12, rasterio, numpy, pytest.

## Global Constraints

- Python **3.12** via `.venv/Scripts/python` (bare `python` = 3.14 ; ne pas l'utiliser).
- Scripts sous `scripts/`, tests sous `tests/`.
- **Aucun test réseau ni vraie BD ORTHO** : test sur un petit raster synthétique EPSG:2154, `--compress deflate` (sans perte → assertions exactes).
- Grille de sortie = grille pixel z19 (mêmes maths `mpp = 2πR/(256·2^z)` et origine que `local_ortho.open_ortho`) → `read_window` mappe directement.
- Le **build réel** (27 Go, heures) est un run manuel différé — non exécuté par l'implémenteur.
- Commits `type: description` ; `git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit` si identité absente.

---

## File Structure

- `scripts/build_cog.py` — **nouveau** : reprojection unique → GeoTIFF tuilé 3857.
- `README.md` — **modifié** : étape `build_cog` dans le workflow départemental.
- `tests/test_build_cog.py` — **nouveau** : round-trip sur raster synthétique.

---

## Task 1: Script build_cog.py + test round-trip

**Files:**
- Create: `scripts/build_cog.py`
- Test: `tests/test_build_cog.py`

**Interfaces:**
- Consumes: `rasterio`, `rasterio.warp.transform_bounds`, `rasterio.windows.Window`, `rasterio.Affine`, `local_ortho.open_ortho`, `math`.
- Produces: script `build_cog.py` (`--src --out [--compress jpeg|deflate] [--zoom 19] [--blocksize 512]`) écrivant un GeoTIFF tuilé EPSG:3857 aligné z19.

- [ ] **Step 1: Écrire le test (round-trip via raster synthétique 2154)**

```python
# tests/test_build_cog.py
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import build_cog  # noqa: E402
from detection_ortho.local_ortho import open_ortho, read_window  # noqa: E402


def _make_src_2154(path, lon, lat, color_rgb=(10, 200, 60), size=1500, res=0.2):
    """Petit raster EPSG:2154 uni avec un gros marqueur central, centré sur (lon,lat)."""
    mx, my = Transformer.from_crs(4326, 2154, always_xy=True).transform(lon, lat)
    transform = from_origin(mx - size / 2 * res, my + size / 2 * res, res, res)
    data = np.zeros((3, size, size), np.uint8)          # fond noir
    c = size // 2
    for b in range(3):                                  # marqueur central ~ 200 px
        data[b, c - 100:c + 100, c - 100:c + 100] = color_rgb[b]
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size,
                       count=3, dtype="uint8", crs="EPSG:2154",
                       transform=transform) as dst:
        dst.write(data)


def test_build_cog_roundtrip(tmp_path, monkeypatch):
    lon, lat, z, win = 0.65, 47.33, 19, 640
    src = tmp_path / "src2154.tif"
    out = tmp_path / "cog3857.tif"
    _make_src_2154(src, lon, lat, color_rgb=(10, 200, 60))

    monkeypatch.setattr(sys, "argv", [
        "build_cog.py", "--src", str(src), "--out", str(out),
        "--compress", "deflate", "--zoom", str(z)])
    build_cog.main()

    # La sortie est un GeoTIFF tuilé en EPSG:3857
    with rasterio.open(out) as d:
        assert d.crs.to_epsg() == 3857
        assert d.profile.get("tiled") is True

    # read_window sur la sortie retrouve le marqueur au centre, RGB->BGR
    vrt = open_ortho(out, zoom=z)
    try:
        img, _, _ = read_window(vrt, lon, lat, z, win)
    finally:
        vrt.close()
    assert img.shape == (win, win, 3)
    b, g, r = img[win // 2, win // 2]
    assert abs(int(b) - 60) < 40 and abs(int(g) - 200) < 40 and abs(int(r) - 10) < 40
    # un coin est le fond noir
    assert tuple(int(v) for v in img[5, 5]) == (0, 0, 0)
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_build_cog.py -q`
Expected: FAIL (`build_cog` inexistant / `ModuleNotFoundError`).

- [ ] **Step 3: Écrire `scripts/build_cog.py`**

```python
# scripts/build_cog.py
"""Pré-reprojette l'ortho en GeoTIFF tuilé EPSG:3857 aligné sur la grille z19.

Fait UNE fois le décodage JP2 + la reprojection (via le WarpedVRT de
local_ortho), en recopiant bloc par bloc. Ensuite read_window lit ce raster
déjà en 3857 et tuilé -> lectures rapides. Aucun outil GDAL externe requis.

Usage:
    python scripts/build_cog.py --src ortho37.vrt --out ortho37_3857.tif
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rasterio
from rasterio import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from detection_ortho.local_ortho import open_ortho

_R = 6378137.0


def _mpp(zoom: int, tile_size: int = 256) -> float:
    return (2 * math.pi * _R) / (tile_size * (2 ** zoom))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True,
                    help="ortho source (VRT/raster, ex. ortho37.vrt)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compress", choices=["jpeg", "deflate"], default="jpeg")
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--blocksize", type=int, default=512)
    args = ap.parse_args()

    mpp = _mpp(args.zoom)

    # 1. Bornes source -> EPSG:3857 -> plage de pixels globaux z19.
    with rasterio.open(args.src) as s:
        nbands = s.count
        minx, miny, maxx, maxy = transform_bounds(
            s.crs, "EPSG:3857", *s.bounds, densify_pts=21)
    gx0 = int(math.floor((minx + math.pi * _R) / mpp))
    gx1 = int(math.ceil((maxx + math.pi * _R) / mpp))
    gy0 = int(math.floor((math.pi * _R - maxy) / mpp))   # maxy -> haut -> gy petit
    gy1 = int(math.ceil((math.pi * _R - miny) / mpp))
    W, H = gx1 - gx0, gy1 - gy0
    transform = Affine(mpp, 0.0, -math.pi * _R + gx0 * mpp,
                       0.0, -mpp, math.pi * _R - gy0 * mpp)

    profile = dict(driver="GTiff", width=W, height=H, count=nbands,
                   dtype="uint8", crs="EPSG:3857", transform=transform,
                   tiled=True, blockxsize=args.blocksize,
                   blockysize=args.blocksize, BIGTIFF="YES")
    if args.compress == "jpeg" and nbands == 3:
        profile.update(compress="JPEG", photometric="YCBCR", jpeg_quality=85)
    else:
        profile.update(compress="DEFLATE")

    print(f"Sortie {W}x{H} px, {nbands} bandes, compress={profile['compress']}")
    n_blocks = ((H + args.blocksize - 1) // args.blocksize) * \
               ((W + args.blocksize - 1) // args.blocksize)
    step = max(1, n_blocks // 50)

    vrt = open_ortho(args.src, zoom=args.zoom)
    try:
        with rasterio.open(args.out, "w", **profile) as dst:
            done = 0
            for r0 in range(0, H, args.blocksize):
                for c0 in range(0, W, args.blocksize):
                    w = min(args.blocksize, W - c0)
                    h = min(args.blocksize, H - r0)
                    data = vrt.read(
                        indexes=list(range(1, nbands + 1)),
                        window=Window(gx0 + c0, gy0 + r0, w, h))
                    dst.write(data, window=Window(c0, r0, w, h))
                    done += 1
                    if done % step == 0 or done == n_blocks:
                        print(f"  build_cog: {done}/{n_blocks} "
                              f"({done * 100 // n_blocks}%)", flush=True)
    finally:
        vrt.close()

    print(f"COG écrit : {args.out}. Passez-le à infer_area.py via --ortho.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_build_cog.py -q`
Expected: 1 passed.

- [ ] **Step 5: Vérifier la syntaxe + suite complète**

Run: `.venv\Scripts\python -c "import ast; ast.parse(open('scripts/build_cog.py').read())"` puis `.venv\Scripts\python -m pytest -q`
Expected: syntaxe OK ; tous les tests passent. **Ne PAS lancer build_cog sur la vraie BD ORTHO** (heures).

- [ ] **Step 6: Commit**

```bash
git add scripts/build_cog.py tests/test_build_cog.py
git commit -m "feat: build_cog (pré-reprojection ortho -> GeoTIFF tuilé 3857 z19)"
```

---

## Task 2: Documentation du workflow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Insérer l'étape build_cog dans la section « Échelle départementale » du `README.md`**

Juste après l'étape de construction du VRT (`gdalbuildvrt` / `build_ortho_vrt.py`) et avant l'étape d'inférence, ajouter :

```markdown
### 1bis. Pré-reprojeter en GeoTIFF tuilé 3857 (perf — une seule fois)
La lecture directe des dalles JP2 (reprojection par fenêtre) est trop lente à
l'échelle départementale (~2,4 s/fenêtre). On pré-reprojette **une fois** en un
GeoTIFF tuilé Web-Mercator aligné sur la grille du modèle :

    python scripts/build_cog.py --src ortho37.vrt --out ortho37_3857.tif

(quelques heures, une seule fois). Ensuite les lectures fenêtrées sont rapides.
```

Puis, dans l'étape 3 (inférence), remplacer `--ortho ortho37.vrt` par
`--ortho ortho37_3857.tif` (le GeoTIFF pré-reprojeté).

- [ ] **Step 2: Lancer la suite complète**

Run: `.venv\Scripts\python -m pytest -q`
Expected: tous les tests passent.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: étape build_cog (pré-reprojection) dans le workflow départemental"
```

---

## Self-Review (fait à la rédaction)

- **Couverture du spec :** build_cog reprojette une fois via le WarpedVRT existant → GeoTIFF tuilé 3857 aligné z19 (Task 1) ✓ ; JPEG par défaut, `--compress deflate` pour test/sans-perte (Task 1) ✓ ; `read_window`/`infer_area` inchangés — juste `--ortho <cog>` (Task 2 doc) ✓ ; build réel = run manuel différé ✓ ; test round-trip au pixel sur synthétique 2154 (Task 1) ✓.
- **Placeholders :** aucun. Le build réel est explicitement différé.
- **Cohérence des types :** `build_cog` produit un GeoTIFF dont la transform/`mpp`/origine sont EXACTEMENT celles de `open_ortho` (grille z19) → `read_window` (via `open_ortho(cog)`) mappe `lonlat_to_global_px` directement sur ses pixels, contrat `(img BGR, ogx, ogy)` inchangé. `transform_bounds` densifié (21 pts) pour des bornes 2154→3857 correctes.
