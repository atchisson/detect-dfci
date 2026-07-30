# Test de valeur NIR ([R,G,NIR]) : Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produire des imagettes 3 canaux `[R, G, NIR]` (bleu ← NIR de l'IRC) pour entraîner/évaluer un YOLO standard et comparer au modèle RVB — sans plomberie 4-canaux.

**Architecture :** `tiles.py` gagne un paramètre `layer` (+ cache tagué). `dataset.py` gagne `compose_rgn` et un passe-plat `layer` sur `assemble_window`. `build_dataset.py` gagne `--nir`. `train.py`/`evaluate.py` inchangés (jpg 3 canaux standard).

**Tech Stack :** Python 3.12, requests, opencv-python, numpy, pytest.

## Global Constraints

- Python **3.12** via `.venv/Scripts/python` (bare `python` = 3.14 ; ne pas l'utiliser).
- Code sous `detection_ortho/`, scripts sous `scripts/`, tests sous `tests/`.
- **Aucun test réseau réel** : fixtures / tuiles pré-écrites en cache.
- Coordonnées (lon, lat). NIR = **canal rouge (index 2 BGR) de la couche IRC** `ORTHOIMAGERY.ORTHOPHOTOS.IRC`.
- Rétro-compat cache : la couche RVB par défaut garde le nom de fichier actuel `{zoom}_{x}_{y}.jpg` (ne pas invalider les caches existants).
- Le run réel (dataset NIR + entraînement) est **différé** (manuel).
- Commits `type: description` ; `git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit` si identité absente.

---

## File Structure

- `detection_ortho/tiles.py` — **modifié** : `layer` sur `tile_url`/`download_tile` + cache tagué + constante `LAYER_IRC`.
- `detection_ortho/dataset.py` — **modifié** : `layer` sur `assemble_window` + `compose_rgn`.
- `scripts/build_dataset.py` — **modifié** : option `--nir`.
- `README.md` — **modifié** : note test NIR.
- `tests/` — tests couche/cache, `compose_rgn`, intégration `--nir`.

---

## Task 1: Couche paramétrable + cache tagué (tiles.py)

**Files:**
- Modify: `detection_ortho/tiles.py`
- Test: `tests/test_tiles_layer.py`

**Interfaces:**
- Produces:
  - Constante `LAYER_IRC = "ORTHOIMAGERY.ORTHOPHOTOS.IRC"`.
  - `tile_url(x, y, zoom, layer=LAYER)`.
  - `download_tile(x, y, zoom, cache_dir, session=None, layer=LAYER)` — cache spécifique à la couche (RVB inchangé, IRC suffixé `_irc`).

- [ ] **Step 1: Écrire les tests**

```python
# tests/test_tiles_layer.py
from detection_ortho.tiles import tile_url, download_tile, LAYER, LAYER_IRC


def test_tile_url_layer():
    assert f"LAYER={LAYER_IRC}" in tile_url(1, 2, 19, layer=LAYER_IRC)
    assert "ORTHOIMAGERY.ORTHOPHOTOS.IRC" in tile_url(1, 2, 19, layer=LAYER_IRC)
    # défaut = RVB
    assert f"LAYER={LAYER}" in tile_url(1, 2, 19)


class _Resp:
    content = b"\xff\xd8\xff\xe0FAKE"

    def raise_for_status(self):
        pass


class _Session:
    def __init__(self):
        self.urls = []

    def get(self, url, headers=None, timeout=30):
        self.urls.append(url)
        return _Resp()


def test_cache_is_layer_specific(tmp_path):
    s = _Session()
    rgb = download_tile(5, 6, 19, tmp_path, session=s)            # RVB
    irc = download_tile(5, 6, 19, tmp_path, session=s, layer=LAYER_IRC)
    # deux fichiers de cache distincts (pas de collision)
    assert rgb != irc
    assert rgb.name == "19_5_6.jpg"          # RVB : nom inchangé
    assert "irc" in irc.name                  # IRC : suffixé
    assert LAYER_IRC in s.urls[1]             # 2e requête = couche IRC
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_tiles_layer.py -q`
Expected: FAIL avec `ImportError` sur `LAYER_IRC` (ou signature).

- [ ] **Step 3: Modifier `tiles.py`**

Ajouter la constante sous `LAYER` :

```python
LAYER_IRC = "ORTHOIMAGERY.ORTHOPHOTOS.IRC"  # composite CIR : bande 1 = NIR
```

Remplacer `tile_url` par (ajout du paramètre `layer`) :

```python
def tile_url(x: int, y: int, zoom: int, layer: str = LAYER) -> str:
    return (
        f"{WMTS_BASE}?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
        f"&LAYER={layer}&STYLE=normal&TILEMATRIXSET=PM"
        f"&TILEMATRIX={zoom}&TILEROW={y}&TILECOL={x}&FORMAT=image/jpeg"
    )
```

Remplacer `download_tile` par (ajout `layer` + cache tagué) :

```python
def download_tile(
    x: int, y: int, zoom: int, cache_dir, session=None, layer: str = LAYER
) -> Path:
    """Télécharge la tuile (x, y, zoom) de la couche `layer` si absente du cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if layer == LAYER else "_" + layer.rsplit(".", 1)[-1].lower()
    path = cache_dir / f"{zoom}_{x}_{y}{tag}.jpg"
    if path.exists():
        return path
    sess = session or requests.Session()
    resp = sess.get(
        tile_url(x, y, zoom, layer),
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_tiles_layer.py tests/test_tiles_download.py -q`
Expected: tous passent (dont les anciens tests de download inchangés).

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/tiles.py tests/test_tiles_layer.py
git commit -m "feat: couche WMTS paramétrable (tile_url/download_tile) + cache tagué + LAYER_IRC"
```

---

## Task 2: assemble_window(layer) + compose_rgn (dataset.py)

**Files:**
- Modify: `detection_ortho/dataset.py`
- Test: `tests/test_compose_rgn.py`

**Interfaces:**
- Consumes: `tiles.download_tile`, `tiles.LAYER`.
- Produces:
  - `assemble_window(..., layer=LAYER)` — passe `layer` à `download_tile`.
  - `compose_rgn(rgb_bgr, irc_bgr) -> np.ndarray` — image BGR où le bleu est remplacé par le NIR (= canal rouge de l'IRC).

- [ ] **Step 1: Écrire les tests**

```python
# tests/test_compose_rgn.py
import numpy as np
from detection_ortho.dataset import compose_rgn


def test_blue_channel_becomes_nir():
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[..., 0] = 10   # B
    rgb[..., 1] = 20   # G
    rgb[..., 2] = 30   # R
    irc = np.zeros((4, 4, 3), np.uint8)
    irc[..., 2] = 200  # canal rouge IRC = NIR
    out = compose_rgn(rgb, irc)
    assert (out[..., 0] == 200).all()   # bleu <- NIR
    assert (out[..., 1] == 20).all()    # G conservé
    assert (out[..., 2] == 30).all()    # R conservé
    # n'altère pas l'entrée
    assert (rgb[..., 0] == 10).all()
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_compose_rgn.py -q`
Expected: FAIL avec `ImportError` sur `compose_rgn`.

- [ ] **Step 3: Modifier `dataset.py`**

Dans l'import depuis `tiles`, ajouter `LAYER`. Remplacer :

```python
from detection_ortho.tiles import download_tile
```

par :

```python
from detection_ortho.tiles import download_tile, LAYER
```

Dans `assemble_window`, ajouter le paramètre `layer` et le passer à `download_tile`. Remplacer la signature et l'appel :

```python
def assemble_window(
    center_lon, center_lat, zoom, window_px, cache_dir, session=None,
    tile_size=256, layer=LAYER,
):
```
et dans la boucle de téléchargement :
```python
        path = download_tile(x, y, zoom, cache_dir, session=session, layer=layer)
```

Ajouter `compose_rgn` à la fin de `dataset.py` :

```python
def compose_rgn(rgb_bgr, irc_bgr):
    """Image BGR où le bleu est remplacé par le NIR (= canal rouge de l'IRC).

    Donne un proxy 3 canaux [R, G, NIR] pour tester l'apport du NIR sans
    plomberie 4-canaux.
    """
    out = rgb_bgr.copy()
    out[:, :, 0] = irc_bgr[:, :, 2]
    return out
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_compose_rgn.py tests/test_dataset_window.py -q`
Expected: tous passent (assemble_window inchangé par défaut).

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/dataset.py tests/test_compose_rgn.py
git commit -m "feat: assemble_window(layer) + compose_rgn ([R,G,NIR])"
```

---

## Task 3: build_dataset --nir

**Files:**
- Modify: `scripts/build_dataset.py`
- Test: `tests/test_build_dataset_nir.py`

**Interfaces:**
- Consumes: `dataset.compose_rgn`, `tiles.LAYER_IRC`, `assemble_window(layer=)`.
- Produces: option `--nir` sur `build_dataset.py` : chaque imagette est composée `[R,G,NIR]` (fenêtre RVB + fenêtre IRC).

- [ ] **Step 1: Écrire le test d'intégration offline (--nir)**

```python
# tests/test_build_dataset_nir.py
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import build_dataset  # noqa: E402
from detection_ortho.dataset import window_tiles  # noqa: E402
from detection_ortho.tiles import LAYER_IRC  # noqa: E402


def _seed(cache, lon, lat, layer_tag, value):
    cache.mkdir(parents=True, exist_ok=True)
    tiles, _, _ = window_tiles(lon, lat, 19, 640)
    suffix = f"_{layer_tag}" if layer_tag else ""
    for x, y in tiles:
        cv2.imwrite(str(cache / f"19_{x}_{y}{suffix}.jpg"),
                    np.full((256, 256, 3), value, np.uint8))


def test_nir_chip_blue_from_irc(tmp_path, monkeypatch):
    monkeypatch.setattr(build_dataset, "fetch_features_geom", lambda *a, **k: [])
    out = tmp_path / "ds"
    cache = out / "tiles_cache"
    lon, lat = 0.65, 47.33
    # tuiles RVB (gris 100) et IRC (rouge=200) pré-semées
    _seed(cache, lon, lat, "", 100)
    irc_tag = LAYER_IRC.rsplit(".", 1)[-1].lower()   # "irc"
    _seed(cache, lon, lat, irc_tag, 200)
    verdicts = tmp_path / "v.csv"
    verdicts.write_text(f"index,lat,lon,score,verdict\n1,{lat},{lon},0.9,vrai\n",
                        encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "build_dataset.py", "--bbox", "0.6", "47.3", "0.7", "47.4",
        "--negatives", "0", "--max-pools", "0", "--nir",
        "--verdicts", str(verdicts), "--out", str(out)])
    build_dataset.main()

    imgs = list((out / "images").rglob("revpos_*.jpg"))
    assert len(imgs) == 1
    chip = cv2.imread(str(imgs[0]))
    # canal bleu du chip = NIR (tuile IRC=200), pas la valeur RVB (100)
    assert int(chip[:, :, 0].mean()) > 150
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_build_dataset_nir.py -q`
Expected: FAIL (option `--nir` inconnue).

- [ ] **Step 3: Modifier `scripts/build_dataset.py`**

Dans le bloc d'imports depuis `detection_ortho.dataset`, ajouter `compose_rgn` ; ajouter l'import `from detection_ortho.tiles import LAYER_IRC`.

Ajouter l'argument (près des autres `ap.add_argument`) :

```python
    ap.add_argument("--nir", action="store_true",
                    help="imagettes [R,G,NIR] (bleu remplacé par le NIR de l'IRC)")
```

Dans la boucle de génération des chips, remplacer l'assemblage :

```python
        try:
            win_img, ogx, ogy = assemble_window(lon, lat, ZOOM, WINDOW, cache)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: échec fenêtre ({exc})", file=sys.stderr)
            continue
```

par :

```python
        try:
            win_img, ogx, ogy = assemble_window(lon, lat, ZOOM, WINDOW, cache)
            if args.nir:
                irc_img, _, _ = assemble_window(
                    lon, lat, ZOOM, WINDOW, cache, layer=LAYER_IRC)
                win_img = compose_rgn(win_img, irc_img)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: échec fenêtre ({exc})", file=sys.stderr)
            continue
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_build_dataset_nir.py -q`
Expected: 1 passed.

- [ ] **Step 5: Syntaxe + suite complète**

Run: `.venv\Scripts\python -c "import ast; ast.parse(open('scripts/build_dataset.py').read())"` puis `.venv\Scripts\python -m pytest -q`
Expected: syntaxe OK ; tous les tests passent. **Ne pas lancer build_dataset --nir en réel** (réseau).

- [ ] **Step 6: Commit**

```bash
git add scripts/build_dataset.py tests/test_build_dataset_nir.py
git commit -m "feat: build_dataset --nir (imagettes [R,G,NIR] via couche IRC)"
```

---

## Task 4: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Ajouter la section « Test de valeur NIR » au `README.md`**

Ajouter à la fin de `README.md` :

```markdown
## Test de valeur NIR (proxy [R,G,NIR])

Mesurer si le proche-infrarouge aide, sans plomberie 4-canaux : on remplace le
bleu par le NIR (canal rouge de la couche IRC WMTS).

    python scripts/build_dataset.py --bbox 0.05 46.72 1.06 47.72 \
        --verdicts verdicts.csv --nir --out dataset_nir
    python scripts/train.py --data dataset_nir/data.yaml --epochs 100 --device cpu
    python scripts/evaluate.py --weights runs/citernes/weights/best.pt \
        --data dataset_nir/data.yaml

Comparer le mAP@50 / la matrice de confusion au modèle RVB (mAP 0,83). Si le
gain est net, investir dans un vrai modèle 4 canaux (RGB+NIR) au pivot natif.
```

- [ ] **Step 2: Lancer la suite complète**

Run: `.venv\Scripts\python -m pytest -q`
Expected: tous les tests passent.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: workflow test de valeur NIR"
```

---

## Self-Review (fait à la rédaction)

- **Couverture du spec :** couche paramétrable + cache tagué (Task 1) ✓ ; `compose_rgn` [R,G,NIR] + `assemble_window(layer)` (Task 2) ✓ ; `build_dataset --nir` (Task 3) ✓ ; protocole (dataset/train/évaluation) via scripts existants inchangés + doc (Task 4) ✓ ; runs réels différés ✓.
- **Placeholders :** aucun. Le run réel (dataset NIR + entraînement + éval) est explicitement manuel.
- **Cohérence des types :** `download_tile(..., layer)` → cache tagué distinct RVB/IRC ; `assemble_window(..., layer=LAYER)` passe `layer` ; `compose_rgn(rgb_bgr, irc_bgr)` → BGR avec bleu = IRC[...,2] (NIR), consommé dans `build_dataset` (win_img composé, labels géométriques inchangés). NIR = canal rouge IRC (index 2 BGR) partout. Rétro-compat cache RVB préservée.
