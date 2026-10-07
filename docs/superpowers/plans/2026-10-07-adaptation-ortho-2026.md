# Adaptation du modèle à l'ortho express 2026 — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rendre possible l'entraînement d'un modèle unique bon sur l'ortho habituelle et sur l'ortho express 2026 : cache de tuiles fiable par couche, préchargement parallèle partout, jeu de données multi-couches, évaluation par couche, runbook.

**Architecture:** Un suffixe de cache propre à chaque couche (`layer_tag`) et une purge qui le reconnaît ; une fonction de préchargement parallèle réutilisée par les scripts d'évaluation et de construction ; `build_dataset.py --layers` produit une imagette par couche disponible pour chaque enregistrement, avec un découpage décidé par enregistrement ; `sweep_threshold.py` et `eval_points.py` gagnent `--layer`.

**Tech Stack:** Python 3.12, pytest, requests, ThreadPoolExecutor, OpenCV, ultralytics (existant).

**Spec:** `docs/superpowers/specs/2026-10-07-adaptation-ortho-2026-design.md`

## Global Constraints

- Tests hors-ligne uniquement : aucun appel réseau dans pytest, aucun appel à la vraie API.
- Tuiles toujours téléchargées **en parallèle** (12 travailleurs par défaut, session partagée) ; jamais tuile par tuile.
- Comportement par défaut inchangé : sans `--layers`/`--layer`, les scripts agissent exactement comme avant (couche `ORTHOIMAGERY.ORTHOPHOTOS`, noms de fichiers de cache et d'imagettes identiques).
- Suffixe de cache : `""` pour la couche standard ; sinon `_` + nom de couche privé du préfixe `ORTHOIMAGERY.ORTHOPHOTOS.`, en minuscules, points et tirets bas remplacés par des tirets (`IRC` → `_irc`, `RVB-EXPRESS.2026` → `_rvb-express-2026`). Caractères autorisés : `[a-z0-9-]`.
- Fenêtre « sans donnée » : moins de 5 % de pixels non blancs (un pixel est non blanc si un de ses canaux est < 250).
- Découpage entraînement/validation/test décidé **par enregistrement**, jamais par imagette.
- `--layers` est incompatible avec `--nir` (erreur claire, code de sortie 2).
- `README.md` est entièrement en CRLF : tout ajout doit rester en CRLF.
- Chaque message de commit se termine par la ligne `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

- Le suffixe `_irc` historique est conservé : les caches IRC existants restent valables (Task 1).
- Un nom de couche avec points, tirets bas ou majuscules ne produit que des caractères `[a-z0-9-]` dans le suffixe (Task 1).
- Un téléchargement qui échoue n'interrompt pas les autres ; liste vide de tuiles sans erreur (Task 2).
- Fenêtre vide dans une couche mais pas dans l'autre : une seule imagette, pas de plantage ; enregistrement vide partout : ignoré (Task 4).
- Les deux rendus d'un même enregistrement tombent dans la même partie (Task 4).
- `--layers` avec `--nir` : erreur ; sans `--layers`, sortie identique à avant (Task 4).
- Un point dont la fenêtre échoue après préchargement est sauté sans plantage dans les scripts d'évaluation (Task 3).

---

## Structure des fichiers

- Modifier `detection_ortho/tiles.py` : `layer_tag`, `prefetch_tiles`, `download_tile` utilise `layer_tag`.
- Modifier `detection_ortho/tilecache.py` : motif de purge.
- Modifier `detection_ortho/dataset.py` : `tiles_for_points`, `prefetch_points`, `window_is_blank`.
- Modifier `scripts/sweep_threshold.py`, `scripts/eval_points.py` : `--layer`, `--workers`, préchargement.
- Modifier `scripts/build_dataset.py` : `--layers`.
- Modifier `README.md` : section de runbook.
- Tests : modifier `tests/test_tiles_download.py`, `tests/test_tiles_layer.py`, `tests/test_tilecache.py` ; créer `tests/test_prefetch.py`, `tests/test_window_is_blank.py`, `tests/test_eval_scripts_layer.py`, `tests/test_build_dataset_layers.py`.

---

### Task 1: Suffixe de cache unique par couche et purge

**Files:**
- Modify: `detection_ortho/tiles.py` (ajout de `layer_tag`, `download_tile` l'utilise)
- Modify: `detection_ortho/tilecache.py:27` (`_TILE_RE`)
- Test: `tests/test_tiles_layer.py`, `tests/test_tiles_download.py`, `tests/test_tilecache.py`

**Interfaces:**
- Consumes: rien.
- Produces: `layer_tag(layer: str) -> str` dans `detection_ortho.tiles` ; le motif de purge accepte un suffixe `_[a-z0-9-]+`.

- [ ] **Step 1: Écrire les tests qui échouent**

Ajouter à la fin de `tests/test_tiles_layer.py` :

```python


import re

from detection_ortho.tiles import layer_tag


def test_layer_tag_standard_is_empty():
    assert layer_tag(LAYER) == ""


def test_layer_tag_irc_keeps_historical_suffix():
    assert layer_tag(LAYER_IRC) == "_irc"


def test_layer_tag_2026_layers_do_not_collide():
    rvb = layer_tag("ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026")
    irc = layer_tag("ORTHOIMAGERY.ORTHOPHOTOS.IRC-EXPRESS.2026")
    assert rvb == "_rvb-express-2026"
    assert irc == "_irc-express-2026"
    assert rvb != irc


def test_layer_tag_only_safe_characters():
    for name in ("ORTHOIMAGERY.ORTHO-SAT.PLEIADES.2026", "ORTHOIMAGERY.ORTHOPHOTOS.A_B.c"):
        assert re.fullmatch(r"_[a-z0-9-]+", layer_tag(name)), name
```

Ajouter à la fin de `tests/test_tiles_download.py` :

```python


def test_download_tile_separates_layers_sharing_a_year(tmp_path):
    sess = FakeSession()
    a = download_tile(1, 2, 19, tmp_path, session=sess,
                      layer="ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026")
    b = download_tile(1, 2, 19, tmp_path, session=sess,
                      layer="ORTHOIMAGERY.ORTHOPHOTOS.IRC-EXPRESS.2026")
    assert a != b
    assert a.exists() and b.exists()
    assert sess.calls == 2
```

Ajouter à la fin de `tests/test_tilecache.py` :

```python


def test_purge_cache_removes_tagged_layer_tiles(tmp_path):
    keep = {(1, 1)}
    for name in ("19_1_1.jpg", "19_2_2.jpg", "19_2_2_irc.jpg",
                 "19_2_2_rvb-express-2026.jpg", "18_2_2_rvb-express-2026.jpg"):
        (tmp_path / name).write_bytes(b"x")
    deleted, kept, _ = purge_cache(tmp_path, keep, 19)
    assert deleted == 3 and kept == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "18_2_2_rvb-express-2026.jpg", "19_1_1.jpg"]
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_tiles_layer.py tests/test_tiles_download.py tests/test_tilecache.py -q`
Expected: FAIL (`ImportError: cannot import name 'layer_tag'`).

- [ ] **Step 3: Implémenter**

Dans `detection_ortho/tiles.py`, juste après la ligne `LAYER_IRC = ...` (ligne 18), ajouter :

```python
_LAYER_PREFIX = "ORTHOIMAGERY.ORTHOPHOTOS."


def layer_tag(layer: str) -> str:
    """Suffixe de nom de fichier de cache propre à la couche.

    Vide pour la couche standard ; sinon `_` + nom court (sans le préfixe
    ORTHOIMAGERY.ORTHOPHOTOS.), en minuscules, avec points et tirets bas
    remplacés par des tirets. IRC donne `_irc`, comme avant.
    """
    if layer == LAYER:
        return ""
    short = layer[len(_LAYER_PREFIX):] if layer.startswith(_LAYER_PREFIX) else layer
    return "_" + short.lower().replace(".", "-").replace("_", "-")
```

Dans `download_tile`, remplacer la ligne :

```python
    tag = "" if layer == LAYER else "_" + layer.rsplit(".", 1)[-1].lower()
```

par :

```python
    tag = layer_tag(layer)
```

Dans `detection_ortho/tilecache.py`, remplacer la ligne 27 :

```python
_TILE_RE = re.compile(r"^(\d+)_(-?\d+)_(-?\d+)(_[a-z]+)?\.jpg$")
```

par :

```python
_TILE_RE = re.compile(r"^(\d+)_(-?\d+)_(-?\d+)(_[a-z0-9-]+)?\.jpg$")
```

- [ ] **Step 4: Vérifier la réussite**

Run: `.venv\Scripts\python -m pytest tests/test_tiles_layer.py tests/test_tiles_download.py tests/test_tilecache.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/tiles.py detection_ortho/tilecache.py tests/test_tiles_layer.py tests/test_tiles_download.py tests/test_tilecache.py
git commit -m "fix: suffixe de cache unique par couche WMTS et purge des tuiles suffixées

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Préchargement parallèle des tuiles

**Files:**
- Modify: `detection_ortho/tiles.py` (ajout de `prefetch_tiles`)
- Modify: `detection_ortho/dataset.py` (import + `tiles_for_points`, `prefetch_points`)
- Test: `tests/test_prefetch.py`

**Interfaces:**
- Consumes: `download_tile(x, y, zoom, cache_dir, session=None, layer=LAYER)` (existant, Task 1 pour le suffixe).
- Produces:
  - `prefetch_tiles(tiles, cache_dir, layer=LAYER, zoom=19, workers=12, session=None, on_progress=None) -> list[str]` : liste des messages d'échec (vide si tout va bien) ; `on_progress(i, total)` appelé après chaque tuile.
  - `tiles_for_points(points, zoom=19, window_px=640) -> set[tuple[int, int]]` (points = dicts avec `lon`, `lat`).
  - `prefetch_points(points, cache_dir, layers, workers=12, zoom=19, window_px=640) -> list[str]`.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_prefetch.py` :

```python
import threading

from detection_ortho import dataset as ds
from detection_ortho import tiles as tiles_mod
from detection_ortho.dataset import prefetch_points, tiles_for_points, window_tiles
from detection_ortho.tiles import prefetch_tiles


def test_prefetch_tiles_runs_downloads_in_parallel(tmp_path, monkeypatch):
    # Trois téléchargements doivent être en cours en même temps pour franchir la barrière.
    barrier = threading.Barrier(3, timeout=5)

    def fake(x, y, zoom, cache_dir, session=None, layer=None, **kw):
        barrier.wait()
        return tmp_path / f"{x}_{y}.jpg"

    monkeypatch.setattr(tiles_mod, "download_tile", fake)
    errors = prefetch_tiles([(1, 1), (2, 2), (3, 3)], tmp_path, workers=3)
    assert errors == []


def test_prefetch_tiles_continues_after_failures_and_reports_them(tmp_path, monkeypatch):
    def fake(x, y, zoom, cache_dir, session=None, layer=None, **kw):
        if x == 2:
            raise RuntimeError("boom")
        return tmp_path / "ok.jpg"

    monkeypatch.setattr(tiles_mod, "download_tile", fake)
    errors = prefetch_tiles([(1, 1), (2, 2), (3, 3)], tmp_path, workers=2)
    assert len(errors) == 1 and "2,2" in errors[0]


def test_prefetch_tiles_empty_list(tmp_path):
    assert prefetch_tiles([], tmp_path) == []


def test_prefetch_tiles_reports_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(tiles_mod, "download_tile",
                        lambda *a, **k: tmp_path / "x.jpg")
    seen = []
    prefetch_tiles([(1, 1), (2, 2)], tmp_path,
                   on_progress=lambda i, n: seen.append((i, n)))
    assert seen == [(1, 2), (2, 2)]


def test_tiles_for_points_matches_window_tiles():
    pts = [{"lon": 0.65, "lat": 47.33}, {"lon": 0.66, "lat": 47.34}]
    expected = set()
    for p in pts:
        expected.update(window_tiles(p["lon"], p["lat"], 19, 640)[0])
    assert tiles_for_points(pts) == expected


def test_prefetch_points_prefetches_each_layer(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        ds, "prefetch_tiles",
        lambda tiles, cache, layer=None, **k: calls.append((layer, len(tiles))) or [])
    errors = prefetch_points([{"lon": 0.65, "lat": 47.33}], tmp_path, ["L1", "L2"])
    assert [c[0] for c in calls] == ["L1", "L2"]
    assert calls[0][1] == calls[1][1] > 0
    assert errors == []
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_prefetch.py -q`
Expected: FAIL (`ImportError: cannot import name 'prefetch_tiles'`).

- [ ] **Step 3: Implémenter**

Dans `detection_ortho/tiles.py`, remplacer l'import `import time` par :

```python
import time
from concurrent.futures import ThreadPoolExecutor
```

(garder les autres imports), puis ajouter après la fonction `download_tile` :

```python
def prefetch_tiles(
    tiles, cache_dir, layer: str = LAYER, zoom: int = 19, workers: int = 12,
    session=None, on_progress=None,
) -> list[str]:
    """Télécharge les tuiles (x, y) en parallèle dans le cache.

    Un échec n'interrompt pas les autres : retourne la liste des messages
    d'échec (vide si tout va bien). `on_progress(i, total)` est appelé, depuis
    le fil appelant, après chaque tuile traitée.
    """
    tiles = list(tiles)
    if not tiles:
        return []
    sess = session or requests.Session()

    def one(xy):
        x, y = xy
        try:
            download_tile(x, y, zoom, cache_dir, session=sess, layer=layer)
            return None
        except Exception as exc:  # noqa: BLE001
            return f"tuile {x},{y} échec ({exc})"

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, err in enumerate(pool.map(one, tiles), 1):
            if err:
                errors.append(err)
            if on_progress:
                on_progress(i, len(tiles))
    return errors
```

Dans `detection_ortho/dataset.py`, remplacer la ligne d'import :

```python
from detection_ortho.tiles import lonlat_to_pixel, download_tile, LAYER
```

par :

```python
from detection_ortho.tiles import lonlat_to_pixel, download_tile, prefetch_tiles, LAYER
```

et ajouter, juste après la fonction `window_tiles` :

```python
def tiles_for_points(points, zoom: int = 19, window_px: int = 640) -> set:
    """Tuiles (x, y) nécessaires pour assembler une fenêtre centrée sur chaque point {lon, lat}."""
    needed: set = set()
    for p in points:
        tiles, _, _ = window_tiles(p["lon"], p["lat"], zoom, window_px)
        needed.update(tiles)
    return needed


def prefetch_points(
    points, cache_dir, layers, workers: int = 12, zoom: int = 19,
    window_px: int = 640,
) -> list[str]:
    """Précharge en parallèle les tuiles des fenêtres de tous les points, pour chaque couche."""
    tiles = tiles_for_points(points, zoom, window_px)
    errors: list[str] = []
    for layer in layers:
        errors += prefetch_tiles(tiles, cache_dir, layer=layer, zoom=zoom, workers=workers)
    return errors
```

- [ ] **Step 4: Vérifier la réussite**

Run: `.venv\Scripts\python -m pytest tests/test_prefetch.py tests/test_tiles_download.py tests/test_tiles_layer.py tests/test_dataset_window.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/tiles.py detection_ortho/dataset.py tests/test_prefetch.py
git commit -m "feat: préchargement parallèle des tuiles (prefetch_tiles, prefetch_points)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `--layer` et préchargement parallèle dans les scripts d'évaluation

**Files:**
- Modify: `scripts/sweep_threshold.py`
- Modify: `scripts/eval_points.py`
- Test: `tests/test_eval_scripts_layer.py`

**Interfaces:**
- Consumes: `prefetch_points(points, cache_dir, layers, workers=12, zoom=19, window_px=640)` (Task 2) ; `LAYER`, `LAYER_IRC`.
- Produces: option `--layer` (défaut : couche standard) et `--workers` (défaut 12) sur les deux scripts ; `--nir` incompatible avec `--layer` autre que la couche standard.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_eval_scripts_layer.py` :

```python
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import eval_points  # noqa: E402
import sweep_threshold  # noqa: E402

L26 = "ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026"


def _csv(tmp_path):
    p = tmp_path / "v.csv"
    p.write_text("index,lat,lon,score,verdict\n1,47.33,0.65,0.9,vrai\n2,47.34,0.66,0.5,faux\n",
                 encoding="utf-8")
    return p


class _FakeModel:
    def __init__(self, *a, **k):
        pass

    def predict(self, img, **k):
        return [types.SimpleNamespace(boxes=None)]


def _patch(monkeypatch, module, seen, read):
    def fake_prefetch(points, cache, layers, workers=12, zoom=19, window_px=640):
        seen["layers"] = list(layers)
        seen["n"] = len(points)
        seen["workers"] = workers
        return []

    def fake_assemble(lon, lat, zoom, window, cache, layer=None, **k):
        read.append(layer)
        return np.zeros((640, 640, 3), np.uint8), 0.0, 0.0

    monkeypatch.setattr(module, "YOLO", _FakeModel)
    monkeypatch.setattr(module, "prefetch_points", fake_prefetch)
    monkeypatch.setattr(module, "assemble_window", fake_assemble)
    monkeypatch.setattr(module, "result_to_boxes", lambda b: [])
    monkeypatch.setattr(module, "boxes_to_points", lambda *a, **k: [])


@pytest.mark.parametrize("module,name", [(sweep_threshold, "sweep_threshold.py"),
                                         (eval_points, "eval_points.py")])
def test_scripts_prefetch_then_read_the_requested_layer(module, name, tmp_path, monkeypatch):
    seen, read = {}, []
    _patch(monkeypatch, module, seen, read)
    monkeypatch.setattr(sys, "argv", [name, "--weights", "w.pt", "--verdicts",
                                      str(_csv(tmp_path)), "--layer", L26,
                                      "--workers", "5", "--cache", str(tmp_path / "c")])
    module.main()
    assert seen == {"layers": [L26], "n": 2, "workers": 5}
    assert read == [L26, L26]


@pytest.mark.parametrize("module,name", [(sweep_threshold, "sweep_threshold.py"),
                                         (eval_points, "eval_points.py")])
def test_scripts_default_layer_is_the_standard_one(module, name, tmp_path, monkeypatch):
    seen, read = {}, []
    _patch(monkeypatch, module, seen, read)
    monkeypatch.setattr(sys, "argv", [name, "--weights", "w.pt", "--verdicts",
                                      str(_csv(tmp_path)), "--cache", str(tmp_path / "c")])
    module.main()
    assert seen["layers"] == ["ORTHOIMAGERY.ORTHOPHOTOS"]
    assert read == ["ORTHOIMAGERY.ORTHOPHOTOS"] * 2


@pytest.mark.parametrize("name", ["sweep_threshold.py", "eval_points.py"])
def test_scripts_reject_nir_with_another_layer(name):
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / name), "--weights", "w.pt",
         "--verdicts", "v.csv", "--nir", "--layer", L26],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 2
    assert "incompatible" in r.stderr


@pytest.mark.parametrize("name", ["sweep_threshold.py", "eval_points.py"])
def test_scripts_help_mentions_layer(name):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / name), "--help"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "--layer" in r.stdout and "--workers" in r.stdout
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_eval_scripts_layer.py -q`
Expected: FAIL (`module 'sweep_threshold' has no attribute 'prefetch_points'`).

- [ ] **Step 3: Implémenter `scripts/sweep_threshold.py`**

1. Remplacer la ligne d'import :
```python
from detection_ortho.dataset import assemble_window, compose_rgn, parse_verdicts
from detection_ortho.tiles import LAYER_IRC
```
par :
```python
from detection_ortho.dataset import (
    assemble_window, compose_rgn, parse_verdicts, prefetch_points,
)
from detection_ortho.tiles import LAYER, LAYER_IRC
```
2. Après la ligne `ap.add_argument("--device", type=str, default="cpu")`, ajouter :
```python
    ap.add_argument("--layer", type=str, default=LAYER,
                    help="couche WMTS à évaluer (défaut : ortho habituelle)")
    ap.add_argument("--workers", type=int, default=12,
                    help="téléchargements de tuiles en parallèle")
```
3. Juste après `args = ap.parse_args()`, ajouter :
```python
    if args.nir and args.layer != LAYER:
        ap.error("--nir et --layer sont incompatibles")
```
4. Juste avant la ligne `model = YOLO(str(args.weights))`, ajouter :
```python
    layers = [args.layer] + ([LAYER_IRC] if args.nir else [])
    errors = prefetch_points(verdicts, args.cache, layers, workers=args.workers,
                             zoom=args.zoom, window_px=args.window)
    print(f"Tuiles préchargées en parallèle ({len(layers)} couche(s)) : "
          f"{len(errors)} en échec.", flush=True)
```
5. Remplacer l'appel :
```python
            img, ogx, ogy = assemble_window(lon, lat, args.zoom, args.window, args.cache)
```
par :
```python
            img, ogx, ogy = assemble_window(lon, lat, args.zoom, args.window,
                                            args.cache, layer=args.layer)
```

- [ ] **Step 4: Implémenter `scripts/eval_points.py`**

Mêmes cinq modifications, avec ces particularités : l'import d'origine est `from detection_ortho.dataset import assemble_window, compose_rgn, parse_verdicts` suivi de `from detection_ortho.tiles import LAYER_IRC` (remplacer comme au point 1 ci-dessus) ; les nouvelles options s'ajoutent après `ap.add_argument("--device", type=str, default="cpu")` ; le bloc de préchargement (point 4) s'insère juste avant `model = YOLO(str(args.weights))` ; l'appel à modifier est :
```python
            img, ogx, ogy = assemble_window(lon, lat, args.zoom, args.window, args.cache)
```
qui devient :
```python
            img, ogx, ogy = assemble_window(lon, lat, args.zoom, args.window,
                                            args.cache, layer=args.layer)
```

- [ ] **Step 5: Vérifier la réussite**

Run: `.venv\Scripts\python -m pytest tests/test_eval_scripts_layer.py tests/test_eval_points.py tests/test_evaluate_help.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/sweep_threshold.py scripts/eval_points.py tests/test_eval_scripts_layer.py
git commit -m "feat: --layer et préchargement parallèle dans sweep_threshold et eval_points

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `build_dataset.py --layers` et runbook

**Files:**
- Modify: `detection_ortho/dataset.py` (ajout de `window_is_blank`)
- Modify: `scripts/build_dataset.py` (imports, option `--layers`, préchargement, boucle d'imagettes)
- Modify: `README.md` (nouvelle section en fin de fichier, CRLF)
- Test: `tests/test_window_is_blank.py`, `tests/test_build_dataset_layers.py`

**Interfaces:**
- Consumes: `layer_tag` (Task 1), `prefetch_tiles` (Task 2), `assemble_window(..., layer=...)` (existant).
- Produces: `window_is_blank(image, min_frac=0.05) -> bool` ; option `--layers L1 [L2 ...]` ; imagettes nommées `<enregistrement>` pour la première couche et `<enregistrement>__<suffixe sans _>` pour les suivantes.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_window_is_blank.py` :

```python
import numpy as np

from detection_ortho.dataset import window_is_blank


def test_all_white_is_blank():
    assert window_is_blank(np.full((64, 64, 3), 255, np.uint8))


def test_gray_window_is_not_blank():
    assert not window_is_blank(np.full((64, 64, 3), 128, np.uint8))


def test_mostly_white_with_a_few_pixels_is_blank():
    img = np.full((100, 100, 3), 255, np.uint8)
    img[:1, :30] = 100          # 30 pixels sur 10 000 = 0,3 %
    assert window_is_blank(img)


def test_a_tenth_of_non_white_pixels_is_not_blank():
    img = np.full((100, 100, 3), 255, np.uint8)
    img[:10, :] = 100           # 10 %
    assert not window_is_blank(img)
```

Créer `tests/test_build_dataset_layers.py` :

```python
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import build_dataset  # noqa: E402
from detection_ortho.dataset import window_tiles  # noqa: E402
from detection_ortho.tiles import LAYER, layer_tag  # noqa: E402

L26 = "ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026"


def _seed(cache, lon, lat, layer, value):
    cache.mkdir(parents=True, exist_ok=True)
    tiles, _, _ = window_tiles(lon, lat, 19, 640)
    for x, y in tiles:
        cv2.imwrite(str(cache / f"19_{x}_{y}{layer_tag(layer)}.jpg"),
                    np.full((256, 256, 3), value, np.uint8))


def test_layers_one_chip_per_available_layer_same_partition(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(build_dataset, "fetch_features_geom", lambda *a, **k: [])
    out = tmp_path / "ds"
    cache = out / "tiles_cache"
    a, b = (0.65, 47.33), (0.66, 47.34)
    _seed(cache, *a, LAYER, 128)
    _seed(cache, *a, L26, 120)
    _seed(cache, *b, LAYER, 128)
    _seed(cache, *b, L26, 255)           # fenêtre blanche en 2026 pour B
    verdicts = tmp_path / "v.csv"
    verdicts.write_text(
        "index,lat,lon,score,verdict\n"
        f"1,{a[1]},{a[0]},0.9,vrai\n"
        f"2,{b[1]},{b[0]},0.9,vrai\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "build_dataset.py", "--bbox", "0.6", "47.3", "0.7", "47.4",
        "--negatives", "0", "--max-pools", "0", "--layers", LAYER, L26,
        "--verdicts", str(verdicts), "--out", str(out)])
    build_dataset.main()

    images = {p.stem: p.parent.name for p in (out / "images").rglob("*.jpg")}
    assert set(images) == {"revpos_0000", "revpos_0001", "revpos_0000__rvb-express-2026"}
    assert images["revpos_0000"] == images["revpos_0000__rvb-express-2026"]
    labels = {p.stem for p in (out / "labels").rglob("*.txt")}
    assert labels == set(images)
    summary = capsys.readouterr().out
    assert "1 fenêtre(s) vide(s)" in summary


def test_layers_incompatible_with_nir(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_dataset.py", "--bbox", "0", "0", "1", "1",
                                      "--nir", "--layers", LAYER, L26])
    with pytest.raises(SystemExit) as exc:
        build_dataset.main()
    assert exc.value.code == 2
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_window_is_blank.py tests/test_build_dataset_layers.py -q`
Expected: FAIL (`ImportError: cannot import name 'window_is_blank'`).

- [ ] **Step 3: Implémenter `window_is_blank`**

Dans `detection_ortho/dataset.py`, ajouter après `prefetch_points` :

```python
def window_is_blank(image, min_frac: float = 0.05) -> bool:
    """Vrai si moins de `min_frac` des pixels ne sont pas (quasi) blancs : pas de donnée à cet endroit."""
    non_white = (image.min(axis=2) < 250).mean()
    return float(non_white) < min_frac
```

- [ ] **Step 4: Implémenter `--layers` dans `scripts/build_dataset.py`**

1. Dans l'import depuis `detection_ortho.dataset`, ajouter `window_is_blank` à la fin de la liste (après `dedup_verdicts, near_any,`). Remplacer la ligne :
```python
from detection_ortho.tiles import download_tile, LAYER_IRC
```
par :
```python
from detection_ortho.tiles import LAYER, LAYER_IRC, layer_tag, prefetch_tiles
```
2. Après l'argument `--nir` (celui dont l'aide est `imagettes [R,G,NIR] ...`), ajouter :
```python
    ap.add_argument("--layers", nargs="+", default=[LAYER], metavar="COUCHE",
                    help="couches WMTS à rendre : chaque enregistrement donne une "
                         "imagette par couche disponible (défaut : ortho habituelle)")
```
3. Juste après `args = ap.parse_args()`, ajouter :
```python
    if args.nir and list(args.layers) != [LAYER]:
        ap.error("--nir est incompatible avec --layers")
```
4. Remplacer tout le bloc compris entre le commentaire `# --- Récupération des images : pré-téléchargement parallèle ...` et le `if errors: print(...)` qui le suit (jusqu'à `file=sys.stderr)` inclus), c'est-à-dire de la ligne `needed = set()` jusqu'à l'impression `... tuile(s) en échec (réessayées à l'assemblage).`, par :
```python
    # --- Récupération des images : préchargement parallèle des tuiles (dédupliquées), par couche ---
    needed = set()
    for _name, lon, lat, _bbox in records:
        tiles, _, _ = window_tiles(lon, lat, ZOOM, WINDOW)
        needed.update(tiles)

    def _tile_progress(i, n):
        if i % max(1, n // 50) == 0 or i == n:
            print(f"  Récupération des tuiles: {i}/{n}", flush=True)

    # Session partagée = keep-alive (évite un handshake TCP/TLS par tuile).
    session = requests.Session()
    for layer in list(args.layers) + ([LAYER_IRC] if args.nir else []):
        print(f"{len(needed)} tuile(s) à récupérer, couche {layer} "
              f"(parallèle x{args.workers})...")
        errors = prefetch_tiles(needed, cache, layer=layer, zoom=ZOOM,
                                workers=args.workers, session=session,
                                on_progress=_tile_progress)
        if errors:
            print(f"  {len(errors)} tuile(s) en échec (réessayées à l'assemblage).",
                  file=sys.stderr)
```
5. Remplacer la boucle de génération des imagettes, de `qa_crops = []` jusqu'à la ligne `write_chip(win_img, labels, imgs / part, lbls / part, name)` incluse, par :
```python
    qa_crops = []
    written = {layer: 0 for layer in args.layers}
    blank = {layer: 0 for layer in args.layers}
    failed = {layer: 0 for layer in args.layers}
    for i, (name, lon, lat, bbox_geo) in enumerate(
        progress(records, len(records), "Génération des chips")
    ):
        part = where[i]  # un enregistrement = une seule partie, quelle que soit la couche
        for k, layer in enumerate(args.layers):
            chip_name = name if k == 0 else f"{name}__{layer_tag(layer).lstrip('_')}"
            try:
                win_img, ogx, ogy = assemble_window(lon, lat, ZOOM, WINDOW, cache,
                                                    layer=layer)
                if args.nir:
                    irc_img, _, _ = assemble_window(
                        lon, lat, ZOOM, WINDOW, cache, layer=LAYER_IRC)
                    win_img = compose_rgn(win_img, irc_img)
            except Exception as exc:  # noqa: BLE001
                failed[layer] += 1
                print(f"  {chip_name}: échec fenêtre ({exc})", file=sys.stderr)
                continue
            if window_is_blank(win_img):
                blank[layer] += 1
                continue
            labels = []
            if bbox_geo is not None:
                px = geo_bbox_to_pixel_bbox(bbox_geo, ogx, ogy, ZOOM, WINDOW)
                line = to_yolo_label(px, WINDOW)
                if line:
                    labels.append(line)
                    if k == 0 and name.startswith("citerne") and len(qa_crops) < 48:
                        x0, y0, x1, y1 = (int(v) for v in px)
                        vis = win_img.copy()
                        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
                        qa_crops.append(cv2.resize(vis, (128, 128)))
            write_chip(win_img, labels, imgs / part, lbls / part, chip_name)
            written[layer] += 1
    for layer in args.layers:
        print(f"Couche {layer} : {written[layer]} imagette(s), "
              f"{blank[layer]} fenêtre(s) vide(s), {failed[layer]} échec(s).")
```

- [ ] **Step 5: Vérifier la réussite (nouveaux tests + non-régression du script)**

Run: `.venv\Scripts\python -m pytest tests/test_window_is_blank.py tests/test_build_dataset_layers.py tests/test_build_dataset_integration.py tests/test_build_dataset_verdicts.py tests/test_build_dataset_multi_verdicts.py tests/test_build_dataset_nir.py -q`
Expected: PASS (les tests existants préremplissent des tuiles grises, jamais blanches : `window_is_blank` ne les affecte pas).

- [ ] **Step 6: Section de runbook dans le README**

Ajouter à la fin de `README.md` (en conservant des fins de ligne CRLF sur toutes les lignes ajoutées) :

````markdown

## Adapter le modèle à l'ortho express 2026

L'ortho express 2026 (couche `ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026`) a un autre rendu (ombres,
reflets, éclairage) : le modèle affiné y perd du rappel. On l'adapte avec des imagettes des deux couches
aux mêmes points de verdicts. Le 36 et le 49 restent mis de côté ; le 44 et le 49 n'ont pas de 2026 (ils
n'apportent que leurs imagettes habituelles, les fenêtres vides sont sautées).

1. **Jeu de données à deux couches** (tout sauf 36 et 49) :

       python scripts/build_dataset.py --bbox 0.05 46.72 1.06 47.72 `
           --layers ORTHOIMAGERY.ORTHOPHOTOS ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026 `
           --verdicts verdicts_maproulette/verdicts_18.csv `
                      verdicts_maproulette/verdicts_28.csv `
                      verdicts_maproulette/verdicts_37.csv `
                      verdicts_maproulette/verdicts_41.csv `
                      verdicts_maproulette/verdicts_44.csv `
                      verdicts_maproulette/verdicts_45.csv `
           --holdout verdicts_maproulette/verdicts_36.csv `
                     verdicts_maproulette/verdicts_49.csv `
           --spatial-split --out dataset_mr2026

   Chaque point donne une imagette par couche disponible (`<nom>` et `<nom>__rvb-express-2026`), toujours
   dans la même partie (entraînement, validation ou test). Un résumé par couche indique les imagettes
   écrites, les fenêtres vides et les échecs.

2. **Affiner** depuis les poids actuels (≈ 20 époques ; environ 10 h de CPU, à confirmer sur une époque) :

       python scripts/train.py --data dataset_mr2026/data.yaml `
           --model models/citernes-yolov8n.pt --epochs 20 --device cpu --name citernes_2026

3. **Évaluer** (les tuiles sont préchargées en parallèle) :

       python scripts/sweep_threshold.py --weights runs/citernes_2026/weights/best.pt `
           --verdicts verdicts_maproulette/verdicts_36.csv `
           --layer ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026
       python scripts/sweep_threshold.py --weights runs/citernes_2026/weights/best.pt `
           --verdicts verdicts_maproulette/verdicts_36.csv
       python scripts/sweep_threshold.py --weights runs/citernes_2026/weights/best.pt `
           --verdicts verdicts_maproulette/verdicts_49.csv

   **Critère d'acceptation** (seuil 0,25) : sur le 36 en 2026, rappel ≥ 0,85 et précision ≥ 0,85 (avant :
   0,60 et 0,92) ; sur l'ortho habituelle, perte ≤ 0,03 de rappel et de précision (avant : 0,895 et 0,879
   au 36, 0,924 et 0,948 au 49). Sinon, ne pas remplacer `models/citernes-yolov8n.pt`.
````

Puis vérifier que le fichier est resté entièrement en CRLF :

Run: `.venv\Scripts\python -c "d=open('README.md','rb').read(); assert d.count(b'\r\n')==d.count(b'\n'); print('CRLF ok', d.count(b'\n'), 'lignes')"`
Expected: `CRLF ok <N> lignes`.

- [ ] **Step 7: Suite complète puis commit**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS (tous les tests).

```bash
git add detection_ortho/dataset.py scripts/build_dataset.py README.md tests/test_window_is_blank.py tests/test_build_dataset_layers.py
git commit -m "feat: build_dataset --layers (imagettes multi-couches, découpage par enregistrement) et runbook ortho 2026

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Étapes manuelles (hors plan, côté utilisateur)

1. Lancer la construction du jeu à deux couches, l'entraînement (≈ 10 h) et les évaluations du runbook.
2. Décider du remplacement de `models/citernes-yolov8n.pt` selon le critère d'acceptation.
3. Sous-projet B (lanceur par zones de couverture) : spec séparée, après mesure.
