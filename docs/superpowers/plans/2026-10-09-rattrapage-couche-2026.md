# Rattrapage des départements déjà traités (couche 2026) — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permettre de repasser un département avec le nouveau modèle sur une couche WMTS au choix (l'ortho express 2026), en n'inférant que les zones couvertes, en streamant les tuiles sans saturer le disque, et sans remontrer les faux positifs déjà rejetés.

**Architecture:** `infer_area.py` gagne `--layer` (streaming et assemblage sur cette couche, suffixe de cache propre déjà en place), un filtre de couverture qui sonde des tuiles de zoom 13 **en mémoire** pour écarter les fenêtres sans donnée, le saut des fenêtres blanches, et `--known-false` (candidats proches d'un verdict « faux » retirés du challenge, listés dans `suppressed.geojson`). Deux petits modules purs et testés : `coverage.py` et `known_false.py`.

**Tech Stack:** Python 3.12, pytest, requests, ThreadPoolExecutor, OpenCV, shapely/ultralytics (existants).

**Design validé avec l'utilisateur (2026-10-09).** Rattrapage des 8 départements déjà traités : ortho 2026 là où elle existe (18, 28, 36, 37, 41, 45, certains couverts en partie), ortho habituelle pour le 44 et le 49 (pas de 2026). Les départements nouveaux suivront la même commande, un par un, plus tard.

## Global Constraints

- **Disque borné, tuiles streamées en parallèle, jamais conservées.** Aucun code ajouté ne laisse de tuile sur le disque : la sonde de couverture travaille uniquement en mémoire (aucune écriture) ; le cache des tuiles d'inférence reste plafonné par `--cache-gb` (défaut 10 Go) et purgé à chaque tranche et à la fin, **y compris pour les tuiles suffixées d'une couche non standard** (`..._rvb-express-2026.jpg`).
- Tuiles toujours téléchargées en parallèle (`--workers`, 12 par défaut) ; jamais tuile par tuile.
- Comportement par défaut inchangé : sans `--layer`, `--known-false`, les résultats, l'empreinte du point de reprise et les noms de fichiers sont identiques à avant.
- Couche non standard : un seul essai par tuile (un 404 signifie « pas de donnée »). Couche standard : 3 essais comme aujourd'hui (404 transitoires documentés).
- Une réponse indéterminée de la sonde (erreur réseau) ne fait **jamais** écarter une fenêtre.
- `--layer` est incompatible avec `--ortho` (erreur claire, code de sortie 2).
- Les fichiers de `--known-false` sont lus **au démarrage** (une faute de chemin doit échouer tout de suite, pas après des heures de calcul). Seuls les verdicts « faux » comptent.
- Tests hors-ligne uniquement : aucun appel réseau dans pytest.
- `README.md` est entièrement en CRLF : tout ajout doit rester en CRLF (vérifier : `.venv\Scripts\python -c "d=open('README.md','rb').read(); assert d.count(b'\r\n')==d.count(b'\n'); print('CRLF ok')"`).
- Chaque message de commit se termine par la ligne `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

- La sonde de couverture n'écrit rien sur le disque et n'écarte jamais une fenêtre sur une erreur indéterminée (Task 1).
- Les tuiles suffixées d'une couche non standard sont bien purgées à chaque tranche et à la fin du streaming : le cache ne grossit pas (Task 3).
- L'empreinte du point de reprise est inchangée pour la couche standard (on ne casse pas un point de reprise existant) (Task 3).
- Une fenêtre entièrement blanche est sautée sans inférence et comptée (Task 3).
- `--known-false` : fichier absent → échec immédiat ; seuls les « faux » suppriment ; les « vrai » jamais ; `suppressed.geojson` écrit (Tasks 2 et 3).
- Départements partiellement couverts : le filtre réduit la grille, l'empreinte suit, et « aucune fenêtre couverte » sort proprement (Task 3).

---

## Structure des fichiers

- Créer `detection_ortho/coverage.py` : sonde de couverture en mémoire.
- Créer `detection_ortho/known_false.py` : lecture des faux et suppression par proximité.
- Modifier `scripts/infer_area.py` : `--layer`, `--known-false`, `--known-false-m`, filtre de couverture, fenêtres blanches, empreinte.
- Modifier `README.md` : section de runbook.
- Tests : créer `tests/test_coverage.py`, `tests/test_known_false.py`, `tests/test_infer_area_layer.py`.

---

### Task 1: Sonde de couverture en mémoire

**Files:**
- Create: `detection_ortho/coverage.py`
- Test: `tests/test_coverage.py`

**Interfaces:**
- Consumes: `detection_ortho.tiles.tile_for_lonlat(lon, lat, zoom) -> (x, y)`, `tile_url(x, y, zoom, layer)`, `USER_AGENT` ; `detection_ortho.dataset.window_is_blank(image, min_frac=0.05) -> bool`.
- Produces:
  - `PROBE_ZOOM = 13`
  - `fetch_probe_tile(x, y, zoom, layer, session=None, tries=3, pause=1.0) -> bytes | None` (None si HTTP 404 ; lève après `tries` échecs autres)
  - `probe_tiles(tiles, layer, workers=12, fetch=None) -> dict[(x, y)] -> True | False | None` (True = données, False = vide, None = indéterminé)
  - `filter_windows_by_coverage(centers, layer, workers=12, fetch=None) -> tuple[list, int, int]` = (fenêtres conservées, nombre écartées, nombre de tuiles de sonde indéterminées)

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_coverage.py` :

```python
import threading

import cv2
import numpy as np

from detection_ortho.coverage import (
    PROBE_ZOOM, fetch_probe_tile, filter_windows_by_coverage, probe_tiles,
)
from detection_ortho.tiles import tile_for_lonlat

LAYER = "ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026"


def _jpeg(value):
    ok, buf = cv2.imencode(".jpg", np.full((64, 64, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


# Quatre centres répartis dans quatre tuiles de zoom 13 distinctes.
CENTERS = [(0.65, 47.33), (0.80, 47.33), (0.95, 47.33), (1.10, 47.33)]
KEYS = [tile_for_lonlat(lon, lat, PROBE_ZOOM) for lon, lat in CENTERS]


def test_the_four_centers_are_in_four_distinct_tiles():
    assert len(set(KEYS)) == 4


def test_filter_drops_empty_and_404_tiles_and_keeps_data_and_unknown():
    answers = {KEYS[0]: _jpeg(128),            # données
               KEYS[1]: _jpeg(255),            # tuile blanche : vide
               KEYS[2]: None,                  # 404 : vide
               KEYS[3]: RuntimeError("boom")}  # erreur réseau : indéterminé

    def fetch(x, y, zoom, layer):
        a = answers[(x, y)]
        if isinstance(a, Exception):
            raise a
        return a

    kept, dropped, unknown = filter_windows_by_coverage(CENTERS, LAYER, workers=2, fetch=fetch)
    assert kept == [CENTERS[0], CENTERS[3]]
    assert dropped == 2 and unknown == 1


def test_unreadable_content_is_unknown_and_never_dropped():
    def fetch(x, y, zoom, layer):
        return b"not a jpeg"

    kept, dropped, unknown = filter_windows_by_coverage(CENTERS, LAYER, fetch=fetch)
    assert kept == CENTERS and dropped == 0 and unknown == 4


def test_empty_centers():
    assert filter_windows_by_coverage([], LAYER, fetch=lambda *a: None) == ([], 0, 0)


def test_windows_sharing_a_tile_cost_a_single_probe():
    calls = []

    def fetch(x, y, zoom, layer):
        calls.append((x, y, zoom, layer))
        return _jpeg(128)

    centers = [(0.65, 47.33), (0.6501, 47.3301), (0.6502, 47.3302)]
    kept, _, _ = filter_windows_by_coverage(centers, LAYER, fetch=fetch)
    assert kept == centers
    assert len(calls) == 1 and calls[0][2] == PROBE_ZOOM and calls[0][3] == LAYER


def test_probing_is_parallel(monkeypatch):
    barrier = threading.Barrier(3, timeout=5)

    def fetch(x, y, zoom, layer):
        barrier.wait()          # ne passe que si 3 sondes sont en cours en même temps
        return _jpeg(128)

    status = probe_tiles([(1, 1), (2, 2), (3, 3)], LAYER, workers=3, fetch=fetch)
    assert list(status.values()) == [True, True, True]


def test_probe_writes_nothing_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    filter_windows_by_coverage(CENTERS, LAYER, fetch=lambda x, y, z, l: _jpeg(128))
    assert list(tmp_path.iterdir()) == []


class _Resp:
    def __init__(self, status, content=b"x"):
        self.status_code = status
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, status):
        self.status = status
        self.calls = 0

    def get(self, url, headers=None, timeout=30):
        self.calls += 1
        return _Resp(self.status)


def test_fetch_probe_tile_404_is_none_without_retry():
    s = _Session(404)
    assert fetch_probe_tile(1, 2, 13, LAYER, session=s, pause=0) is None
    assert s.calls == 1


def test_fetch_probe_tile_retries_then_raises_on_server_errors():
    s = _Session(500)
    try:
        fetch_probe_tile(1, 2, 13, LAYER, session=s, tries=2, pause=0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("une erreur 500 persistante doit lever")
    assert s.calls == 2


def test_fetch_probe_tile_returns_the_bytes_on_success():
    s = _Session(200)
    assert fetch_probe_tile(1, 2, 13, LAYER, session=s, pause=0) == b"x"
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_coverage.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'detection_ortho.coverage'`).

- [ ] **Step 3: Implémenter**

Créer `detection_ortho/coverage.py` :

```python
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
```

- [ ] **Step 4: Vérifier la réussite**

Run: `.venv\Scripts\python -m pytest tests/test_coverage.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/coverage.py tests/test_coverage.py
git commit -m "feat: sonde de couverture d'une couche WMTS en mémoire (filter_windows_by_coverage)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Faux déjà rejetés (`known_false`)

**Files:**
- Create: `detection_ortho/known_false.py`
- Test: `tests/test_known_false.py`

**Interfaces:**
- Consumes: `detection_ortho.dataset.parse_verdicts(lines) -> list[{lon, lat, verdict}]`, `detection_ortho.dataset.near_any(points, ref_points, radius_m) -> list[bool]`.
- Produces:
  - `load_known_false(paths) -> list[tuple[float, float]]` : points (lon, lat) des verdicts « faux » de tous les fichiers (FileNotFoundError si un fichier manque).
  - `suppress_known_false(points, false_points, radius_m=25.0) -> tuple[list, list]` : (conservés, écartés) ; `points` = dicts avec `lon`, `lat`.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_known_false.py` :

```python
import pytest

from detection_ortho.known_false import load_known_false, suppress_known_false


def _csv(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("index,lat,lon,score,verdict\n" + "".join(
        f"{i},{lat},{lon},0.5,{v}\n" for i, (lat, lon, v) in enumerate(rows, 1)),
        encoding="utf-8")
    return p


def test_load_keeps_only_the_faux_of_all_files(tmp_path):
    a = _csv(tmp_path, "a.csv", [(47.33, 0.65, "faux"), (47.34, 0.66, "vrai")])
    b = _csv(tmp_path, "b.csv", [(46.5, 1.5, "faux")])
    assert load_known_false([a, b]) == [(0.65, 47.33), (1.5, 46.5)]


def test_load_missing_file_fails_immediately(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_known_false([tmp_path / "absent.csv"])


def test_load_no_paths_is_empty():
    assert load_known_false([]) == []


def test_suppress_drops_candidates_near_a_known_false():
    pts = [{"lon": 0.65, "lat": 47.33}, {"lon": 0.7, "lat": 47.4}]
    kept, dropped = suppress_known_false(pts, [(0.65005, 47.33)], radius_m=25.0)   # ~4 m
    assert kept == [pts[1]] and dropped == [pts[0]]


def test_suppress_keeps_candidates_beyond_the_radius():
    pts = [{"lon": 0.65, "lat": 47.33}]
    kept, dropped = suppress_known_false(pts, [(0.651, 47.33)], radius_m=25.0)      # ~75 m
    assert kept == pts and dropped == []


def test_suppress_without_false_points_or_candidates_is_a_noop():
    pts = [{"lon": 0.65, "lat": 47.33}]
    assert suppress_known_false(pts, []) == (pts, [])
    assert suppress_known_false([], [(0.65, 47.33)]) == ([], [])


def test_suppress_keeps_the_extra_keys_of_the_points():
    pts = [{"lon": 0.7, "lat": 47.4, "score": 0.9}]
    kept, _ = suppress_known_false(pts, [(0.65, 47.33)])
    assert kept[0]["score"] == 0.9
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_known_false.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'detection_ortho.known_false'`).

- [ ] **Step 3: Implémenter**

Créer `detection_ortho/known_false.py` :

```python
"""Faux positifs déjà rejetés : ne pas les remontrer lors d'une nouvelle passe."""
from __future__ import annotations

from pathlib import Path

from detection_ortho.dataset import near_any, parse_verdicts


def load_known_false(paths) -> list[tuple[float, float]]:
    """Points (lon, lat) des verdicts « faux » des CSV de revue.

    Lève FileNotFoundError si un fichier manque : à appeler au démarrage d'un
    long run, pour échouer tout de suite plutôt qu'après des heures de calcul.
    """
    points: list[tuple[float, float]] = []
    for path in paths:
        for v in parse_verdicts(Path(path).read_text(encoding="utf-8").splitlines()):
            if v["verdict"] == "faux":
                points.append((v["lon"], v["lat"]))
    return points


def suppress_known_false(points, false_points, radius_m: float = 25.0):
    """Sépare les candidats (dicts avec `lon`, `lat`) proches d'un faux connu.

    Retourne (conservés, écartés). Un candidat est écarté s'il est à `radius_m`
    ou moins d'un point de `false_points`.
    """
    points = list(points)
    if not points or not false_points:
        return points, []
    flags = near_any([(p["lon"], p["lat"]) for p in points], false_points, radius_m)
    kept = [p for p, f in zip(points, flags) if not f]
    dropped = [p for p, f in zip(points, flags) if f]
    return kept, dropped
```

- [ ] **Step 4: Vérifier la réussite**

Run: `.venv\Scripts\python -m pytest tests/test_known_false.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/known_false.py tests/test_known_false.py
git commit -m "feat: known_false — ne pas remontrer les faux positifs déjà rejetés

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `infer_area.py` — `--layer`, couverture, fenêtres blanches, `--known-false`, runbook

**Files:**
- Modify: `scripts/infer_area.py`
- Modify: `README.md` (nouvelle section en fin de fichier, CRLF)
- Test: `tests/test_infer_area_layer.py`

**Interfaces:**
- Consumes: `filter_windows_by_coverage(centers, layer, workers, fetch)` (Task 1) ; `load_known_false(paths)`, `suppress_known_false(points, false_points, radius_m)` (Task 2) ; `window_is_blank(image)` ; `download_tile(..., layer=, tries=)` ; `LAYER`.
- Produces: options `--layer`, `--known-false`, `--known-false-m` ; `download_ok(xy, cache, session, layer=LAYER)` ; `stream_windows(..., start=0, layer=LAYER)` ; `build_fingerprint(centers, args)` ; livrables `suppressed.geojson`.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_infer_area_layer.py` :

```python
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import infer_area  # noqa: E402
from detection_ortho import checkpoint, tiles as tiles_mod  # noqa: E402
from detection_ortho.tiles import LAYER  # noqa: E402

L26 = "ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026"


class _Resp:
    content = b"\xff\xd8\xff\xe0FAKE"

    def raise_for_status(self):
        pass


class _Session:
    """Session factice : mémorise les URL demandées ; `fail` fait échouer chaque appel."""

    def __init__(self, fail=False):
        self.urls = []
        self.fail = fail

    def get(self, url, headers=None, timeout=30):
        self.urls.append(url)
        if self.fail:
            raise RuntimeError("404")
        return _Resp()


class _SyncFuture:
    def __init__(self, v):
        self.v = v

    def result(self):
        return self.v


class _SyncPool:
    def submit(self, fn, *args):
        return _SyncFuture(fn(*args))


def _centers(n):
    return [(0.001 * i, 47.5) for i in range(n)]


def test_stream_on_another_layer_fetches_it_and_purges_its_tagged_tiles(tmp_path):
    sess = _Session()
    seen_tagged = False
    for _ in infer_area.stream_windows(_centers(40), tmp_path, 160, _SyncPool(), sess,
                                       layer=L26):
        files = [p.name for p in tmp_path.glob("*.jpg")]
        seen_tagged = seen_tagged or any("rvb-express-2026" in n for n in files)
    assert seen_tagged, "les tuiles de la couche 2026 doivent être suffixées dans le cache"
    assert sess.urls and all(f"LAYER={L26}" in u for u in sess.urls)
    # Disque borné : tout est purgé à la fin, y compris les tuiles suffixées.
    assert list(tmp_path.glob("*.jpg")) == []


def test_download_ok_tries_once_on_another_layer_and_three_times_on_the_standard(monkeypatch, tmp_path):
    monkeypatch.setattr(tiles_mod.time, "sleep", lambda s: None)
    s26, s_std = _Session(fail=True), _Session(fail=True)
    assert infer_area.download_ok((1, 2), tmp_path, s26, L26) is not None
    assert len(s26.urls) == 1
    assert infer_area.download_ok((1, 2), tmp_path, s_std) is not None
    assert len(s_std.urls) == 3


def _args(**kw):
    base = dict(boundary="Cher", conf=0.25, overlap=0.2, weights="models/x.pt",
                ortho=None, insee="18", admin_level=None, layer=LAYER)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_fingerprint_is_unchanged_for_the_standard_layer():
    centers = _centers(10)
    legacy = checkpoint.fingerprint(centers, {
        "boundary": "Cher", "conf": 0.25, "overlap": 0.2, "zoom": 19, "window": 640,
        "weights": "x.pt", "ortho": "", "insee": "18", "admin_level": ""})
    assert infer_area.build_fingerprint(centers, _args()) == legacy


def test_fingerprint_changes_with_a_non_standard_layer():
    centers = _centers(10)
    assert infer_area.build_fingerprint(centers, _args(layer=L26)) != \
        infer_area.build_fingerprint(centers, _args())


def test_layer_with_local_ortho_is_rejected():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "infer_area.py"), "--boundary", "X",
         "--weights", "w.pt", "--layer", L26, "--ortho", "ortho.vrt"],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 2
    assert "incompatible" in r.stderr


def test_missing_known_false_file_fails_before_any_network_work(tmp_path):
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "infer_area.py"), "--boundary", "X",
         "--weights", "w.pt", "--known-false", str(tmp_path / "absent.csv")],
        capture_output=True, text=True, timeout=300)
    assert r.returncode != 0
    assert "absent.csv" in r.stderr


def test_help_mentions_the_new_options():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "infer_area.py"), "--help"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    for opt in ("--layer", "--known-false", "--known-false-m"):
        assert opt in r.stdout
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_infer_area_layer.py -q`
Expected: FAIL (`TypeError: stream_windows() got an unexpected keyword argument 'layer'`, `AttributeError: ... build_fingerprint`).

- [ ] **Step 3: Implémenter dans `scripts/infer_area.py`**

1. Imports : remplacer
```python
from detection_ortho.dataset import assemble_window, window_tiles
```
par
```python
from detection_ortho.dataset import assemble_window, window_is_blank, window_tiles
```
remplacer
```python
from detection_ortho.tiles import download_tile
```
par
```python
from detection_ortho.tiles import LAYER, download_tile
```
et ajouter après `from detection_ortho.zones import apply_zone_filter` :
```python
from detection_ortho.coverage import filter_windows_by_coverage
from detection_ortho.known_false import load_known_false, suppress_known_false
```
2. Remplacer la fonction `download_ok` par :
```python
def download_ok(xy, cache, session, layer=LAYER):
    """Télécharge une tuile ; retourne None si OK, l'exception sinon.

    Couche standard : 3 essais (des 404 transitoires existent). Autre couche : un
    seul essai, un 404 y signifie « pas de donnée » (zone non couverte).
    """
    try:
        download_tile(xy[0], xy[1], ZOOM, cache, session=session, layer=layer,
                      tries=3 if layer == LAYER else 1)
    except Exception as exc:  # noqa: BLE001
        return exc
    return None
```
3. Dans `stream_windows`, changer la signature en `def stream_windows(centers, cache, budget_bytes, pool, session, start=0, layer=LAYER):` et remplacer les deux `pool.submit(download_ok, xy, cache, session)` par `pool.submit(download_ok, xy, cache, session, layer)`.
4. Ajouter après `fetch_retry` :
```python
def build_fingerprint(centers, args):
    """Empreinte du point de reprise. La couche n'y figure que si elle n'est pas
    la couche standard, pour ne pas invalider un point de reprise existant."""
    params = {
        "boundary": args.boundary, "conf": args.conf, "overlap": args.overlap,
        "zoom": ZOOM, "window": WINDOW, "weights": Path(args.weights).name,
        "ortho": args.ortho or "",
        "insee": args.insee or "", "admin_level": args.admin_level or "",
    }
    if args.layer != LAYER:
        params["layer"] = args.layer
    return checkpoint.fingerprint(centers, params)
```
5. Options : après l'argument `--refresh-zones` (juste avant `args = ap.parse_args()`), ajouter :
```python
    ap.add_argument("--layer", type=str, default=LAYER,
                    help="couche WMTS à inférer (défaut : ortho habituelle ; ex. "
                         "ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026). Seules les "
                         "zones couvertes par la couche sont inférées.")
    ap.add_argument("--known-false", type=Path, nargs="+", default=None,
                    help="CSV de verdicts : les candidats à moins de --known-false-m "
                         "d'un point « faux » sont retirés du challenge "
                         "(listés dans suppressed.geojson)")
    ap.add_argument("--known-false-m", type=float, default=25.0,
                    help="rayon (m) de suppression autour des faux déjà rejetés")
```
6. Juste après `args = ap.parse_args()`, ajouter :
```python
    if args.layer != LAYER and args.ortho:
        ap.error("--layer est incompatible avec --ortho (lecture locale)")
    # Lus au démarrage : une faute de chemin doit échouer tout de suite.
    false_pts = load_known_false(args.known_false) if args.known_false else []
    if args.known_false:
        print(f"Faux déjà rejetés : {len(false_pts)} point(s) « faux » lus dans "
              f"{len(args.known_false)} fichier(s).")
```
7. Après le bloc du filtre des zones interdites (après le `else: print("Filtre des zones interdites désactivé ...")`) et avant `# --- A bis. Point de reprise ---`, ajouter :
```python
    if args.layer != LAYER:
        n_avant = len(centers)
        centers, n_hors, n_inc = filter_windows_by_coverage(
            centers, args.layer, workers=args.workers)
        print(f"Couverture de {args.layer} : {n_hors} fenêtre(s) hors couverture "
              f"écartée(s) sur {n_avant} ({n_inc} tuile(s) de sonde indéterminée(s), "
              f"fenêtres conservées).")
        if not centers:
            sys.exit("Aucune fenêtre couverte par cette couche dans l'emprise : "
                     "rien à inférer.")
```
8. Remplacer l'appel `empreinte = checkpoint.fingerprint(centers, {...})` (tout le bloc, de `empreinte = checkpoint.fingerprint(centers, {` jusqu'à `})` inclus) par :
```python
    empreinte = build_fingerprint(centers, args)
```
9. Dans `stream_windows(centers, cache, args.cache_gb * 1e9, tile_pool, session, start=start)` ajouter `layer=args.layer`. Dans la branche pré-téléchargement intégral, remplacer `pool.submit(download_ok, xy, cache, session)` par `pool.submit(download_ok, xy, cache, session, args.layer)`.
10. Avant la boucle d'inférence (juste avant `try:` qui précède `for lon, lat in progress(windows, ...`), ajouter `n_vides = 0`. Remplacer l'appel
```python
                    img, ogx, ogy = assemble_window(lon, lat, ZOOM, WINDOW, cache)
```
par
```python
                    img, ogx, ogy = assemble_window(lon, lat, ZOOM, WINDOW, cache,
                                                    layer=args.layer)
```
et, juste après le bloc `except Exception as exc: ... continue` qui suit la lecture de la fenêtre, ajouter :
```python
            if window_is_blank(img):  # aucune donnée ici : inutile d'inférer
                n_vides += 1
                continue
```
11. Après la boucle (après le `finally:` qui ferme `ortho_vrt`/`tile_pool`), ajouter :
```python
    if n_vides:
        print(f"Fenêtres sans donnée (blanches) sautées : {n_vides}.")
```
12. Après `res = match_detections(detections, osm, radius_m=args.radius)` et avant l'écriture de `matched.geojson`, ajouter :
```python
    suppressed: list = []
    if args.known_false:
        res["detected_only"], suppressed = suppress_known_false(
            res["detected_only"], false_pts, args.known_false_m)
        write_geojson(points_to_geojson(suppressed), args.out / "suppressed.geojson")
        print(f"Faux déjà rejetés : {len(suppressed)} candidat(e)s écarté(e)s "
              f"(à moins de {args.known_false_m:g} m d'un point « faux »).")
```
13. Dans le résumé, après la ligne `Candidates (∉ OSM) -> MapRoul.`, ajouter :
```python
    if args.known_false:
        print(f"  Écartées (déjà rejetées)      : {len(suppressed)}")
```

- [ ] **Step 4: Vérifier la réussite (nouveaux tests + non-régression du script)**

Run: `.venv\Scripts\python -m pytest tests/test_infer_area_layer.py tests/test_infer_area_stream.py tests/test_infer_area_progress.py tests/test_infer_area_help.py tests/test_checkpoint.py tests/test_coverage.py tests/test_known_false.py -q`
Expected: PASS.

- [ ] **Step 5: Runbook dans le README**

Ajouter à la fin de `README.md` (en CRLF) la section suivante :

````markdown

## Rattrapage des départements déjà traités (nouveau modèle, ortho 2026)

Repasser les 8 départements déjà traités avec le modèle actuel. L'ortho express 2026 est utilisée là où
elle existe (18, 28, 36, 37, 41, 45, certains couverts en partie) ; le 44 et le 49 n'en ont pas et restent
sur l'ortho habituelle. Les tuiles sont **streamées en parallèle et purgées au fil de l'eau** : le cache disque
reste plafonné par `--cache-gb` (10 Go par défaut) quelle que soit la couche, et il est vidé en fin de run.

- `--layer` limite l'inférence aux zones couvertes par la couche : une sonde de tuiles (en mémoire, rien
  n'est écrit sur le disque) écarte les fenêtres sans donnée, et les fenêtres encore blanches sont sautées.
- `--known-false` retire du challenge les candidats à moins de 25 m d'un point jugé « faux » dans vos
  verdicts (ils sont listés dans `suppressed.geojson`). Les fichiers sont lus au démarrage. Vous pouvez en
  ajouter d'autres (par exemple les faux du 37 revus à la main, absents des CSV MapRoulette).
- Un run est **reprenable** : relancer la même commande (voir le runbook des passes départementales).

Une fonction PowerShell évite de répéter les options (à coller une fois dans le terminal) :

    function Rattrapage($nom, $insee, $couche = "") {
        $a = @("scripts/infer_area.py", "--boundary", $nom, "--insee", $insee,
               "--weights", "models/citernes-yolov8n.pt", "--conf", "0.25",
               "--known-false", "verdicts_maproulette/verdicts_$insee.csv",
               "--device", "cpu", "--out", "inference_rattrapage_$insee")
        if ($couche) { $a += @("--layer", $couche) }
        & .venv\Scripts\python @a
    }
    $L26 = "ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026"

Puis, un département à la fois (de l'ordre de 15 à 19 h de CPU chacun, moins s'il est couvert en partie) :

    Rattrapage "Cher" "18" $L26
    Rattrapage "Eure-et-Loir" "28" $L26
    Rattrapage "Indre" "36" $L26
    Rattrapage "Indre-et-Loire" "37" $L26
    Rattrapage "Loir-et-Cher" "41" $L26
    Rattrapage "Loiret" "45" $L26
    Rattrapage "Loire-Atlantique" "44"
    Rattrapage "Maine-et-Loire" "49"

Le challenge se filtre ensuite au seuil de qualité du modèle actuel : 0,35 pour les départements en
ortho 2026, 0,60 pour le 44 et le 49 (ortho habituelle) :

    python scripts/export_maproulette.py --input inference_rattrapage_18/detected_only.geojson `
        --out inference_rattrapage_18/challenge_18.geojson --min-score 0.35
````

Puis vérifier le CRLF (commande dans les contraintes globales) et que la fonction PowerShell se lit sans erreur de syntaxe :

Run: `powershell -NoProfile -Command "$t = @'` suivi du corps de la fonction `Rattrapage` (tel qu'écrit ci-dessus) `'@; $e = $null; [void][System.Management.Automation.Language.Parser]::ParseInput($t, [ref]$null, [ref]$e); if ($e.Count) { $e } else { 'syntaxe PowerShell ok' }"` (ou équivalent : écrire le corps de la fonction dans un fichier temporaire hors du dépôt et l'analyser avec `[System.Management.Automation.Language.Parser]::ParseFile`).
Expected: `syntaxe PowerShell ok`.

- [ ] **Step 6: Suite complète puis commit**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS (tous les tests).

```bash
git add scripts/infer_area.py README.md tests/test_infer_area_layer.py
git commit -m "feat: infer_area --layer (couverture, fenêtres blanches) et --known-false ; runbook de rattrapage

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Étapes manuelles (hors plan, côté utilisateur)

1. Lancer les rattrapages un département à la fois (comptez 15 à 19 h de CPU chacun).
2. Exporter chaque challenge au seuil indiqué, le relire, puis l'importer dans MapRoulette (import manuel).
3. Départements nouveaux : même commande, un par un, avec le nom, le code INSEE et la couche 2026 si la zone est couverte.
