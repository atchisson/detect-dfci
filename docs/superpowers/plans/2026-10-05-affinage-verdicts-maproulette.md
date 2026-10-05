# Affinage avec les verdicts MapRoulette — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Récupérer les verdicts de revue (fixed / not an issue) du projet MapRoulette 64996 et les utiliser (avec dédoublonnage) pour construire un dataset d'affinage du modèle, évalué sur des départements mis de côté.

**Architecture:** Logique pure (statut → verdict, nom de challenge → département, tâche → ligne CSV, dédoublonnage) dans `detection_ortho/` ; réseau confiné dans un script de récupération ; `build_dataset.py` accepte plusieurs CSV de verdicts et écarte les vrais qui doublonnent un positif déjà présent. L'entraînement et l'évaluation réutilisent `train.py --model`, `sweep_threshold.py` et `eval_points.py` sans modification.

**Tech Stack:** Python 3.12, pytest, urllib (stdlib), ultralytics (existant), API MapRoulette v2 (lecture publique).

**Spec:** `docs/superpowers/specs/2026-10-05-affinage-verdicts-maproulette-design.md`

## Global Constraints

- Aucun upload vers MapRoulette ou OSM : le script de récupération ne fait que des `GET`.
- Format CSV de verdicts inchangé : `index,lat,lon,score,verdict` (verdict `vrai` ou `faux`), lisible par `dataset.parse_verdicts`.
- Statuts MapRoulette retenus : `1` (fixed) → `vrai`, `2` (falsePositive / not an issue) → `faux`. Tous les autres statuts sont ignorés.
- User-Agent HTTP = `https://github.com/atchisson/detect-dfci`.
- Tests hors-ligne uniquement (aucun appel réseau dans pytest). Interpréteur : `.venv\Scripts\python`.
- Scripts : `sys.stdout.reconfigure(encoding="utf-8")` en tête de `main`, comme les scripts existants.
- Boîte des vrais : `DEFAULT_BOX_M` (13 m), comme l'itération 2.
- Rétrocompatibilité : `build_dataset.py --verdicts un_seul.csv` continue de fonctionner.

## Review Focus

- Tâche avec `location` absente ou coordonnées non numériques → ligne ignorée, pas de crash (Task 1).
- Score absent ou non numérique dans les propriétés de la géométrie → score `0.0`, la ligne est conservée (Task 1).
- Nom de challenge qui n'est pas `DECI <code>` → challenge ignoré avec avertissement, pas d'arrêt (Task 2).
- Pagination : un challenge dont le nombre de tâches est un multiple exact de 500 doit se terminer sur une page vide (Task 2).
- Échec réseau en cours de challenge → exception, aucun CSV partiel pour ce département (Task 2).
- Départements corses `2A` / `2B` : le code est conservé tel quel, pas converti en entier (Task 1).
- Deux vrais proches l'un de l'autre mais de part et d'autre d'une frontière de cellule de la grille → doublon quand même détecté (Task 3).
- Un faux situé sur une citerne OSM n'est jamais écarté (c'est un négatif dur légitime) (Task 3).

---

### Task 1: Fonctions pures MapRoulette → verdicts

**Files:**
- Modify: `detection_ortho/maproulette.py`
- Test: `tests/test_maproulette_verdicts.py`

**Interfaces:**
- Consumes: rien.
- Produces:
  - `STATUS_VERDICT: dict[int, str]` = `{1: "vrai", 2: "faux"}`
  - `dept_from_challenge_name(name: str | None) -> str | None` (ex. `"DECI 44"` → `"44"`, `"DECI 2A"` → `"2A"`, sinon `None`)
  - `task_to_row(task: dict) -> dict | None` → `{"lon": float, "lat": float, "score": float, "verdict": str}` ou `None`
  - `rows_to_csv(rows: list[dict]) -> str` (en-tête `index,lat,lon,score,verdict`, index à partir de 1)

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_maproulette_verdicts.py` :

```python
from detection_ortho.dataset import parse_verdicts
from detection_ortho.maproulette import (
    dept_from_challenge_name, task_to_row, rows_to_csv,
)


def _task(status=1, coords=(0.65, 47.33), score="0.91"):
    t = {
        "status": status,
        "location": {"type": "Point", "coordinates": list(coords)},
        "geometries": {"features": [{"properties": {"score": score}}]},
    }
    return t


def test_dept_from_challenge_name():
    assert dept_from_challenge_name("DECI 44") == "44"
    assert dept_from_challenge_name("DECI 2A") == "2A"
    assert dept_from_challenge_name("deci 37") == "37"
    assert dept_from_challenge_name("Autre challenge") is None
    assert dept_from_challenge_name(None) is None


def test_task_to_row_fixed_is_vrai_lonlat_order():
    row = task_to_row(_task(status=1, coords=(0.65, 47.33), score="0.91"))
    assert row == {"lon": 0.65, "lat": 47.33, "score": 0.91, "verdict": "vrai"}


def test_task_to_row_not_an_issue_is_faux():
    assert task_to_row(_task(status=2))["verdict"] == "faux"


def test_task_to_row_ignores_other_statuses():
    for status in (0, 3, 4, 5, 6, 9):
        assert task_to_row(_task(status=status)) is None


def test_task_to_row_missing_or_bad_location_is_skipped():
    t = _task()
    del t["location"]
    assert task_to_row(t) is None
    assert task_to_row(_task(coords=("x", "y"))) is None


def test_task_to_row_bad_score_defaults_to_zero():
    t = _task(score="abc")
    assert task_to_row(t)["score"] == 0.0
    t2 = _task()
    t2["geometries"] = {"features": []}
    assert task_to_row(t2)["score"] == 0.0


def test_rows_to_csv_roundtrips_through_parse_verdicts():
    rows = [
        {"lon": 0.65, "lat": 47.33, "score": 0.91, "verdict": "vrai"},
        {"lon": 1.1, "lat": 46.9, "score": 0.5, "verdict": "faux"},
    ]
    text = rows_to_csv(rows)
    assert text.splitlines()[0] == "index,lat,lon,score,verdict"
    parsed = parse_verdicts(text.splitlines())
    assert parsed == [
        {"lon": 0.65, "lat": 47.33, "verdict": "vrai"},
        {"lon": 1.1, "lat": 46.9, "verdict": "faux"},
    ]


def test_rows_to_csv_empty_has_only_header():
    assert rows_to_csv([]) == "index,lat,lon,score,verdict\n"
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_maproulette_verdicts.py -v`
Expected: FAIL (`ImportError: cannot import name 'dept_from_challenge_name'`).

- [ ] **Step 3: Implémenter**

Dans `detection_ortho/maproulette.py`, remplacer l'en-tête (docstring + imports) par :

```python
"""Tâches MapRoulette : génération d'un fichier d'import et lecture de verdicts.

IMPORTANT : ce module est PUR (aucun accès réseau). `to_maproulette_tasks`
GÉNÈRE UN FICHIER que l'utilisateur importe lui-même dans MapRoulette ; les
fonctions de verdict transforment des tâches DÉJÀ téléchargées (la récupération,
en lecture seule, vit dans scripts/fetch_maproulette_verdicts.py). Aucun envoi
vers MapRoulette ni OSM.
"""
from __future__ import annotations

import re

# Statuts de tâche MapRoulette : 1 = fixed (vraie citerne ajoutée à OSM),
# 2 = false positive / « not an issue ». Les autres sont ignorés.
STATUS_VERDICT = {1: "vrai", 2: "faux"}

_DEPT_RE = re.compile(r"\bDECI\s+(\w+)\s*$", re.IGNORECASE)
```

Puis ajouter à la fin du fichier :

```python


def dept_from_challenge_name(name: str | None) -> str | None:
    """Code département d'un challenge nommé « DECI <code> », sinon None."""
    m = _DEPT_RE.search(name or "")
    return m.group(1).upper() if m else None


def task_to_row(task: dict) -> dict | None:
    """Tâche MapRoulette → {lon, lat, score, verdict}, ou None si à ignorer."""
    verdict = STATUS_VERDICT.get(task.get("status"))
    if verdict is None:
        return None
    try:
        lon, lat = (float(v) for v in task["location"]["coordinates"][:2])
    except (KeyError, TypeError, ValueError):
        return None
    try:
        score = float(task["geometries"]["features"][0]["properties"]["score"])
    except (KeyError, IndexError, TypeError, ValueError):
        score = 0.0
    return {"lon": lon, "lat": lat, "score": score, "verdict": verdict}


def rows_to_csv(rows: list[dict]) -> str:
    """CSV `index,lat,lon,score,verdict` (index dès 1), lisible par parse_verdicts."""
    lines = ["index,lat,lon,score,verdict"]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i},{r['lat']:.9f},{r['lon']:.9f},{r['score']:.4f},{r['verdict']}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Vérifier la réussite**

Run: `.venv\Scripts\python -m pytest tests/test_maproulette_verdicts.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/maproulette.py tests/test_maproulette_verdicts.py
git commit -m "feat: fonctions pures MapRoulette -> verdicts (statut, département, CSV)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Script `fetch_maproulette_verdicts.py`

**Files:**
- Create: `scripts/fetch_maproulette_verdicts.py`
- Test: `tests/test_fetch_maproulette_verdicts.py`

**Interfaces:**
- Consumes (Task 1): `dept_from_challenge_name`, `task_to_row`, `rows_to_csv`, `STATUS_VERDICT` de `detection_ortho.maproulette`.
- Produces:
  - `http_get_json(url: str, tries: int = 4, pause: float = 3.0, timeout: int = 120) -> object`
  - `iter_tasks(challenge_id: int) -> Iterator[dict]` (pagine par 500, appelle `http_get_json` via le nom global du module)
  - `main()` (CLI : `--project` défaut `64996`, `--out` défaut `verdicts_maproulette`) ; écrit `<out>/verdicts_<DEPT>.csv`.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_fetch_maproulette_verdicts.py` :

```python
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import fetch_maproulette_verdicts as fmv  # noqa: E402
from detection_ortho.dataset import parse_verdicts  # noqa: E402


def _task(status, lon, lat, score="0.8"):
    return {
        "status": status,
        "location": {"coordinates": [lon, lat]},
        "geometries": {"features": [{"properties": {"score": score}}]},
    }


def _fake_get(challenges, tasks_by_id, fail_ids=()):
    """http_get_json factice : sert la liste des challenges et les pages de tâches."""
    def get(url, *a, **k):
        if "/project/" in url:
            return challenges
        cid = int(url.split("/challenge/")[1].split("/")[0])
        if cid in fail_ids:
            raise RuntimeError("réseau coupé")
        page = int(url.split("page=")[1])
        size = fmv.PAGE
        return tasks_by_id[cid][page * size:(page + 1) * size]
    return get


def test_iter_tasks_paginates_and_stops_on_exact_multiple(monkeypatch):
    monkeypatch.setattr(fmv, "PAGE", 2)
    tasks = [_task(1, 0.1 * i, 47.0) for i in range(4)]  # multiple exact de PAGE
    monkeypatch.setattr(fmv, "http_get_json", _fake_get([], {7: tasks}))
    assert len(list(fmv.iter_tasks(7))) == 4


def test_main_writes_one_csv_per_dept(tmp_path, monkeypatch, capsys):
    challenges = [
        {"id": 1, "name": "DECI 44"},
        {"id": 2, "name": "DECI 2A"},
        {"id": 3, "name": "Test sans rapport"},
    ]
    tasks = {
        1: [_task(1, 0.65, 47.33), _task(2, 0.66, 47.34),
            _task(5, 0.67, 47.35), _task(6, 0.68, 47.36)],
        2: [_task(2, 9.0, 42.0)],
    }
    monkeypatch.setattr(fmv, "http_get_json", _fake_get(challenges, tasks))
    monkeypatch.setattr(sys, "argv", [
        "fetch_maproulette_verdicts.py", "--out", str(tmp_path)])
    fmv.main()

    f44 = tmp_path / "verdicts_44.csv"
    parsed = parse_verdicts(f44.read_text(encoding="utf-8").splitlines())
    assert [v["verdict"] for v in parsed] == ["vrai", "faux"]  # 5 et 6 ignorés
    assert (tmp_path / "verdicts_2A.csv").exists()
    assert not list(tmp_path.glob("*sans*"))  # challenge non DECI ignoré
    assert len(list(tmp_path.glob("verdicts_*.csv"))) == 2
    out = capsys.readouterr().out
    assert "44" in out and "ignoré" in out.lower()


def test_main_network_failure_leaves_no_partial_csv(tmp_path, monkeypatch):
    challenges = [{"id": 1, "name": "DECI 44"}]
    monkeypatch.setattr(
        fmv, "http_get_json", _fake_get(challenges, {1: []}, fail_ids=(1,)))
    monkeypatch.setattr(sys, "argv", [
        "fetch_maproulette_verdicts.py", "--out", str(tmp_path)])
    with pytest.raises(RuntimeError):
        fmv.main()
    assert list(tmp_path.glob("verdicts_*.csv")) == []


def test_help_runs():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "fetch_maproulette_verdicts.py"),
         "--help"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "--project" in r.stdout
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_fetch_maproulette_verdicts.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'fetch_maproulette_verdicts'`).

- [ ] **Step 3: Implémenter**

Créer `scripts/fetch_maproulette_verdicts.py` :

```python
"""Récupère les verdicts de revue d'un projet MapRoulette (lecture seule).

Pour chaque challenge « DECI <dept> » du projet, télécharge les tâches via
l'API publique (GET uniquement, aucune clé, aucun envoi) et écrit un CSV
`verdicts_<dept>.csv` au format `index,lat,lon,score,verdict` :
statut 1 (fixed) -> vrai, statut 2 (not an issue) -> faux, autres ignorés.

Usage:
    python scripts/fetch_maproulette_verdicts.py --out verdicts_maproulette
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_ortho.maproulette import (
    dept_from_challenge_name, task_to_row, rows_to_csv, STATUS_VERDICT,
)

API = "https://maproulette.org/api/v2"
UA = "https://github.com/atchisson/detect-dfci"
PAGE = 500


def http_get_json(url: str, tries: int = 4, pause: float = 3.0,
                  timeout: int = 120):
    """GET JSON avec nouvelles tentatives ; lève RuntimeError si tout échoue."""
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  GET retry {i + 1}/{tries} ({exc})", file=sys.stderr)
            time.sleep(pause)
    raise RuntimeError(f"GET {url} : {last}") from last


def iter_tasks(challenge_id: int):
    """Toutes les tâches d'un challenge, par pages de PAGE."""
    page = 0
    while True:
        batch = http_get_json(
            f"{API}/challenge/{challenge_id}/tasks?limit={PAGE}&page={page}")
        if not batch:
            return
        yield from batch
        if len(batch) < PAGE:
            return
        page += 1


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=int, default=64996,
                    help="identifiant du projet MapRoulette")
    ap.add_argument("--out", type=Path, default=Path("verdicts_maproulette"))
    args = ap.parse_args()

    challenges = http_get_json(
        f"{API}/project/{args.project}/challenges?limit=200")
    args.out.mkdir(parents=True, exist_ok=True)

    for ch in challenges:
        dept = dept_from_challenge_name(ch.get("name"))
        if dept is None:
            print(f"Challenge « {ch.get('name')} » ignoré (nom != 'DECI <dept>').")
            continue
        statuses: Counter = Counter()
        rows = []
        for task in iter_tasks(ch["id"]):
            statuses[task.get("status")] += 1
            row = task_to_row(task)
            if row is not None:
                rows.append(row)
        # Écriture seulement une fois le challenge entièrement lu : pas de CSV partiel.
        path = args.out / f"verdicts_{dept}.csv"
        path.write_text(rows_to_csv(rows), encoding="utf-8")
        n_vrai = sum(1 for r in rows if r["verdict"] == "vrai")
        n_ignores = sum(n for s, n in statuses.items() if s not in STATUS_VERDICT)
        print(f"Dép. {dept}: {n_vrai} vrai(s), {len(rows) - n_vrai} faux, "
              f"{n_ignores} ignoré(s) -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Vérifier la réussite**

Run: `.venv\Scripts\python -m pytest tests/test_fetch_maproulette_verdicts.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_maproulette_verdicts.py tests/test_fetch_maproulette_verdicts.py
git commit -m "feat: fetch_maproulette_verdicts — verdicts par département via l'API (lecture seule)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Dédoublonnage des verdicts `vrai`

**Files:**
- Modify: `detection_ortho/dataset.py` (import + nouvelle fonction après `parse_verdicts`)
- Test: `tests/test_dedup_verdicts.py`

**Interfaces:**
- Consumes: `detection_ortho.geo.haversine_m(lon1, lat1, lon2, lat2) -> float`.
- Produces: `dedup_verdicts(verdicts: list[dict], ref_points: list[tuple[float, float]], radius_m: float) -> tuple[list[dict], int]` — `verdicts` au format `parse_verdicts` ; `ref_points` = liste de `(lon, lat)` ; retourne `(verdicts_conservés_dans_l_ordre, nombre_de_vrais_écartés)`.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_dedup_verdicts.py` :

```python
from detection_ortho.dataset import dedup_verdicts

LAT = 47.0


def _v(lon, verdict="vrai", lat=LAT):
    return {"lon": lon, "lat": lat, "verdict": verdict}


def test_vrai_near_reference_is_dropped():
    # ~5 m d'écart
    kept, dropped = dedup_verdicts([_v(0.50006)], [(0.5, LAT)], radius_m=15)
    assert kept == [] and dropped == 1


def test_vrai_far_from_reference_is_kept():
    kept, dropped = dedup_verdicts([_v(0.51)], [(0.5, LAT)], radius_m=15)
    assert len(kept) == 1 and dropped == 0


def test_faux_on_reference_is_never_dropped():
    kept, dropped = dedup_verdicts([_v(0.5, "faux")], [(0.5, LAT)], radius_m=15)
    assert len(kept) == 1 and dropped == 0


def test_second_close_vrai_is_dropped_first_kept():
    kept, dropped = dedup_verdicts([_v(0.5), _v(0.50006)], [], radius_m=15)
    assert len(kept) == 1 and dropped == 1
    assert kept[0]["lon"] == 0.5


def test_duplicate_detected_across_grid_cell_boundary():
    # radius 15 m -> cellules de 0.00025° ; ces deux points (~4,6 m) sont de
    # part et d'autre de la frontière de cellule lon = 0.00025.
    kept, dropped = dedup_verdicts(
        [_v(0.00028)], [(0.00022, LAT)], radius_m=15)
    assert kept == [] and dropped == 1


def test_order_and_non_vrai_preserved():
    vs = [_v(0.3, "faux"), _v(0.4), _v(0.5, "faux")]
    kept, dropped = dedup_verdicts(vs, [], radius_m=15)
    assert kept == vs and dropped == 0
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_dedup_verdicts.py -v`
Expected: FAIL (`ImportError: cannot import name 'dedup_verdicts'`).

- [ ] **Step 3: Implémenter**

Dans `detection_ortho/dataset.py`, ajouter après la ligne `from detection_ortho.tiles import lonlat_to_pixel, download_tile, LAYER` :

```python
from detection_ortho.geo import haversine_m
```

Puis ajouter la fonction juste après `parse_verdicts` (avant `compose_rgn`) :

```python
def dedup_verdicts(
    verdicts: list[dict],
    ref_points: list[tuple[float, float]],
    radius_m: float,
) -> tuple[list[dict], int]:
    """Écarte les verdicts `vrai` qui doublonnent un point déjà connu.

    Un `vrai` à moins de `radius_m` d'un point de `ref_points` (positifs OSM
    déjà chargés, en (lon, lat)) ou d'un `vrai` déjà retenu est écarté : c'est
    la même citerne. Les `faux` ne sont jamais écartés (négatifs durs
    légitimes, même posés sur une citerne OSM voisine). L'ordre est préservé.
    Grille de cellules ≥ radius_m (60 km/° est un minorant sûr en France) pour
    éviter le O(n²) ; les 8 cellules voisines sont examinées.
    Retourne (verdicts conservés, nombre de `vrai` écartés).
    """
    cell = max(radius_m, 1.0) / 60000.0
    grid: dict = {}

    def key(lon: float, lat: float) -> tuple[int, int]:
        return math.floor(lon / cell), math.floor(lat / cell)

    def add(lon: float, lat: float) -> None:
        grid.setdefault(key(lon, lat), []).append((lon, lat))

    def near(lon: float, lat: float) -> bool:
        kx, ky = key(lon, lat)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for x, y in grid.get((kx + dx, ky + dy), ()):
                    if haversine_m(lon, lat, x, y) <= radius_m:
                        return True
        return False

    for lon, lat in ref_points:
        add(lon, lat)
    kept: list[dict] = []
    dropped = 0
    for v in verdicts:
        if v["verdict"] == "vrai":
            if near(v["lon"], v["lat"]):
                dropped += 1
                continue
            add(v["lon"], v["lat"])
        kept.append(v)
    return kept, dropped
```

- [ ] **Step 4: Vérifier la réussite (nouveaux tests + non-régression dataset)**

Run: `.venv\Scripts\python -m pytest tests/test_dedup_verdicts.py tests/test_dataset_boxes.py tests/test_dataset_geo.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/dataset.py tests/test_dedup_verdicts.py
git commit -m "feat: dedup_verdicts — écarter les vrais qui doublonnent un positif connu

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `build_dataset.py` — verdicts multiples + dédoublonnage, runbook

**Files:**
- Modify: `scripts/build_dataset.py:28-33` (import), `:81-82` (argument `--verdicts`), `:128-140` (ingestion)
- Modify: `README.md` (nouvelle section en fin de fichier)
- Test: `tests/test_build_dataset_multi_verdicts.py`

**Interfaces:**
- Consumes: `parse_verdicts` (existant), `dedup_verdicts(verdicts, ref_points, radius_m)` (Task 3).
- Produces: CLI `build_dataset.py --verdicts A.csv B.csv ... [--dedup-m 15]`.

- [ ] **Step 1: Écrire le test qui échoue**

Créer `tests/test_build_dataset_multi_verdicts.py` :

```python
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import build_dataset  # noqa: E402
from detection_ortho.dataset import window_tiles  # noqa: E402


def _seed_tiles(cache, lon, lat):
    cache.mkdir(parents=True, exist_ok=True)
    tiles, _, _ = window_tiles(lon, lat, 19, 640)
    for x, y in tiles:
        cv2.imwrite(str(cache / f"19_{x}_{y}.jpg"),
                    np.full((256, 256, 3), 128, np.uint8))


def test_multiple_verdict_files_with_dedup(tmp_path, monkeypatch, capsys):
    osm_lon, osm_lat = 0.65, 47.33

    def fake_fetch(selectors, *a, **k):
        if selectors[0][0] == "emergency":  # une citerne OSM (nœud)
            return [{"type": "node", "lon": osm_lon, "lat": osm_lat, "tags": {}}]
        return []  # pas de piscines
    monkeypatch.setattr(build_dataset, "fetch_features_geom", fake_fetch)

    out = tmp_path / "ds"
    cache = out / "tiles_cache"
    far_lon, far_lat = 0.66, 47.34
    faux_lon, faux_lat = 0.67, 47.35
    for lon, lat in ((osm_lon, osm_lat), (far_lon, far_lat), (faux_lon, faux_lat)):
        _seed_tiles(cache, lon, lat)

    a = tmp_path / "a.csv"  # vrai sur la citerne OSM (~3 m) -> doublon
    a.write_text("index,lat,lon,score,verdict\n"
                 f"1,{osm_lat},{osm_lon + 0.00004},0.9,vrai\n", encoding="utf-8")
    b = tmp_path / "b.csv"  # vrai lointain + faux
    b.write_text("index,lat,lon,score,verdict\n"
                 f"1,{far_lat},{far_lon},0.9,vrai\n"
                 f"2,{faux_lat},{faux_lon},0.5,faux\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "build_dataset.py", "--bbox", "0.6", "47.3", "0.7", "47.4",
        "--negatives", "0", "--max-pools", "0",
        "--verdicts", str(a), str(b), "--out", str(out)])
    build_dataset.main()

    names = [p.stem for p in (out / "labels").rglob("*.txt")]
    assert sum(n.startswith("citerne") for n in names) == 1
    assert sum(n.startswith("revpos") for n in names) == 1   # le doublon est écarté
    assert sum(n.startswith("hardneg") for n in names) == 1
    assert "1 doublon" in capsys.readouterr().out
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_build_dataset_multi_verdicts.py -v`
Expected: FAIL (argparse : `unrecognized arguments` pour le second fichier).

- [ ] **Step 3: Implémenter**

Dans `scripts/build_dataset.py` :

1. Import : ajouter `dedup_verdicts` à la liste importée de `detection_ortho.dataset` (ligne 29-33) :

```python
from detection_ortho.dataset import (
    element_to_box, assemble_window, geo_bbox_to_pixel_bbox, to_yolo_label,
    write_chip, split_indices, spatial_split_indices, write_data_yaml,
    window_tiles, fixed_box_geo, DEFAULT_BOX_M, parse_verdicts, compose_rgn,
    dedup_verdicts,
)
```

2. Arguments : remplacer le bloc `--verdicts` par :

```python
    ap.add_argument("--verdicts", type=Path, nargs="+", default=None,
                    help="CSV de revue (un ou plusieurs) : faux -> négatifs "
                         "durs, vrai -> positifs")
    ap.add_argument("--dedup-m", type=float, default=15.0,
                    help="rayon (m) sous lequel un vrai doublonne un positif "
                         "déjà présent et est écarté")
```

3. Ingestion : remplacer le bloc `if args.verdicts:` (de `vs = parse_verdicts(...)` jusqu'au `print("Verdicts ingérés ...")`) par :

```python
    if args.verdicts:
        vs = []
        for path in args.verdicts:
            vs += parse_verdicts(path.read_text(encoding="utf-8").splitlines())
        vs, n_dup = dedup_verdicts(
            vs, [(b["lon"], b["lat"]) for b in boxes], args.dedup_m)
        n_hard = n_rev = 0
        for v in vs:
            if v["verdict"] == "faux":
                records.append((f"hardneg_{n_hard:04d}", v["lon"], v["lat"], None))
                n_hard += 1
            else:  # vrai
                bbox = fixed_box_geo(v["lon"], v["lat"], DEFAULT_BOX_M)
                records.append((f"revpos_{n_rev:04d}", v["lon"], v["lat"], bbox))
                n_rev += 1
        print(f"Verdicts ingérés : {n_hard} négatif(s) dur(s), {n_rev} positif(s), "
              f"{n_dup} doublon(s) écarté(s).")
```

(Le message contient `"1 doublon"`, requis par le test.)

- [ ] **Step 4: Vérifier la réussite (nouveau test + non-régression)**

Run: `.venv\Scripts\python -m pytest tests/test_build_dataset_multi_verdicts.py tests/test_build_dataset_verdicts.py tests/test_build_dataset_integration.py tests/test_build_dataset_nir.py -v`
Expected: PASS (le test historique à un seul fichier passe toujours).

- [ ] **Step 5: Runbook README**

Ajouter à la fin de `README.md` :

````markdown

## Affiner avec les verdicts MapRoulette

Réentraîner à partir des poids actuels avec les revues faites dans MapRoulette
(`fixed` = vrai, `not an issue` = faux). Les départements **36 et 49** sont mis de
côté pour mesurer le gain sur des zones jamais vues.

1. **Récupérer les verdicts** (lecture seule, API publique) :

       python scripts/fetch_maproulette_verdicts.py --out verdicts_maproulette

   Écrit `verdicts_maproulette\verdicts_<dept>.csv` et affiche le décompte par
   département (les statuts autres que fixed / not an issue sont ignorés).

2. **Dataset d'affinage** — tous les départements SAUF 36 et 49 (sous PowerShell
   les chemins sont listés explicitement, pas de globbing) :

       python scripts/build_dataset.py --bbox 0.05 46.72 1.06 47.72 \
           --verdicts verdicts_maproulette\verdicts_18.csv verdicts_maproulette\verdicts_28.csv \
                      verdicts_maproulette\verdicts_37.csv verdicts_maproulette\verdicts_41.csv \
                      verdicts_maproulette\verdicts_44.csv verdicts_maproulette\verdicts_45.csv \
           --spatial-split --out dataset_mr

   Les vrais à moins de `--dedup-m` (15 m) d'une citerne OSM déjà chargée sont
   écartés (ex. le 37, déjà dans OSM). Un CSV local supplémentaire (ex. les faux
   du 37 revus à la main) peut être ajouté à `--verdicts`.

3. **Affiner** (30 époques depuis les poids actuels ; long sur CPU — de nuit, ou
   `notebooks/train_yolo.ipynb` sur Colab) :

       python scripts/train.py --data dataset_mr/data.yaml \
           --model models/citernes-yolov8n.pt --epochs 30 --device cpu --name citernes_mr

4. **Comparer avant/après sur les départements mis de côté** (une inférence par
   point puis balayage de seuil ; mêmes commandes pour `verdicts_49.csv`) :

       python scripts/sweep_threshold.py --weights models/citernes-yolov8n.pt \
           --verdicts verdicts_maproulette\verdicts_36.csv
       python scripts/sweep_threshold.py --weights runs/citernes_mr/weights/best.pt \
           --verdicts verdicts_maproulette\verdicts_36.csv

   Lire la précision et les vrais conservés à 0,40 / 0,55 / 0,70. ⚠️ Tous ces
   points sont des détections ≥ 0,40 de l'ancien modèle : son rappel y vaut 100 %
   par construction. Seuls le gain de **précision** et la rétention des vrais par
   le nouveau modèle sont concluants (ce n'est pas une mesure du rappel absolu).
````

- [ ] **Step 6: Suite complète puis commit**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS (tous les tests, y compris les ~106 existants).

```bash
git add scripts/build_dataset.py tests/test_build_dataset_multi_verdicts.py README.md
git commit -m "feat: build_dataset — verdicts multiples + dédoublonnage ; runbook d'affinage MapRoulette

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Étapes manuelles (hors plan, côté utilisateur)

1. `fetch_maproulette_verdicts.py` puis `build_dataset.py` (téléchargement de ~6 000 tuiles WMTS, quelques dizaines de minutes).
2. Entraînement (10-20 h CPU estimé, à confirmer sur la 1ʳᵉ époque) ou Colab.
3. Comparaison avant/après sur 36 et 49, puis décision de remplacer `models/citernes-yolov8n.pt`.
