# Itération 2 — Hard-negative mining : Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingérer les verdicts de revue (101 faux → négatifs durs, 12 vrais → positifs) dans le dataset, permettre le réentraînement, et fournir un outil de comparaison avant/après sur Tours Métropole.

**Architecture :** Extension de `dataset.py` (`parse_verdicts`) et de `build_dataset.py` (`--verdicts`), + une fonction pure `compare_to_verdicts` dans `compare.py` avec un script CLI. Réentraînement et re-inférence réutilisent `train.py` / `infer_area.py` inchangés (runs manuels différés).

**Tech Stack :** Python 3.12, ultralytics, shapely, opencv-python, numpy, pytest. Réutilise tout l'existant.

## Global Constraints

- Python **3.12** via `.venv/Scripts/python` (bare `python` = 3.14 ; ne pas l'utiliser).
- Code sous `detection_ortho/`, scripts sous `scripts/`, tests sous `tests/`.
- **Aucun test ne fait d'appel réseau réel ni ne charge de modèle** (fixtures / cache pré-semé / fonctions pures).
- Coordonnées **(lon, lat)**. CSV de verdicts au format `index,lat,lon,score,verdict` (⚠️ lat en colonne 1, lon en colonne 2).
- Boîte des nouveaux positifs = `fixed_box_geo(lon, lat, DEFAULT_BOX_M)` (13 m, déjà dans `dataset.py`).
- Étapes **[RÉSEAU/MODÈLE — DIFFÉRÉ]** non exécutées par l'implémenteur (syntaxe vérifiée, run manuel ultérieur).
- Commits fréquents `type: description` ; `git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit` si identité absente.

---

## File Structure

- `detection_ortho/dataset.py` — **modifié** : ajout `parse_verdicts`.
- `detection_ortho/compare.py` — **modifié** : ajout `compare_to_verdicts`.
- `scripts/build_dataset.py` — **modifié** : option `--verdicts`.
- `scripts/compare_to_verdicts.py` — **nouveau** : rapport avant/après.
- `README.md` — **modifié** : section itération 2.
- `tests/` — tests des fonctions pures + intégration `--verdicts`.

---

## Task 1: Parsing des verdicts de revue

**Files:**
- Modify: `detection_ortho/dataset.py`
- Test: `tests/test_parse_verdicts.py`

**Interfaces:**
- Consumes: rien (pur).
- Produces: `parse_verdicts(lines: list[str]) -> list[dict]` → `{"lon","lat","verdict"}` pour les lignes `vrai`/`faux` ; ignore en-tête, `skip`, `non_revu`, lignes malformées.

- [ ] **Step 1: Écrire le test**

```python
# tests/test_parse_verdicts.py
from detection_ortho.dataset import parse_verdicts


def test_parses_vrai_faux_ignores_rest():
    lines = [
        "index,lat,lon,score,verdict",       # en-tête -> ignoré
        "1,47.361519,0.524931,0.997,vrai",
        "2,47.300208,0.525744,0.992,faux",
        "3,47.10,0.50,0.60,skip",            # ignoré
        "4,47.11,0.51,0.55,non_revu",        # ignoré
        "malformée",                          # ignoré
    ]
    out = parse_verdicts(lines)
    assert len(out) == 2
    assert out[0] == {"lon": 0.524931, "lat": 47.361519, "verdict": "vrai"}
    assert out[1]["verdict"] == "faux"
    # ordre (lon, lat) correct : lon vient de la colonne 2, lat de la colonne 1
    assert out[1]["lon"] == 0.525744 and out[1]["lat"] == 47.300208


def test_empty():
    assert parse_verdicts([]) == []
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_parse_verdicts.py -q`
Expected: FAIL avec `ImportError` sur `parse_verdicts`.

- [ ] **Step 3: Ajouter `parse_verdicts` à `dataset.py`**

Ajouter à la fin de `detection_ortho/dataset.py` :

```python
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
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_parse_verdicts.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/dataset.py tests/test_parse_verdicts.py
git commit -m "feat: parse_verdicts (lecture des verdicts de revue)"
```

---

## Task 2: Ingestion des verdicts dans build_dataset

**Files:**
- Modify: `scripts/build_dataset.py`
- Test: `tests/test_build_dataset_verdicts.py`

**Interfaces:**
- Consumes: `parse_verdicts`, `fixed_box_geo`, `DEFAULT_BOX_M` (dataset.py).
- Produces: option `--verdicts <csv>` sur `build_dataset.py` : `faux` → record négatif `hardneg_i` (bbox None) ; `vrai` → record positif `revpos_i` (boîte fixe 13 m). Ajoutés aux `records` avant le pré-téléchargement et le split.

- [ ] **Step 1: Écrire le test d'intégration offline (avec --verdicts)**

```python
# tests/test_build_dataset_verdicts.py
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


def test_verdicts_add_hardneg_and_revpos(tmp_path, monkeypatch):
    # Pas de citernes/piscines OSM : on isole l'effet des verdicts.
    monkeypatch.setattr(build_dataset, "fetch_features_geom", lambda *a, **k: [])
    out = tmp_path / "ds"
    cache = out / "tiles_cache"
    faux_lon, faux_lat = 0.65, 47.33
    vrai_lon, vrai_lat = 0.66, 47.34
    _seed_tiles(cache, faux_lon, faux_lat)
    _seed_tiles(cache, vrai_lon, vrai_lat)
    verdicts = tmp_path / "v.csv"
    verdicts.write_text(
        "index,lat,lon,score,verdict\n"
        f"1,{faux_lat},{faux_lon},0.5,faux\n"
        f"2,{vrai_lat},{vrai_lon},0.9,vrai\n",
        encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "build_dataset.py", "--bbox", "0.6", "47.3", "0.7", "47.4",
        "--negatives", "0", "--max-pools", "0",
        "--verdicts", str(verdicts), "--out", str(out)])
    build_dataset.main()

    # un chip hardneg_* (label vide) et un chip revpos_* (label non vide)
    labels = list((out / "labels").rglob("*.txt"))
    hard = [p for p in labels if p.stem.startswith("hardneg")]
    rev = [p for p in labels if p.stem.startswith("revpos")]
    assert len(hard) == 1 and hard[0].read_text().strip() == ""
    assert len(rev) == 1 and rev[0].read_text().strip().startswith("0 ")
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_build_dataset_verdicts.py -q`
Expected: FAIL (option `--verdicts` inconnue → SystemExit/erreur argparse).

- [ ] **Step 3: Étendre `scripts/build_dataset.py`**

Dans le bloc d'imports depuis `detection_ortho.dataset`, ajouter `fixed_box_geo`,
`DEFAULT_BOX_M`, `parse_verdicts` à la liste importée. Concrètement, remplacer :

```python
from detection_ortho.dataset import (
    element_to_box, assemble_window, geo_bbox_to_pixel_bbox, to_yolo_label,
    write_chip, split_indices, write_data_yaml, window_tiles,
)
```

par :

```python
from detection_ortho.dataset import (
    element_to_box, assemble_window, geo_bbox_to_pixel_bbox, to_yolo_label,
    write_chip, split_indices, write_data_yaml, window_tiles,
    fixed_box_geo, DEFAULT_BOX_M, parse_verdicts,
)
```

Ajouter l'argument (près des autres `ap.add_argument`) :

```python
    ap.add_argument("--verdicts", type=Path, default=None,
                    help="CSV de revue : faux -> négatifs durs, vrai -> positifs")
```

Juste APRÈS la boucle des négatifs de fond aléatoires (celle qui ajoute les
`bg_*` à `records`) et AVANT le pré-téléchargement des tuiles, insérer :

```python
    # --- Chips issus de la revue (hard-negative mining) ---
    if args.verdicts:
        vs = parse_verdicts(args.verdicts.read_text(encoding="utf-8").splitlines())
        n_hard = n_rev = 0
        for v in vs:
            if v["verdict"] == "faux":
                records.append((f"hardneg_{n_hard:04d}", v["lon"], v["lat"], None))
                n_hard += 1
            else:  # vrai
                bbox = fixed_box_geo(v["lon"], v["lat"], DEFAULT_BOX_M)
                records.append((f"revpos_{n_rev:04d}", v["lon"], v["lat"], bbox))
                n_rev += 1
        print(f"Verdicts ingérés : {n_hard} négatif(s) dur(s), {n_rev} positif(s).")
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_build_dataset_verdicts.py -q`
Expected: 1 passed.

- [ ] **Step 5: Vérifier la syntaxe + la suite complète**

Run: `.venv\Scripts\python -c "import ast; ast.parse(open('scripts/build_dataset.py').read())"` puis `.venv\Scripts\python -m pytest -q`
Expected: syntaxe OK ; tous les tests passent. **Ne pas exécuter `build_dataset.py` en réel** (réseau).

- [ ] **Step 6: Commit**

```bash
git add scripts/build_dataset.py tests/test_build_dataset_verdicts.py
git commit -m "feat: build_dataset --verdicts (hard-negatives + positifs de revue)"
```

---

## Task 3: Comparaison des détections aux verdicts connus

**Files:**
- Modify: `detection_ortho/compare.py`
- Test: `tests/test_compare_to_verdicts.py`

**Interfaces:**
- Consumes: `geo.haversine_m` (déjà importable).
- Produces: `compare_to_verdicts(detections: list[dict], verdicts: list[dict], radius_m: float) -> dict` → `{fp_total, fp_still_detected, fp_suppressed, tp_total, tp_kept, n_candidates_new}`.

- [ ] **Step 1: Écrire le test**

```python
# tests/test_compare_to_verdicts.py
from detection_ortho.compare import compare_to_verdicts


def test_counts_fp_suppressed_and_tp_kept():
    verdicts = [
        {"lon": 0.10, "lat": 47.0, "verdict": "faux"},   # encore détecté
        {"lon": 0.20, "lat": 47.0, "verdict": "faux"},   # supprimé
        {"lon": 0.30, "lat": 47.0, "verdict": "vrai"},   # conservé
        {"lon": 0.40, "lat": 47.0, "verdict": "vrai"},   # perdu
    ]
    detections = [
        {"lon": 0.100001, "lat": 47.0, "score": 0.9},    # proche du faux #1
        {"lon": 0.300001, "lat": 47.0, "score": 0.8},    # proche du vrai #3
        {"lon": 0.90, "lat": 47.0, "score": 0.7},        # ailleurs
    ]
    r = compare_to_verdicts(detections, verdicts, radius_m=25)
    assert r["fp_total"] == 2
    assert r["fp_still_detected"] == 1
    assert r["fp_suppressed"] == 1
    assert r["tp_total"] == 2
    assert r["tp_kept"] == 1
    assert r["n_candidates_new"] == 3


def test_empty():
    r = compare_to_verdicts([], [], radius_m=25)
    assert r["fp_total"] == 0 and r["tp_total"] == 0 and r["n_candidates_new"] == 0
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_compare_to_verdicts.py -q`
Expected: FAIL avec `ImportError` sur `compare_to_verdicts`.

- [ ] **Step 3: Ajouter `compare_to_verdicts` à `compare.py`**

Ajouter à la fin de `detection_ortho/compare.py` :

```python
def compare_to_verdicts(
    detections: list[dict], verdicts: list[dict], radius_m: float
) -> dict:
    """Croise de nouvelles détections avec des verdicts connus (par proximité).

    Mesure le gain avant/après : combien de faux positifs connus ne sont plus
    détectés (fp_suppressed) et combien de vrais positifs restent détectés
    (tp_kept).
    """
    def detected(pt: dict) -> bool:
        return any(
            haversine_m(pt["lon"], pt["lat"], d["lon"], d["lat"]) <= radius_m
            for d in detections
        )

    faux = [v for v in verdicts if v.get("verdict") == "faux"]
    vrai = [v for v in verdicts if v.get("verdict") == "vrai"]
    fp_still = sum(1 for v in faux if detected(v))
    tp_kept = sum(1 for v in vrai if detected(v))
    return {
        "fp_total": len(faux),
        "fp_still_detected": fp_still,
        "fp_suppressed": len(faux) - fp_still,
        "tp_total": len(vrai),
        "tp_kept": tp_kept,
        "n_candidates_new": len(detections),
    }
```

> Note : `compare.py` importe déjà `from detection_ortho.geo import haversine_m` (utilisé par `match_detections`) ; ne pas le réimporter.

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_compare_to_verdicts.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/compare.py tests/test_compare_to_verdicts.py
git commit -m "feat: compare_to_verdicts (mesure du gain avant/après)"
```

---

## Task 4: Script de comparaison + doc

**Files:**
- Create: `scripts/compare_to_verdicts.py`
- Modify: `README.md`
- Test: `tests/test_compare_to_verdicts_help.py`

**Interfaces:**
- Consumes: `dataset.parse_verdicts`, `compare.compare_to_verdicts`.
- Produces: `scripts/compare_to_verdicts.py` — charge `detected_only.geojson` (nouveau run) + `verdicts.csv` (revue), imprime le rapport avant/après.

- [ ] **Step 1: Écrire le test smoke (--help)**

```python
# tests/test_compare_to_verdicts_help.py
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_compare_help_runs():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "compare_to_verdicts.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "ModuleNotFoundError" not in r.stderr
    assert "--detections" in r.stdout
    assert "--verdicts" in r.stdout
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_compare_to_verdicts_help.py -q`
Expected: FAIL (script inexistant).

- [ ] **Step 3: Écrire `scripts/compare_to_verdicts.py`**

```python
# scripts/compare_to_verdicts.py
"""Itération 2 — Compare de nouvelles détections aux verdicts de revue connus.

Usage:
    python scripts/compare_to_verdicts.py \
        --detections inference_out/detected_only.geojson \
        --verdicts verdicts.csv --radius 25

Rapport avant/après : faux positifs supprimés, vrais positifs conservés,
nombre de candidats. Génération/lecture de fichiers uniquement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_ortho.dataset import parse_verdicts
from detection_ortho.compare import compare_to_verdicts


def _load_points(geojson_path: Path) -> list[dict]:
    fc = json.loads(geojson_path.read_text(encoding="utf-8"))
    pts = []
    for feat in fc.get("features", []):
        coords = feat["geometry"]["coordinates"]
        pts.append({"lon": coords[0], "lat": coords[1]})
    return pts


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", type=Path, required=True,
                    help="GeoJSON des nouvelles détections (detected_only.geojson)")
    ap.add_argument("--verdicts", type=Path, required=True,
                    help="CSV de revue (verdicts.csv)")
    ap.add_argument("--radius", type=float, default=25.0)
    args = ap.parse_args()

    detections = _load_points(args.detections)
    verdicts = parse_verdicts(args.verdicts.read_text(encoding="utf-8").splitlines())
    r = compare_to_verdicts(detections, verdicts, radius_m=args.radius)

    print("\n=== Comparaison aux verdicts connus ===")
    print(f"  Candidats (nouveau run)        : {r['n_candidates_new']}")
    print(f"  Faux positifs supprimés        : {r['fp_suppressed']}/{r['fp_total']}")
    print(f"  Faux positifs encore détectés  : {r['fp_still_detected']}/{r['fp_total']}")
    print(f"  Vrais positifs conservés       : {r['tp_kept']}/{r['tp_total']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_compare_to_verdicts_help.py -q`
Expected: 1 passed.

- [ ] **Step 5: Ajouter la section itération 2 au `README.md`**

Ajouter à la fin de `README.md` :

```markdown
## Itération 2 — Hard-negative mining

À partir de `verdicts.csv` (revue de la carte, cf. `make_map.py`) :

1. Régénérer le dataset augmenté (négatifs durs + positifs de revue) :

       python scripts/build_dataset.py --bbox 0.05 46.72 1.06 47.72 \
           --verdicts verdicts.csv --out dataset

2. Réentraîner :

       python scripts/train.py --data dataset/data.yaml --epochs 100 --device cpu

3. Re-inférer Tours Métropole avec les nouveaux poids (cf. Jalon 3), puis
   mesurer le gain avant/après :

       python scripts/compare_to_verdicts.py \
           --detections inference_out/detected_only.geojson --verdicts verdicts.csv
```

- [ ] **Step 6: Lancer la suite complète**

Run: `.venv\Scripts\python -m pytest -q`
Expected: tous les tests passent.

- [ ] **Step 7: Commit**

```bash
git add scripts/compare_to_verdicts.py README.md tests/test_compare_to_verdicts_help.py
git commit -m "feat: script de comparaison aux verdicts + doc itération 2"
```

---

## Self-Review (fait à la rédaction)

- **Couverture du spec :** parse_verdicts (Task 1) ✓ ; ingestion `--verdicts` avec faux→hardneg / vrai→revpos boîte fixe (Task 2) ✓ ; réentraînement (réutilise `train.py`, run manuel — pas de tâche code) ✓ ; comparaison avant/après `compare_to_verdicts` + script (Tasks 3-4) ✓ ; re-inférence (réutilise `infer_area.py`, run manuel) ✓ ; doc workflow (Task 4) ✓.
- **Placeholders :** aucun. Les runs réels (dataset, entraînement, inférence) sont explicitement des étapes manuelles différées, cohérent avec les jalons précédents.
- **Cohérence des types :** `parse_verdicts -> [{lon,lat,verdict}]` consommé par build_dataset (faux→record bbox=None ; vrai→record `fixed_box_geo`) ET par `compare_to_verdicts` ; `_load_points -> [{lon,lat}]` consommé par `compare_to_verdicts` ; retour dict `{fp_total, fp_suppressed, fp_still_detected, tp_total, tp_kept, n_candidates_new}` consommé fidèlement par le script. Colonnes CSV (lat col 1, lon col 2) respectées. Convention (lon, lat) partout.
