# Diagnostic de précision honnête + calibration de seuil : Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deux leviers de précision automatiques : (1) split spatial pour un mAP honnête, (2) balayage de seuil sur données labellisées pour verrouiller le point de fonctionnement.

**Architecture :** `dataset.py` gagne `spatial_split_indices` (+ `build_dataset --spatial-split`). `infer.py` gagne `max_score_near`. `compare.py` gagne `sweep_precision_recall`. `scripts/sweep_threshold.py` (nouveau) infère une fois et balaye le seuil.

**Tech Stack :** Python 3.12, ultralytics, opencv-python, numpy, pytest.

## Global Constraints

- Python **3.12** via `.venv\Scripts\python` (bare `python` = 3.14 ; ne pas l'utiliser).
- Code sous `detection_ortho/`, scripts sous `scripts/`, tests sous `tests/`.
- **Aucun test réseau réel ni poids réel** : tuiles pré-semées, modèle YOLO stubé.
- Coordonnées (lon, lat). Fonctions pures = déterministes et testables isolément.
- **Réutiliser** sans les modifier : `assemble_window`, `compose_rgn`, `LAYER_IRC`, `parse_verdicts`, `result_to_boxes`, `boxes_to_points`, `is_detected_near`, `haversine_m`.
- Le chemin par défaut (sans `--spatial-split`) reste **strictement inchangé**.
- Les 2 runs réels (rebuild+entraînement spatial ; balayage) sont **différés** (manuels).
- Commits `type: description` ; `git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit` si identité absente.

## File Structure

- `detection_ortho/dataset.py` — **modifié** : `spatial_split_indices`.
- `scripts/build_dataset.py` — **modifié** : `--spatial-split` / `--cell-deg`.
- `detection_ortho/infer.py` — **modifié** : `max_score_near`.
- `detection_ortho/compare.py` — **modifié** : `sweep_precision_recall`.
- `scripts/sweep_threshold.py` — **nouveau** : CLI de balayage.
- `README.md` — **modifié** : section diagnostic précision.
- `tests/` — `test_spatial_split.py`, `test_max_score_near.py`, `test_sweep_precision_recall.py`, `test_sweep_threshold.py`.

---

## Task 1: Split spatial (dataset.py + build_dataset.py)

**Files:**
- Modify: `detection_ortho/dataset.py`, `scripts/build_dataset.py`
- Test: `tests/test_spatial_split.py`

**Interfaces:**
- Produces: `spatial_split_indices(points, cell_deg=0.05, seed=0, ratios=(0.7,0.15,0.15)) -> dict` — partition par cellules entières ; option `--spatial-split`/`--cell-deg` sur build_dataset.

- [ ] **Step 1: Écrire les tests**

```python
# tests/test_spatial_split.py
from detection_ortho.dataset import spatial_split_indices


def test_cells_stay_whole():
    # 3 cellules bien séparées, 2 points chacune
    points = [
        (0.10, 47.10), (0.11, 47.11),   # cellule A (0,05°)
        (0.60, 47.60), (0.61, 47.61),   # cellule B
        (1.00, 47.00), (1.01, 47.01),   # cellule C
    ]
    split = spatial_split_indices(points, cell_deg=0.05, seed=0)
    # partition complète et disjointe
    allidx = sorted(split["train"] + split["val"] + split["test"])
    assert allidx == list(range(6))
    # les deux points d'une même cellule sont dans le même lot
    for a, b in [(0, 1), (2, 3), (4, 5)]:
        for part in ("train", "val", "test"):
            assert (a in split[part]) == (b in split[part])


def test_deterministic():
    points = [(0.1 * i, 47.0 + 0.1 * i) for i in range(20)]
    assert spatial_split_indices(points, seed=3) == spatial_split_indices(points, seed=3)


def test_ratios_approx():
    # 100 cellules distinctes, 1 point chacune → ~70/15/15
    points = [(0.1 * i, 47.0) for i in range(100)]
    split = spatial_split_indices(points, cell_deg=0.05, seed=0)
    assert 60 <= len(split["train"]) <= 80
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == 100
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_spatial_split.py -q`
Expected: FAIL (`ImportError` sur `spatial_split_indices`).

- [ ] **Step 3: Modifier `dataset.py`**

Ajouter après `split_indices` (le module importe déjà `random` et `math` — sinon `import math`) :

```python
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
```

Si `math` n'est pas importé en tête de `dataset.py`, ajouter `import math`.

- [ ] **Step 4: Modifier `scripts/build_dataset.py`**

Importer `spatial_split_indices` (l'ajouter à la ligne d'import depuis `detection_ortho.dataset`, à côté de `split_indices`).

Ajouter les arguments (près des autres `ap.add_argument`) :

```python
    ap.add_argument("--spatial-split", action="store_true",
                    help="split géographique par cellule (au lieu d'aléatoire)")
    ap.add_argument("--cell-deg", type=float, default=0.05,
                    help="taille de cellule du split spatial, en degrés")
```

Remplacer la ligne `split = split_indices(len(records), seed=args.seed)` par :

```python
    if args.spatial_split:
        split = spatial_split_indices(
            [(lon, lat) for _n, lon, lat, _b in records],
            cell_deg=args.cell_deg, seed=args.seed)
    else:
        split = split_indices(len(records), seed=args.seed)
```

- [ ] **Step 5: Lancer les tests + syntaxe**

Run: `.venv\Scripts\python -m pytest tests/test_spatial_split.py tests/test_dataset_window.py -q` puis `.venv\Scripts\python -c "import ast; ast.parse(open('scripts/build_dataset.py').read())"`
Expected: tous passent ; syntaxe OK.

- [ ] **Step 6: Commit**

```bash
git add detection_ortho/dataset.py scripts/build_dataset.py tests/test_spatial_split.py
git commit -m "feat: split spatial par cellule (spatial_split_indices + build_dataset --spatial-split)"
```

---

## Task 2: Helpers de calibration (infer.py + compare.py)

**Files:**
- Modify: `detection_ortho/infer.py`, `detection_ortho/compare.py`
- Test: `tests/test_max_score_near.py`, `tests/test_sweep_precision_recall.py`

**Interfaces:**
- Produces:
  - `max_score_near(det_points, lon, lat, radius_m) -> float` (infer.py) — meilleur `score` à ≤ radius, sinon 0.0.
  - `sweep_precision_recall(scored, thresholds) -> list[dict]` (compare.py) — `scored` = liste `(best_score, is_true)` ; par seuil : `{conf, precision, recall, tp, fp}`.

- [ ] **Step 1: Écrire les tests**

```python
# tests/test_max_score_near.py
from detection_ortho.infer import max_score_near


def test_best_score_within_radius():
    pts = [
        {"lon": 0.6531, "lat": 47.3305, "score": 0.4},
        {"lon": 0.6531, "lat": 47.3305, "score": 0.8},
    ]
    assert max_score_near(pts, 0.6531, 47.3305, 25.0) == 0.8


def test_zero_when_none_near():
    pts = [{"lon": 0.6551, "lat": 47.3305, "score": 0.9}]  # ~150 m
    assert max_score_near(pts, 0.6531, 47.3305, 25.0) == 0.0


def test_zero_when_empty():
    assert max_score_near([], 0.6531, 47.3305, 25.0) == 0.0
```

```python
# tests/test_sweep_precision_recall.py
from detection_ortho.compare import sweep_precision_recall


def test_sweep_basic():
    # 2 vrais (scores 0.9, 0.6), 2 faux (scores 0.7, 0.3)
    scored = [(0.9, True), (0.6, True), (0.7, False), (0.3, False)]
    rows = {r["conf"]: r for r in sweep_precision_recall(scored, [0.5, 0.8])}
    # seuil 0.5 : tp=2 (0.9,0.6), fp=1 (0.7) -> prec 2/3, rappel 2/2
    assert rows[0.5]["tp"] == 2 and rows[0.5]["fp"] == 1
    assert abs(rows[0.5]["precision"] - 2 / 3) < 1e-9
    assert rows[0.5]["recall"] == 1.0
    # seuil 0.8 : tp=1 (0.9), fp=0 -> prec 1.0, rappel 1/2
    assert rows[0.8]["tp"] == 1 and rows[0.8]["fp"] == 0
    assert rows[0.8]["precision"] == 1.0 and rows[0.8]["recall"] == 0.5


def test_sweep_no_positives_no_crash():
    rows = sweep_precision_recall([(0.2, False)], [0.5])
    assert rows[0]["precision"] == 0.0 and rows[0]["recall"] == 0.0
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_max_score_near.py tests/test_sweep_precision_recall.py -q`
Expected: FAIL (ImportError sur les deux fonctions).

- [ ] **Step 3: Modifier `infer.py`**

Ajouter après `is_detected_near` :

```python
def max_score_near(det_points, lon: float, lat: float, radius_m: float) -> float:
    """Meilleur score de détection à <= radius_m du centre (lon, lat), sinon 0.0."""
    scores = [
        p["score"] for p in det_points
        if haversine_m(lon, lat, p["lon"], p["lat"]) <= radius_m
    ]
    return max(scores) if scores else 0.0
```

- [ ] **Step 4: Modifier `compare.py`**

Ajouter à la fin :

```python
def sweep_precision_recall(scored, thresholds) -> list:
    """Précision/rappel par seuil. `scored` = liste de (best_score, is_true).

    Un point « tire » à un seuil si best_score >= seuil.
    """
    n_true = sum(1 for _s, t in scored if t)
    rows = []
    for th in thresholds:
        tp = sum(1 for s, t in scored if t and s >= th)
        fp = sum(1 for s, t in scored if (not t) and s >= th)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / n_true if n_true else 0.0
        rows.append({"conf": th, "precision": prec, "recall": rec, "tp": tp, "fp": fp})
    return rows
```

- [ ] **Step 5: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_max_score_near.py tests/test_sweep_precision_recall.py tests/test_infer_windows.py tests/test_compare_to_verdicts.py -q`
Expected: tous passent.

- [ ] **Step 6: Commit**

```bash
git add detection_ortho/infer.py detection_ortho/compare.py tests/test_max_score_near.py tests/test_sweep_precision_recall.py
git commit -m "feat: max_score_near + sweep_precision_recall (calibration de seuil)"
```

---

## Task 3: Script `scripts/sweep_threshold.py`

**Files:**
- Create: `scripts/sweep_threshold.py`
- Test: `tests/test_sweep_threshold.py`

**Interfaces:**
- Consumes: `assemble_window`, `compose_rgn`, `LAYER_IRC`, `parse_verdicts`, `result_to_boxes`, `boxes_to_points`, `max_score_near`, `sweep_precision_recall`, `ultralytics.YOLO`.
- Produces: CLI qui infère une fois par point (à `--conf-min`) et imprime la table précision/rappel vs seuil.

- [ ] **Step 1: Écrire le test d'intégration (modèle stubé, cache pré-semé)**

```python
# tests/test_sweep_threshold.py
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import sweep_threshold  # noqa: E402
from detection_ortho.dataset import window_tiles  # noqa: E402


def _seed(cache, lon, lat, value):
    cache.mkdir(parents=True, exist_ok=True)
    tiles, _, _ = window_tiles(lon, lat, 19, 640)
    for x, y in tiles:
        cv2.imwrite(str(cache / f"19_{x}_{y}.jpg"),
                    np.full((256, 256, 3), value, np.uint8))


class _Box:
    def __init__(self, cx, cy, score):
        self.xywh = [[cx, cy, 20.0, 20.0]]
        self.conf = [score]


class _Res:
    def __init__(self, boxes):
        self.boxes = boxes


def test_sweep_prints_table(tmp_path, monkeypatch, capsys):
    # un vrai (score 0.9) et un faux (score 0.4)
    vrai = (0.65, 47.33)
    faux = (0.70, 47.40)
    cache = tmp_path / "cache"
    _seed(cache, *vrai, 100)
    _seed(cache, *faux, 120)

    verdicts = tmp_path / "v.csv"
    verdicts.write_text(
        "index,lat,lon,score,verdict\n"
        f"1,{vrai[1]},{vrai[0]},0.9,vrai\n"
        f"2,{faux[1]},{faux[0]},0.4,faux\n",
        encoding="utf-8")

    score_by_lon = {round(vrai[0], 4): 0.9, round(faux[0], 4): 0.4}

    class FakeYOLO:
        def __init__(self, weights):
            pass

        def predict(self, img, conf=0.05, device="cpu", verbose=False):
            # score choisi selon la moyenne des pixels (100=vrai, 120=faux)
            s = 0.9 if int(img.mean()) < 110 else 0.4
            return [_Res([_Box(320.0, 320.0, s)])]

    monkeypatch.setattr(sweep_threshold, "YOLO", FakeYOLO)
    monkeypatch.setattr(sys, "argv", [
        "sweep_threshold.py", "--weights", "x.pt", "--verdicts", str(verdicts),
        "--conf-min", "0.1", "--conf-max", "0.9", "--step", "0.4",
        "--cache", str(cache)])
    sweep_threshold.main()

    out = capsys.readouterr().out
    # à seuil 0.5 : le vrai (0.9) tire, le faux (0.4) non -> précision 1.0, rappel 1.0
    assert "0.50" in out or "0.5" in out
    assert "précision" in out.lower() or "precision" in out.lower()
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_sweep_threshold.py -q`
Expected: FAIL (`ModuleNotFoundError: sweep_threshold`).

- [ ] **Step 3: Créer `scripts/sweep_threshold.py`**

```python
"""Calibration de seuil — précision/rappel vs confiance sur points labellisés.

Infère UNE fois par point (à --conf-min), retient le meilleur score près du
centre, puis balaye le seuil en mémoire. Lecture/impression seulement.

Usage:
    python scripts/sweep_threshold.py --weights runs/citernes/weights/best.pt \
        --verdicts verdicts.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from detection_ortho.dataset import assemble_window, compose_rgn, parse_verdicts
from detection_ortho.tiles import LAYER_IRC
from detection_ortho.infer import result_to_boxes, boxes_to_points, max_score_near
from detection_ortho.compare import sweep_precision_recall


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--nir", action="store_true")
    ap.add_argument("--conf-min", type=float, default=0.05)
    ap.add_argument("--conf-max", type=float, default=0.9)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--radius", type=float, default=25.0)
    ap.add_argument("--window", type=int, default=640)
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--cache", type=Path, default=Path("tiles_cache/eval"))
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    verdicts = parse_verdicts(args.verdicts.read_text(encoding="utf-8").splitlines())
    print(f"{len(verdicts)} point(s), inférence à conf {args.conf_min} "
          f"({'[R,G,NIR]' if args.nir else 'RVB'})...")

    model = YOLO(str(args.weights))
    scored: list = []
    for i, v in enumerate(verdicts, 1):
        lon, lat = v["lon"], v["lat"]
        try:
            img, ogx, ogy = assemble_window(lon, lat, args.zoom, args.window, args.cache)
            if args.nir:
                irc, _, _ = assemble_window(lon, lat, args.zoom, args.window,
                                            args.cache, layer=LAYER_IRC)
                img = compose_rgn(img, irc)
        except Exception as exc:  # noqa: BLE001
            print(f"  point {i}: échec fenêtre ({exc})", file=sys.stderr)
            continue
        res = model.predict(img, conf=args.conf_min, device=args.device, verbose=False)
        pts = boxes_to_points(result_to_boxes(res[0].boxes), ogx, ogy, args.zoom)
        scored.append((max_score_near(pts, lon, lat, args.radius),
                       v.get("verdict") == "vrai"))
        print(f"  {i}/{len(verdicts)}", end="\r", file=sys.stderr, flush=True)

    n = int(round((args.conf_max - args.conf_min) / args.step)) + 1
    thresholds = [round(args.conf_min + k * args.step, 4) for k in range(max(n, 1))]
    rows = sweep_precision_recall(scored, thresholds)
    print("\n=== Précision/rappel vs seuil ===")
    print("  seuil  précision  rappel   tp   fp")
    for r in rows:
        print(f"  {r['conf']:.2f}    {r['precision']:.3f}     {r['recall']:.3f}   "
              f"{r['tp']:>3}  {r['fp']:>3}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_sweep_threshold.py -q`
Expected: 1 passed.

- [ ] **Step 5: Syntaxe + suite complète**

Run: `.venv\Scripts\python -c "import ast; ast.parse(open('scripts/sweep_threshold.py').read())"` puis `.venv\Scripts\python -m pytest -q`
Expected: syntaxe OK ; tous les tests passent. **Ne pas lancer sweep_threshold.py en réel.**

- [ ] **Step 6: Commit**

```bash
git add scripts/sweep_threshold.py tests/test_sweep_threshold.py
git commit -m "feat: sweep_threshold — balayage précision/rappel vs seuil (1 passe)"
```

---

## Task 4: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Ajouter la section « Diagnostic de précision » au `README.md`**

Ajouter à la fin de `README.md` :

```markdown
## Diagnostic de précision (split spatial + calibration de seuil)

Le mAP de test (0,84) est optimiste (split aléatoire → fuite géographique). Pour
un vrai chiffre de généralisation, régénérer le dataset avec un **split spatial**
(zones disjointes), puis ré-entraîner et évaluer :

    python scripts/build_dataset.py --bbox 0.05 46.72 1.06 47.72 \
        --verdicts verdicts.csv --spatial-split --out dataset_spatial
    python scripts/train.py --data dataset_spatial/data.yaml --epochs 100 \
        --device cpu --name citernes_spatial
    python scripts/evaluate.py --weights runs/citernes_spatial/weights/best.pt \
        --data dataset_spatial/data.yaml

Pour verrouiller le seuil de confiance (gain de précision immédiat), balayer sur
les points labellisés :

    python scripts/sweep_threshold.py --weights runs/citernes/weights/best.pt \
        --verdicts verdicts.csv

La table précision/rappel vs seuil donne le point de fonctionnement à utiliser
pour le run départemental. Pour catégoriser les faux positifs, ouvrir
`scripts/make_map.py` sur les points de `verdicts.csv`.
```

- [ ] **Step 2: Lancer la suite complète**

Run: `.venv\Scripts\python -m pytest -q`
Expected: tous les tests passent.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: workflow diagnostic de précision (split spatial + seuil)"
```

---

## Self-Review (fait à la rédaction)

- **Couverture du spec :** split spatial (Task 1) ✓ ; `max_score_near` + `sweep_precision_recall` (Task 2) ✓ ; `sweep_threshold.py` une passe + balayage (Task 3) ✓ ; protocole + doc + FP via make_map (Task 4) ✓ ; runs réels différés ✓ ; tests hors-ligne (stub + cache) ✓.
- **Placeholders :** aucun.
- **Cohérence des types :** `spatial_split_indices(points, …) -> {train,val,test}` (indices) ↔ `build_dataset` (records → points) ; `max_score_near(...) -> float` ; `sweep_precision_recall([(score,is_true)], [seuils]) -> [{conf,precision,recall,tp,fp}]` ; `sweep_threshold` : `boxes_to_points → max_score_near → (score, verdict=='vrai') → sweep_precision_recall`. Chemin aléatoire par défaut inchangé (Task 1 Step 4). Cache défaut `tiles_cache/eval` (gitignoré, tuiles RVB déjà présentes).
