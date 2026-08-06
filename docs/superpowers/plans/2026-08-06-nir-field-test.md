# Test terrain NIR aux points de verdict : Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un script `scripts/eval_points.py` évaluant un modèle YOLO aux points labellisés de `verdicts.csv` (fenêtre RVB ou [R,G,NIR] via `--nir`), réutilisant `compare_to_verdicts` pour compter FP supprimés / vrais conservés. On le lance une fois par modèle et on compare.

**Architecture :** `infer.py` gagne un helper pur `is_detected_near`. `scripts/eval_points.py` (nouveau) boucle sur les points, assemble la/les fenêtre(s), prédit, décide « tiré ici ? », alimente `compare_to_verdicts`.

**Tech Stack :** Python 3.12, ultralytics, opencv-python, numpy, pytest.

## Global Constraints

- Python **3.12** via `.venv\Scripts\python` (bare `python` = 3.14 ; ne pas l'utiliser).
- Code sous `detection_ortho/`, scripts sous `scripts/`, tests sous `tests/`.
- **Aucun test réseau réel ni poids réel** : tuiles pré-semées en cache, modèle YOLO stubé (monkeypatch).
- Coordonnées (lon, lat). NIR = canal rouge (index 2 BGR) de la couche `LAYER_IRC`.
- **Réutiliser** sans les modifier : `assemble_window`, `compose_rgn`, `LAYER_IRC`, `parse_verdicts`, `result_to_boxes`, `boxes_to_points`, `compare_to_verdicts`, `haversine_m`.
- Le script **n'upload rien** et ne fait que lire/imprimer (cohérent avec la contrainte projet). Les 2 runs réels sont **différés** (manuels).
- Commits `type: description` ; `git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit` si identité absente.

## File Structure

- `detection_ortho/infer.py` — **modifié** : helper `is_detected_near` + import `haversine_m`.
- `scripts/eval_points.py` — **nouveau** : CLI d'évaluation aux points.
- `README.md` — **modifié** : section test terrain NIR.
- `tests/` — `test_is_detected_near.py`, `test_eval_points.py`.

---

## Task 1: Helper `is_detected_near` (infer.py)

**Files:**
- Modify: `detection_ortho/infer.py`
- Test: `tests/test_is_detected_near.py`

**Interfaces:**
- Consumes: `detection_ortho.geo.haversine_m`.
- Produces: `is_detected_near(det_points, lon, lat, radius_m) -> bool` — vrai si un point `{lon,lat,...}` est à ≤ `radius_m` du centre `(lon,lat)`.

- [ ] **Step 1: Écrire les tests**

```python
# tests/test_is_detected_near.py
from detection_ortho.infer import is_detected_near


def test_detected_within_radius():
    pts = [{"lon": 0.6531, "lat": 47.3305, "score": 0.9}]
    assert is_detected_near(pts, 0.6531, 47.3305, 25.0)


def test_not_detected_outside_radius():
    # ~150 m à l'est → hors du rayon 25 m
    pts = [{"lon": 0.6551, "lat": 47.3305, "score": 0.9}]
    assert not is_detected_near(pts, 0.6531, 47.3305, 25.0)


def test_empty_points():
    assert not is_detected_near([], 0.6531, 47.3305, 25.0)
```

- [ ] **Step 2: Lancer les tests pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_is_detected_near.py -q`
Expected: FAIL (`ImportError` sur `is_detected_near`).

- [ ] **Step 3: Modifier `infer.py`**

Ajouter l'import (à côté des imports `detection_ortho`) :

```python
from detection_ortho.geo import haversine_m
```

Ajouter la fonction (après `boxes_to_points`) :

```python
def is_detected_near(det_points, lon: float, lat: float, radius_m: float) -> bool:
    """Vrai si une détection tombe à <= radius_m du centre (lon, lat)."""
    return any(
        haversine_m(lon, lat, p["lon"], p["lat"]) <= radius_m
        for p in det_points
    )
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_is_detected_near.py tests/test_infer.py -q`
Expected: tous passent (les tests infer existants inchangés).

- [ ] **Step 5: Commit**

```bash
git add detection_ortho/infer.py tests/test_is_detected_near.py
git commit -m "feat: is_detected_near (détection à proximité d'un point)"
```

---

## Task 2: Script `scripts/eval_points.py`

**Files:**
- Create: `scripts/eval_points.py`
- Test: `tests/test_eval_points.py`

**Interfaces:**
- Consumes: `assemble_window`, `compose_rgn`, `LAYER_IRC`, `parse_verdicts`, `result_to_boxes`, `boxes_to_points`, `is_detected_near`, `compare_to_verdicts`, `ultralytics.YOLO`.
- Produces: CLI `eval_points.py --weights --verdicts [--conf 0.55] [--nir] [--radius 25] [--window 640] [--zoom 19] [--cache tiles_cache/eval] [--device cpu]` → imprime le rapport de comparaison aux verdicts.

- [ ] **Step 1: Écrire le test d'intégration (modèle stubé, cache pré-semé)**

```python
# tests/test_eval_points.py
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import eval_points  # noqa: E402
from detection_ortho.dataset import window_tiles  # noqa: E402
from detection_ortho.tiles import LAYER_IRC  # noqa: E402


def _seed(cache, lon, lat, tag, value):
    cache.mkdir(parents=True, exist_ok=True)
    tiles, _, _ = window_tiles(lon, lat, 19, 640)
    suffix = f"_{tag}" if tag else ""
    for x, y in tiles:
        cv2.imwrite(str(cache / f"19_{x}_{y}{suffix}.jpg"),
                    np.full((256, 256, 3), value, np.uint8))


class _Box:
    def __init__(self, cx, cy, score):
        self.xywh = [[cx, cy, 20.0, 20.0]]
        self.conf = [score]


class _Res:
    def __init__(self, boxes):
        self.boxes = boxes


def test_nir_eval_composes_and_tallies(tmp_path, monkeypatch, capsys):
    lon, lat = 0.65, 47.33
    cache = tmp_path / "cache"
    _seed(cache, lon, lat, "", 100)                                  # RVB
    _seed(cache, lon, lat, LAYER_IRC.rsplit(".", 1)[-1].lower(), 200)  # IRC

    verdicts = tmp_path / "v.csv"
    verdicts.write_text(f"index,lat,lon,score,verdict\n1,{lat},{lon},0.9,faux\n",
                        encoding="utf-8")

    seen = []

    class FakeYOLO:
        def __init__(self, weights):
            pass

        def predict(self, img, conf=0.25, device="cpu", verbose=False):
            seen.append(img)
            return [_Res([_Box(320.0, 320.0, 0.9)])]  # tire au centre de la fenêtre

    monkeypatch.setattr(eval_points, "YOLO", FakeYOLO)
    monkeypatch.setattr(sys, "argv", [
        "eval_points.py", "--weights", "x.pt", "--verdicts", str(verdicts),
        "--conf", "0.55", "--nir", "--cache", str(cache)])
    eval_points.main()

    # --nir a composé : le bleu de l'image vue par le modèle vient de l'IRC (200)
    assert seen and int(seen[0][:, :, 0].mean()) > 150
    # le point faux est encore détecté (le stub tire) -> 0 supprimé sur 1
    out = capsys.readouterr().out
    assert "Faux positifs supprimés        : 0/1" in out
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_eval_points.py -q`
Expected: FAIL (`ModuleNotFoundError: eval_points`).

- [ ] **Step 3: Créer `scripts/eval_points.py`**

```python
"""Test terrain NIR — évalue un modèle YOLO aux points labellisés de verdicts.

Pour chaque point de verdicts.csv, assemble la fenêtre (RVB, ou [R,G,NIR] avec
--nir), lance le modèle, et note si le modèle « tire » à cet emplacement. Les
points tirés alimentent compare_to_verdicts (FP supprimés / vrais conservés).
Aucun upload, lecture/impression seulement.

Usage:
    python scripts/eval_points.py --weights runs/citernes/weights/best.pt \
        --verdicts verdicts.csv --conf 0.55
    python scripts/eval_points.py --weights runs/citernes_nir/weights/best.pt \
        --verdicts verdicts.csv --conf 0.55 --nir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from detection_ortho.dataset import assemble_window, compose_rgn, parse_verdicts
from detection_ortho.tiles import LAYER_IRC
from detection_ortho.infer import result_to_boxes, boxes_to_points, is_detected_near
from detection_ortho.compare import compare_to_verdicts


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--nir", action="store_true",
                    help="composer [R,G,NIR] (bleu remplacé par le NIR de l'IRC)")
    ap.add_argument("--radius", type=float, default=25.0)
    ap.add_argument("--window", type=int, default=640)
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--cache", type=Path, default=Path("tiles_cache/eval"))
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    verdicts = parse_verdicts(args.verdicts.read_text(encoding="utf-8").splitlines())
    print(f"{len(verdicts)} point(s) à évaluer "
          f"({'[R,G,NIR]' if args.nir else 'RVB'}, conf {args.conf}).")

    model = YOLO(str(args.weights))
    detections: list[dict] = []
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
        res = model.predict(img, conf=args.conf, device=args.device, verbose=False)
        pts = boxes_to_points(result_to_boxes(res[0].boxes), ogx, ogy, args.zoom)
        if is_detected_near(pts, lon, lat, args.radius):
            detections.append({"lon": lon, "lat": lat})
        print(f"  {i}/{len(verdicts)}", end="\r", file=sys.stderr, flush=True)

    r = compare_to_verdicts(detections, verdicts, args.radius)
    print("\n=== Évaluation aux points de verdict ===")
    print(f"  Points où le modèle tire       : {r['n_candidates_new']}")
    print(f"  Faux positifs supprimés        : {r['fp_suppressed']}/{r['fp_total']}")
    print(f"  Faux positifs encore détectés  : {r['fp_still_detected']}/{r['fp_total']}")
    print(f"  Vrais positifs conservés       : {r['tp_kept']}/{r['tp_total']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `.venv\Scripts\python -m pytest tests/test_eval_points.py -q`
Expected: 1 passed.

- [ ] **Step 5: Syntaxe + suite complète**

Run: `.venv\Scripts\python -c "import ast; ast.parse(open('scripts/eval_points.py').read())"` puis `.venv\Scripts\python -m pytest -q`
Expected: syntaxe OK ; tous les tests passent. **Ne pas lancer eval_points.py en réel** (réseau + poids).

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_points.py tests/test_eval_points.py
git commit -m "feat: eval_points — test terrain d'un modèle aux points de verdict (--nir)"
```

---

## Task 3: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Ajouter la section « Test terrain NIR » au `README.md`**

Ajouter à la fin de `README.md` :

```markdown
## Test terrain NIR (aux points de verdict)

Juger le NIR sur des données réelles labellisées (`verdicts.csv`) plutôt que sur
le mAP de test. On évalue chaque modèle aux 44 points connus et on compare
combien de faux positifs chacun supprime, à vrais conservés égaux.

    python scripts/eval_points.py --weights runs/citernes/weights/best.pt \
        --verdicts verdicts.csv --conf 0.55            # RVB (garde-fou ~20/32 FP, 12/12 vrais)
    python scripts/eval_points.py --weights runs/citernes_nir/weights/best.pt \
        --verdicts verdicts.csv --conf 0.55 --nir      # NIR

Le NIR est concluant s'il supprime strictement plus de faux positifs que le RVB
en conservant au moins autant de vrais. Sinon, le pivot natif se fera en 3
canaux RVB (perf seule).
```

- [ ] **Step 2: Lancer la suite complète**

Run: `.venv\Scripts\python -m pytest -q`
Expected: tous les tests passent.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: workflow test terrain NIR aux points de verdict"
```

---

## Self-Review (fait à la rédaction)

- **Couverture du spec :** helper `is_detected_near` (Task 1) ✓ ; `eval_points.py` avec `--nir`, réutilisant assemble_window/compose_rgn/result_to_boxes/boxes_to_points/compare_to_verdicts (Task 2) ✓ ; protocole 2 runs + décision documentés (Task 3) ✓ ; runs réels différés ✓ ; aucun réseau/poids en test (stub + cache) ✓.
- **Placeholders :** aucun.
- **Cohérence des types :** `assemble_window(...)→(img,ogx,ogy)` ; `compose_rgn(rgb,irc)→BGR` (bleu=IRC[...,2]) ; `result_to_boxes(res[0].boxes)→[(cx,cy,score)]` ; `boxes_to_points(...,ogx,ogy,zoom)→[{lon,lat,score}]` ; `is_detected_near(pts,lon,lat,r)→bool` ; `compare_to_verdicts(detections,verdicts,r)→dict`. Libellés d'impression alignés sur le test (`"Faux positifs supprimés        : X/Y"`). Cache défaut `tiles_cache/eval` (sous `tiles_cache/` déjà gitignoré).
