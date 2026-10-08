# Boîtes d'entraînement d'après les polygones OSM — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pour les vraies citernes issues des verdicts, dessiner la boîte d'entraînement d'après le polygone OSM voisin (quand il existe) au lieu du carré fixe de 13 m, de façon optionnelle (`--osm-geom`).

**Architecture:** Une fonction pure `match_osm_polygons` dans `dataset.py` apparie chaque point « vrai » au polygone OSM (way) le plus proche ; `build_dataset.py --osm-geom` interroge Overpass une fois par fichier de verdicts (emprise des points « vrai » plus une marge), attache la boîte trouvée au verdict, et la boîte devient l'étiquette de l'imagette (repli : carré de 13 m).

**Tech Stack:** Python 3.12, pytest, Overpass via `detection_ortho.osm.fetch_features_geom` (existant, via `fetch_retry` de build_dataset.py).

**Contexte mesuré (département 36) :** 343 des 346 éléments `emergency=water_tank` sont des polygones ; 154 des 162 « vrai » ont un polygone à moins de 25 m ; plus grand côté médian 13 m (extrêmes 5 et 34 m) ; l'IoU entre le carré fixe de 13 m et la boîte du polygone a pour médiane 0,67, et 25 % des étiquettes sont sous 0,5.

## Global Constraints

- Tests hors-ligne uniquement : aucun appel réseau dans pytest (Overpass est simulé en remplaçant `build_dataset.fetch_features_geom`).
- Comportement par défaut inchangé : sans `--osm-geom`, aucune requête Overpass supplémentaire et étiquettes identiques à avant.
- Seuls les éléments OSM de type `way` avec une géométrie d'au moins 3 sommets sont utilisés (un nœud n'a pas d'emprise). Rayon d'appariement par défaut : 15 m, mesuré entre le point de verdict et le centre de la boîte du polygone. Taille plausible : plus grand côté de la boîte entre 3 m et 40 m (sinon repli sur le carré de 13 m).
- Les verdicts « faux » ne sont jamais concernés. Aucune requête pour un fichier sans verdict « vrai ».
- `README.md` est entièrement en CRLF : tout ajout doit rester en CRLF (vérifier : `.venv\Scripts\python -c "d=open('README.md','rb').read(); assert d.count(b'\r\n')==d.count(b'\n'); print('CRLF ok')"`).
- Chaque message de commit se termine par la ligne `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

- Un élément OSM sans géométrie, avec moins de 3 sommets, ou de type `node` ne plante pas et n'est jamais choisi (Task 1).
- Deux polygones à portée : le plus proche l'emporte ; un polygone trop grand (bâtiment, réservoir) ou trop petit est ignoré, donc repli (Task 1).
- Fichier de verdicts sans « vrai » : aucune requête Overpass (Task 1).
- Sans `--osm-geom`, les étiquettes et le nombre d'appels à `fetch_features_geom` sont inchangés (Task 1).
- La boîte du polygone peut être décalée du point de verdict : l'imagette reste centrée sur le point et l'étiquette est bien placée d'après la boîte du polygone (Task 1).

---

### Task 1: Appariement aux polygones OSM et option `--osm-geom`

**Files:**
- Modify: `detection_ortho/dataset.py` (ajout de `match_osm_polygons` après `element_to_box`)
- Modify: `scripts/build_dataset.py` (import, `attach_osm_boxes`, options `--osm-geom` et `--osm-geom-m`, ingestion des verdicts)
- Modify: `README.md` (paragraphe en fin de la section « Adapter le modèle à l'ortho express 2026 », CRLF)
- Test: `tests/test_match_osm_polygons.py`, `tests/test_build_dataset_osm_geom.py`

**Interfaces:**
- Consumes: `polygon_bounds(geometry) -> (west, south, east, north)`, `haversine_m(lon1, lat1, lon2, lat2)` (`detection_ortho.geo`), `fetch_retry(selectors, w, s, e, n)` de build_dataset.py.
- Produces: `match_osm_polygons(points, elements, radius_m=15.0, min_side_m=3.0, max_side_m=40.0) -> list` — pour chaque point `{lon, lat}`, la boîte `(west, south, east, north)` du polygone retenu ou `None` ; `attach_osm_boxes(verdicts, radius_m, margin=0.02) -> tuple[int, int]` dans build_dataset.py ; options `--osm-geom`, `--osm-geom-m`.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_match_osm_polygons.py` :

```python
import math

from detection_ortho.dataset import match_osm_polygons

LON, LAT = 0.65, 47.33
M = 111320.0


def _rect_way(lon, lat, w_m, h_m, tags=None):
    dlat = h_m / 2 / M
    dlon = w_m / 2 / (M * math.cos(math.radians(lat)))
    pts = [(lon - dlon, lat - dlat), (lon + dlon, lat - dlat),
           (lon + dlon, lat + dlat), (lon - dlon, lat + dlat), (lon - dlon, lat - dlat)]
    return {"type": "way", "tags": tags or {},
            "geometry": [{"lon": x, "lat": y} for x, y in pts]}


def _bounds(el):
    xs = [p["lon"] for p in el["geometry"]]
    ys = [p["lat"] for p in el["geometry"]]
    return min(xs), min(ys), max(xs), max(ys)


def test_returns_bounds_of_the_close_polygon():
    way = _rect_way(LON, LAT, 20, 8)
    assert match_osm_polygons([{"lon": LON, "lat": LAT}], [way]) == [_bounds(way)]


def test_nearest_polygon_wins():
    near = _rect_way(LON + 0.00003, LAT, 12, 12)   # ~2 m
    far = _rect_way(LON + 0.00010, LAT, 12, 12)    # ~7 m
    out = match_osm_polygons([{"lon": LON, "lat": LAT}], [far, near])
    assert out == [_bounds(near)]


def test_node_and_malformed_elements_are_ignored():
    node = {"type": "node", "lon": LON, "lat": LAT, "tags": {}}
    no_geom = {"type": "way", "tags": {}}
    two_vertices = {"type": "way", "tags": {},
                    "geometry": [{"lon": LON, "lat": LAT}, {"lon": LON + 0.0001, "lat": LAT}]}
    assert match_osm_polygons([{"lon": LON, "lat": LAT}], [node, no_geom, two_vertices]) == [None]


def test_polygon_beyond_radius_is_ignored():
    way = _rect_way(LON + 0.0005, LAT, 12, 12)     # ~37 m
    assert match_osm_polygons([{"lon": LON, "lat": LAT}], [way]) == [None]


def test_implausible_sizes_fall_back_to_none():
    too_big = _rect_way(LON, LAT, 60, 30)
    too_small = _rect_way(LON, LAT, 2, 1.5)
    assert match_osm_polygons([{"lon": LON, "lat": LAT}], [too_big]) == [None]
    assert match_osm_polygons([{"lon": LON, "lat": LAT}], [too_small]) == [None]


def test_empty_inputs():
    assert match_osm_polygons([], []) == []
    assert match_osm_polygons([{"lon": LON, "lat": LAT}], []) == [None]


def test_one_result_per_point_in_order():
    a = _rect_way(LON, LAT, 12, 12)
    pts = [{"lon": LON, "lat": LAT}, {"lon": LON + 0.01, "lat": LAT}]
    out = match_osm_polygons(pts, [a])
    assert out == [_bounds(a), None]
```

Créer `tests/test_build_dataset_osm_geom.py` :

```python
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import build_dataset  # noqa: E402
from detection_ortho.dataset import window_tiles  # noqa: E402

M = 111320.0
A = (0.65, 47.33)      # a un polygone OSM de 24 m x 8 m
B = (0.66, 47.34)      # aucun polygone OSM


def _seed(cache, lon, lat):
    cache.mkdir(parents=True, exist_ok=True)
    tiles, _, _ = window_tiles(lon, lat, 19, 640)
    for x, y in tiles:
        cv2.imwrite(str(cache / f"19_{x}_{y}.jpg"), np.full((256, 256, 3), 128, np.uint8))


def _way(lon, lat, w_m, h_m):
    dlat = h_m / 2 / M
    dlon = w_m / 2 / (M * math.cos(math.radians(lat)))
    pts = [(lon - dlon, lat - dlat), (lon + dlon, lat - dlat), (lon + dlon, lat + dlat),
           (lon - dlon, lat + dlat), (lon - dlon, lat - dlat)]
    return {"type": "way", "tags": {}, "geometry": [{"lon": x, "lat": y} for x, y in pts]}


def _label(out, stem):
    p = next((out / "labels").rglob(f"{stem}.txt"))
    _cls, _cx, _cy, w, h = p.read_text().split()
    return float(w), float(h)


def _run(tmp_path, monkeypatch, osm_geom):
    calls = []

    def fake_fetch(selectors, w, s, e, n):
        calls.append((selectors, w, s, e, n))
        if selectors[0][0] == "leisure":
            return []
        if (w, s, e, n) == (0.6, 47.3, 0.7, 47.4):     # l'emprise --bbox : aucun positif OSM
            return []
        return [_way(A[0], A[1], 24, 8)]                 # requête --osm-geom du fichier de verdicts

    monkeypatch.setattr(build_dataset, "fetch_features_geom", fake_fetch)
    out = tmp_path / "ds"
    for pt in (A, B, (B[0], B[1] + 0.01)):       # les trois points créent une imagette : tuiles préremplies
        _seed(out / "tiles_cache", *pt)
    verdicts = tmp_path / "v.csv"
    verdicts.write_text("index,lat,lon,score,verdict\n"
                        f"1,{A[1]},{A[0]},0.9,vrai\n"
                        f"2,{B[1]},{B[0]},0.9,vrai\n"
                        f"3,{B[1] + 0.01},{B[0]},0.4,faux\n", encoding="utf-8")
    argv = ["build_dataset.py", "--bbox", "0.6", "47.3", "0.7", "47.4", "--negatives", "0",
            "--max-pools", "0", "--verdicts", str(verdicts), "--out", str(out)]
    if osm_geom:
        argv.append("--osm-geom")
    monkeypatch.setattr(sys, "argv", argv)
    build_dataset.main()
    return out, calls


def test_osm_geom_uses_polygon_box_and_falls_back(tmp_path, monkeypatch, capsys):
    out, calls = _run(tmp_path, monkeypatch, osm_geom=True)
    wa, ha = _label(out, "revpos_0000")     # polygone 24 m x 8 m (~0,2026 m/px, fenêtre 640 px)
    assert abs(wa - 24 / 0.2026 / 640) < 0.02 and abs(ha - 8 / 0.2026 / 640) < 0.02
    wb, hb = _label(out, "revpos_0001")     # repli : carré de 13 m
    assert abs(wb - 13 / 0.2026 / 640) < 0.02 and abs(hb - 13 / 0.2026 / 640) < 0.02
    summary = capsys.readouterr().out
    assert "1 polygone(s)" in summary and "1 repli(s)" in summary
    # une seule requête OSM supplémentaire (un fichier de verdicts) hors --bbox et piscines
    extra = [c for c in calls if c[0][0][0] == "emergency" and (c[1], c[2], c[3], c[4]) != (0.6, 47.3, 0.7, 47.4)]
    assert len(extra) == 1


def test_default_is_unchanged_without_osm_geom(tmp_path, monkeypatch):
    out, calls = _run(tmp_path, monkeypatch, osm_geom=False)
    for stem in ("revpos_0000", "revpos_0001"):
        w, h = _label(out, stem)
        assert abs(w - 13 / 0.2026 / 640) < 0.02 and abs(h - 13 / 0.2026 / 640) < 0.02
    assert len(calls) == 2           # positifs de --bbox + piscines, rien d'autre


def test_no_overpass_call_for_a_file_without_vrai(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(selectors, w, s, e, n):
        calls.append((w, s, e, n))
        return []

    monkeypatch.setattr(build_dataset, "fetch_features_geom", fake_fetch)
    n_poly, n_fb = build_dataset.attach_osm_boxes(
        [{"lon": 0.65, "lat": 47.33, "verdict": "faux"}], 15.0)
    assert (n_poly, n_fb) == (0, 0) and calls == []
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_match_osm_polygons.py tests/test_build_dataset_osm_geom.py -q`
Expected: FAIL (`ImportError: cannot import name 'match_osm_polygons'`).

- [ ] **Step 3: Implémenter `match_osm_polygons`**

Dans `detection_ortho/dataset.py`, ajouter juste après la fonction `element_to_box` (et ajouter `from detection_ortho.geo import haversine_m` s'il n'est pas déjà importé : il l'est depuis la tâche de dédoublonnage) :

```python
def match_osm_polygons(
    points, elements, radius_m: float = 15.0,
    min_side_m: float = 3.0, max_side_m: float = 40.0,
) -> list:
    """Pour chaque point {lon, lat}, boîte géo (west, south, east, north) du polygone OSM le plus proche.

    Seuls les éléments `way` ayant au moins 3 sommets sont candidats. Le polygone
    retenu est celui dont le centre de boîte est le plus proche du point, à
    `radius_m` au plus, et dont le plus grand côté mesure entre `min_side_m` et
    `max_side_m` (sinon None : le point garde le carré fixe). Une entrée de
    sortie par point, dans l'ordre.
    """
    cands = []
    for el in elements:
        geom = el.get("geometry") if el.get("type") == "way" else None
        if not geom or len(geom) < 3:
            continue
        w, s, e, n = polygon_bounds(geom)
        lat_c = (s + n) / 2
        side = max((e - w) * _M_PER_DEG_LAT * math.cos(math.radians(lat_c)),
                   (n - s) * _M_PER_DEG_LAT)
        if not (min_side_m <= side <= max_side_m):
            continue
        cands.append(((w + e) / 2, lat_c, (w, s, e, n)))
    out = []
    for p in points:
        best, best_d = None, radius_m
        for lon_c, lat_c, box in cands:
            d = haversine_m(p["lon"], p["lat"], lon_c, lat_c)
            if d <= best_d:
                best, best_d = box, d
        out.append(best)
    return out
```

- [ ] **Step 4: Implémenter `--osm-geom` dans `scripts/build_dataset.py`**

1. Dans l'import depuis `detection_ortho.dataset`, ajouter `match_osm_polygons` à la liste.
2. Ajouter, au niveau module, juste après la fonction `fetch_retry` :
```python
def attach_osm_boxes(verdicts, radius_m, margin=0.02):
    """Ajoute `bbox_geo` aux verdicts `vrai` ayant un polygone OSM proche.

    Une requête Overpass par appel (emprise des points « vrai » + marge) ;
    aucune requête s'il n'y a pas de « vrai ». Retourne (polygones retenus, replis).
    """
    vrai = [v for v in verdicts if v["verdict"] == "vrai"]
    if not vrai:
        return 0, 0
    west = min(v["lon"] for v in vrai) - margin
    east = max(v["lon"] for v in vrai) + margin
    south = min(v["lat"] for v in vrai) - margin
    north = max(v["lat"] for v in vrai) + margin
    elements = fetch_retry([("emergency", "water_tank")], west, south, east, north)
    n_poly = 0
    for v, box in zip(vrai, match_osm_polygons(vrai, elements, radius_m)):
        if box is not None:
            v["bbox_geo"] = box
            n_poly += 1
    return n_poly, len(vrai) - n_poly
```
3. Après l'argument `--dedup-m`, ajouter :
```python
    ap.add_argument("--osm-geom", action="store_true",
                    help="boîte des vrais d'après le polygone OSM voisin (repli : carré fixe)")
    ap.add_argument("--osm-geom-m", type=float, default=15.0,
                    help="rayon (m) d'appariement d'un vrai à un polygone OSM")
```
4. Remplacer le bloc d'ingestion des verdicts (de `vs = []` jusqu'à l'impression `Verdicts ingérés ...` incluse) par :
```python
        vs = []
        n_poly = n_fallback = 0
        for path in args.verdicts:
            file_vs = parse_verdicts(path.read_text(encoding="utf-8").splitlines())
            if args.osm_geom:
                n_p, n_f = attach_osm_boxes(file_vs, args.osm_geom_m)
                n_poly += n_p
                n_fallback += n_f
            vs += file_vs
        vs, n_dup = dedup_verdicts(
            vs, [(b["lon"], b["lat"]) for b in boxes], args.dedup_m)
        n_hard = n_rev = 0
        for v in vs:
            if v["verdict"] == "faux":
                records.append((f"hardneg_{n_hard:04d}", v["lon"], v["lat"], None))
                n_hard += 1
            else:  # vrai
                bbox = v.get("bbox_geo") or fixed_box_geo(v["lon"], v["lat"], DEFAULT_BOX_M)
                records.append((f"revpos_{n_rev:04d}", v["lon"], v["lat"], bbox))
                n_rev += 1
        print(f"Verdicts ingérés : {n_hard} négatif(s) dur(s), {n_rev} positif(s), "
              f"{n_dup} doublon(s) écarté(s).")
        if args.osm_geom:
            print(f"Géométrie OSM : {n_poly} polygone(s) retenu(s), {n_fallback} repli(s) "
                  f"sur le carré de {DEFAULT_BOX_M:g} m.")
```

- [ ] **Step 5: Vérifier la réussite (nouveaux tests + non-régression)**

Run: `.venv\Scripts\python -m pytest tests/test_match_osm_polygons.py tests/test_build_dataset_osm_geom.py tests/test_build_dataset_layers.py tests/test_build_dataset_integration.py tests/test_build_dataset_verdicts.py tests/test_build_dataset_multi_verdicts.py tests/test_build_dataset_nir.py -q`
Expected: PASS.

- [ ] **Step 6: Runbook dans le README**

À la fin de la section « Adapter le modèle à l'ortho express 2026 » (donc à la fin du fichier), ajouter, en CRLF :

````markdown

### Boîtes d'après les polygones OSM (recommandé avant un réentraînement)

Par défaut, une vraie citerne issue des verdicts reçoit un carré fixe de 13 m. Dans le 36, un quart de
ces carrés recouvre moins de la moitié du vrai contour (IoU < 0,5 avec le polygone OSM). `--osm-geom`
dessine plutôt la boîte du polygone OSM voisin (à moins de `--osm-geom-m`, 15 m par défaut ; taille entre
3 et 40 m ; repli sur le carré de 13 m sinon). Il fait une requête Overpass par fichier de verdicts.

    Copy-Item -Recurse dataset_mr2026/tiles_cache dataset_mr2026_geom/tiles_cache
    python scripts/build_dataset.py --bbox 0.05 46.72 1.06 47.72 `
        --layers ORTHOIMAGERY.ORTHOPHOTOS ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026 `
        --osm-geom `
        --verdicts verdicts_maproulette/verdicts_18.csv `
                   verdicts_maproulette/verdicts_28.csv `
                   verdicts_maproulette/verdicts_37.csv `
                   verdicts_maproulette/verdicts_41.csv `
                   verdicts_maproulette/verdicts_44.csv `
                   verdicts_maproulette/verdicts_45.csv `
        --holdout verdicts_maproulette/verdicts_36.csv `
                  verdicts_maproulette/verdicts_49.csv `
        --spatial-split --out dataset_mr2026_geom

La première commande réutilise les tuiles déjà téléchargées (rien à retélécharger). Le résumé
« Géométrie OSM : N polygone(s) retenu(s), M repli(s) » indique la part de boîtes passées au contour réel.
Pour entraîner, repartir des poids déjà adaptés à la 2026 (≈ 15 époques) :

    python scripts/train.py --data dataset_mr2026_geom/data.yaml `
        --model runs/citernes_2026/weights/best.pt --epochs 15 --device cpu --name citernes_2026_geom
````

Puis vérifier le CRLF (commande dans les contraintes globales).

- [ ] **Step 7: Suite complète puis commit**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS (tous les tests).

```bash
git add detection_ortho/dataset.py scripts/build_dataset.py README.md tests/test_match_osm_polygons.py tests/test_build_dataset_osm_geom.py
git commit -m "feat: build_dataset --osm-geom — boîtes des vrais d'après les polygones OSM

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```
