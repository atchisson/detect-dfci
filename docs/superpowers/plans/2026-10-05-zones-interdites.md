# Exclusion des zones interdites (ZIPTV / ZICAD) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** `infer_area.py` ne télécharge ni n'infère les fenêtres dont l'emprise intersecte une zone ZIPTV ou ZICAD.

**Architecture :** un module pur `detection_ortho/zones.py` (parseurs GeoJSON/KML, emprise de fenêtre, filtre, chargement avec cache) et un branchement de quelques lignes dans `scripts/infer_area.py`, juste après `windows_over_polygon`. Le filtre porte sur la liste `centers`, donc il précède le point de reprise, le WMTS et l'inférence.

**Tech Stack :** Python, shapely (déjà utilisé), `xml.etree.ElementTree` (stdlib), requests, pytest.

**Spec :** `docs/superpowers/specs/2026-10-05-zones-interdites-design.md`

## Global Constraints

- Aucune nouvelle dépendance (KML lu avec la stdlib).
- Fichiers de zones téléchargés à l'exécution dans `data/zones/`, **jamais versionnés** (déjà dans `.gitignore`) : la licence de ZICAD n'est pas précisée.
- ZIPTV : `https://static.data.gouv.fr/resources/zones-interdites-a-la-prise-de-vue-aerienne/20181007-134434/2017-10.geojson` ; ZICAD : `https://data.geopf.fr/annexes/ressources/documentation/Arrete_ZICAD_10-2024.kml`.
- Règle : une fenêtre est écartée dès que son **emprise** (`WINDOW` px à `ZOOM`) intersecte l'union des deux jeux.
- Option `--skip-restricted-zones / --no-skip-restricted-zones` (`BooleanOptionalAction`, **actif par défaut**) et `--refresh-zones`.
- Réseau KO et cache absent : `RuntimeError` explicite, jamais « aucune zone » en silence.
- Le code et les tests suivent le style du dépôt : commentaires et messages en français, tests dans `tests/`, lancés avec `python -m pytest`.

## Review Focus

- KML dont un `Placemark` contient un `MultiGeometry` de plusieurs `Polygon` : tous doivent être lus (Task 1).
- Coordonnées KML avec altitude, retours à la ligne et espaces multiples : lues sans erreur (Task 1).
- Entités GeoJSON `geometry: null` ou `Point` : ignorées sans lever (Task 1).
- Polygone invalide (auto-intersection) : réparé, pas de crash dans l'union (Task 1).
- Toutes les fenêtres écartées, ou liste de centres vide : sortie propre, pas de traceback (Tasks 1 et 2).

---

### Task 1 : module `zones.py`

**Files :**
- Create: `detection_ortho/zones.py`
- Test: `tests/test_zones.py`

**Interfaces :**
- Consumes: `lonlat_to_global_px(lon, lat, zoom, tile_size=256) -> (gx, gy)` et `global_px_to_lonlat(gx, gy, zoom, tile_size=256) -> (lon, lat)` de `detection_ortho.dataset`.
- Produces:
  - `parse_geojson_zones(data: dict) -> list[Polygon]`
  - `parse_kml_zones(data: bytes | str) -> list[Polygon]`
  - `window_footprint(lon, lat, zoom, window_px) -> Polygon` (WGS84, rectangle aligné sur lon/lat)
  - `filter_windows(centers, zones, zoom, window_px) -> tuple[list, int]`
  - `load_zones(cache_dir: Path, refresh=False, session=None) -> BaseGeometry`
  - `apply_zone_filter(centers, cache_dir, zoom, window_px, refresh=False, session=None) -> tuple[list, int]`
  - constantes `ZIPTV_URL`, `ZICAD_URL`, `ZIPTV_FILE = "ziptv.geojson"`, `ZICAD_FILE = "zicad.kml"`

- [ ] **Step 1 : écrire les tests qui échouent**

Créer `tests/test_zones.py` :

```python
import pytest
import requests
from shapely.geometry import Point, Polygon, box

from detection_ortho import zones
from detection_ortho.zones import (
    ZICAD_FILE, ZICAD_URL, ZIPTV_FILE, ZIPTV_URL, apply_zone_filter,
    filter_windows, load_zones, parse_geojson_zones, parse_kml_zones,
    window_footprint,
)

ZOOM, WINDOW = 19, 640

GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {}, "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}},
        {"type": "Feature", "properties": {}, "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [[[2, 2], [2, 3], [3, 3], [3, 2], [2, 2]]],
                [[[4, 4], [4, 5], [5, 5], [5, 4], [4, 4]]]]}},
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Point", "coordinates": [9, 9]}},
        {"type": "Feature", "properties": {}, "geometry": None},
    ],
}

KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<Placemark><MultiGeometry>
<Polygon><outerBoundaryIs><LinearRing><coordinates>
   0,0,0   0,1,0
   1,1,0
   1,0,0 0,0,0
</coordinates></LinearRing></outerBoundaryIs></Polygon>
<Polygon><outerBoundaryIs><LinearRing><coordinates>2,2 2,3 3,3 3,2 2,2</coordinates></LinearRing></outerBoundaryIs></Polygon>
</MultiGeometry></Placemark>
<Placemark><Point><coordinates>9,9,0</coordinates></Point></Placemark>
</Document></kml>"""


# --- parseurs -------------------------------------------------------------

def test_parse_geojson_polygon_multipolygon_et_ignore_point_et_null():
    polys = parse_geojson_zones(GEOJSON)
    assert len(polys) == 3  # 1 Polygon + 2 du MultiPolygon, Point/null ignorés
    assert all(p.area == pytest.approx(1.0) for p in polys)


def test_parse_geojson_sans_features():
    assert parse_geojson_zones({"type": "FeatureCollection"}) == []


def test_parse_kml_multigeometry_altitude_et_retours_a_la_ligne():
    polys = parse_kml_zones(KML)
    assert len(polys) == 2  # les deux Polygon du MultiGeometry, Point ignoré
    assert all(p.area == pytest.approx(1.0) for p in polys)


def test_parse_kml_accepte_les_octets():
    assert len(parse_kml_zones(KML.encode("utf-8"))) == 2


# --- emprise et filtre ------------------------------------------------------

def test_window_footprint_taille_et_centrage():
    fp = window_footprint(2.0, 47.0, ZOOM, WINDOW)
    west, south, east, north = fp.bounds
    largeur = east - west
    assert largeur == pytest.approx(640 * 360 / (256 * 2**19), rel=1e-4)
    assert (west + east) / 2 == pytest.approx(2.0, abs=1e-6)
    assert (south + north) / 2 == pytest.approx(47.0, abs=1e-4)
    assert fp.contains(Point(2.0, 47.0))


def test_filter_windows_dedans_bord_et_loin():
    zone = box(0, 0, 1, 1)
    centres = [(0.5, 0.5),      # dedans
               (1.0005, 0.5),   # emprise qui déborde sur le bord
               (1.01, 0.5)]     # loin
    gardees, n = filter_windows(centres, zone, ZOOM, WINDOW)
    assert gardees == [(1.01, 0.5)]
    assert n == 2


def test_filter_windows_zones_vides_ou_pas_de_centres():
    assert filter_windows([(0.5, 0.5)], Polygon(), ZOOM, WINDOW) == ([(0.5, 0.5)], 0)
    assert filter_windows([], box(0, 0, 1, 1), ZOOM, WINDOW) == ([], 0)


def test_filter_windows_toutes_ecartees():
    gardees, n = filter_windows([(0.5, 0.5), (0.6, 0.6)], box(0, 0, 1, 1),
                                ZOOM, WINDOW)
    assert gardees == [] and n == 2


# --- chargement et cache ------------------------------------------------------

class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.appels = []

    def get(self, url, timeout=None):
        self.appels.append(url)
        return FakeResponse(self.routes[url])


class DeadSession:
    def get(self, url, timeout=None):
        raise requests.ConnectionError("réseau coupé")


def _routes():
    import json
    return {ZIPTV_URL: json.dumps(GEOJSON).encode("utf-8"),
            ZICAD_URL: KML.encode("utf-8")}


def test_load_zones_telecharge_met_en_cache_et_fusionne(tmp_path):
    s = FakeSession(_routes())
    z = load_zones(tmp_path, session=s)
    assert (tmp_path / ZIPTV_FILE).exists() and (tmp_path / ZICAD_FILE).exists()
    assert len(s.appels) == 2
    # 3 polygones ZIPTV + 2 ZICAD ; (0-1) et (2-3) en double, fusionnés
    assert z.area == pytest.approx(3.0)  # carrés (0-1), (2-3), (4-5) ; doublons fusionnés


def test_load_zones_relit_le_cache_sans_reseau(tmp_path):
    load_zones(tmp_path, session=FakeSession(_routes()))
    z = load_zones(tmp_path, session=DeadSession())
    assert z.area == pytest.approx(3.0)


def test_load_zones_refresh_retelecharge(tmp_path):
    load_zones(tmp_path, session=FakeSession(_routes()))
    s = FakeSession(_routes())
    load_zones(tmp_path, refresh=True, session=s)
    assert len(s.appels) == 2


def test_load_zones_refresh_en_echec_retombe_sur_le_cache(tmp_path, capsys):
    load_zones(tmp_path, session=FakeSession(_routes()))
    z = load_zones(tmp_path, refresh=True, session=DeadSession())
    assert z.area == pytest.approx(3.0)
    assert "cache" in capsys.readouterr().err


def test_load_zones_reseau_ko_sans_cache_leve(tmp_path):
    with pytest.raises(RuntimeError, match="zones"):
        load_zones(tmp_path, session=DeadSession())


def test_load_zones_repare_un_polygone_invalide(tmp_path):
    import json
    noeud = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {}, "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}}]}
    routes = {ZIPTV_URL: json.dumps(noeud).encode("utf-8"), ZICAD_URL: KML.encode("utf-8")}
    z = load_zones(tmp_path, session=FakeSession(routes))  # ne lève pas
    assert z.is_valid


def test_apply_zone_filter_de_bout_en_bout(tmp_path):
    s = FakeSession(_routes())
    gardees, n = apply_zone_filter([(0.5, 0.5), (50.0, 10.0)], tmp_path,
                                   ZOOM, WINDOW, session=s)
    assert gardees == [(50.0, 10.0)] and n == 1


def test_apply_zone_filter_liste_vide(tmp_path):
    assert apply_zone_filter([], tmp_path, ZOOM, WINDOW,
                             session=FakeSession(_routes())) == ([], 0)
```

- [ ] **Step 2 : vérifier l'échec**

Run: `python -m pytest tests/test_zones.py -v`
Expected: erreur de collecte `ImportError: cannot import name ... from detection_ortho.zones` (le module n'existe pas).

- [ ] **Step 3 : implémenter `detection_ortho/zones.py`**

```python
"""Zones interdites à la prise de vue aérienne (ZIPTV) et à la captation
aérienne de données (ZICAD).

L'ortho IGN floute ces sites : toute détection y est un faux positif. On écarte
donc les fenêtres d'inférence qui touchent ces zones, avant téléchargement.

Les fichiers sont téléchargés à l'exécution et gardés en cache (non versionnés :
la licence de ZICAD n'est pas précisée).
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from detection_ortho.dataset import global_px_to_lonlat, lonlat_to_global_px

ZIPTV_URL = ("https://static.data.gouv.fr/resources/"
             "zones-interdites-a-la-prise-de-vue-aerienne/20181007-134434/"
             "2017-10.geojson")
ZICAD_URL = ("https://data.geopf.fr/annexes/ressources/documentation/"
             "Arrete_ZICAD_10-2024.kml")
ZIPTV_FILE = "ziptv.geojson"
ZICAD_FILE = "zicad.kml"


def _polygons(geom) -> list[Polygon]:
    """Polygones contenus dans une géométrie (points et lignes ignorés)."""
    if isinstance(geom, Polygon):
        return [geom]
    return [p for g in getattr(geom, "geoms", []) for p in _polygons(g)]


def parse_geojson_zones(data: dict) -> list[Polygon]:
    out: list[Polygon] = []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if geom:
            out.extend(_polygons(shape(geom)))
    return out


def _local(tag: str) -> str:
    """Nom de balise sans espace de noms XML."""
    return tag.rsplit("}", 1)[-1]


def _parse_coords(text: str) -> list[tuple[float, float]]:
    pts = []
    for tok in text.split():
        parts = tok.split(",")  # lon,lat[,alt] — sans espace dans un triplet
        pts.append((float(parts[0]), float(parts[1])))
    return pts


def parse_kml_zones(data: bytes | str) -> list[Polygon]:
    """Anneaux extérieurs de tous les <Polygon> d'un KML (trous ignorés : on
    préfère écarter trop que pas assez)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    root = ET.fromstring(data)
    out: list[Polygon] = []
    for poly in root.iter():
        if _local(poly.tag) != "Polygon":
            continue
        for outer in poly:
            if _local(outer.tag) != "outerBoundaryIs":
                continue
            for el in outer.iter():
                if _local(el.tag) == "coordinates":
                    pts = _parse_coords(el.text or "")
                    if len(pts) >= 3:
                        out.append(Polygon(pts))
    return out


def window_footprint(lon: float, lat: float, zoom: int, window_px: int) -> Polygon:
    """Emprise WGS84 d'une fenêtre window_px centrée sur (lon, lat).

    Même géométrie que windows_over_polygon / window_tiles : carré en pixels
    Web Mercator, donc rectangle aligné sur lon/lat.
    """
    gx, gy = lonlat_to_global_px(lon, lat, zoom)
    half = window_px / 2
    west, north = global_px_to_lonlat(gx - half, gy - half, zoom)
    east, south = global_px_to_lonlat(gx + half, gy + half, zoom)
    return box(west, south, east, north)


def filter_windows(centers, zones, zoom: int, window_px: int):
    """(centres gardés, nombre écartés) : écarte les fenêtres qui touchent `zones`."""
    if zones.is_empty or not centers:
        return list(centers), 0
    prepared = prep(zones)
    gardees = [c for c in centers
               if not prepared.intersects(window_footprint(c[0], c[1], zoom, window_px))]
    return gardees, len(centers) - len(gardees)


def _fetch(url: str, path: Path, refresh: bool, session) -> bytes:
    """Contenu du fichier : cache si présent (sauf refresh), sinon téléchargement."""
    if path.exists() and not refresh:
        return path.read_bytes()
    try:
        resp = (session or requests).get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        if path.exists():
            print(f"  Téléchargement impossible ({exc}) : cache conservé "
                  f"{path}.", file=sys.stderr)
            return path.read_bytes()
        raise RuntimeError(
            f"Impossible de télécharger les zones interdites ({url}) et aucun "
            f"cache dans {path.parent} : {exc}. Relancez avec "
            f"--no-skip-restricted-zones pour inférer sans filtre.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return resp.content


def load_zones(cache_dir: Path, refresh: bool = False, session=None):
    """Union ZIPTV + ZICAD (géométrie shapely WGS84)."""
    cache_dir = Path(cache_dir)
    ziptv = _fetch(ZIPTV_URL, cache_dir / ZIPTV_FILE, refresh, session)
    zicad = _fetch(ZICAD_URL, cache_dir / ZICAD_FILE, refresh, session)
    import json
    polys = parse_geojson_zones(json.loads(ziptv)) + parse_kml_zones(zicad)
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    return unary_union(polys)


def apply_zone_filter(centers, cache_dir: Path, zoom: int, window_px: int,
                      refresh: bool = False, session=None):
    """Charge les zones puis filtre les centres : (gardés, nombre écartés)."""
    if not centers:
        return [], 0
    zones = load_zones(cache_dir, refresh=refresh, session=session)
    return filter_windows(centers, zones, zoom, window_px)
```

- [ ] **Step 4 : vérifier que les tests passent**

Run: `python -m pytest tests/test_zones.py -v`
Expected: tous PASS.

- [ ] **Step 5 : vérifier sur les vrais fichiers (une fois, à la main)**

Run :

```
python -c "from pathlib import Path; from detection_ortho import zones; import json; z=zones.load_zones(Path('data/zones')); g=zones.parse_geojson_zones(json.loads((Path('data/zones')/zones.ZIPTV_FILE).read_bytes())); k=zones.parse_kml_zones((Path('data/zones')/zones.ZICAD_FILE).read_bytes()); print('ZIPTV', len(g), 'ZICAD', len(k), 'union', z.geom_type, round(z.area,6))"
```

Expected : `ZIPTV` ≥ 250 polygones, `ZICAD` ≥ 1, union non vide. **Si ZIPTV donne 0 polygone ou très peu (zones décrites en points + rayon) ou si le KML a un format inattendu (ValueError dans `_parse_coords`) : s'arrêter et le signaler au lieu de contourner.**

- [ ] **Step 6 : commit**

```bash
git add detection_ortho/zones.py tests/test_zones.py
git commit -m "feat: zones — chargement ZIPTV/ZICAD et filtrage des fenêtres d'inférence"
```

(terminer le message par la ligne `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` après une ligne vide)

---

### Task 2 : branchement dans `infer_area.py` + README + amendement de la spec

**Files :**
- Modify: `scripts/infer_area.py` (imports ; arguments ~l.166-173 ; après `windows_over_polygon` ~l.208-209)
- Modify: `README.md` (section ajoutée en fin de fichier)
- Modify: `docs/superpowers/specs/2026-10-05-zones-interdites-design.md` (paragraphe checkpoint)
- Test: `tests/test_infer_area_zones.py`

**Interfaces :**
- Consumes: `apply_zone_filter(centers, cache_dir, zoom, window_px, refresh=False, session=None) -> (list, int)` de `detection_ortho.zones` ; constantes `ZOOM`, `WINDOW` de `infer_area.py`.
- Produces: options CLI `--skip-restricted-zones` / `--no-skip-restricted-zones` (défaut actif) et `--refresh-zones`.

- [ ] **Step 1 : écrire le test qui échoue**

Créer `tests/test_infer_area_zones.py` :

```python
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_aide_expose_les_options_de_zones():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "infer_area.py"), "--help"],
        capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr
    assert "--skip-restricted-zones" in r.stdout
    assert "--no-skip-restricted-zones" in r.stdout
    assert "--refresh-zones" in r.stdout
```

- [ ] **Step 2 : vérifier l'échec**

Run: `python -m pytest tests/test_infer_area_zones.py -v`
Expected: FAIL (`--skip-restricted-zones` absent de l'aide).

- [ ] **Step 3 : implémenter**

Dans `scripts/infer_area.py` :

1. Ajouter l'import après `from detection_ortho.maproulette import to_maproulette_tasks` :

```python
from detection_ortho.zones import apply_zone_filter
```

2. Ajouter les arguments juste avant `args = ap.parse_args()` :

```python
    ap.add_argument("--skip-restricted-zones",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="ne pas inférer les fenêtres qui touchent une zone "
                         "interdite ZIPTV/ZICAD (floutées dans l'ortho) ; "
                         "--no-skip-restricted-zones pour désactiver")
    ap.add_argument("--refresh-zones", action="store_true",
                    help="retélécharger les fichiers ZIPTV/ZICAD au lieu de "
                         "relire le cache data/zones/")
```

3. Juste après `print(f"{len(centers)} fenêtre(s) d'inférence.")`, ajouter :

```python
    if args.skip_restricted_zones:
        zones_cache = Path(__file__).resolve().parent.parent / "data" / "zones"
        centers, n_ecartees = apply_zone_filter(
            centers, zones_cache, ZOOM, WINDOW, refresh=args.refresh_zones)
        print(f"Zones interdites (ZIPTV/ZICAD) : {n_ecartees} fenêtre(s) "
              f"écartée(s) sur {n_ecartees + len(centers)}.")
        if not centers:
            sys.exit("Toutes les fenêtres touchent une zone interdite : rien à "
                     "inférer.")
```

L'empreinte du point de reprise n'est **pas** modifiée : `centers` filtrés y entrent déjà, et ne pas ajouter de clé évite de rejeter inutilement un checkpoint en cours quand aucune zone ne touche l'emprise.

- [ ] **Step 4 : vérifier que le test passe**

Run: `python -m pytest tests/test_infer_area_zones.py tests/test_zones.py -v`
Expected: PASS.

- [ ] **Step 5 : amender la spec**

Dans `docs/superpowers/specs/2026-10-05-zones-interdites-design.md`, remplacer le paragraphe « L'empreinte du point de reprise reçoit `skip_zones` … » (section `scripts/infer_area.py`) par :

```
- L'empreinte du point de reprise n'est pas modifiée : `centers` filtrés y
  entrent déjà. Un checkpoint antérieur reste donc valide si aucune zone ne
  touche l'emprise, et est refusé (message existant, `--restart`) si le filtre
  change la grille.
```

- [ ] **Step 6 : documenter dans le README**

Ajouter en fin de `README.md` :

````markdown
## Zones interdites (ZIPTV / ZICAD)

L'ortho IGN floute les sites militaires et sensibles : toute détection y est un
faux positif. `scripts/infer_area.py` ne télécharge ni n'infère donc les
fenêtres dont l'emprise touche une zone **ZIPTV** (prise de vue aérienne, arrêté
de 2018) ou **ZICAD** (captation aérienne de données, version 10-2024). Le
filtre est actif par défaut et affiche son bilan avant l'inférence.

```powershell
python scripts/infer_area.py --boundary "Indre" --insee 36 --weights models/citernes-yolov8n.pt
python scripts/infer_area.py ... --refresh-zones              # retélécharger les zones
python scripts/infer_area.py ... --no-skip-restricted-zones   # désactiver le filtre
```

Les fichiers sont téléchargés au premier run dans `data/zones/` (non versionné :
la licence de ZICAD n'est pas précisée). Une fenêtre qui touche seulement le
bord d'une zone est écartée en entier : on perd les citernes voisines dans cette
fenêtre (fenêtres de 130-190 m de côté). Les anciens résultats ne sont pas
re-filtrés.
````

- [ ] **Step 7 : suite complète**

Run: `python -m pytest -q`
Expected: tout PASS (aucune régression).

- [ ] **Step 8 : commit**

```bash
git add scripts/infer_area.py tests/test_infer_area_zones.py README.md docs/superpowers/specs/2026-10-05-zones-interdites-design.md
git commit -m "feat: infer_area écarte les fenêtres en zones interdites ZIPTV/ZICAD"
```

(terminer le message par la ligne `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` après une ligne vide)

- [ ] **Step 9 (optionnel, à la main) : essai réel**

Lancer `infer_area.py` sur une commune connue pour contenir un site militaire et vérifier que la ligne « Zones interdites (ZIPTV/ZICAD) : N fenêtre(s) écartée(s) » affiche N > 0. Interrompre après le bilan (Ctrl+C) : le but est seulement de vérifier le filtre.
