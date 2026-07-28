# Jalon 3 — Inférence à l'échelle + comparaison OSM + MapRoulette : Design

**Date :** 2026-07-28
**Statut :** validé (brainstorming)
**Prérequis :** Jalons 0-2 livrés. Modèle entraîné : `runs/detect/runs/citernes/weights/best.pt`
(mAP@50 test = 0,83).

## Objectif

Exécuter le détecteur YOLO sur une **vraie zone géographique** (Tours Métropole
Val de Loire), comparer les détections à OpenStreetMap, et produire un **fichier
de challenge MapRoulette** listant les citernes candidates absentes d'OSM — pour
validation humaine.

## Portée

- **Zone : Tours Métropole Val de Loire** (~380 km², ~22 communes). Emprise =
  **polygone administratif récupéré d'OSM**, pas une simple bbox.
- **Dans le périmètre :** fenêtrage glissant + inférence YOLO sur la zone,
  post-traitement géospatial, comparaison OSM (3 catégories), génération du
  fichier MapRoulette.
- **Hors périmètre :** l'inférence départementale/nationale (repoussée), tout
  **upload/publication automatique vers MapRoulette** (contrainte explicite —
  voir ci-dessous), le ré-entraînement.

## Contrainte ferme — aucun upload automatique

Le code **génère seulement des fichiers**. Il ne fait **aucun appel à l'API
MapRoulette**, **aucune publication**, **aucun envoi** vers un service externe.
L'utilisateur charge lui-même le `.geojson` dans l'interface MapRoulette quand
il le décide.

## Décisions issues du brainstorming

- **Emprise = polygone admin OSM** : on n'infère que sur les fenêtres dont le
  centre tombe dans le polygone → pas de calcul ni de candidat hors métropole.
- **Fenêtrage glissant 640 px** (comme à l'entraînement) avec **chevauchement
  ~20-25 %** (évite de couper une citerne en bord de fenêtre).
- **Dédoublonnage par proximité** des détections répétées entre fenêtres
  chevauchantes (réutilise `geo.dedup_points`, rayon ~10 m).
- **Seuil de confiance : 0,4 par défaut** (option `--conf`). On privilégie le
  rappel (un humain valide dans MapRoulette) ; on ajuste après le premier run
  selon le volume de candidats.
- **La comparaison OSM sur Tours Métropole EST l'évaluation honnête** (données
  géographiques réelles complètes). **Pas de split spatial séparé** (YAGNI).
- Une détection « Détecté \ OSM » est **soit** un faux positif du modèle,
  **soit** une vraie citerne manquante — c'est MapRoulette (humain) qui tranche.
  Le code ne présuppose rien.

## Architecture — pipeline

```
[A] Emprise Tours Métropole   ──►  [B] Fenêtrage + inférence YOLO  ──►  [C] Post-traitement géo
    (polygone OSM, Overpass)          (best.pt, conf=0.4)               (boîtes→lon/lat, dédup)
                                                                              │
[F] Export MapRoulette  ◄──  [E] Comparaison OSM  ◄───────────────────────────┘
    (GeoJSON tâches,             (match_detections : 3 catégories)      [D] detections.geojson
     génération fichier seule)
```

## Composants

### A. Récupération de l'emprise (`detection_ortho/osm.py`, extension)

- `fetch_boundary_polygon(name, session=None) -> list[dict]` : via Overpass,
  récupère la relation administrative de `name` (« Tours Métropole Val de
  Loire ») avec géométrie, retourne le(s) anneau(x) de coordonnées `{lon,lat}`.
  (Tags exacts confirmés à l'implémentation — cf. démarche Jalon 0.)
- Le polygone est converti en `shapely.geometry.Polygon` côté logique inférence.

### B. Fenêtrage & inférence (`detection_ortho/infer.py`, nouveau)

Fonctions **pures (TDD)** :
- `windows_over_polygon(polygon, zoom, window_px, overlap) -> list[tuple[float,float]]`
  : centres `(lon, lat)` des fenêtres couvrant la bbox du polygone (pas =
  `window_px*(1-overlap)` en pixels), **filtrées** pour ne garder que les centres
  **dans le polygone** (shapely `contains`).
- `global_px_to_lonlat(gx, gy, zoom, tile_size=256) -> (lon, lat)` : inverse de
  `dataset.lonlat_to_global_px` (via `tiles.pixel_to_lonlat`).

Orchestration **I/O** (`run_inference`) : pour chaque centre → `assemble_window`
(depuis le cache, tuiles pré-téléchargées en parallèle comme au Jalon 2) →
`model.predict(image, conf)` → pour chaque boîte, centre pixel → global px →
`global_px_to_lonlat` → `{lon, lat, score}`. Charge le modèle via
`ultralytics.YOLO(weights)`.

### C. Post-traitement géospatial (réutilise `geo.py`)

- `geo.dedup_points(detections, radius_m=10)` : fusionne les détections répétées
  entre fenêtres chevauchantes, garde le meilleur score. (Déjà écrit.)
- Sortie : **`detections.geojson`** (via `geojson_io.points_to_geojson`, déjà écrit).

### D. Comparaison OSM (réutilise `compare.py`)

- `compare.match_detections(detections, osm_points, radius_m)` (déjà écrit) →
  `{matched, detected_only, osm_only}`. `osm_points` = citernes OSM de la zone
  (via `osm.fetch_citernes` sur la bbox de la métropole, filtrées au polygone).
- Écrit `matched.geojson`, `detected_only.geojson`, `osm_only.geojson`.

### E. Export MapRoulette (`detection_ortho/maproulette.py`, nouveau)

- `to_maproulette_tasks(points, instruction) -> dict` (pur) : FeatureCollection
  GeoJSON, une Feature par candidat, avec propriété d'instruction. Format
  compatible import MapRoulette (GeoJSON de tâches).
- Écrit **`maproulette_challenge.geojson`**. **Aucun appel réseau.**

### F. Scripts d'orchestration

- `scripts/infer_area.py` : `--boundary "Tours Métropole Val de Loire"`
  `--weights ... --conf 0.4 --out inference_out`. Enchaîne A→E, écrit tous les
  GeoJSON + un **résumé chiffré** (nb détections, rappel réel ∩OSM/OSM total,
  nb candidats) + un **overlay** de vérification.
- `scripts/export_maproulette.py` : `detected_only.geojson` → fichier de tâches
  MapRoulette (peut aussi être fait en fin de `infer_area.py`).

## Livrables (dans `inference_out/`)

- `detections.geojson` — toutes les détections (points + score)
- `matched.geojson`, `detected_only.geojson`, `osm_only.geojson` — 3 catégories
- `maproulette_challenge.geojson` — fichier de tâches (candidats)
- `overlay.png` — visualisation d'inspection
- résumé chiffré imprimé (console)

## Critère de succès

Le pipeline produit sur Tours Métropole un `maproulette_challenge.geojson`
exploitable, avec un **rappel réel (∩OSM) cohérent avec le Jalon 2 (~80 %)** et
un **volume de candidats raisonnable** (ajustable via `--conf`).

## Stack technique (ajouts)

- `shapely` (déjà dans requirements) pour le point-dans-polygone.
- Réutilise : `ultralytics`, `detection_ortho.{tiles, dataset, geo, compare,
  geojson_io, osm}`, téléchargement parallèle du Jalon 2.

## Intégration avec l'existant

- Réutilise massivement : `assemble_window` / téléchargement parallèle (Jalon 2),
  `dedup_points` / `haversine_m` (Jalon 1), `match_detections` (Jalon 1),
  `points_to_geojson` / `write_geojson` (Jalon 1), `fetch_citernes` (Jalon 0).
- `global_px_to_lonlat` est l'inverse de `lonlat_to_global_px` (Jalon 2).

## Tests (hors-ligne, mockés)

- `windows_over_polygon` : couverture d'un polygone simple, filtrage
  point-dans-polygone (un carré → centres à l'intérieur seulement).
- `global_px_to_lonlat` : aller-retour avec `lonlat_to_global_px`.
- `dedup_points`, `match_detections` : déjà testés (Jalon 1).
- `to_maproulette_tasks` : structure GeoJSON + instruction présente.
- `fetch_boundary_polygon` : parsing d'une réponse Overpass mockée.
- Le **run réel** sur Tours Métropole (réseau IGN/OSM + inférence) est **différé**
  à une exécution manuelle.

## Risques & mitigations

- **Volume de faux positifs candidats** → seuil `--conf` ajustable ; MapRoulette
  filtre par l'humain ; on regarde le volume après le 1er run.
- **Récupération du polygone admin** (tags EPCI variables dans OSM) → démarche de
  reconnaissance à l'implémentation (comme les tags citernes au Jalon 0) ;
  repli possible sur une bbox si la relation est introuvable.
- **Coût de calcul** (~60-65k tuiles) → téléchargement parallèle (déjà écrit) +
  inférence en tâche de fond ; borné à la métropole (faisable en une soirée).
- **Citernes en bord de fenêtre** → chevauchement des fenêtres + dédoublonnage.
- **Fuite/optimisme** : la comparaison OSM réelle remplace le split spatial et
  donne une mesure terrain non biaisée.
