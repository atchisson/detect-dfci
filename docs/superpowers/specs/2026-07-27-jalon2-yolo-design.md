# Jalon 2 — Détecteur YOLO de citernes : Design

**Date :** 2026-07-27
**Statut :** validé (brainstorming)
**Prérequis :** Jalons 0 & 1 livrés (package `detection_ortho/`, scripts recon/baseline).

## Objectif

**Prouver qu'un détecteur YOLO apprend à repérer les citernes souples** sur
l'ortho IGN : construire un jeu de données depuis OSM, entraîner sur Colab,
évaluer sur un jeu de test tenu à l'écart. On ne traite PAS l'inférence à
grande échelle ni MapRoulette dans ce jalon.

## Portée

- **Dans le périmètre :** génération automatique du dataset, entraînement
  YOLOv8n (transfer learning), évaluation (précision / rappel / mAP@50) +
  vérification visuelle.
- **Hors périmètre :** inférence à l'échelle départementale (millions de
  tuiles — problème repoussé au Jalon 3), génération MapRoulette, multi-classes,
  multi-départements.

## Décisions issues du brainstorming (fondées sur les données réelles)

- **Une seule classe `citerne`.** Vivier de positifs = **tous les
  `emergency=water_tank`** (~203 sur le dpt 37), PAS filtrés sur
  `water_tank:type=flexible` : ce sous-tag n'est renseigné que sur ~40 % des
  objets, alors que visuellement les `water_tank` non tagués sont en écrasante
  majorité les mêmes bâches souples (vérifié sur planches-contact). La curation
  des rares intrus se fait à l'œil au moment de la génération.
- **Annotation automatique.** Sur les 203 positifs, **190 sont des ways
  (polygones)** dans OSM (94 %) → la boîte englobante se dérive du polygone. Les
  13 points restants reçoivent une boîte de taille fixe (médiane observée
  ≈ 13 × 13 m). **Pas de session d'annotation manuelle.**
- **Négatifs difficiles inclus** : piscines (`leisure=swimming_pool`, le
  confondeur n°1 — même teinte turquoise) + tuiles de fond aléatoires (toits,
  routes, champs). Imagettes sans boîte.
- **Données : Indre-et-Loire (dpt 37) uniquement.** BBox de travail :
  `west=0.05, south=46.72, east=1.06, north=47.72`.
- **Modèle : YOLOv8n**, transfer learning depuis COCO, image 640, **entraîné
  en local sur CPU** (Ryzen 2600X) ; Colab GPU en option.
- **Taille des objets (mesurée) :** médiane 13 × 13 m, plage 6–62 m → ≈ 44 px
  de côté à zoom 19 (≈ 0,3 m/px). Détectable.

## Architecture — pipeline en étapes

```
[A] OSM géométries positifs  ┐
[B] OSM piscines + négatifs   ├─►  [C] Construction imagettes + labels  ─►  [D] Split train/val/test
    (emergency=water_tank      │        (fenêtres 640px, boîtes YOLO)          (dataset YOLO + data.yaml)
     via out geom;)            ┘                                                      │
                                                                                      ▼
[F] Évaluation (P/R/mAP@50   ◄─────────────  [E] Entraînement YOLOv8n (Colab)  ◄──────┘
    + mosaïque TP/FP/FN)                          (best.pt)
```

## Composants

### A. Récupération des positifs avec géométrie (`detection_ortho/osm.py`, extension)

- Nouvelle fonction `fetch_features_geom(west, south, east, north, selectors,
  session=None) -> list[dict]` utilisant **`out geom;`** : retourne les éléments
  avec, pour chaque objet, `type`, `tags`, et soit `lon/lat` (node) soit
  `geometry` (liste de `{lon,lat}` pour un way).
- Fonction dérivée `water_tank_boxes(...)` : à partir des `emergency=water_tank`,
  produit pour chaque objet un dict `{"lon","lat","bbox_geo","tags","source"}`
  où `bbox_geo = (west, south, east, north)` en degrés — dérivé du polygone
  (way) ou d'une boîte fixe de `DEFAULT_BOX_M` mètres centrée (node).
- La fonction existante `fetch_citernes` (centroïdes, `out center;`) est
  **conservée intacte** pour le pipeline de comparaison (Jalon 1/3).

### B. Récupération des négatifs (`detection_ortho/osm.py` + logique dataset)

- Piscines : `fetch_features_geom(..., selectors=[("leisure","swimming_pool")])`
  → mêmes boîtes géo (pour centrer les imagettes ; label vide).
- Fonds aléatoires : tuiles `(x, y)` tirées aléatoirement (graine fixe) dans la
  bbox du département — pas de requête OSM. Générés par la logique dataset.

### C. Construction imagettes + labels (`detection_ortho/dataset.py`, nouveau, pur + I/O)

Fonctions **pures et testables (TDD)** :
- `polygon_bounds(geometry: list[dict]) -> tuple[float,float,float,float]` :
  bbox géo (w,s,e,n) d'une liste de sommets `{lon,lat}`.
- `fixed_box_geo(lon, lat, size_m) -> tuple[...]` : bbox géo carrée centrée.
- `geo_bbox_to_pixel_bbox(bbox_geo, origin_x, origin_y, zoom, window_px) -> tuple`
  : convertit une bbox géo en pixels DANS une fenêtre, via `lonlat_to_pixel`.
- `to_yolo_label(px_bbox, window_px) -> str` : ligne YOLO
  `0 cx cy w h` (normalisée 0–1), clampée au cadre.

Fonctions I/O :
- `assemble_window(center_lon, center_lat, zoom, window_px, cache_dir,
  session=None) -> (np.ndarray, origin_x, origin_y)` : assemble une mosaïque de
  tuiles couvrant une fenêtre `window_px × window_px` centrée sur le point
  (gère le chevauchement de tuiles), retourne l'image + l'origine pixel absolue
  pour la conversion géo→pixel.
- `write_chip(image, label_lines, images_dir, labels_dir, name)` : écrit
  l'imagette `.jpg` + le `.txt` label (vide pour un négatif).

### D. Génération du dataset + split (`scripts/build_dataset.py`)

- Orchestration : positifs (A) + piscines (B) + N tuiles de fond aléatoires.
- Pour chaque objet : `assemble_window` → conversion boîte → label YOLO ;
  négatifs → label vide.
- **Mosaïque de QA** : image récapitulative avec boîtes dessinées, pour écarter
  visuellement les intrus (liste d'exclusion passée en option `--exclude`).
- **Split train/val/test = 70/15/15**, graine fixe (`random.Random(0)`).
  *Note : split aléatoire (pas spatial) accepté pour cette preuve ; un split
  géographique — anti-fuite — sera préférable au Jalon 3.*
- Sortie : arborescence YOLO
  `dataset/images/{train,val,test}/`, `dataset/labels/{train,val,test}/`,
  `dataset/data.yaml` (1 classe `citerne`).

### E. Entraînement (`scripts/train.py` principal ; Colab optionnel)

- **Chemin principal : entraînement LOCAL sur CPU** (Ryzen 2600X). Le matériel
  GPU disponible est écarté : Quadro P620 = 2 Go VRAM (trop juste, risque
  d'OOM + config CUDA) ; RX 5700 XT = AMD sous Windows (pas de CUDA/ROCm).
- **`scripts/train.py`** : wrapper Ultralytics YOLOv8n,
  `model.train(data="data.yaml", epochs=…, imgsz=640, pretrained=True,
  device=…)`. Option `--device` (défaut **`cpu`**) ; augmentation par défaut
  d'Ultralytics (flips, rotations, jitter HSV — pertinent vu la variabilité de
  couleur). Pour ~200 images sur un modèle nano, un run CPU est de l'ordre de
  1–3 h — acceptable en tâche de fond.
- **Colab reste une option** (`notebooks/train_yolo.ipynb`, T4 gratuit) pour qui
  veut de la vitesse, mais n'est PAS le chemin nominal.
- Sortie : `best.pt` rapatrié / produit en local (git-ignoré).

### F. Évaluation (`scripts/evaluate.py`)

- Charge `best.pt`, lance `model.val(split="test")` → **précision, rappel,
  mAP@50, mAP@50-95** sur le jeu de test tenu à l'écart.
- Génère une **mosaïque TP / FP / FN** sur le test (les prédictions vs les
  labels), pour juger à l'œil — notamment le rejet des piscines.

## Critère de succès

- **mAP@50 ≥ 0,6** sur le jeu de test **et** rejet visible des piscines/toits
  (précision nettement supérieure aux ~2 % de la baseline OpenCV).
- Si atteint : YOLO est validé → on écrira le plan du Jalon 3 (inférence à
  l'échelle + MapRoulette).

## Stack technique (ajouts au projet)

- `ultralytics` (YOLOv8) — ajouté à `requirements.txt` (installe PyTorch CPU).
- Réutilise : `detection_ortho.tiles` (téléchargement, `lonlat_to_pixel`),
  `numpy`, `opencv-python`, `requests`.
- Entraînement local CPU (Ryzen 2600X) ; Colab GPU en option.

## Intégration avec l'existant

- `assemble_window` s'appuie sur `download_tile` / `lonlat_to_pixel` /
  `tile_for_lonlat` (Jalon 0) — mêmes conventions (lon, lat), zoom 19, WMTS PM.
- `fetch_features_geom` vit dans `osm.py` à côté de `fetch_citernes` sans le
  modifier.
- Le `best.pt` et le module d'inférence seront consommés au Jalon 3 par un
  pipeline réutilisant `tiles`, `dedup_points`, `match_detections`,
  `geojson_io`.

## Risques & mitigations

- **Jeu petit (~190 boîtes)** → transfer learning + augmentation ; objet
  distinct ⇒ suffisant pour une preuve. Mesuré via mAP sur test.
- **Bruit de tags OSM** → on ignore le sous-tag, curation visuelle des intrus.
- **Confusion piscines** → négatifs difficiles explicites (piscines OSM).
- **Fuite train/test (split aléatoire, citernes voisines)** → acceptée pour la
  preuve ; split spatial noté pour le Jalon 3.
- **Overpass instable (504/406 déjà rencontrés)** → User-Agent (déjà en place)
  + logique de retry dans les scripts de récupération.
- **Boîtes axis-aligned depuis polygones tournés** → la bbox min/max est
  légèrement plus large que le réservoir tourné ; acceptable (YOLO = boîtes
  axis-aligned).
