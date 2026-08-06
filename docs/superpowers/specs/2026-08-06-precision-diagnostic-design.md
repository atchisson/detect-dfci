# Diagnostic de précision honnête + calibration de seuil : Design

**Date :** 2026-08-06
**Statut :** validé (brainstorming)
**Prérequis :** Modèles `runs/citernes/weights/best.pt` (RVB itér. 2). `verdicts.csv`
racine (12 vrais / 101 faux, 113 points labellisés). Cache RVB des points déjà
présent (`tiles_cache/eval`, run eval_points).

## Objectif

Attaquer la **précision terrain** (~27 % à conf 0,4 sur Tours), qui est la vraie
douleur du projet. Deux leviers automatiques : (1) un **split spatial** pour
remplacer le mAP optimiste (0,84, fuite géographique) par un vrai chiffre de
généralisation ; (2) un **balayage de seuil** sur données labellisées pour
verrouiller le point de fonctionnement (gain immédiat, 27 %→50 % déjà observé à
0,55). La catégorisation des faux positifs se fait avec l'outil existant
(`make_map.py`) — aucun code.

## Contexte

- `split_indices` (dataset.py) fait un 70/15/15 **aléatoire** → citernes voisines
  des deux côtés → test qui fuit → mAP 0,84 optimiste (précision terrain réelle
  11–27 %).
- On dispose de `verdicts.csv` (labels terrain) et des briques
  `assemble_window`/`compose_rgn`/`result_to_boxes`/`boxes_to_points`/
  `is_detected_near`.
- Décision NIR : abandonné (marginal) ; on reste **3 canaux RVB**. `--nir` reste
  dispo dans les outils par cohérence, non utilisé ici par défaut.

## Décisions issues du brainstorming

- **Split spatial par cellule de grille** : les imagettes d'une même cellule
  géographique vont dans le même lot → zones de test disjointes du train.
- **Balayage de seuil en une passe** : inférer une fois à conf basse, retenir le
  meilleur score près de chaque point, puis balayer le seuil **en mémoire**.
- **Catégorisation des FP** : via `make_map.py` (existant), hors périmètre code.

## Composants

### A. Split spatial (`detection_ortho/dataset.py` + `scripts/build_dataset.py`)

- `spatial_split_indices(points, cell_deg=0.05, seed=0, ratios=(0.7,0.15,0.15)) -> dict`
  (pur) : groupe les indices par cellule `(floor(lon/cell_deg), floor(lat/cell_deg))`,
  mélange les cellules de façon déterministe (seed), puis remplit
  train→val→test **par cellules entières** jusqu'à approcher les ratios (comptés
  en nombre d'imagettes). Une cellule n'est jamais scindée.
- `build_dataset.py` : options `--spatial-split` (flag) et `--cell-deg 0.05`.
  Quand actif, utilise `spatial_split_indices([(lon,lat) …], …)` au lieu de
  `split_indices(len(records), …)`. Reste inchangé sinon.

### B. Balayage de seuil (`detection_ortho/infer.py` + `detection_ortho/compare.py` + `scripts/sweep_threshold.py`)

- `max_score_near(det_points, lon, lat, radius_m) -> float` (infer.py, pur) : le
  meilleur `score` parmi les détections à ≤ `radius_m` du centre, sinon `0.0`.
- `sweep_precision_recall(scored, thresholds) -> list[dict]` (compare.py, pur) :
  `scored` = liste de `(best_score, is_true)` ; pour chaque seuil, calcule
  `tp`/`fp`/`precision`/`recall` (un point « tire » si `best_score >= seuil`).
- `scripts/sweep_threshold.py --weights --verdicts [--nir] [--conf-min 0.05]
  [--conf-max 0.9] [--step 0.05] [--radius 25] [--window 640] [--cache
  tiles_cache/eval] [--device cpu]` : pour chaque point, assemble la fenêtre,
  inférence **une fois** à `--conf-min`, `max_score_near` → `(score, is_true)` ;
  puis `sweep_precision_recall` → imprime la table précision/rappel vs seuil.

## Runs (manuels)

1. **Split spatial** : `build_dataset --spatial-split --bbox … --verdicts verdicts.csv --out dataset_spatial` → `train.py --data dataset_spatial/data.yaml --name citernes_spatial` → `evaluate.py --weights runs/citernes_spatial/weights/best.pt --data dataset_spatial/data.yaml` = **vrai mAP** de généralisation.
2. **Seuil** : `sweep_threshold.py --weights runs/citernes/weights/best.pt --verdicts verdicts.csv` = table précision/rappel → point de fonctionnement.

## Critère de succès

- Le split spatial donne un mAP de test **honnête** (probablement < 0,84), qui
  devient la référence pour juger les prochaines itérations.
- Le balayage identifie le seuil **maximisant la précision à rappel acceptable**
  (garder les ~12 vrais). Verrouiller ce seuil pour le run départemental.

## Stack technique

Réutilise `dataset`/`tiles`/`infer`/`compare` + Ultralytics. Aucune nouvelle
dépendance.

## Tests (hors-ligne)

- `spatial_split_indices` : points synthétiques en cellules distinctes → chaque
  cellule entièrement dans un seul lot ; deux points d'une même cellule jamais
  séparés ; déterminisme (même seed → même partition).
- `max_score_near` : meilleur score dans le rayon ; `0.0` si aucun / liste vide.
- `sweep_precision_recall` : `scored` synthétique → précision/rappel corrects à
  quelques seuils (dont bords).
- `sweep_threshold` : modèle stubé (scores variés) + tuiles pré-semées → une
  passe, table cohérente. Aucun poids/réseau réel.
- Les **runs réels** (rebuild+entraînement spatial ; balayage) sont **différés**.

## Risques & mitigations

- **Split spatial bruité** (~187 citernes → peu en test) → chiffre plus honnête
  mais à variance élevée ; l'assumer, ne pas sur-interpréter un point.
- **Balayage sur un jeu labellisé issu d'un run RVB conf 0,4** → mesure la
  précision sur CES candidats ; représentatif du terrain Tours, pas d'une zone
  neuve — cohérent avec l'usage (verrouiller le seuil du run départemental).
- **Cellule mal dimensionnée** (`cell_deg`) → trop grande = peu de séparation ;
  trop petite = ratios mal respectés ; défaut 0,05° (~5 km) raisonnable,
  ajustable en argument.
