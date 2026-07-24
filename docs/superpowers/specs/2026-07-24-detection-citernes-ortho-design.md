# Détection de citernes souples sur ortho IGN — Design

**Date :** 2026-07-24
**Statut :** validé (brainstorming)

## Objectif

Détecter automatiquement les **citernes souples de secours** (réserves incendie
DFCI) sur l'imagerie aérienne (ortho IGN), comparer les détections à la base
OpenStreetMap (OSM), et générer un challenge **MapRoulette** pour faire ajouter
par la communauté les citernes absentes d'OSM.

Double finalité : **monter en compétence en vision par ordinateur** tout en
produisant un **outil réellement utile** d'inventaire.

## Portée

- **Étendue cible :** un département (zone à risque incendie), en commençant par
  un petit secteur pour la mise au point.
- **Objet détecté :** une seule classe, `citerne`. Sortie = un point géolocalisé
  par citerne (pas de contour précis — un point suffit pour la comparaison OSM).
- **Hors périmètre :** la publication automatique du challenge MapRoulette (on
  génère le fichier, l'utilisateur le charge lui-même) ; l'inférence temps réel ;
  toute autre classe d'objet.

## Contexte matériel & coûts

- **Machines disponibles :** Quadro P620 2 Go (inférence légère seulement),
  RX 5700 XT 8 Go (AMD, écartée — ROCm mal supporté sous Windows), Coral USB
  Edge TPU (option future d'accélération d'inférence quantifiée), gros CPU
  (pré/post-traitement géospatial).
- **Entraînement :** Google Colab (T4 16 Go gratuit, Pro ~10 €/mois si besoin).
- **Inférence à l'échelle :** en local sur CPU / Quadro P620, en tâche de fond.
- **Coût estimé :** 0 à ~15 €. Aucun achat de matériel nécessaire.

## Approche retenue

- **Cœur : détection d'objets par deep learning avec Ultralytics YOLO**
  (v8/v11, taille `n` ou `s`), transfer learning depuis les poids COCO. Sortie
  boîte + score ; le centroïde devient un point géolocalisé.
- **Rampe d'accès : baseline OpenCV** (seuillage couleur + filtrage
  morphologique/forme) pour explorer les données et obtenir un premier point de
  comparaison sans attendre l'entraînement.
- Segmentation sémantique (U-Net) écartée : annotation trop lourde et inutile
  puisqu'on ne veut qu'un point par citerne.

## Architecture — pipeline en étapes découplées

```
[1] Acquisition ortho IGN  ──►  [2] Labels OSM + annotation  ──►  [3] Entraînement YOLO
        (tuiles)                    (jeu d'entraînement)              (modèle .pt)
                                                                          │
[6] Export MapRoulette  ◄──  [5] Comparaison OSM  ◄──  [4] Inférence à l'échelle
    (GeoJSON challenge)          (matching spatial)        (détections → points géo)
```

Chaque étape produit un **artefact fichier** concret (tuiles PNG, dataset YOLO,
poids `.pt`, GeoJSON de détections, GeoJSON MapRoulette). Le pipeline est
inspectable et reprenable à n'importe quelle étape.

## Découpage en jalons

- **Jalon 0 — Reconnaissance des données** *(indispensable, rapide)* : sur un
  petit secteur, récupérer les citernes taguées dans OSM, télécharger l'ortho
  correspondante, et **regarder**. Combien y en a-t-il ? Sont-elles
  identifiables visuellement ? Quels tags OSM sont réellement utilisés ? Valide
  la faisabilité et dimensionne le reste **avant** tout investissement.
- **Jalon 1 — Baseline OpenCV** : détection couleur/forme sur le petit secteur →
  première comparaison à OSM. Comprendre les données, obtenir une référence.
- **Jalon 2 — Détecteur YOLO** : annotation assistée par OSM, entraînement sur
  Colab, évaluation (précision / rappel / mAP).
- **Jalon 3 — Passage à l'échelle** : inférence par tuilage glissant sur le
  département, post-traitement géospatial (dédoublonnage, reprojection).
- **Jalon 4 — Comparaison OSM & MapRoulette** : matching spatial, génération du
  challenge des citernes manquantes.

Chaque jalon donne un résultat exploitable et peut servir de point de pause.

## Composants

### 1. Acquisition ortho (`ortho_tiles.py`)

- Source : flux **WMTS Géoplateforme IGN** (`data.geopf.fr/wmts`, couche
  `ORTHOIMAGERY.ORTHOPHOTOS`), gratuit et **sans clé**.
- Tuiles 256×256, grille Web Mercator (EPSG:3857), zoom ~19 (≈20–30 cm/pixel).
- Télécharge et **met en cache** les tuiles d'une emprise donnée.
- Fournit les conversions **pixel ↔ coordonnées géographiques**.
- Ne télécharge que les zones utiles (pas de dalles JP2 massives).

### 2. Labels OSM & annotation

- **`osm_fetch.py`** : via l'**API Overpass**, récupère les objets citernes dans
  une emprise. Tags à couvrir (liste confirmée au Jalon 0) : `emergency=water_tank`,
  `man_made=water_tank`, `emergency=fire_water_pond`, éventuellement
  `emergency=suction_point`. Sortie : points géolocalisés = citernes connues.
- **Annotation** : autour de chaque point OSM, extraire une imagette
  (ex. 256×256 centrée) ; tracer les **boîtes englobantes** avec **Roboflow**
  (gratuit, navigateur ; alternative locale : `labelImg`). Compléter avec des
  **négatifs** (toits, champs, bâches agricoles). Export au **format YOLO**.
- Les points OSM servent deux fois : amorce d'annotation (on sait où regarder)
  **et** vérité-terrain de la comparaison finale.

### 3. Entraînement (`train.ipynb` + `train.py`)

- Ultralytics YOLO, une classe `citerne`, transfer learning depuis COCO.
- Entraînement sur Colab ; notebook versionné dans le repo. Sortie : `best.pt`
  rapatrié en local.
- Découpe **train / val / test** ; jeu de test (~20 %) **jamais vu** à
  l'entraînement, réservé à la mesure honnête de performance.

### 4. Inférence à l'échelle (`infer.py` + `geo_postprocess.py`)

- **Fenêtre glissante avec chevauchement** sur l'emprise (évite de couper une
  citerne à la frontière de deux tuiles).
- YOLO tuile par tuile, en local (CPU / Quadro P620), en tâche de fond.
- Post-traitement : boîtes pixel → **lat/lon**, **dédoublonnage** des détections
  dans les zones de chevauchement (fusion par proximité + score), filtrage par
  seuil de confiance.
- Sortie : **GeoJSON de détections** (un point + score par citerne).

### 5. Comparaison OSM (`osm_compare.py`)

- Appariement de proximité (rayon de tolérance ~15–25 m) entre détections et
  points OSM connus. Trois catégories :
  - **Détecté ∩ OSM** : confirme le modèle.
  - **Détecté \ OSM** : candidat à ajouter → alimente MapRoulette.
  - **OSM \ Détecté** : faux négatif ou citerne disparue → évalue le rappel.
- Sortie : GeoJSON avec attribut `statut` (ou trois fichiers).

### 6. Export MapRoulette (`maproulette_export.py`)

- Transforme les candidats « Détecté \ OSM » en fichier de tâches **GeoJSON**
  (une tâche par point) avec instruction : « Une citerne semble présente ici sur
  l'ortho IGN. Vérifiez et ajoutez-la si confirmé. »
- **S'arrête à la génération du fichier** — la création du challenge public est
  une action manuelle de l'utilisateur (rien n'est publié automatiquement).

## Performance visée

- Objet caractéristique → viser un **rappel de 70–90 %**, précision ajustable
  via le seuil de confiance.
- **Compromis assumé** : on privilégie le **rappel** (quitte à sur-proposer),
  car un humain valide chaque candidat dans MapRoulette — mieux vaut proposer
  trop que rater des citernes.
- Vitesse d'inférence : quelques tuiles/seconde sur CPU → un département =
  quelques heures à une nuit de calcul en arrière-plan. Acceptable pour un
  traitement ponctuel.

## Validation & tests

- **Qualité du modèle** : précision / rappel / mAP sur le jeu de test tenu à
  l'écart ; courbe précision-rappel pour choisir le seuil de confiance.
- **Tests unitaires (pytest)** sur les fonctions pures et critiques — celles
  qui, si buguées, faussent silencieusement le résultat géographique :
  conversion pixel ↔ géo, dédoublonnage des détections, appariement spatial OSM,
  format d'export MapRoulette.
- **Vérification visuelle** : utilitaire superposant détections + OSM sur les
  tuiles pour inspection à l'œil.

## Stack technique

- **Python 3.11+**, environnement virtuel (venv/conda).
- **CV / modèle :** ultralytics, opencv-python, numpy.
- **Géospatial :** pyproj, shapely, geopandas, mercantile (tuilage XYZ),
  requests (Overpass / WMTS).
- **Annotation :** Roboflow (navigateur) ou labelImg (local).
- **Entraînement :** Google Colab (T4).
- **Tests :** pytest.

## Risque principal & mitigation

**Risque :** trop peu de citernes dans OSM sur la zone (< 30–50) → jeu
d'entraînement trop maigre.

**Mitigation :** (a) le Jalon 0 mesure ce nombre d'abord ; (b) si trop faible,
élargir l'emprise OSM à plusieurs communes / le département entier pour ramasser
plus d'exemples connus ; (c) augmentation de données + transfer learning
permettent d'apprendre avec peu d'exemples pour un objet aussi distinct.
