# Adaptation du modèle à l'ortho express 2026 — design (sous-projet A)

Date : 2026-10-07

## Contexte et constats

On veut repasser le modèle affiné (`models/citernes-yolov8n.pt`) sur toute la couverture de l'ortho 2026,
y compris là où des départements ont déjà été scannés. Cette ambition se découpe en deux sous-projets :

- **A (ce document)** : adapter le modèle au rendu de l'ortho 2026.
- **B (autre spec, après A)** : lanceur par zones de couverture (zones qui ne sont pas des départements
  entiers, non-redite des faux positifs déjà rejetés, reprise, ordre des zones).

Constats mesurés le 2026-10-07 :

- « L'ortho 2026 » est la couche WMTS Géoplateforme `ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026` (ortho
  express). Son emprise déclarée est mondiale ; la couverture réelle se découvre en sondant les tuiles
  (404 ou tuile blanche de 1 651 octets = vide). Elle représente environ 139 000 km² (un quart de la France),
  en blocs : Centre-Val de Loire (28, 36, 37, 41, 45, 18 en grande partie), Normandie, Bretagne nord,
  Auvergne, région lyonnaise, Alpes, Provence. Le 44 et le 49 n'y sont pas.
- **Écart de rendu** : sur les 498 points du 36 (mis de côté à l'entraînement), le modèle affiné donne au
  seuil 0,25 un rappel de 0,60 et une précision de 0,92 sur l'ortho express 2026, contre 0,90 et 0,88 sur
  l'ortho habituelle. Ce sont les mêmes bâches : l'image 2026 a d'autres ombres, d'autres reflets et un autre
  éclairage.
- **Les bâches sont toujours là** : sur 150 points « vrai » tirés dans 18, 28, 37, 41 et 45, 119 sont
  détectés en 2026. Sur les 31 autres, vus un à un, 26 montrent la bâche, 3 ont disparu, 2 sont indécidables
  (ombre). Les positifs 2026 sont donc valides à environ 97 %.

## Objectif

Un seul modèle, bon sur les deux rendus, obtenu en affinant les poids actuels avec des imagettes des deux
couches aux mêmes points de verdicts.

**Critère d'acceptation** (mesuré sur le 36, mis de côté, au seuil 0,25) :

- sur l'ortho 2026 : rappel ≥ 0,85 et précision ≥ 0,85 (aujourd'hui 0,60 et 0,92) ;
- sur l'ortho habituelle : ni le rappel ni la précision ne perdent plus de 0,03 (aujourd'hui 0,895 et
  0,879 au 36, 0,924 et 0,948 au 49).

Si le critère n'est pas atteint, on le dit et on ne remplace pas `models/citernes-yolov8n.pt`.

**Hors périmètre** : scanner des départements (sous-projet B), remplacer le modèle versionné (décision
après mesure), la couche infrarouge, l'ortho de 2025.

## Exigences transversales

- **Tuiles en parallèle, toujours.** Tout script qui lit beaucoup de fenêtres précharge d'abord ses tuiles
  en parallèle (12 travailleurs, session partagée), puis lit depuis le cache. `sweep_threshold.py` et
  `eval_points.py` les chargent aujourd'hui une par une (environ 1,2 tuile par seconde contre 45 en
  parallèle).
- **Pas de conservation à l'échelle départementale.** Les passes départementales continuent de streamer en
  parallèle avec cache borné et purge (`infer_area.py`, `--cache-gb`). Les jeux de données d'entraînement
  gardent leur cache (quelques centaines de Mo).

## Composants

### 1. Cache de tuiles : suffixe unique et purge (`detection_ortho/tiles.py`, `tilecache.py`)

Deux défauts constatés :

- Le suffixe de cache d'une couche est le dernier segment de son nom (`_2026`). `RVB-EXPRESS.2026` et
  `IRC-EXPRESS.2026` produisent donc le même fichier et se télescopent.
- Le motif de purge de `tilecache.py` n'accepte que des lettres dans le suffixe : les fichiers
  `…_2026.jpg` ne sont jamais purgés, ce qui ferait exploser le disque lors d'une passe départementale.

Correction : une fonction `layer_tag(layer)` donne `""` pour la couche standard et `_<nom court>` sinon,
où le nom court est le nom de couche privé du préfixe `ORTHOIMAGERY.ORTHOPHOTOS.`, en minuscules, avec les
points remplacés par des tirets (`IRC` → `_irc`, inchangé ; `RVB-EXPRESS.2026` → `_rvb-express-2026`).
`download_tile` l'utilise, et le motif de purge accepte `[a-z0-9-]+` dans le suffixe.

### 2. Préchargement parallèle (`detection_ortho/tiles.py`)

`prefetch_tiles(tiles, cache_dir, layer, workers=12, zoom=19) -> list` télécharge les tuiles demandées en
parallèle et retourne les échecs. Utilisé par `build_dataset.py` (qui garde son comportement), par
`sweep_threshold.py` et par `eval_points.py`.

### 3. Jeu de données multi-couches (`scripts/build_dataset.py`)

- Option `--layers L1 [L2 …]` (défaut : la couche standard seule, donc comportement inchangé). Incompatible
  avec `--nir` (erreur claire).
- Chaque enregistrement produit une imagette par couche disponible. Le nom de la première couche reste
  `<enregistrement>` ; les autres portent le suffixe de couche (`<enregistrement>__rvb-express-2026`).
- Une fenêtre sans donnée (tuile absente ou moins de 5 % de pixels non blancs) est sautée et comptée dans
  un résumé par couche.
- **Le découpage entraînement/validation/test se décide par enregistrement**, pas par imagette : les deux
  rendus d'un même endroit tombent toujours dans la même partie, ce qui évite qu'un point soit en
  entraînement dans un rendu et en test dans l'autre.
- `--holdout` s'applique aux enregistrements, donc aux deux rendus.

### 4. Évaluation par couche (`sweep_threshold.py`, `eval_points.py`)

Option `--layer` (défaut : couche standard) et préchargement parallèle. Les commandes d'évaluation du
runbook donnent le tableau pour chaque couche.

### 5. Entraînement et runbook (README)

Construire le jeu avec les deux couches et les verdicts de 18, 28, 37, 41, 45 (plus 44 qui n'aura que sa
couche standard), `--holdout` sur le 36 et le 49, `--spatial-split` ; affiner environ 20 époques depuis
`models/citernes-yolov8n.pt` ; évaluer sur le 36 pour les deux couches et sur le 49 pour la couche
standard ; comparer au critère d'acceptation.

## Décision sur les données

Les imagettes 2026 positives sont gardées sans filtre automatique. Un filtre fondé sur la détection
retirerait précisément les cas sombres ou ombrés que le modèle doit apprendre. Le bruit estimé (environ 3 %
de positifs absents ou indécidables) est tolérable pour un détecteur d'objets. Le contrôle inverse (un
« faux » devenu une vraie citerne en 2026) n'a pas été fait : cas jugé très improbable.

## Tests (hors-ligne, TDD)

- `layer_tag` : standard, `IRC` (suffixe `_irc` conservé), couches 2026 RVB et IRC distinctes.
- Motif de purge : un fichier `…_rvb-express-2026.jpg` est purgé, un fichier d'un autre zoom ne l'est pas.
- `prefetch_tiles` : appelle le téléchargeur pour chaque tuile, en parallèle, retourne les échecs.
- `build_dataset --layers` : deux couches avec caches préremplis, une fenêtre vide dans la seconde couche ;
  vérifier le nombre d'imagettes par couche, les noms, le même découpage pour les deux rendus d'un
  enregistrement, et le résumé.
- `--layers` avec `--nir` : erreur.
- `sweep_threshold.py --help` et `eval_points.py --help` mentionnent `--layer`.

## Risques

- **Durée** : environ 8 200 images au lieu de 4 900. À 0,33 s par image et par époque (mesuré sur
  l'entraînement précédent), 20 époques représentent environ 10 h de CPU. À confirmer sur une époque.
- **Édition 2026 mouvante** : l'ortho express 2026 s'enrichit à mesure que de nouvelles zones sont
  traitées. Les imagettes sont figées au moment de la construction du jeu ; la couverture est à resonder
  avant le sous-projet B.
- **Critère non atteint** : un rendu d'ombres trop différent peut limiter le rappel. Pistes si besoin :
  plus d'époques, plus d'augmentation de luminosité, imagettes supplémentaires.
- **Les 44 et 49 sans 2026** : ils n'apportent que le rendu standard. Le 49 reste le seul test mis de côté
  pour la couche standard ; le 36 sert aux deux couches.
