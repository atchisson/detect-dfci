# Affinage du modèle avec les verdicts MapRoulette — design

Date : 2026-10-05

## Contexte et objectif

Le modèle `models/citernes-yolov8n.pt` (entraîné sur le 37 seul : ~190 positifs OSM
+ 101 négatifs durs de la revue de Tours) a été exécuté sur les départements 18, 28,
36, 37, 41, 44, 45, 49. Les détections ≥ 0,40 ont été revues dans MapRoulette
(projet 64996, un challenge « DECI NN » par département) :

- `fixed` (status 1) → vraie citerne, ajoutée dans OSM ;
- `not an issue` (status 2) → faux positif ;
- autres statuts (3 skipped, 4 deleted, 5 already fixed, 6 too hard) : marginaux
  (9 tâches), **ignorés**.

Volume : 2 313 vrais, 3 338 faux, scores 0,40–0,96. L'API MapRoulette est publique
en lecture (pas de clé). Chaque tâche expose position, score du modèle actuel et statut.

**Objectif** : réduire les faux positifs (et accessoirement élargir la variété de
bâches reconnues) en affinant le modèle actuel sur ces verdicts, avec une évaluation
honnête sur des départements jamais vus à l'entraînement.

**Hors périmètre** : rappel sur les types atypiques (lagunes, cuves rondes) — ces
citernes n'apparaissent pas dans les challenges (seules les détections ≥ 0,40 ont été
revues). Aucun upload MapRoulette/OSM (contrainte ferme du projet : lecture seule).

## Approche retenue (A)

Affinage (fine-tuning) des poids actuels sur : dataset OSM du 37 + tous les verdicts
d'entraînement. ~30 époques. Entraînement manuel côté utilisateur (CPU de nuit ou
Colab), comme les itérations précédentes.

## Composants

### 1. `scripts/fetch_maproulette_verdicts.py` (nouveau)
- `--project 64996` (défaut), `--out verdicts_maproulette/`.
- Liste les challenges du projet, pagine les tâches (`/api/v2/challenge/{id}/tasks`,
  `limit=500&page=N`), User-Agent = URL du dépôt.
- Écrit un CSV par département `verdicts_<NN>.csv` au **format existant**
  `index,lat,lon,score,verdict` (verdict `vrai`/`faux`). Le numéro de département
  est extrait du nom du challenge (`DECI NN`).
- Affiche le décompte par département et par statut (y compris les ignorés).
- Logique pure (statut → verdict, extraction dept, tâche → ligne CSV) dans
  `detection_ortho/maproulette.py`, testée hors-ligne ; le réseau reste dans le script.

### 2. `scripts/build_dataset.py` (modifié)
- `--verdicts` accepte **plusieurs fichiers** (`nargs="+"`) ; rétrocompatible.
- **Dédoublonnage** : un verdict `vrai` situé à moins de `--dedup-m` (défaut 15 m)
  du centre d'un positif OSM déjà chargé (ou d'un autre `vrai` déjà retenu) est
  écarté. Nécessaire : les vrais du 37 sont maintenant dans OSM et ressortiraient
  doublement. Le décompte des doublons écartés est affiché.
- Aucun changement du format des chips ni du split (`--spatial-split` recommandé).
- Le `--bbox` reste celui du 37 : il alimente positifs OSM, piscines et fonds. Les
  autres départements ne contribuent que par leurs verdicts (vrais = positifs boîte
  13 m, faux = négatifs durs), car leurs vraies citernes sont précisément ces
  vrais.

### 3. Évaluation (peu ou pas de code nouveau)
- **Départements mis de côté : 36 et 49** → leurs CSV ne sont **jamais** passés à
  `build_dataset`. Ils servent uniquement d'ensemble de test.
- Avant/après : `scripts/eval_points.py` et `scripts/sweep_threshold.py` (déjà
  prévus pour `--verdicts <csv>`) lancés avec l'ancien modèle puis le nouveau, sur
  `verdicts_36.csv` et `verdicts_49.csv`. Comparaison à conf 0,40 / 0,55 / 0,70 :
  faux supprimés, vrais conservés, précision.
- Biais à documenter : tous les points de test sont des détections ≥ 0,40 de
  l'ancien modèle, donc son « rappel » est 100 % par construction à 0,40 ; seul
  le gain de **précision** et la rétention des vrais par le nouveau modèle sont
  concluants. Le gain n'est pas une mesure du rappel absolu.

### 4. Runbook (README)
Section « Affiner avec les verdicts MapRoulette » : fetch → build (tous les
départements sauf 36 et 49, `--spatial-split`, `--verdicts` multiples) → train
(`--model models/citernes-yolov8n.pt --epochs 30 --name citernes_mr`) → eval
avant/après sur 36 et 49.

## Flux de données

API MapRoulette → `verdicts_maproulette/verdicts_NN.csv` → `build_dataset.py`
(positifs OSM 37 + vrais dédoublonnés + faux + piscines + fonds ; tuiles WMTS aux
coordonnées) → `train.py --model <poids actuels>` → `runs/citernes_mr/weights/best.pt`
→ eval sur 36/49 vs ancien modèle.

## Gestion d'erreurs
- API : timeout et nouvel essai par page ; échec d'un challenge = erreur claire
  (pas de CSV partiel silencieux).
- Statut inconnu/ignoré : compté, pas écrit.
- Tuile WMTS en échec au build : comportement existant (point sauté, signalé).

## Tests (hors-ligne, TDD)
- mapping statut → verdict ; extraction du numéro de département ;
  tâche JSON → ligne CSV (lat/lon, score, ordre des coordonnées GeoJSON lon,lat).
- parsing de `--verdicts` multiples ; dédoublonnage (vrai proche d'un positif
  écarté, vrai lointain conservé, faux jamais écarté).
- Round-trip : CSV écrit par le fetch relu par `parse_verdicts`.

## Risques
- **Durée d'entraînement CPU** : ~6 000 images vs ~600 avant ; 30 époques ≈ 10-20 h
  CPU (à confirmer sur une époque). Colab (`notebooks/train_yolo.ipynb`) reste une
  option.
- **Dérive** : l'affinage pourrait dégrader des cas bien reconnus ; c'est le rôle du
  test sur 36/49 + comparaison avec l'ancien modèle.
- Boîte 13 m uniforme pour les vrais (position seule disponible) : approximation
  acceptée, cohérente avec l'itération 2.
- Les 433 faux du 37 (hors MapRoulette) peuvent être ajoutés en passant leur CSV
  local supplémentaire à `--verdicts` s'il existe encore.
