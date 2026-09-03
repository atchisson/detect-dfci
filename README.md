# detection_ortho

Détection de citernes souples de secours sur l'ortho IGN, comparaison OSM,
export MapRoulette. Voir `docs/superpowers/specs/`.

## Installation

    python -m venv .venv
    .venv\Scripts\activate   # Windows
    pip install -r requirements.txt

## Tests

    pytest

## Docker (environnement reproductible)

Construire l'image (lance aussi les tests) :

    docker build -t detection_ortho .
    docker run --rm detection_ortho          # exécute pytest

Lancer un script en persistant les sorties dans ./out :

    docker run --rm -v "$PWD/out:/app/out" detection_ortho \
        python scripts/recon.py --bbox 6.14 43.41 6.16 43.43 --zoom 19 --out /app/out/recon_out

Ou via docker compose :

    docker compose run --rm app python scripts/run_baseline.py \
        --bbox 6.14 43.41 6.16 43.43 --out /app/out/baseline_out

## Jalon 2 — Détecteur YOLO

1. Générer le dataset depuis OSM + ortho IGN :

       python scripts/build_dataset.py --bbox 0.05 46.72 1.06 47.72 --out dataset

   Vérifier `dataset/qa_positives.png` et relancer avec `--exclude <indices>`
   pour écarter les intrus.

2. Entraîner (local CPU par défaut) :

       python scripts/train.py --data dataset/data.yaml --epochs 100 --device cpu

   (Option GPU : `notebooks/train_yolo.ipynb` sur Colab.)

3. Évaluer sur le jeu de test :

       python scripts/evaluate.py --weights runs/citernes/weights/best.pt --data dataset/data.yaml

   Objectif : mAP@50 ≥ 0,60 et rejet visible des piscines/toits.

## Jalon 3 — Inférence sur une zone + MapRoulette

Inférer sur une emprise (relation OSM), comparer à OSM, générer le challenge :

    python scripts/infer_area.py --boundary "Tours Métropole Val de Loire" \
        --weights runs/citernes/weights/best.pt --conf 0.4 --out inference_out

Livrables dans `inference_out/` : `detections.geojson`, `matched/detected_only/
osm_only.geojson`, `maproulette_challenge.geojson`, et `overlay.png` (aperçu
visuel emprise + détections vs OSM, généré en best-effort).

Regénérer seulement le fichier MapRoulette depuis les candidats :

    python scripts/export_maproulette.py --input inference_out/detected_only.geojson \
        --out inference_out/maproulette_challenge.geojson

**Aucun upload automatique** : importez `maproulette_challenge.geojson`
vous-même dans l'interface MapRoulette.

## Itération 2 — Hard-negative mining

À partir de `verdicts.csv` (revue de la carte, cf. `make_map.py`) :

1. Régénérer le dataset augmenté (négatifs durs + positifs de revue) :

       python scripts/build_dataset.py --bbox 0.05 46.72 1.06 47.72 \
           --verdicts verdicts.csv --out dataset

2. Réentraîner :

       python scripts/train.py --data dataset/data.yaml --epochs 100 --device cpu

3. Re-inférer Tours Métropole avec les nouveaux poids (cf. Jalon 3), puis
   mesurer le gain avant/après :

       python scripts/compare_to_verdicts.py \
           --detections inference_out/detected_only.geojson --verdicts verdicts.csv

**Note** : le mAP test rapporté par `train.py` n'est **pas comparable
directement** d'une itération à l'autre — ajouter `--verdicts` change
`len(records)`, ce qui recompose le split 70/15/15. Le vrai gain se lit via
`compare_to_verdicts.py` (ré-inférence sur la même emprise).

## Échelle départementale (BD ORTHO locale + GPU)

### 1. Récupérer la BD ORTHO du département
Télécharger la **BD ORTHO 20 cm RVB Lambert-93** du département depuis
cartes.gouv.fr (jeu IGNF_BD-ORTHO), décompresser les dalles `.jp2` dans un
dossier. Construire un mosaïque virtuelle (une fois) :

    gdalbuildvrt ortho37.vrt chemin/vers/dalles/*.jp2

(ou QGIS → Raster → Divers → Construire un raster virtuel). Si `gdalbuildvrt`
n'est pas installé, un script maison ne dépendant que de rasterio fait le
même travail :

    python scripts/build_ortho_vrt.py --dir chemin/vers/dalles --out ortho37.vrt

### 1bis. Pré-reprojeter en tuiles 3857 (perf — une seule fois)
La lecture directe des dalles JP2 (reprojection par fenêtre) est trop lente à
l'échelle départementale (~2,4 s/fenêtre). On pré-reprojette **une fois** chaque
dalle en un GeoTIFF Web-Mercator aligné sur la grille du modèle, **en parallèle**,
puis on assemble une VRT que l'inférence lit vite (~3 ms/fenêtre) :

    python scripts/build_cog_tiles.py \
        --src BDORTHO_2-0_RVB-0M20_JP2-E080_LAMB93_D037_2025-01-01 \
        --out-dir cog_tiles --workers 6

(quelques heures, une seule fois ; reprise auto si interrompu). Produit
`cog_tiles/mosaic.vrt`, à passer à `infer_area.py --ortho`.

> ⚠️ **NE PAS** faire `build_cog.py --src <une VRT>` : `build_cog` traiterait la
> mosaïque comme un seul raster et tenterait de la charger entière en RAM →
> sortie **noire**. Toujours pointer sur des **dalles** (dossier). `build_cog.py`
> reste utile pour reprojeter une **seule** dalle en un fichier unique.

### 2. (Optionnel) GPU : installer PyTorch CUDA pour la Quadro P620
Par défaut le venv est en CPU. Pour utiliser la P620 :

    .venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    .venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"   # True attendu

### 3. Inférer sur le département (lecture locale, GPU)

    python scripts/infer_area.py --boundary "Indre-et-Loire" \
        --weights models/citernes-yolov8n.pt --ortho cog_tiles/mosaic.vrt \
        --conf 0.55 --device 0 --out inference_dept37

Si PyTorch CUDA n'est pas installé, utiliser `--device cpu` : il n'y a **pas
de repli automatique** — un `--device 0` sans CUDA disponible échoue (erreur
PyTorch), il faut explicitement repasser en CPU. Livrables identiques au
Jalon 3 (detections/detected_only/... + maproulette_challenge.geojson +
overlay.png), avec **imagerie** 100 % locale (l'emprise et les citernes de
référence restent récupérées via OSM/Overpass).

## Reproduire sur un autre PC — tous les départements

Tout le nécessaire est dans le dépôt : le code, les scripts, **et le modèle
entraîné** (`models/citernes-yolov8n.pt`, YOLOv8n, ~6 Mo). Seule la BD ORTHO
n'est pas versionnée (trop volumineuse) — elle se télécharge par département, ou
se lit en streaming via le WMTS.

**Mise en place (une fois) :**

    py -3.12 -m venv .venv   # 3.13 fonctionne aussi
    .venv\Scripts\python -m pip install -r requirements.txt

**Pour chaque département `NN` (nom OSM `"<Département>"`) :**

1. **Imagerie** — deux options :
   - *Locale (rapide, ~2,4 To pour toute la France)* : télécharger la BD ORTHO
     20 cm RVB Lambert-93 du département sur cartes.gouv.fr, décompresser les
     `.jp2`, puis pré-reprojeter en parallèle (une fois par département) :

         python scripts/build_cog_tiles.py --src <dossier_dalles_NN> \
             --out-dir cog_tilesNN --workers 6

     Vérifier que la mosaïque n'est pas noire avant de lancer l'inférence
     (lire une fenêtre sur une citerne connue).
   - *WMTS streaming (rien à télécharger à la main)* : ne pas passer `--ortho` ;
     `infer_area` récupère les tuiles lui-même. Par défaut le cache disque est
     **plafonné à 10 Go** (`--cache-gb`) : les tuiles sont téléchargées par
     tranches en parallèle pendant que l'inférence tourne, puis purgées dès
     qu'elles ne servent plus, et le cache est vidé en fin de run. Un
     département entier demanderait sinon ~21 Go d'un coup (2,7 M tuiles à
     ~7,8 ko), pré-téléchargées avant de commencer. `--cache-gb 0` rétablit ce
     pré-téléchargement intégral (cache non borné, réutilisable d'un run à
     l'autre). Le WMTS IGN n'est pas soumis à limite d'usage.

2. **Inférence** (ajouter `--ortho cog_tilesNN/mosaic.vrt` si option locale) :

       python scripts/infer_area.py --boundary "<Département>" \
           --weights models/citernes-yolov8n.pt --conf 0.40 \
           --device cpu --out inferenceNN

3. **Challenge MapRoulette** — filtrer au seuil de qualité (≥0.7 ≈ 88 % de
   précision sur le 37) plutôt que tout publier :

       python scripts/export_maproulette.py \
           --input inferenceNN/detected_only.geojson \
           --out inferenceNN/challenge_NN.geojson --min-score 0.7

   (Import manuel dans MapRoulette — aucun upload automatique.) Pour réviser à
   la main d'abord : `python scripts/make_map.py --dir inferenceNN`.

> Le modèle a été entraîné sur le 37 : il excelle sur la **bâche souple
> turquoise** mais rate les types atypiques (grands bassins, cuves rondes). Pour
> le national, envisager un ré-entraînement sur des exemples plus divers.

## Test de valeur NIR (proxy [R,G,NIR])

Mesurer si le proche-infrarouge aide, sans plomberie 4-canaux : on remplace le
bleu par le NIR (canal rouge de la couche IRC WMTS).

    python scripts/build_dataset.py --bbox 0.05 46.72 1.06 47.72 \
        --verdicts verdicts.csv --nir --out dataset_nir
    python scripts/train.py --data dataset_nir/data.yaml --epochs 100 --device cpu --name citernes_nir
    python scripts/evaluate.py --weights runs/citernes_nir/weights/best.pt \
        --data dataset_nir/data.yaml

Comparer le mAP@50 / la matrice de confusion au modèle RVB (mAP 0,83). Si le
gain est net, investir dans un vrai modèle 4 canaux (RGB+NIR) au pivot natif.

## Test terrain NIR (aux points de verdict)

Juger le NIR sur des données réelles labellisées (`verdicts.csv`) plutôt que sur
le mAP de test. On évalue chaque modèle aux points labellisés connus et on compare
combien de faux positifs chacun supprime, à vrais conservés égaux.

    python scripts/eval_points.py --weights runs/citernes/weights/best.pt \
        --verdicts verdicts.csv --conf 0.55            # RVB (garde-fou : supprime la grande majorité des faux, garde ~tous les vrais)
    python scripts/eval_points.py --weights runs/citernes_nir/weights/best.pt \
        --verdicts verdicts.csv --conf 0.55 --nir      # NIR

Le NIR est concluant s'il supprime strictement plus de faux positifs que le RVB
en conservant au moins autant de vrais. Sinon, le pivot natif se fera en 3
canaux RVB (perf seule).

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
