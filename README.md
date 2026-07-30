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

### 2. (Optionnel) GPU : installer PyTorch CUDA pour la Quadro P620
Par défaut le venv est en CPU. Pour utiliser la P620 :

    .venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    .venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"   # True attendu

### 3. Inférer sur le département (lecture locale, GPU)

    python scripts/infer_area.py --boundary "Indre-et-Loire" \
        --weights runs/citernes/weights/best.pt --ortho ortho37.vrt \
        --conf 0.55 --device 0 --out inference_dept37

Si PyTorch CUDA n'est pas installé, utiliser `--device cpu` : il n'y a **pas
de repli automatique** — un `--device 0` sans CUDA disponible échoue (erreur
PyTorch), il faut explicitement repasser en CPU. Livrables identiques au
Jalon 3 (detections/detected_only/... + maproulette_challenge.geojson +
overlay.png), avec **imagerie** 100 % locale (l'emprise et les citernes de
référence restent récupérées via OSM/Overpass).
