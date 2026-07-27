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
