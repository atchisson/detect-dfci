# Test de valeur NIR (proxy [R,G,NIR]) : Design

**Date :** 2026-07-30
**Statut :** validé (brainstorming)
**Prérequis :** Jalons 0-3 + itérations 2/COG livrés. Modèle RVB de référence : `runs/citernes/weights/best.pt` (mAP@50 test 0,83).

## Objectif

Mesurer **à moindre coût** si le proche-infrarouge (NIR) améliore la détection,
avant d'investir dans un pivot 4-canaux / natif. On teste un **proxy 3 canaux
`[R, G, NIR]`** (le bleu, canal le moins utile ici, remplacé par le NIR) →
entraînement/évaluation YOLOv8 **standard**, aucune plomberie 4-canaux.

## Contexte

- Visuellement (couche IRC WMTS), le NIR rejette fortement la **végétation**
  (rouge vif) et rend la **cible saillante** (eau/bâche = bleu sombre). Mais
  **les piscines aussi sont bleues** → le NIR ne règle pas la confusion
  citerne/piscine. Gain réel **à mesurer**, pas à supposer.
- Le NIR est la **bande 1 de l'IRC** = **canal rouge** de la couche
  `ORTHOIMAGERY.ORTHOPHOTOS.IRC` (composite CIR : NIR, R, V).

## Décisions issues du brainstorming

- **Proxy `[R, G, NIR]`** : imagette 3 canaux où le bleu est remplacé par le NIR
  (canal rouge de l'IRC). jpg standard → `train.py`/`evaluate.py` **inchangés**.
- **Mêmes params qu'itération 2** (bbox dpt 37, `--verdicts`, graine, split) →
  comparaison **équitable** RVB vs [R,G,NIR].
- **Comparaison par mAP@50 de test** (+ matrice de confusion / rejet
  végétation-eau). `infer_area` non modifié pour ce test — le terrain viendra au
  pivot natif si le gain est confirmé.

## Composants

### A. Récupération multi-couches (`detection_ortho/tiles.py`, extension)

- `tile_url(x, y, zoom, layer=LAYER)` : paramètre `layer` (défaut = ortho RVB
  actuelle).
- `download_tile(x, y, zoom, cache_dir, session=None, layer=LAYER)` : télécharge
  la tuile de la couche demandée. **Cache spécifique à la couche** : le nom de
  fichier inclut un tag dérivé de `layer` (RVB → nom actuel inchangé pour ne pas
  invalider les caches existants ; IRC → suffixe `_irc`). Évite le télescopage
  RVB/IRC.
- `dataset.assemble_window(..., layer=LAYER)` : passe `layer` à `download_tile`
  (assemble une fenêtre RVB **ou** IRC).

### B. Composition [R,G,NIR] (`detection_ortho/dataset.py`, extension)

- `compose_rgn(rgb_bgr, irc_bgr) -> np.ndarray` (pur) : retourne une image BGR
  où le canal **bleu est remplacé par le NIR** (= canal rouge de l'IRC).
  Concrètement `out = rgb.copy(); out[:,:,0] = irc[:,:,2]`. Testable sur arrays
  synthétiques.

### C. Dataset NIR (`scripts/build_dataset.py`, extension)

- Option `--nir` : pour chaque enregistrement, assembler la fenêtre RVB **et** la
  fenêtre IRC (`assemble_window(..., layer=IRC)`), composer via `compose_rgn`,
  écrire l'imagette [R,G,NIR]. Labels/split/QA inchangés. (Double le nombre de
  tuiles téléchargées — acceptable pour un test.)

## Protocole (runs manuels différés)

1. `python scripts/build_dataset.py --bbox 0.05 46.72 1.06 47.72 --verdicts verdicts.csv --nir --out dataset_nir`
2. `python scripts/train.py --data dataset_nir/data.yaml --epochs 100 --device cpu`
3. `python scripts/evaluate.py --weights runs/citernes*/weights/best.pt --data dataset_nir/data.yaml`
4. **Comparer** mAP@50 / précision / matrice de confusion au modèle RVB.

## Critère de succès

Décider, sur des chiffres : le modèle [R,G,NIR] fait-il **mieux** que le RVB
(mAP@50 0,83, et surtout moins de faux positifs végétation/eau) ?
- **Oui** → pivot natif B en **4 canaux** (RGB+NIR).
- **Non/marginal** → pivot natif B en **3 canaux RVB** (juste pour la perf).

## Stack technique

- Réutilise tout : `tiles`, `dataset` (assemble_window, compose_rgn),
  `build_dataset`, `train`, `evaluate`. Couche IRC via WMTS (pas de téléchargement
  BD ORTHO IRC pour ce test).

## Tests (hors-ligne)

- `tile_url(layer=...)` : la couche apparaît dans l'URL ; `download_tile` cache
  la tuile IRC sous un nom distinct (pas de collision RVB/IRC) — réseau mocké.
- `compose_rgn` : le canal bleu de sortie == canal rouge (NIR) de l'IRC, R et G
  conservés.
- Le **run réel** (dataset NIR + entraînement + éval) est **différé** (manuel).

## Risques & mitigations

- **Collision de cache RVB/IRC** → cache tagué par couche.
- **mAP de test optimiste** (déjà constaté) → on regarde AUSSI la matrice de
  confusion / le rejet des faux positifs ; le vrai terrain viendra au pivot.
- **Double téléchargement de tuiles** (RVB + IRC) → acceptable pour un test ; le
  cache évite les re-téléchargements.
- **Couverture IRC WMTS** → couche nationale, coïncide avec le RVB.
