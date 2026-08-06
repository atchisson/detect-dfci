# Test terrain NIR aux points de verdict : Design

**Date :** 2026-08-06
**Statut :** validé (brainstorming)
**Prérequis :** Itération A livrée (modèles `runs/citernes/weights/best.pt` RVB et
`runs/citernes_nir/weights/best.pt` NIR entraînés). `verdicts.csv` (44 points
labellisés : 12 vrais / 32 faux) à la racine.

## Objectif

Trancher, sur le **terrain** et non sur un mAP de test peu fiable, si le NIR
réduit réellement les faux positifs. Méthode : évaluer chaque modèle **aux
emplacements déjà labellisés** et comparer combien de faux positifs connus
chacun supprime, à vrais positifs conservés égaux.

## Contexte

- Le test de valeur (itération A) a donné un mAP quasi identique (RVB 0,840 /
  NIR 0,845), signal marginal. Mais le mAP de test est un mauvais prédicteur du
  terrain (précision réelle Tours 11–27 % vs test 84 %). Le vrai juge = les FP
  sur données réelles labellisées.
- On dispose de `verdicts.csv` (labels terrain) et de `compare.compare_to_verdicts`
  (décompte FP supprimés / vrais conservés) déjà testé.
- À conf 0,55, le RVB (itér. 2) supprimait ~20/32 FP en gardant 12/12 vrais →
  garde-fou : le run RVB doit reproduire ces chiffres.

## Décisions issues du brainstorming

- **Méthode « aux points de verdict »** (pas de ré-inférence pleine zone) :
  ~88 fenêtres, automatique, aucun re-review. Limite assumée : ne voit pas les
  nouveaux FP que le NIR créerait ailleurs — mais c'est un filtre décisif de
  premier ordre (si le NIR ne bat pas le RVB sur ses propres FP connus, le
  4-canaux est mort).
- **IRC via WMTS** (couche `ORTHOIMAGERY.ORTHOPHOTOS.IRC`, même grille z19
  EPSG:3857 que l'entraînement du modèle NIR → alignement pixel garanti). Les
  dalles BD ORTHO IRC locales (Lambert-93) sont réservées à l'itération B.
- **Comparaison** : lancer le script une fois par modèle, comparer FP
  supprimés / vrais conservés.

## Composants

### A. Helper `is_detected_near` (`detection_ortho/infer.py`, extension)

- `is_detected_near(det_points, lon, lat, radius_m) -> bool` (pur) : vrai si un
  point de détection (`{lon,lat,...}`) est à ≤ `radius_m` du centre `(lon,lat)`.
  Réutilise `haversine_m`. Testable sur points synthétiques.

### B. Script `scripts/eval_points.py` (nouveau)

- Args : `--weights` (obligatoire), `--verdicts` (obligatoire), `--conf 0.55`,
  `--nir` (flag), `--radius 25`, `--window 640`, `--cache` (défaut
  `tiles_cache_eval`), `--device cpu`, `--zoom 19`.
- Flux, pour chaque point de `parse_verdicts(verdicts.csv)` :
  1. `assemble_window(lon, lat, zoom, window, cache)` → `(rgb, ogx, ogy)`.
  2. si `--nir` : `assemble_window(..., layer=LAYER_IRC)` → `irc` ;
     `img = compose_rgn(rgb, irc)`. sinon `img = rgb`.
  3. `model.predict(img, conf, device, verbose=False)[0].boxes` →
     `result_to_boxes` → `boxes_to_points(…, ogx, ogy, zoom)`.
  4. si `is_detected_near(points, lon, lat, radius)` → le modèle « tire » ici →
     ajouter `{lon, lat}` à la liste `detections`.
  5. échec d'assemblage d'un point → message stderr + `continue` (comme
     `build_dataset`).
- `compare_to_verdicts(detections, verdicts, radius)` → imprimer le rapport
  (mêmes libellés que `compare_to_verdicts.py` : candidats, FP supprimés, FP
  encore détectés, vrais conservés).

## L'expérience (runs manuels)

```
python scripts/eval_points.py --weights runs/citernes/weights/best.pt \
    --verdicts verdicts.csv --conf 0.55                      # RVB (garde-fou)
python scripts/eval_points.py --weights runs/citernes_nir/weights/best.pt \
    --verdicts verdicts.csv --conf 0.55 --nir                # NIR
```

## Critère de succès

Le **NIR gagne** s'il **supprime strictement plus de faux positifs** que le RVB
tout en **conservant au moins autant de vrais**. Sinon, gain non concluant.
- **NIR gagne nettement** → itération B en **4 canaux** (RGB+NIR).
- **Ex æquo / NIR perd** → itération B en **3 canaux RVB** (pivot natif juste
  pour la perf).

## Stack technique

Réutilise `tiles`/`dataset` (`assemble_window`, `compose_rgn`, `LAYER_IRC`,
`parse_verdicts`), `infer` (`result_to_boxes`, `boxes_to_points`), `compare`
(`compare_to_verdicts`). Modèle YOLO via Ultralytics (poids déjà entraînés).

## Tests (hors-ligne)

- `is_detected_near` : détection dans/hors du rayon (points synthétiques).
- Test d'intégration `eval_points` : **modèle stubé** (monkeypatch de `YOLO`
  renvoyant des boîtes canées) + tuiles RVB **et** IRC pré-semées en cache ;
  vérifie que `--nir` compose bien (l'image vue par le modèle a le bleu = NIR),
  que les points « tirés » alimentent `compare_to_verdicts`, et que le rapport
  a la bonne forme. Aucun poids réel, aucun réseau.
- Les **runs réels** (2 lancements ci-dessus) sont **différés** (manuels).

## Risques & mitigations

- **Garde-fou RVB non reproduit** (≠ ~20/32 FP, 12/12 vrais) → bug de méthode ;
  à investiguer avant d'interpréter le NIR.
- **Fenêtre 640px ≈ 192 m couvre des points voisins** → une détection peut
  matcher un verdict voisin ; sans effet sur le décompte (proximité par
  verdict), acceptable.
- **Double téléchargement RVB+IRC** aux 44 points → négligeable (~88 fenêtres),
  cache dédié `tiles_cache_eval`.
- **Angle mort** (nouveaux FP NIR ailleurs) → assumé ; test plus large seulement
  si le NIR passe ce premier filtre.
