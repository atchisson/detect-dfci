# Itération 2 — Hard-negative mining : Design

**Date :** 2026-07-28
**Statut :** validé (brainstorming)
**Prérequis :** Jalons 0-3 livrés. Modèle itér. 1 : `runs/detect/runs/citernes/weights/best.pt`.

## Objectif

Améliorer la **précision terrain** du détecteur en réinjectant à l'entraînement
les erreurs réelles révélées par la revue manuelle de Tours Métropole
(hard-negative mining), puis **mesurer le gain réel** en relançant l'inférence
sur la même métropole.

## Contexte (résultats itération 1)

- Inférence Tours Métropole @conf 0.4 : **113 candidats, 12 vrais / 101 faux →
  précision ≈ 11 %**. Rappel sur citernes OSM connues : 13/14 ≈ 93 %.
- Le score de confiance est **bien calibré** (les 6 détections ≥0.90 sont toutes
  vraies), mais la précision globale est basse : le modèle **sur-détecte** sur la
  diversité réelle. Le test du Jalon 2 (mAP 0.83) était optimiste (négatifs peu
  variés).
- La revue manuelle a produit **`verdicts.csv`** : 113 candidats étiquetés
  `vrai`/`faux` (12 vrais, 101 faux) — matière première de cette itération.

## Portée

- **Dans le périmètre :** augmenter le jeu avec les 101 FP (négatifs durs) + 12
  TP (positifs), réentraîner YOLOv8n, évaluer (test), **et re-inférer Tours
  Métropole** pour mesurer le gain réel (avant/après).
- **Hors périmètre :** changement d'architecture de modèle, extension à d'autres
  départements, publication MapRoulette.

## Décisions issues du brainstorming

- **Ingestion directe de `verdicts.csv`** dans `build_dataset.py` via
  `--verdicts` : `faux` → imagette négative (label vide, `hardneg_XXXX`) ;
  `vrai` → imagette positive avec **boîte fixe 13 m** centrée (`revpos_XXXX`,
  faute de polygone OSM) ; `skip`/`non_revu` → ignorés.
- Les chips de revue s'**ajoutent** au jeu OSM existant (187 citernes + 300
  piscines + 120 fonds) ; tout est régénéré (reproductible depuis OSM +
  verdicts). Bilan : ~199 positifs / 521 négatifs (ratio ~1:2,6).
- **Mesure du gain = re-inférence sur Tours Métropole** + croisement avec les
  verdicts connus (pas seulement le mAP de test).
- On **conserve** les négatifs génériques existants (piscines + fonds) en plus
  des 101 durs, pour ne pas sur-ajuster aux 101 points précis.

## Architecture — pipeline

```
verdicts.csv ─┐
OSM (citernes, │  [A] build_dataset --verdicts  ──►  [B] train.py (YOLOv8n, CPU)  ──►  best.pt (itér.2)
piscines) ─────┘      (dataset augmenté)                                                    │
                                                                                            ▼
[D] compare_to_verdicts.py   ◄──  [C] infer_area.py sur Tours Métropole (nouveaux poids)
    (FP supprimés / TP gardés /       (re-inférence, run différé)
     candidats avant-après)
```

## Composants

### A. Augmentation du dataset (`detection_ortho/dataset.py` + `scripts/build_dataset.py`, extension)

- Nouvelle fonction pure `parse_verdicts(rows) -> list[dict]` : depuis les lignes
  du CSV (`index,lat,lon,score,verdict`), retourne des enregistrements
  `{"lon","lat","verdict"}` en ignorant `skip`/`non_revu`/en-tête.
- `build_dataset.py` gagne `--verdicts <csv>` : pour chaque verdict, `faux` →
  record négatif (`hardneg_i`, bbox_geo=None) ; `vrai` → record positif
  (`revpos_i`) avec `fixed_box_geo(lon, lat, DEFAULT_BOX_M)`. Ajoutés à la liste
  `records` avant le split. Réutilise `assemble_window`/`write_chip`.

### B. Réentraînement (`scripts/train.py`, inchangé)

- YOLOv8n depuis COCO sur le dataset augmenté, local CPU (~3-4 h). Sortie
  `best.pt` (itér. 2), rangé dans `runs/citernes*/weights/`.

### C. Re-inférence (`scripts/infer_area.py`, inchangé)

- Relancer sur Tours Métropole avec les nouveaux poids → nouveau
  `detected_only.geojson` (et les autres livrables). Run différé (~1 soirée).

### D. Comparaison avant/après (`detection_ortho/evalcompare.py` + `scripts/compare_to_verdicts.py`, nouveau)

- Fonction pure `compare_to_verdicts(detections, verdicts, radius_m) -> dict` :
  croise les nouvelles détections avec les verdicts connus (par proximité,
  `haversine_m`) et retourne :
  - `fp_suppressed` / `fp_total` : parmi les 101 FP connus, combien ne sont plus
    détectés (↑ = mieux) ;
  - `tp_kept` / `tp_total` : parmi les 12 TP connus, combien sont toujours
    détectés (doit rester haut) ;
  - `n_candidates_new` : nombre total de candidats du nouveau run.
- `scripts/compare_to_verdicts.py` : charge `detected_only.geojson` + `verdicts.csv`,
  appelle la fonction, imprime le rapport avant/après.

## Critère de succès

- Sur Tours Métropole (nouveaux poids) : **forte baisse des faux positifs**
  (majorité des 101 FP connus supprimés) tout en **conservant la plupart des 12
  TP connus** → précision nettement supérieure aux ~11 % de l'itération 1.

## Stack technique

- Réutilise tout l'existant : `dataset` (assemble_window, fixed_box_geo,
  write_chip, split), `geo.haversine_m`, `infer_area`, `train`, `evaluate`,
  `ultralytics`.

## Tests (hors-ligne)

- `parse_verdicts` : faux→négatif, vrai→positif, skip/non_revu/en-tête ignorés.
- `build_dataset` avec `--verdicts` : test d'intégration offline (fake fetch +
  cache pré-semé + petit CSV) → chips `hardneg_*` sans label, `revpos_*` avec
  label.
- `compare_to_verdicts` : FP supprimés / TP gardés sur un petit jeu synthétique.
- Les runs réels (régénération dataset, réentraînement, re-inférence) sont
  **différés** (manuels).

## Risques & mitigations

- **Sur-apprentissage aux 101 FP précis** → on garde les négatifs génériques
  (piscines + fonds) + augmentation Ultralytics ; 101 points variés sur 390 km².
- **12 positifs, c'est peu** → gain positif marginal ; le gros gain vient des
  négatifs durs. Boîte fixe 13 m acceptable pour si peu d'exemples.
- **mAP de test peu représentatif** → la mesure décisive est la re-inférence
  terrain (composant D), pas le mAP seul.
- **Dépendance au format `verdicts.csv`** → `parse_verdicts` tolère en-tête,
  colonnes en trop, verdicts inconnus (ignorés).
