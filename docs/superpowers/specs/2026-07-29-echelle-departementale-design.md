# Échelle départementale — BD ORTHO locale + GPU : Design

**Date :** 2026-07-29
**Statut :** validé (brainstorming)
**Prérequis :** Jalons 0-3 + itération 2 livrés. Modèle : `runs/citernes/weights/best.pt`.

## Objectif

Faire tourner le détecteur sur **tout le département de l'Indre-et-Loire (37)**,
en lisant l'ortho **en local** (BD ORTHO + rasterio) et en inférant sur le **GPU
P620**, pour produire une liste de citernes candidates absentes d'OSM + un
challenge MapRoulette départemental — sans dépendre de millions de requêtes WMTS.

## Contexte

- La métropole (~390 km², ~147k tuiles WMTS, ~36k fenêtres) est passée en WMTS.
- Le département (~6100 km², ~3 M tuiles z19, **~570k fenêtres**) rend le
  téléchargement tuile-par-tuile impraticable (3 M requêtes, 3 M fichiers).
- Disque disponible : ~334 Go libres (C:) → la BD ORTHO 37 (~30-60 Go) tient
  largement.

## Décisions issues du brainstorming

- **Acquisition = BD ORTHO locale + rasterio.** Téléchargement manuel unique des
  dalles JP2 depuis geoservices.ign.fr ; lecture de fenêtres en local, zéro
  réseau ensuite.
- **Cohérence avec l'entraînement via reprojection.** Le modèle a été entraîné
  sur des tuiles WMTS **Web Mercator (EPSG:3857)** à ~30 cm/px ; la BD ORTHO est
  en **Lambert-93 (EPSG:2154)** à 20 cm/px. On lit donc via un **`WarpedVRT`
  reprojeté en EPSG:3857** et on extrait **exactement la même fenêtre
  géographique** que le chemin WMTS (mêmes bornes, rééchantillonnées en 640 px)
  → imagettes pixel-cohérentes avec l'entraînement.
- **Calcul = GPU P620** (`--device 0`), install PyTorch CUDA documentée, **repli
  CPU** (`--device cpu`). Inférence YOLOv8n 640 px batch 1 tient dans 2 Go.
- **Fenêtrage accéléré** : `windows_over_polygon` utilise une géométrie préparée
  (`shapely.prepared.prep`) pour le point-dans-polygone à grande échelle.
- **Emprise** = relation OSM « Indre-et-Loire » (via `fetch_relation_ways`).
- Seuil conseillé : **conf 0,55** (issu de l'analyse itération 2).

## Architecture — pipeline

```
BD ORTHO locale (dalles JP2, EPSG:2154)
        │  WarpedVRT -> EPSG:3857
        ▼
[A] local_ortho.read_window(lon,lat,z,640) ─► [B] YOLO GPU (P620, --device 0) ─► [C] boxes_to_points + dedup
   (mêmes bornes que le chemin WMTS)                                                     │
                                                                                         ▼
[F] MapRoulette + overlay  ◄──  [E] comparaison OSM  ◄──────────────────────────  [D] detections.geojson
```

Seule la brique d'acquisition ([A]) change ; [B]-[F] réutilisent le Jalon 3.

## Composants

### A. Lecture locale de l'ortho (`detection_ortho/local_ortho.py`, nouveau)

- `open_ortho(ortho_dir) -> WarpedVRT` : construit un mosaïque (VRT/`rasterio`
  sur les dalles du dossier) reprojetée en EPSG:3857.
- `read_window(vrt, lon, lat, zoom, window_px, tile_size=256) -> (np.ndarray, float, float)`
  : calcule les bornes EPSG:3857 de la fenêtre centrée (mêmes maths que
  `lonlat_to_global_px` : origine = pixel global du coin haut-gauche = centre −
  window_px/2), lit ce rectangle via le VRT rééchantillonné en `window_px`×
  `window_px`, retourne **(image BGR uint8, origin_gx, origin_gy)** — **même
  contrat que `dataset.assemble_window`** (donc `boxes_to_points` fonctionne à
  l'identique). Conversion RGB→BGR pour coller au format d'entraînement (chips
  écrits via cv2).

### B/C/D/E/F. Inférence & aval (`scripts/infer_area.py`, extension)

- Option **`--ortho-dir <dossier>`** : si fournie, l'inférence lit via
  `local_ortho.read_window` au lieu du WMTS `assemble_window` (et **saute** le
  pré-téléchargement des tuiles). Sinon, comportement WMTS actuel inchangé.
- `--device 0` pour la P620. Le reste (dedup, comparaison OSM, MapRoulette,
  overlay) est identique au Jalon 3.

### Perf du fenêtrage (`detection_ortho/infer.py`, modif)

- `windows_over_polygon` utilise `shapely.prepared.prep(polygon)` pour les tests
  `contains` — nécessaire à l'échelle départementale (grille ~1-2 M points).

### Dépendances & install

- `rasterio` ajouté à `requirements.txt`.
- PyTorch CUDA : commande d'install documentée (README), séparée de l'install
  CPU par défaut ; vérif `torch.cuda.is_available()`.

## Livrables

- `detections/matched/detected_only/osm_only.geojson`, `maproulette_challenge.geojson`,
  `overlay.png` (dans un dossier de sortie) + carte de revue via `make_map.py`.
- Le tout produit en **lecture 100 % locale**, en quelques heures sur la P620.

## Critère de succès

Inférence complète sur l'Indre-et-Loire aboutie (lecture locale, GPU), produisant
une liste de candidats et un `maproulette_challenge.geojson` exploitables.

## Tests (hors-ligne)

- `read_window` sur un **petit raster synthétique** créé avec rasterio dans le
  test (georéférencé, EPSG:3857 ou 2154) : vérifie la taille de sortie
  (window_px²), le contrat de retour `(img, ogx, ogy)` cohérent avec
  `lonlat_to_global_px`, et la lecture d'un motif connu. **Aucune vraie BD ORTHO,
  aucun réseau.**
- `windows_over_polygon` avec `prep()` : mêmes assertions qu'avant (comportement
  inchangé, seule la perf change).
- Le **run réel** (téléchargement BD ORTHO + inférence GPU) est **différé**
  (manuel).

## Risques & mitigations

- **Décalage projection/résolution** (Lambert-93 20 cm vs Web Mercator 30 cm) →
  `WarpedVRT` EPSG:3857 + mêmes bornes que l'entraînement ; c'est le point de
  correction central.
- **Setup CUDA P620** (driver, wheels Pascal) → repli `--device cpu` documenté ;
  vérif `torch.cuda.is_available()` avant lancement.
- **Perf point-dans-polygone** à l'échelle → `prep()`.
- **Requête OSM départementale** (504 déjà vus) → retry (déjà en place).
- **Couverture des dalles** (trous, bord de département) → `read_window` tolère
  les zones sans données (fenêtre noire → aucune détection, non bloquant).
- **Format BGR/RGB** → conversion explicite pour coller à l'entraînement.
