# Pré-reprojection COG — perf de la lecture locale : Design

**Date :** 2026-07-30
**Statut :** validé (brainstorming)
**Prérequis :** jalon échelle départementale livré (`local_ortho`, `--ortho`, `build_ortho_vrt`).

## Objectif

Éliminer le goulot de lecture (**2408 ms/fenêtre**, 96 % du temps) en
**pré-reprojetant l'ortho une seule fois** en un **GeoTIFF tuilé EPSG:3857
aligné sur la grille pixel z19**. `read_window` lit alors ce raster déjà en 3857
et tuilé → lectures **rapides** (ni reprojection ni décodage JP2 par fenêtre).

## Contexte (benchmark)

- Lecture JP2 + reprojection à la volée : **2408 ms/fenêtre** ; inférence CPU :
  87 ms. → département ~395 h (≈ 16 j) en l'état. **Le GPU n'aide pas** (il
  n'accélère que les 87 ms).
- Cause : `read_window` reprojette (2154→3857) et décode un JP2 25000×25000 en
  accès aléatoire **à chaque fenêtre**.

## Décisions issues du brainstorming

- **Pré-reprojection unique** en GeoTIFF tuilé 3857, aligné z19. Construit **avec
  rasterio seul** (ni `gdalbuildvrt` ni `osgeo`, absents).
- **`read_window` et `infer_area.py` INCHANGÉS** : on pointe simplement
  `--ortho ortho37_3857.tif`. Le raster étant déjà en 3857 sur la grille z19, le
  `WarpedVRT` de `open_ortho` devient un passage quasi-identité (rapide) et la
  lecture de tuiles GeoTIFF est rapide.
- **Compression JPEG** (YCbCr, q~85) : fichier petit (dizaines de Go), lectures
  rapides ; la source JP2 est déjà avec perte, le modèle n'a pas besoin du
  sans-perte. (`--compress deflate` dispo pour un besoin sans perte / les tests.)

## Composant

### `scripts/build_cog.py` (nouveau)

`--src ortho37.vrt --out ortho37_3857.tif [--compress jpeg|deflate] [--zoom 19]`

1. Ouvre la source, reprojette ses bornes en EPSG:3857 → plage de **pixels
   globaux** z19 `(gx_min..gx_max, gy_min..gy_max)` (via `mpp = 2πR/(256·2^z)`).
2. Crée un GeoTIFF de sortie : taille `(gx_max-gx_min)×(gy_max-gy_min)`,
   `transform = Affine(mpp, 0, -πR + gx_min·mpp, 0, -mpp, πR - gy_min·mpp)`,
   `crs=EPSG:3857`, **tuilé 512**, `BIGTIFF=YES`, compression choisie
   (JPEG/YCbCr par défaut).
3. Ouvre le `WarpedVRT` z19 via `local_ortho.open_ortho(src)` (grille monde 3857).
4. **Recopie bloc par bloc** : pour chaque tuile de sortie `(c, r)`, lit la
   fenêtre `Window(gx_min+c, gy_min+r, w, h)` du WarpedVRT et l'écrit dans la
   sortie. Le décodage JP2 + la reprojection ne se font **qu'une fois**,
   séquentiellement (efficace).
5. (Optionnel) overviews pour la visualisation.

Alignement : la sortie occupe exactement la grille pixel z19, donc les maths de
`read_window` (`lonlat_to_global_px`) mappent directement sur ses pixels.

## Utilisation (workflow mis à jour)

```
python scripts/build_ortho_vrt.py --dir <dalles> --out ortho37.vrt        # (déjà fait)
python scripts/build_cog.py --src ortho37.vrt --out ortho37_3857.tif       # UNE fois (~heures)
python scripts/infer_area.py --boundary "Indre-et-Loire" \
    --weights runs/citernes/weights/best.pt --ortho ortho37_3857.tif \
    --conf 0.55 --device cpu --out inference_dept37                        # rapide
```

## Critère de succès

Après pré-reprojection, `read_window` sur `ortho37_3857.tif` tombe de ~2400 ms à
**quelques dizaines de ms/fenêtre** → département faisable en **quelques heures
CPU**. Contenu identique (même géoréférencement, mêmes détections).

## Stack technique

- Réutilise `local_ortho.open_ortho` (WarpedVRT z19). rasterio pour l'écriture
  tuilée + compression. Aucun outil GDAL externe.

## Tests (hors-ligne)

- `build_cog` sur un **petit raster source synthétique EPSG:2154** (avec un
  marqueur localisé, `--compress deflate` pour un test sans perte) → produit un
  GeoTIFF 3857 tuilé ; puis `open_ortho(sortie)` + `read_window` au point du
  marqueur retrouve le marqueur au **pixel attendu** (round-trip complet).
- Le **build réel** (27 Go → GeoTIFF, plusieurs heures) et l'inférence
  départementale restent des **runs manuels différés**.

## Risques & mitigations

- **JPEG avec perte** → accepté (source déjà lossy ; `--compress deflate` en
  repli).
- **Temps de build** (reprojeter 27 Go une fois) → séquentiel/bloc, one-time ;
  bien plus rapide que 16 j d'inférence naïve.
- **Sortie > 4 Go** → `BIGTIFF=YES`.
- **JPEG + petites images de test** (contraintes YCbCr) → les tests utilisent
  `--compress deflate`.
- **Alignement grille z19** (le point critique) → test de round-trip au pixel ;
  mêmes maths `mpp`/origine que `open_ortho`.
