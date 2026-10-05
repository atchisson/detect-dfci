# Exclusion des zones interdites (ZIPTV / ZICAD) de l'inférence — conception

Date : 2026-10-05

## Objectif

L'ortho IGN floute les sites militaires ou sensibles. Ces zones produisent des
faux positifs revus pour rien sur MapRoulette. On **n'infère pas** sur les
fenêtres qui touchent ces zones : on économise le téléchargement WMTS et le
calcul, et on supprime la source de faux positifs.

Critère de succès : sur un run `infer_area.py`, les fenêtres dont l'emprise
intersecte une zone ZIPTV ou ZICAD ne sont ni téléchargées ni inférées, le bilan
est affiché, et le reste du comportement est inchangé.

## Hors périmètre

- Pas de re-filtrage des anciens résultats ni de `build_dataset` (les faux
  positifs déjà revus restent utilisables en entraînement).
- Pas d'étiquetage ni de sortie séparée des détections en zone : elles ne sont
  simplement pas calculées.

## Sources de données

| Jeu | Format | URL | Licence |
|---|---|---|---|
| ZIPTV (arrêté 2018, 289 zones) | GeoJSON | `https://static.data.gouv.fr/resources/zones-interdites-a-la-prise-de-vue-aerienne/20181007-134434/2017-10.geojson` | Licence Ouverte |
| ZICAD (arrêté 2023, version 10-2024) | KML | `https://data.geopf.fr/annexes/ressources/documentation/Arrete_ZICAD_10-2024.kml` | non précisée |

ZIPTV est ancien ; ZICAD est la liste en vigueur et la plus à jour. Les deux sont
fusionnés : l'union est conservative (on écarte plus, jamais moins).
La licence de ZICAD n'étant pas précisée, les fichiers sont **téléchargés à
l'exécution et mis en cache localement, jamais versionnés** (`data/zones/`,
ajouté au `.gitignore`).

## Composants

### `detection_ortho/zones.py` (nouveau)

- `load_zones(cache_dir, refresh=False, session=None) -> BaseGeometry` :
  télécharge les deux fichiers si absents du cache (ou si `refresh`), les parse
  et renvoie l'union des polygones (WGS84). Échec réseau avec cache absent :
  `RuntimeError` explicite (on ne dégrade pas silencieusement en « aucune zone »).
- `parse_geojson_zones(data) -> list[Polygon]` et `parse_kml_zones(text) ->
  list[Polygon]` : fonctions pures. KML lu avec `xml.etree.ElementTree` (pas de
  nouvelle dépendance) ; on extrait les `Polygon/outerBoundaryIs/.../coordinates`
  (lon,lat[,alt]). Géométries `Point`/`LineString` ignorées ; les cercles
  éventuels sont déjà des polygones dans les fichiers.
- `window_footprint(lon, lat, zoom, window_px)` : polygone WGS84 de la fenêtre,
  construit avec `lonlat_to_global_px` / `global_px_to_lonlat` (dataset.py), donc
  cohérent avec `windows_over_polygon`.
- `filter_windows(centers, zones, zoom, window_px) -> (gardées, nb_écartées)` :
  écarte toute fenêtre dont l'emprise intersecte `zones` (géométrie préparée
  `shapely.prepared.prep`). Les polygones invalides sont réparés (`buffer(0)`).

### `scripts/infer_area.py`

- Options : `--skip-restricted-zones / --no-skip-restricted-zones`
  (`argparse.BooleanOptionalAction`, **actif par défaut**) et `--refresh-zones`.
- Juste après `windows_over_polygon` (centers), si actif : chargement des zones
  (cache `data/zones/`), filtrage, affichage du bilan
  (« N fenêtre(s) écartée(s) sur M (zones interdites) »).
- Si toutes les fenêtres sont écartées : sortie avec message clair.
- L'empreinte du point de reprise reçoit `skip_zones` (booléen) en plus des
  paramètres actuels. `centers` filtrés entrent déjà dans l'empreinte ; un
  checkpoint antérieur est donc refusé avec le message existant (`--restart`).

## Règle d'intersection

Une fenêtre qui touche seulement le bord d'une zone est écartée entière : on
perd les citernes voisines de cette fenêtre. Acceptable : fenêtres de 130-190 m
de côté et zones peu nombreuses. Une règle « centre dans la zone » ferait
inférer des fenêtres à moitié floutées, donc des faux positifs.

## Tests (TDD, sans réseau)

`tests/test_zones.py`, fixtures GeoJSON et KML minuscules :
- parse GeoJSON (Polygon, MultiPolygon) et KML (Polygon, ignorer Point) ;
- `window_footprint` : taille et centrage cohérents avec `windows_over_polygon` ;
- `filter_windows` : fenêtre dedans écartée, fenêtre touchant le bord écartée,
  fenêtre éloignée gardée, liste vide, zones vides ;
- `load_zones` : usage du cache sans réseau, `refresh`, erreur si réseau KO et
  cache absent (session factice) ;
- branchement `infer_area` : fonction d'aide testable séparément du `main`.

## Documentation

Section README « Zones interdites (ZIPTV / ZICAD) » : pourquoi, options, cache,
rafraîchissement, limite de la règle d'intersection.

## Exécution

Plan exécuté en subagent-driven (préférence établie).
