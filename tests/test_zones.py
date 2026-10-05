import pytest
import requests
from shapely.geometry import Point, Polygon, box

from detection_ortho import zones
from detection_ortho.zones import (
    ZICAD_FILE, ZICAD_URL, ZIPTV_FILE, ZIPTV_URL, apply_zone_filter,
    filter_windows, load_zones, parse_geojson_zones, parse_kml_zones,
    window_footprint,
)

ZOOM, WINDOW = 19, 640

GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {}, "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}},
        {"type": "Feature", "properties": {}, "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [[[2, 2], [2, 3], [3, 3], [3, 2], [2, 2]]],
                [[[4, 4], [4, 5], [5, 5], [5, 4], [4, 4]]]]}},
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Point", "coordinates": [9, 9]}},
        {"type": "Feature", "properties": {}, "geometry": None},
    ],
}

KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<Placemark><MultiGeometry>
<Polygon><outerBoundaryIs><LinearRing><coordinates>
   0,0,0   0,1,0
   1,1,0
   1,0,0 0,0,0
</coordinates></LinearRing></outerBoundaryIs></Polygon>
<Polygon><outerBoundaryIs><LinearRing><coordinates>2,2 2,3 3,3 3,2 2,2</coordinates></LinearRing></outerBoundaryIs></Polygon>
</MultiGeometry></Placemark>
<Placemark><Point><coordinates>9,9,0</coordinates></Point></Placemark>
</Document></kml>"""


# --- parseurs -------------------------------------------------------------

def test_parse_geojson_polygon_multipolygon_et_ignore_point_et_null():
    polys = parse_geojson_zones(GEOJSON)
    assert len(polys) == 3  # 1 Polygon + 2 du MultiPolygon, Point/null ignorés
    assert all(p.area == pytest.approx(1.0) for p in polys)


def test_parse_geojson_sans_features():
    assert parse_geojson_zones({"type": "FeatureCollection"}) == []


def test_parse_kml_multigeometry_altitude_et_retours_a_la_ligne():
    polys = parse_kml_zones(KML)
    assert len(polys) == 2  # les deux Polygon du MultiGeometry, Point ignoré
    assert all(p.area == pytest.approx(1.0) for p in polys)


def test_parse_kml_accepte_les_octets():
    assert len(parse_kml_zones(KML.encode("utf-8"))) == 2


# --- emprise et filtre ------------------------------------------------------

def test_window_footprint_taille_et_centrage():
    fp = window_footprint(2.0, 47.0, ZOOM, WINDOW)
    west, south, east, north = fp.bounds
    largeur = east - west
    assert largeur == pytest.approx(640 * 360 / (256 * 2**19), rel=1e-4)
    assert (west + east) / 2 == pytest.approx(2.0, abs=1e-6)
    assert (south + north) / 2 == pytest.approx(47.0, abs=1e-4)
    assert fp.contains(Point(2.0, 47.0))


def test_filter_windows_dedans_bord_et_loin():
    zone = box(0, 0, 1, 1)
    centres = [(0.5, 0.5),      # dedans
               (1.0005, 0.5),   # emprise qui déborde sur le bord
               (1.01, 0.5)]     # loin
    gardees, n = filter_windows(centres, zone, ZOOM, WINDOW)
    assert gardees == [(1.01, 0.5)]
    assert n == 2


def test_filter_windows_zones_vides_ou_pas_de_centres():
    assert filter_windows([(0.5, 0.5)], Polygon(), ZOOM, WINDOW) == ([(0.5, 0.5)], 0)
    assert filter_windows([], box(0, 0, 1, 1), ZOOM, WINDOW) == ([], 0)


def test_filter_windows_toutes_ecartees():
    gardees, n = filter_windows([(0.5, 0.5), (0.6, 0.6)], box(0, 0, 1, 1),
                                ZOOM, WINDOW)
    assert gardees == [] and n == 2


# --- chargement et cache ------------------------------------------------------

class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.appels = []

    def get(self, url, timeout=None):
        self.appels.append(url)
        return FakeResponse(self.routes[url])


class DeadSession:
    def get(self, url, timeout=None):
        raise requests.ConnectionError("réseau coupé")


def _routes():
    import json
    return {ZIPTV_URL: json.dumps(GEOJSON).encode("utf-8"),
            ZICAD_URL: KML.encode("utf-8")}


def test_load_zones_telecharge_met_en_cache_et_fusionne(tmp_path):
    s = FakeSession(_routes())
    z = load_zones(tmp_path, session=s)
    assert (tmp_path / ZIPTV_FILE).exists() and (tmp_path / ZICAD_FILE).exists()
    assert len(s.appels) == 2
    # 3 polygones ZIPTV + 2 ZICAD ; (0-1) et (2-3) en double, fusionnés
    assert z.area == pytest.approx(3.0)  # carrés (0-1), (2-3), (4-5) ; doublons fusionnés


def test_load_zones_relit_le_cache_sans_reseau(tmp_path):
    load_zones(tmp_path, session=FakeSession(_routes()))
    z = load_zones(tmp_path, session=DeadSession())
    assert z.area == pytest.approx(3.0)


def test_load_zones_refresh_retelecharge(tmp_path):
    load_zones(tmp_path, session=FakeSession(_routes()))
    s = FakeSession(_routes())
    load_zones(tmp_path, refresh=True, session=s)
    assert len(s.appels) == 2


def test_load_zones_refresh_en_echec_retombe_sur_le_cache(tmp_path, capsys):
    load_zones(tmp_path, session=FakeSession(_routes()))
    z = load_zones(tmp_path, refresh=True, session=DeadSession())
    assert z.area == pytest.approx(3.0)
    assert "cache" in capsys.readouterr().err


def test_load_zones_reseau_ko_sans_cache_leve(tmp_path):
    with pytest.raises(RuntimeError, match="zones"):
        load_zones(tmp_path, session=DeadSession())


def test_load_zones_repare_un_polygone_invalide(tmp_path):
    import json
    noeud = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {}, "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}}]}
    routes = {ZIPTV_URL: json.dumps(noeud).encode("utf-8"), ZICAD_URL: KML.encode("utf-8")}
    z = load_zones(tmp_path, session=FakeSession(routes))  # ne lève pas
    assert z.is_valid


def test_apply_zone_filter_de_bout_en_bout(tmp_path):
    s = FakeSession(_routes())
    gardees, n = apply_zone_filter([(0.5, 0.5), (50.0, 10.0)], tmp_path,
                                   ZOOM, WINDOW, session=s)
    assert gardees == [(50.0, 10.0)] and n == 1


def test_apply_zone_filter_liste_vide(tmp_path):
    assert apply_zone_filter([], tmp_path, ZOOM, WINDOW,
                             session=FakeSession(_routes())) == ([], 0)
