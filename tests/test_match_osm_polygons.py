import math

from detection_ortho.dataset import match_osm_polygons

LON, LAT = 0.65, 47.33
M = 111320.0


def _rect_way(lon, lat, w_m, h_m, tags=None):
    dlat = h_m / 2 / M
    dlon = w_m / 2 / (M * math.cos(math.radians(lat)))
    pts = [(lon - dlon, lat - dlat), (lon + dlon, lat - dlat),
           (lon + dlon, lat + dlat), (lon - dlon, lat + dlat), (lon - dlon, lat - dlat)]
    return {"type": "way", "tags": tags or {},
            "geometry": [{"lon": x, "lat": y} for x, y in pts]}


def _bounds(el):
    xs = [p["lon"] for p in el["geometry"]]
    ys = [p["lat"] for p in el["geometry"]]
    return min(xs), min(ys), max(xs), max(ys)


def test_returns_bounds_of_the_close_polygon():
    way = _rect_way(LON, LAT, 20, 8)
    assert match_osm_polygons([{"lon": LON, "lat": LAT}], [way]) == [_bounds(way)]


def test_nearest_polygon_wins():
    near = _rect_way(LON + 0.00003, LAT, 12, 12)   # ~2 m
    far = _rect_way(LON + 0.00010, LAT, 12, 12)    # ~7 m
    out = match_osm_polygons([{"lon": LON, "lat": LAT}], [far, near])
    assert out == [_bounds(near)]


def test_node_and_malformed_elements_are_ignored():
    node = {"type": "node", "lon": LON, "lat": LAT, "tags": {}}
    no_geom = {"type": "way", "tags": {}}
    two_vertices = {"type": "way", "tags": {},
                    "geometry": [{"lon": LON, "lat": LAT}, {"lon": LON + 0.0001, "lat": LAT}]}
    assert match_osm_polygons([{"lon": LON, "lat": LAT}], [node, no_geom, two_vertices]) == [None]


def test_polygon_beyond_radius_is_ignored():
    way = _rect_way(LON + 0.0005, LAT, 12, 12)     # ~37 m
    assert match_osm_polygons([{"lon": LON, "lat": LAT}], [way]) == [None]


def test_implausible_sizes_fall_back_to_none():
    too_big = _rect_way(LON, LAT, 60, 30)
    too_small = _rect_way(LON, LAT, 2, 1.5)
    assert match_osm_polygons([{"lon": LON, "lat": LAT}], [too_big]) == [None]
    assert match_osm_polygons([{"lon": LON, "lat": LAT}], [too_small]) == [None]


def test_empty_inputs():
    assert match_osm_polygons([], []) == []
    assert match_osm_polygons([{"lon": LON, "lat": LAT}], []) == [None]


def test_one_result_per_point_in_order():
    a = _rect_way(LON, LAT, 12, 12)
    pts = [{"lon": LON, "lat": LAT}, {"lon": LON + 0.01, "lat": LAT}]
    out = match_osm_polygons(pts, [a])
    assert out == [_bounds(a), None]
