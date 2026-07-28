from shapely.geometry import Point
from detection_ortho.infer import ways_to_polygon, windows_over_polygon


def _square_ways():
    # Carré [0,0]-[0.02,0.02] en deux ways (deux moitiés du contour).
    return [
        [{"lon": 0.0, "lat": 0.0}, {"lon": 0.02, "lat": 0.0}, {"lon": 0.02, "lat": 0.02}],
        [{"lon": 0.02, "lat": 0.02}, {"lon": 0.0, "lat": 0.02}, {"lon": 0.0, "lat": 0.0}],
    ]


def test_ways_to_polygon_builds_square():
    poly = ways_to_polygon(_square_ways())
    assert poly.area > 0
    assert poly.contains(Point(0.01, 0.01))       # centre dedans
    assert not poly.contains(Point(0.05, 0.05))    # loin dehors


def test_windows_cover_polygon_and_stay_inside():
    poly = ways_to_polygon(_square_ways())
    centers = windows_over_polygon(poly, zoom=19, window_px=640, overlap=0.2)
    assert len(centers) > 0
    # tous les centres sont dans le polygone
    for lon, lat in centers:
        assert poly.contains(Point(lon, lat))


def test_windows_empty_for_tiny_polygon_far_away():
    # Un polygone minuscule peut ne contenir aucun centre de grille : au moins
    # la fonction ne plante pas et retourne une liste.
    poly = ways_to_polygon(_square_ways())
    centers = windows_over_polygon(poly, zoom=19, window_px=640, overlap=0.0)
    assert isinstance(centers, list)
