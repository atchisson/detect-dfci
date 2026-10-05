from detection_ortho.dataset import near_any

LAT = 47.0


def test_near_and_far():
    refs = [(0.5, LAT)]
    pts = [(0.50006, LAT), (0.51, LAT)]  # ~4,5 m puis ~760 m
    assert near_any(pts, refs, 100) == [True, False]


def test_empty_refs_and_empty_points():
    assert near_any([(0.5, LAT)], [], 100) == [False]
    assert near_any([], [(0.5, LAT)], 100) == []


def test_radius_matters():
    refs = [(0.5, LAT)]
    pts = [(0.5 + 0.0015, LAT)]  # ~114 m
    assert near_any(pts, refs, 100) == [False]
    assert near_any(pts, refs, 150) == [True]


def test_across_grid_cell_boundary():
    radius = 100
    cell = radius / 60000.0
    b = round(0.5 / cell) * cell  # frontière exacte de cellule
    refs = [(b - 1e-6, LAT)]
    pts = [(b + 1e-6, LAT)]  # ~0,15 m, mais dans la cellule voisine
    assert near_any(pts, refs, radius) == [True]
