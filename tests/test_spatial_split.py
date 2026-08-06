from detection_ortho.dataset import spatial_split_indices


def test_cells_stay_whole():
    # 3 cellules bien séparées, 2 points chacune
    points = [
        (0.10, 47.10), (0.11, 47.11),   # cellule A (0,05°)
        (0.60, 47.60), (0.61, 47.61),   # cellule B
        (1.00, 47.00), (1.01, 47.01),   # cellule C
    ]
    split = spatial_split_indices(points, cell_deg=0.05, seed=0)
    # partition complète et disjointe
    allidx = sorted(split["train"] + split["val"] + split["test"])
    assert allidx == list(range(6))
    # les deux points d'une même cellule sont dans le même lot
    for a, b in [(0, 1), (2, 3), (4, 5)]:
        for part in ("train", "val", "test"):
            assert (a in split[part]) == (b in split[part])


def test_deterministic():
    points = [(0.1 * i, 47.0 + 0.1 * i) for i in range(20)]
    assert spatial_split_indices(points, seed=3) == spatial_split_indices(points, seed=3)


def test_ratios_approx():
    # 100 cellules distinctes, 1 point chacune → ~70/15/15
    points = [(0.1 * i, 47.0) for i in range(100)]
    split = spatial_split_indices(points, cell_deg=0.05, seed=0)
    assert 60 <= len(split["train"]) <= 80
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == 100
