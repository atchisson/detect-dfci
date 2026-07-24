from detection_ortho.geo import haversine_m, dedup_points


def test_haversine_known_distance():
    # ~1 degré de latitude ≈ 111 km.
    d = haversine_m(6.0, 43.0, 6.0, 44.0)
    assert 110_000 < d < 112_000


def test_haversine_zero():
    assert haversine_m(6.0, 43.0, 6.0, 43.0) == 0.0


def test_dedup_merges_close_points_keeps_best_score():
    pts = [
        {"lon": 6.0000, "lat": 43.0000, "score": 0.7},
        {"lon": 6.00002, "lat": 43.00001, "score": 0.9},  # ~2 m -> doublon
        {"lon": 6.0100, "lat": 43.0000, "score": 0.5},    # ~800 m -> distinct
    ]
    out = dedup_points(pts, radius_m=20)
    assert len(out) == 2
    kept = max(out, key=lambda p: p["score"])
    assert kept["score"] == 0.9


def test_dedup_empty():
    assert dedup_points([], radius_m=20) == []
