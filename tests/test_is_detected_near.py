from detection_ortho.infer import is_detected_near


def test_detected_within_radius():
    pts = [{"lon": 0.6531, "lat": 47.3305, "score": 0.9}]
    assert is_detected_near(pts, 0.6531, 47.3305, 25.0)


def test_not_detected_outside_radius():
    # ~150 m à l'est → hors du rayon 25 m
    pts = [{"lon": 0.6551, "lat": 47.3305, "score": 0.9}]
    assert not is_detected_near(pts, 0.6531, 47.3305, 25.0)


def test_empty_points():
    assert not is_detected_near([], 0.6531, 47.3305, 25.0)
