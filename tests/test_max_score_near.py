from detection_ortho.infer import max_score_near


def test_best_score_within_radius():
    pts = [
        {"lon": 0.6531, "lat": 47.3305, "score": 0.4},
        {"lon": 0.6531, "lat": 47.3305, "score": 0.8},
    ]
    assert max_score_near(pts, 0.6531, 47.3305, 25.0) == 0.8


def test_zero_when_none_near():
    pts = [{"lon": 0.6551, "lat": 47.3305, "score": 0.9}]  # ~150 m
    assert max_score_near(pts, 0.6531, 47.3305, 25.0) == 0.0


def test_zero_when_empty():
    assert max_score_near([], 0.6531, 47.3305, 25.0) == 0.0
