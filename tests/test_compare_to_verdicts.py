from detection_ortho.compare import compare_to_verdicts


def test_counts_fp_suppressed_and_tp_kept():
    verdicts = [
        {"lon": 0.10, "lat": 47.0, "verdict": "faux"},   # encore détecté
        {"lon": 0.20, "lat": 47.0, "verdict": "faux"},   # supprimé
        {"lon": 0.30, "lat": 47.0, "verdict": "vrai"},   # conservé
        {"lon": 0.40, "lat": 47.0, "verdict": "vrai"},   # perdu
    ]
    detections = [
        {"lon": 0.100001, "lat": 47.0, "score": 0.9},    # proche du faux #1
        {"lon": 0.300001, "lat": 47.0, "score": 0.8},    # proche du vrai #3
        {"lon": 0.90, "lat": 47.0, "score": 0.7},        # ailleurs
    ]
    r = compare_to_verdicts(detections, verdicts, radius_m=25)
    assert r["fp_total"] == 2
    assert r["fp_still_detected"] == 1
    assert r["fp_suppressed"] == 1
    assert r["tp_total"] == 2
    assert r["tp_kept"] == 1
    assert r["n_candidates_new"] == 3


def test_empty():
    r = compare_to_verdicts([], [], radius_m=25)
    assert r["fp_total"] == 0 and r["tp_total"] == 0 and r["n_candidates_new"] == 0
