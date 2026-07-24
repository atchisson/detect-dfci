from detection_ortho.compare import match_detections


def test_three_categories():
    dets = [
        {"lon": 6.0000, "lat": 43.0000, "score": 0.9},   # colle à osm A
        {"lon": 6.0500, "lat": 43.0000, "score": 0.8},   # nouveau (detected_only)
    ]
    osm = [
        {"lon": 6.00001, "lat": 43.00000, "tags": {"emergency": "water_tank"}},  # A
        {"lon": 6.2000, "lat": 43.0000, "tags": {"man_made": "water_tank"}},     # osm_only
    ]
    res = match_detections(dets, osm, radius_m=25)
    assert len(res["matched"]) == 1
    assert len(res["detected_only"]) == 1
    assert len(res["osm_only"]) == 1
    assert res["detected_only"][0]["lon"] == 6.05


def test_one_osm_point_matched_once():
    # Deux détections proches du même point OSM : une seule s'apparie.
    dets = [
        {"lon": 6.00000, "lat": 43.0, "score": 0.9},
        {"lon": 6.00003, "lat": 43.0, "score": 0.8},
    ]
    osm = [{"lon": 6.00001, "lat": 43.0, "tags": {}}]
    res = match_detections(dets, osm, radius_m=25)
    assert len(res["matched"]) == 1
    assert len(res["osm_only"]) == 0
    assert len(res["detected_only"]) == 1


def test_empty_inputs():
    res = match_detections([], [], radius_m=25)
    assert res == {"matched": [], "detected_only": [], "osm_only": []}
