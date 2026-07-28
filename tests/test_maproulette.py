# tests/test_maproulette.py
from detection_ortho.maproulette import to_maproulette_tasks


def test_tasks_structure():
    pts = [{"lon": 0.65, "lat": 47.33, "score": 0.82}]
    fc = to_maproulette_tasks(pts, "Vérifiez cette citerne.")
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    feat = fc["features"][0]
    assert feat["geometry"] == {"type": "Point", "coordinates": [0.65, 47.33]}
    assert feat["properties"]["instruction"] == "Vérifiez cette citerne."
    assert feat["properties"]["score"] == 0.82


def test_empty_points():
    fc = to_maproulette_tasks([], "x")
    assert fc["features"] == []
