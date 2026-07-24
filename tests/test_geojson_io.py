import json
from detection_ortho.geojson_io import points_to_geojson, write_geojson


def test_points_to_geojson_structure():
    pts = [{"lon": 6.1, "lat": 43.4, "score": 0.9}]
    fc = points_to_geojson(pts)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    feat = fc["features"][0]
    assert feat["geometry"] == {"type": "Point", "coordinates": [6.1, 43.4]}
    assert feat["properties"]["score"] == 0.9


def test_write_and_reload(tmp_path):
    fc = points_to_geojson([{"lon": 6.1, "lat": 43.4}])
    p = tmp_path / "out.geojson"
    write_geojson(fc, p)
    reloaded = json.loads(p.read_text(encoding="utf-8"))
    assert reloaded["features"][0]["geometry"]["coordinates"] == [6.1, 43.4]
