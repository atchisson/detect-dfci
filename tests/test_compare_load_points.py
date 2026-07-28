import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import compare_to_verdicts as cmp  # noqa: E402


def test_load_points_reads_lon_lat_and_skips_bad(tmp_path):
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0.65, 47.33]},
         "properties": {}},
        {"type": "Feature", "geometry": None, "properties": {}},          # skipped
        {"type": "Feature", "properties": {}},                            # no geometry -> skipped
    ]}
    p = tmp_path / "d.geojson"
    p.write_text(json.dumps(fc), encoding="utf-8")
    pts = cmp._load_points(p)
    assert pts == [{"lon": 0.65, "lat": 47.33}]
