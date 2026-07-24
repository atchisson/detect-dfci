from detection_ortho.osm import (
    build_overpass_query,
    parse_overpass_response,
    fetch_citernes,
    CITERNE_TAGS,
)


def test_query_contains_bbox_and_tags():
    q = build_overpass_query(6.14, 43.41, 6.16, 43.43)
    assert "[out:json]" in q
    assert "43.41,6.14,43.43,6.16" in q  # ordre Overpass : s,w,n,e
    assert '"emergency"="water_tank"' in q
    assert "out center;" in q


def test_parse_node_and_way_center():
    data = {
        "elements": [
            {"type": "node", "lon": 6.1, "lat": 43.4, "tags": {"emergency": "water_tank"}},
            {"type": "way", "center": {"lon": 6.2, "lat": 43.5}, "tags": {"man_made": "water_tank"}},
            {"type": "way", "tags": {"man_made": "water_tank"}},  # sans center -> ignoré
        ]
    }
    pts = parse_overpass_response(data)
    assert len(pts) == 2
    assert pts[0] == {"lon": 6.1, "lat": 43.4, "tags": {"emergency": "water_tank"}}
    assert pts[1]["lon"] == 6.2 and pts[1]["lat"] == 43.5


def test_fetch_citernes_uses_session(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"elements": [
                {"type": "node", "lon": 6.1, "lat": 43.4, "tags": {"emergency": "water_tank"}},
            ]}

    class FakeSession:
        def post(self, url, data, timeout=90):
            return FakeResp()

    pts = fetch_citernes(6.14, 43.41, 6.16, 43.43, session=FakeSession())
    assert len(pts) == 1
    assert CITERNE_TAGS  # non vide
