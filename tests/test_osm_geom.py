from detection_ortho.osm import (
    build_geom_query,
    parse_geom_response,
    fetch_features_geom,
)


def test_geom_query_has_selectors_bbox_and_out_geom():
    q = build_geom_query([("emergency", "water_tank")], 0.05, 46.72, 1.06, 47.72)
    assert "[out:json]" in q
    assert '"emergency"="water_tank"' in q
    assert "46.72,0.05,47.72,1.06" in q  # s,w,n,e
    assert "out geom;" in q


def test_parse_geom_keeps_node_and_way_geometry():
    data = {
        "elements": [
            {"type": "node", "lon": 0.65, "lat": 47.33, "tags": {"emergency": "water_tank"}},
            {"type": "way", "tags": {"emergency": "water_tank"},
             "geometry": [{"lon": 0.6, "lat": 47.3}, {"lon": 0.6009, "lat": 47.3007}]},
            {"type": "way", "tags": {"x": "y"}},  # sans geometry -> ignoré
        ]
    }
    out = parse_geom_response(data)
    assert len(out) == 2
    assert out[0]["type"] == "node" and out[0]["lon"] == 0.65
    assert out[1]["type"] == "way" and len(out[1]["geometry"]) == 2


def test_fetch_features_geom_uses_session():
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"elements": [
                {"type": "node", "lon": 0.65, "lat": 47.33, "tags": {}},
            ]}

    class FakeSession:
        def post(self, url, data, headers=None, timeout=90):
            assert headers and "User-Agent" in headers
            return FakeResp()

    out = fetch_features_geom([("leisure", "swimming_pool")],
                              0.05, 46.72, 1.06, 47.72, session=FakeSession())
    assert len(out) == 1
