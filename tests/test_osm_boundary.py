from detection_ortho.osm import (
    build_boundary_query, parse_relation_ways, fetch_relation_ways,
)


def test_boundary_query_targets_relation_with_geom():
    q = build_boundary_query("Tours Métropole Val de Loire")
    assert 'relation' in q
    assert '"name"="Tours Métropole Val de Loire"' in q
    assert '"boundary"="administrative"' in q
    assert "out geom;" in q


def test_parse_relation_ways_extracts_member_geometries():
    data = {"elements": [{
        "type": "relation",
        "members": [
            {"type": "way", "role": "outer",
             "geometry": [{"lon": 0.0, "lat": 0.0}, {"lon": 1.0, "lat": 0.0}]},
            {"type": "way", "role": "outer",
             "geometry": [{"lon": 1.0, "lat": 0.0}, {"lon": 0.0, "lat": 0.0}]},
            {"type": "node", "role": "admin_centre", "lon": 0.5, "lat": 0.5},
        ],
    }]}
    ways = parse_relation_ways(data)
    assert len(ways) == 2
    assert ways[0][0] == {"lon": 0.0, "lat": 0.0}


def test_fetch_relation_ways_uses_session():
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"elements": [{"type": "relation", "members": [
                {"type": "way", "geometry": [{"lon": 0.0, "lat": 0.0}]},
            ]}]}

    class FakeSession:
        def post(self, url, data, headers=None, timeout=180):
            return FakeResp()

    ways = fetch_relation_ways("X", session=FakeSession())
    assert len(ways) == 1
