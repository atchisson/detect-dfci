from detection_ortho.osm import build_boundary_query, parse_relations


def test_query_sans_filtre_inchangee():
    q = build_boundary_query("Cher")
    assert 'relation["name"="Cher"]["boundary"];' in q
    assert "admin_level" not in q and "ref:INSEE" not in q


def test_query_avec_admin_level():
    q = build_boundary_query("Indre", admin_level=6)
    assert 'relation["name"="Indre"]["boundary"]["admin_level"="6"];' in q


def test_query_avec_insee():
    q = build_boundary_query("Indre", insee="36")
    assert 'relation["name"="Indre"]["boundary"]["ref:INSEE"="36"];' in q


def test_query_avec_les_deux_filtres():
    q = build_boundary_query("Indre", admin_level="6", insee="36")
    assert '["admin_level"="6"]["ref:INSEE"="36"]' in q


def _reponse_deux_homonymes():
    return {"elements": [
        {"type": "relation", "id": 7417,
         "tags": {"admin_level": "6", "ref:INSEE": "36", "boundary": "administrative"},
         "members": [{"type": "way", "geometry": [{"lon": 1.0, "lat": 46.8}]}]},
        {"type": "relation", "id": 65571,
         "tags": {"admin_level": "8", "ref:INSEE": "44074", "boundary": "administrative"},
         "members": [{"type": "way", "geometry": [{"lon": -1.6, "lat": 47.2}]}]},
    ]}


def test_parse_relations_separe_les_homonymes():
    rels = parse_relations(_reponse_deux_homonymes())
    assert [r["id"] for r in rels] == [7417, 65571]
    assert rels[0]["tags"]["ref:INSEE"] == "36"
    assert rels[0]["ways"] == [[{"lon": 1.0, "lat": 46.8}]]
    # Chaque relation garde ses ways : rien n'est fusionné.
    assert rels[1]["ways"] == [[{"lon": -1.6, "lat": 47.2}]]


def test_parse_relations_ignore_les_non_relations():
    data = {"elements": [{"type": "node", "id": 1},
                         {"type": "relation", "id": 2, "tags": {}, "members": []}]}
    rels = parse_relations(data)
    assert len(rels) == 1 and rels[0]["id"] == 2 and rels[0]["ways"] == []


def test_parse_relations_sur_reponse_vide():
    assert parse_relations({"elements": []}) == []
