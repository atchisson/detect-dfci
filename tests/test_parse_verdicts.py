from detection_ortho.dataset import parse_verdicts


def test_parses_vrai_faux_ignores_rest():
    lines = [
        "index,lat,lon,score,verdict",       # en-tête -> ignoré
        "1,47.361519,0.524931,0.997,vrai",
        "2,47.300208,0.525744,0.992,faux",
        "3,47.10,0.50,0.60,skip",            # ignoré
        "4,47.11,0.51,0.55,non_revu",        # ignoré
        "malformée",                          # ignoré
    ]
    out = parse_verdicts(lines)
    assert len(out) == 2
    assert out[0] == {"lon": 0.524931, "lat": 47.361519, "verdict": "vrai"}
    assert out[1]["verdict"] == "faux"
    # ordre (lon, lat) correct : lon vient de la colonne 2, lat de la colonne 1
    assert out[1]["lon"] == 0.525744 and out[1]["lat"] == 47.300208


def test_empty():
    assert parse_verdicts([]) == []
