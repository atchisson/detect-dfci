from detection_ortho.dataset import dedup_verdicts

LAT = 47.0


def _v(lon, verdict="vrai", lat=LAT):
    return {"lon": lon, "lat": lat, "verdict": verdict}


def test_vrai_near_reference_is_dropped():
    # ~5 m d'écart
    kept, dropped = dedup_verdicts([_v(0.50006)], [(0.5, LAT)], radius_m=15)
    assert kept == [] and dropped == 1


def test_vrai_far_from_reference_is_kept():
    kept, dropped = dedup_verdicts([_v(0.51)], [(0.5, LAT)], radius_m=15)
    assert len(kept) == 1 and dropped == 0


def test_faux_on_reference_is_never_dropped():
    kept, dropped = dedup_verdicts([_v(0.5, "faux")], [(0.5, LAT)], radius_m=15)
    assert len(kept) == 1 and dropped == 0


def test_second_close_vrai_is_dropped_first_kept():
    kept, dropped = dedup_verdicts([_v(0.5), _v(0.50006)], [], radius_m=15)
    assert len(kept) == 1 and dropped == 1
    assert kept[0]["lon"] == 0.5


def test_duplicate_detected_across_grid_cell_boundary():
    # radius 15 m -> cellules de 0.00025° ; ces deux points (~4,6 m) sont de
    # part et d'autre de la frontière de cellule lon = 0.00025.
    kept, dropped = dedup_verdicts(
        [_v(0.00028)], [(0.00022, LAT)], radius_m=15)
    assert kept == [] and dropped == 1


def test_order_and_non_vrai_preserved():
    vs = [_v(0.3, "faux"), _v(0.4), _v(0.5, "faux")]
    kept, dropped = dedup_verdicts(vs, [], radius_m=15)
    assert kept == vs and dropped == 0
