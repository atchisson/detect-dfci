import pytest

from detection_ortho.known_false import load_known_false, suppress_known_false


def _csv(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("index,lat,lon,score,verdict\n" + "".join(
        f"{i},{lat},{lon},0.5,{v}\n" for i, (lat, lon, v) in enumerate(rows, 1)),
        encoding="utf-8")
    return p


def test_load_keeps_only_the_faux_of_all_files(tmp_path):
    a = _csv(tmp_path, "a.csv", [(47.33, 0.65, "faux"), (47.34, 0.66, "vrai")])
    b = _csv(tmp_path, "b.csv", [(46.5, 1.5, "faux")])
    assert load_known_false([a, b]) == [(0.65, 47.33), (1.5, 46.5)]


def test_load_missing_file_fails_immediately(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_known_false([tmp_path / "absent.csv"])


def test_load_no_paths_is_empty():
    assert load_known_false([]) == []


def test_suppress_drops_candidates_near_a_known_false():
    pts = [{"lon": 0.65, "lat": 47.33}, {"lon": 0.7, "lat": 47.4}]
    kept, dropped = suppress_known_false(pts, [(0.65005, 47.33)], radius_m=25.0)   # ~4 m
    assert kept == [pts[1]] and dropped == [pts[0]]


def test_suppress_keeps_candidates_beyond_the_radius():
    pts = [{"lon": 0.65, "lat": 47.33}]
    kept, dropped = suppress_known_false(pts, [(0.651, 47.33)], radius_m=25.0)      # ~75 m
    assert kept == pts and dropped == []


def test_suppress_without_false_points_or_candidates_is_a_noop():
    pts = [{"lon": 0.65, "lat": 47.33}]
    assert suppress_known_false(pts, []) == (pts, [])
    assert suppress_known_false([], [(0.65, 47.33)]) == ([], [])


def test_suppress_keeps_the_extra_keys_of_the_points():
    pts = [{"lon": 0.7, "lat": 47.4, "score": 0.9}]
    kept, _ = suppress_known_false(pts, [(0.65, 47.33)])
    assert kept[0]["score"] == 0.9
