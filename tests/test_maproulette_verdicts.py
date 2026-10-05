from detection_ortho.dataset import parse_verdicts
from detection_ortho.maproulette import (
    dept_from_challenge_name, task_to_row, rows_to_csv,
)


def _task(status=1, coords=(0.65, 47.33), score="0.91"):
    t = {
        "status": status,
        "location": {"type": "Point", "coordinates": list(coords)},
        "geometries": {"features": [{"properties": {"score": score}}]},
    }
    return t


def test_dept_from_challenge_name():
    assert dept_from_challenge_name("DECI 44") == "44"
    assert dept_from_challenge_name("DECI 2A") == "2A"
    assert dept_from_challenge_name("deci 37") == "37"
    assert dept_from_challenge_name("Autre challenge") is None
    assert dept_from_challenge_name(None) is None


def test_task_to_row_fixed_is_vrai_lonlat_order():
    row = task_to_row(_task(status=1, coords=(0.65, 47.33), score="0.91"))
    assert row == {"lon": 0.65, "lat": 47.33, "score": 0.91, "verdict": "vrai"}


def test_task_to_row_not_an_issue_is_faux():
    assert task_to_row(_task(status=2))["verdict"] == "faux"


def test_task_to_row_ignores_other_statuses():
    for status in (0, 3, 4, 5, 6, 9):
        assert task_to_row(_task(status=status)) is None


def test_task_to_row_missing_or_bad_location_is_skipped():
    t = _task()
    del t["location"]
    assert task_to_row(t) is None
    assert task_to_row(_task(coords=("x", "y"))) is None


def test_task_to_row_bad_score_defaults_to_zero():
    t = _task(score="abc")
    assert task_to_row(t)["score"] == 0.0
    t2 = _task()
    t2["geometries"] = {"features": []}
    assert task_to_row(t2)["score"] == 0.0


def test_rows_to_csv_roundtrips_through_parse_verdicts():
    rows = [
        {"lon": 0.65, "lat": 47.33, "score": 0.91, "verdict": "vrai"},
        {"lon": 1.1, "lat": 46.9, "score": 0.5, "verdict": "faux"},
    ]
    text = rows_to_csv(rows)
    assert text.splitlines()[0] == "index,lat,lon,score,verdict"
    parsed = parse_verdicts(text.splitlines())
    assert parsed == [
        {"lon": 0.65, "lat": 47.33, "verdict": "vrai"},
        {"lon": 1.1, "lat": 46.9, "verdict": "faux"},
    ]


def test_rows_to_csv_empty_has_only_header():
    assert rows_to_csv([]) == "index,lat,lon,score,verdict\n"
