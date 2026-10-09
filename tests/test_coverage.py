import threading

import cv2
import numpy as np

from detection_ortho.coverage import (
    PROBE_ZOOM, fetch_probe_tile, filter_windows_by_coverage, probe_tile_count,
    probe_tiles,
)
from detection_ortho.tiles import tile_for_lonlat

LAYER = "ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026"


def _jpeg(value):
    ok, buf = cv2.imencode(".jpg", np.full((64, 64, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


# Quatre centres répartis dans quatre tuiles de zoom 13 distinctes.
CENTERS = [(0.65, 47.33), (0.80, 47.33), (0.95, 47.33), (1.10, 47.33)]
KEYS = [tile_for_lonlat(lon, lat, PROBE_ZOOM) for lon, lat in CENTERS]


def test_the_four_centers_are_in_four_distinct_tiles():
    assert len(set(KEYS)) == 4


def test_filter_drops_empty_and_404_tiles_and_keeps_data_and_unknown():
    answers = {KEYS[0]: _jpeg(128),            # données
               KEYS[1]: _jpeg(255),            # tuile blanche : vide
               KEYS[2]: None,                  # 404 : vide
               KEYS[3]: RuntimeError("boom")}  # erreur réseau : indéterminé

    def fetch(x, y, zoom, layer):
        a = answers[(x, y)]
        if isinstance(a, Exception):
            raise a
        return a

    kept, dropped, unknown = filter_windows_by_coverage(CENTERS, LAYER, workers=2, fetch=fetch)
    assert kept == [CENTERS[0], CENTERS[3]]
    assert dropped == 2 and unknown == 1


def test_unreadable_content_is_unknown_and_never_dropped():
    def fetch(x, y, zoom, layer):
        return b"not a jpeg"

    kept, dropped, unknown = filter_windows_by_coverage(CENTERS, LAYER, fetch=fetch)
    assert kept == CENTERS and dropped == 0 and unknown == 4


def test_empty_centers():
    assert filter_windows_by_coverage([], LAYER, fetch=lambda *a: None) == ([], 0, 0)


def test_windows_sharing_a_tile_cost_a_single_probe():
    calls = []

    def fetch(x, y, zoom, layer):
        calls.append((x, y, zoom, layer))
        return _jpeg(128)

    centers = [(0.65, 47.33), (0.6501, 47.3301), (0.6502, 47.3302)]
    kept, _, _ = filter_windows_by_coverage(centers, LAYER, fetch=fetch)
    assert kept == centers
    assert len(calls) == 1 and calls[0][2] == PROBE_ZOOM and calls[0][3] == LAYER


def test_probing_is_parallel(monkeypatch):
    barrier = threading.Barrier(3, timeout=5)

    def fetch(x, y, zoom, layer):
        barrier.wait()          # ne passe que si 3 sondes sont en cours en même temps
        return _jpeg(128)

    status = probe_tiles([(1, 1), (2, 2), (3, 3)], LAYER, workers=3, fetch=fetch)
    assert list(status.values()) == [True, True, True]


def test_probe_writes_nothing_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    filter_windows_by_coverage(CENTERS, LAYER, fetch=lambda x, y, z, l: _jpeg(128))
    assert list(tmp_path.iterdir()) == []


class _Resp:
    def __init__(self, status, content=b"x"):
        self.status_code = status
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, status):
        self.status = status
        self.calls = 0

    def get(self, url, headers=None, timeout=30):
        self.calls += 1
        return _Resp(self.status)


def test_fetch_probe_tile_404_is_confirmed_once_before_returning_none():
    s = _Session(404)
    assert fetch_probe_tile(1, 2, 13, LAYER, session=s, pause=0) is None
    assert s.calls == 2


def test_fetch_probe_tile_transient_404_then_200_returns_the_content():
    class Seq(_Session):
        def get(self, url, headers=None, timeout=30):
            self.calls += 1
            return _Resp(404 if self.calls == 1 else 200)

    s = Seq(200)
    assert fetch_probe_tile(1, 2, 13, LAYER, session=s, pause=0) == b"x"
    assert s.calls == 2


def test_fetch_probe_tile_retries_then_raises_on_server_errors():
    s = _Session(500)
    try:
        fetch_probe_tile(1, 2, 13, LAYER, session=s, tries=2, pause=0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("une erreur 500 persistante doit lever")
    assert s.calls == 2


def test_fetch_probe_tile_returns_the_bytes_on_success():
    s = _Session(200)
    assert fetch_probe_tile(1, 2, 13, LAYER, session=s, pause=0) == b"x"


# --- Persistance de la sonde (reprise déterministe) ---

def _recording_fetch(answers):
    calls = []

    def fetch(x, y, zoom, layer):
        calls.append((x, y))
        return answers[(x, y)]

    fetch.calls = calls
    return fetch


def _full_answers():
    return {KEYS[0]: _jpeg(128), KEYS[1]: _jpeg(255), KEYS[2]: None, KEYS[3]: _jpeg(100)}


def test_probe_cache_is_reused_without_new_requests(tmp_path):
    path = tmp_path / "coverage.json"
    first = _recording_fetch(_full_answers())
    r1 = filter_windows_by_coverage(CENTERS, LAYER, fetch=first, cache_path=path)
    assert len(first.calls) == 4

    second = _recording_fetch({})
    r2 = filter_windows_by_coverage(CENTERS, LAYER, fetch=second, cache_path=path)
    assert r2 == r1 and second.calls == []


def test_probe_cache_probes_only_missing_keys(tmp_path):
    path = tmp_path / "coverage.json"
    filter_windows_by_coverage(CENTERS[:2], LAYER, fetch=_recording_fetch(_full_answers()),
                               cache_path=path)
    second = _recording_fetch(_full_answers())
    kept, dropped, unknown = filter_windows_by_coverage(CENTERS, LAYER, fetch=second,
                                                        cache_path=path)
    assert sorted(second.calls) == sorted(KEYS[2:])
    assert kept == [CENTERS[0], CENTERS[3]] and dropped == 2 and unknown == 0


def test_probe_cache_for_another_layer_is_ignored(tmp_path):
    path = tmp_path / "coverage.json"
    filter_windows_by_coverage(CENTERS, "AUTRE.COUCHE", fetch=_recording_fetch(_full_answers()),
                               cache_path=path)
    f = _recording_fetch(_full_answers())
    filter_windows_by_coverage(CENTERS, LAYER, fetch=f, cache_path=path)
    assert len(f.calls) == 4


def test_probe_cache_file_format_and_parent_folder(tmp_path):
    import json
    path = tmp_path / "sub" / "dir" / "coverage.json"
    filter_windows_by_coverage(CENTERS, LAYER, fetch=_recording_fetch(_full_answers()),
                               cache_path=path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["layer"] == LAYER and data["zoom"] == PROBE_ZOOM
    got = {(x, y): s for x, y, s in data["tiles"]}
    assert got == {KEYS[0]: True, KEYS[1]: False, KEYS[2]: False, KEYS[3]: True}
    assert list(path.parent.glob("*.tmp")) == []


def test_probe_cache_stored_null_is_reused_as_is(tmp_path):
    import json
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"layer": LAYER, "zoom": PROBE_ZOOM,
                                "tiles": [[x, y, None] for x, y in KEYS]}), encoding="utf-8")

    f = _recording_fetch({})
    kept, dropped, unknown = filter_windows_by_coverage(CENTERS, LAYER, fetch=f,
                                                        cache_path=path)
    assert f.calls == []
    assert kept == CENTERS and dropped == 0 and unknown == 4


def test_probe_tile_count():
    assert probe_tile_count(CENTERS) == 4
    assert probe_tile_count([(0.65, 47.33), (0.6501, 47.3301), (0.6502, 47.3302)]) == 1
    assert probe_tile_count([]) == 0
