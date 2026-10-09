import threading

import cv2
import numpy as np

from detection_ortho.coverage import (
    PROBE_ZOOM, fetch_probe_tile, filter_windows_by_coverage, probe_tiles,
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


def test_fetch_probe_tile_404_is_none_without_retry():
    s = _Session(404)
    assert fetch_probe_tile(1, 2, 13, LAYER, session=s, pause=0) is None
    assert s.calls == 1


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
