import threading

from detection_ortho import dataset as ds
from detection_ortho import tiles as tiles_mod
from detection_ortho.dataset import prefetch_points, tiles_for_points, window_tiles
from detection_ortho.tiles import prefetch_tiles


def test_prefetch_tiles_runs_downloads_in_parallel(tmp_path, monkeypatch):
    # Trois téléchargements doivent être en cours en même temps pour franchir la barrière.
    barrier = threading.Barrier(3, timeout=5)

    def fake(x, y, zoom, cache_dir, session=None, layer=None, **kw):
        barrier.wait()
        return tmp_path / f"{x}_{y}.jpg"

    monkeypatch.setattr(tiles_mod, "download_tile", fake)
    errors = prefetch_tiles([(1, 1), (2, 2), (3, 3)], tmp_path, workers=3)
    assert errors == []


def test_prefetch_tiles_continues_after_failures_and_reports_them(tmp_path, monkeypatch):
    def fake(x, y, zoom, cache_dir, session=None, layer=None, **kw):
        if x == 2:
            raise RuntimeError("boom")
        return tmp_path / "ok.jpg"

    monkeypatch.setattr(tiles_mod, "download_tile", fake)
    errors = prefetch_tiles([(1, 1), (2, 2), (3, 3)], tmp_path, workers=2)
    assert len(errors) == 1 and "2,2" in errors[0]


def test_prefetch_tiles_empty_list(tmp_path):
    assert prefetch_tiles([], tmp_path) == []


def test_prefetch_tiles_reports_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(tiles_mod, "download_tile",
                        lambda *a, **k: tmp_path / "x.jpg")
    seen = []
    prefetch_tiles([(1, 1), (2, 2)], tmp_path,
                   on_progress=lambda i, n: seen.append((i, n)))
    assert seen == [(1, 2), (2, 2)]


def test_tiles_for_points_matches_window_tiles():
    pts = [{"lon": 0.65, "lat": 47.33}, {"lon": 0.66, "lat": 47.34}]
    expected = set()
    for p in pts:
        expected.update(window_tiles(p["lon"], p["lat"], 19, 640)[0])
    assert tiles_for_points(pts) == expected


def test_prefetch_points_prefetches_each_layer(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        ds, "prefetch_tiles",
        lambda tiles, cache, layer=None, **k: calls.append((layer, len(tiles))) or [])
    errors = prefetch_points([{"lon": 0.65, "lat": 47.33}], tmp_path, ["L1", "L2"])
    assert [c[0] for c in calls] == ["L1", "L2"]
    assert calls[0][1] == calls[1][1] > 0
    assert errors == []


def test_prefetch_tiles_forwards_tries(tmp_path, monkeypatch):
    seen = []

    def fake(x, y, zoom, cache_dir, session=None, layer=None, **kw):
        seen.append(kw.get("tries"))
        return tmp_path / "ok.jpg"

    monkeypatch.setattr(tiles_mod, "download_tile", fake)
    prefetch_tiles([(1, 1), (2, 2)], tmp_path, workers=1, tries=1)
    assert seen == [1, 1]


def test_tile_cache_path_standard_and_tagged(tmp_path):
    assert tiles_mod.tile_cache_path(5, 6, 19, tmp_path) == tmp_path / "19_5_6.jpg"
    assert (tiles_mod.tile_cache_path(
        5, 6, 19, tmp_path, "ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026")
        == tmp_path / "19_5_6_rvb-express-2026.jpg")
