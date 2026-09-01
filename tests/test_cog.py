import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from detection_ortho.cog import (
    zoom_mpp, world_grid_window, source_read_bytes, reproject_dalle, write_vrt,
    MAX_SOURCE_BYTES,
)

_R = 6378137.0


def test_zoom_mpp_z19():
    assert abs(zoom_mpp(19) - (2 * math.pi * _R) / (256 * 2 ** 19)) < 1e-9


def test_world_grid_window_snapped_and_covers():
    mpp = zoom_mpp(19)
    west, south, east, north = 70000.0, 5996000.0, 70400.0, 5996400.0
    c0, r0, wpx, hpx, tr = world_grid_window(west, south, east, north, mpp)
    # origine calée sur la grille mondiale (multiple de mpp depuis -πR / +πR)
    assert abs((tr.c + math.pi * _R) / mpp - round((tr.c + math.pi * _R) / mpp)) < 1e-6
    assert abs((math.pi * _R - tr.f) / mpp - round((math.pi * _R - tr.f) / mpp)) < 1e-6
    # la fenêtre englobe l'emprise demandée
    assert tr.c <= west and tr.f >= north
    assert tr.c + wpx * mpp >= east and tr.f - hpx * mpp <= south
    assert wpx > 0 and hpx > 0


def test_source_read_bytes():
    assert source_read_bytes(25000, 25000, 3) == 25000 * 25000 * 3


def _make_src(path, val=150):
    prof = dict(driver="GTiff", width=100, height=100, count=3, dtype="uint8",
                crs="EPSG:2154", transform=from_origin(500000, 6700000, 20, 20))
    with rasterio.open(path, "w", **prof) as o:
        o.write(np.full((3, 100, 100), val, np.uint8))


def test_reproject_dalle_produces_3857_data(tmp_path):
    src = tmp_path / "src.tif"
    _make_src(src, 150)
    out = reproject_dalle(src, tmp_path / "out.tif", zoom=19, compress="deflate")
    with rasterio.open(out) as d:
        assert d.crs.to_epsg() == 3857
        arr = d.read()
    # cœur non-noir (les bords peuvent avoir des zéros dus au reprojet)
    h, w = arr.shape[1], arr.shape[2]
    core = arr[:, h // 4:3 * h // 4, w // 4:3 * w // 4]
    assert core.mean() > 120


def test_reproject_dalle_refuses_huge_source(tmp_path, monkeypatch):
    src = tmp_path / "src.tif"
    _make_src(src)
    monkeypatch.setattr("detection_ortho.cog.MAX_SOURCE_BYTES", 10)  # 100x100x3 > 10
    with pytest.raises(ValueError, match="mosaïque"):
        reproject_dalle(src, tmp_path / "out.tif")


def test_write_vrt_mosaics_tiles(tmp_path):
    # deux tuiles 3857 adjacentes horizontalement, mêmes res
    outs = []
    for i, val in enumerate((100, 200)):
        p = tmp_path / f"t{i}.tif"
        prof = dict(driver="GTiff", width=50, height=50, count=3, dtype="uint8",
                    crs="EPSG:3857",
                    transform=from_origin(1000 + i * 50 * 2.0, 2000, 2.0, 2.0))
        with rasterio.open(p, "w", **prof) as o:
            o.write(np.full((3, 50, 50), val, np.uint8))
        outs.append(p)
    vrt = write_vrt(outs, tmp_path / "m.vrt")
    with rasterio.open(vrt) as d:
        assert d.width == 100 and d.height == 50
        arr = d.read(1)
    assert arr[:, :50].mean() == 100 and arr[:, 50:].mean() == 200


def test_write_vrt_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        write_vrt([], tmp_path / "m.vrt")
