# tests/test_build_cog.py
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import build_cog  # noqa: E402
from detection_ortho.local_ortho import open_ortho, read_window  # noqa: E402


def _make_src_2154(path, lon, lat, color_rgb=(10, 200, 60), size=1500, res=0.2):
    """Petit raster EPSG:2154 uni avec un gros marqueur central, centré sur (lon,lat)."""
    mx, my = Transformer.from_crs(4326, 2154, always_xy=True).transform(lon, lat)
    transform = from_origin(mx - size / 2 * res, my + size / 2 * res, res, res)
    data = np.zeros((3, size, size), np.uint8)          # fond noir
    c = size // 2
    for b in range(3):                                  # marqueur central ~ 200 px
        data[b, c - 100:c + 100, c - 100:c + 100] = color_rgb[b]
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size,
                       count=3, dtype="uint8", crs="EPSG:2154",
                       transform=transform) as dst:
        dst.write(data)


def test_build_cog_roundtrip(tmp_path, monkeypatch):
    lon, lat, z, win = 0.65, 47.33, 19, 640
    src = tmp_path / "src2154.tif"
    out = tmp_path / "cog3857.tif"
    _make_src_2154(src, lon, lat, color_rgb=(10, 200, 60))

    monkeypatch.setattr(sys, "argv", [
        "build_cog.py", "--src", str(src), "--out", str(out),
        "--compress", "deflate", "--zoom", str(z)])
    build_cog.main()

    # La sortie est un GeoTIFF tuilé en EPSG:3857
    with rasterio.open(out) as d:
        assert d.crs.to_epsg() == 3857
        assert d.profile.get("tiled") is True

    # read_window sur la sortie retrouve le marqueur au centre, RGB->BGR
    vrt = open_ortho(out, zoom=z)
    try:
        img, _, _ = read_window(vrt, lon, lat, z, win)
    finally:
        vrt.close()
    assert img.shape == (win, win, 3)
    b, g, r = img[win // 2, win // 2]
    assert abs(int(b) - 60) < 40 and abs(int(g) - 200) < 40 and abs(int(r) - 10) < 40
    # un coin est le fond noir
    assert tuple(int(v) for v in img[5, 5]) == (0, 0, 0)
