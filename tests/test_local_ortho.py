import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer

from detection_ortho.local_ortho import open_ortho, read_window
from detection_ortho.dataset import lonlat_to_global_px


def _make_ortho(path, lon, lat, color_rgb=(10, 200, 60), size=2000, res=0.29):
    """GeoTIFF EPSG:3857 uni, centré sur (lon,lat), couvrant ~size*res mètres."""
    mx, my = Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
    transform = from_origin(mx - size / 2 * res, my + size / 2 * res, res, res)
    data = np.zeros((3, size, size), np.uint8)
    for b in range(3):
        data[b] = color_rgb[b]
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=3,
        dtype="uint8", crs="EPSG:3857", transform=transform,
    ) as dst:
        dst.write(data)


def test_read_window_shape_origin_and_bgr(tmp_path):
    lon, lat, z, win = 0.65, 47.33, 19, 640
    tif = tmp_path / "ortho.tif"
    _make_ortho(tif, lon, lat, color_rgb=(10, 200, 60))
    vrt = open_ortho(tif, zoom=z)
    try:
        img, ogx, ogy = read_window(vrt, lon, lat, z, win)
    finally:
        vrt.close()
    # taille et type
    assert img.shape == (win, win, 3) and img.dtype == np.uint8
    # origine cohérente avec le chemin WMTS
    gx, gy = lonlat_to_global_px(lon, lat, z)
    assert abs(ogx - (gx - win / 2)) < 1e-6
    assert abs(ogy - (gy - win / 2)) < 1e-6
    # pixel central = couleur du raster, convertie RGB(10,200,60) -> BGR(60,200,10)
    b, g, r = img[win // 2, win // 2]
    assert (int(b), int(g), int(r)) == (60, 200, 10)


def test_read_window_outside_data_is_black(tmp_path):
    # Fenêtre loin du raster -> lecture boundless remplie de 0 (pas de crash).
    tif = tmp_path / "ortho.tif"
    _make_ortho(tif, 0.65, 47.33)
    vrt = open_ortho(tif, zoom=19)
    try:
        img, _, _ = read_window(vrt, 2.0, 48.5, 19, 640)  # ailleurs
    finally:
        vrt.close()
    assert img.shape == (640, 640, 3)
    assert int(img.sum()) == 0


def test_close_releases_source_file(tmp_path):
    tif = tmp_path / "ortho.tif"
    _make_ortho(tif, 0.65, 47.33)
    vrt = open_ortho(tif, zoom=19)
    read_window(vrt, 0.65, 47.33, 19, 640)
    vrt.close()
    assert vrt.src_dataset.closed  # la source est bien fermée
