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


def test_read_window_registration(tmp_path):
    import math
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from pyproj import Transformer

    lon, lat, z, win = 0.65, 47.33, 19, 640
    mpp = (2 * math.pi * 6378137.0) / (256 * 2 ** z)      # m/px grille z19
    cx, cy = Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
    size = 1200
    left, top = cx - (size // 2) * mpp, cy + (size // 2) * mpp
    transform = from_origin(left, top, mpp, mpp)          # raster EPSG:3857 @ mpp
    data = np.zeros((3, size, size), np.uint8)            # fond noir
    # marqueur RVB (10,200,60) à +100 px est / +60 px sud du CENTRE
    mrow, mcol = size // 2 + 60, size // 2 + 100
    data[0, mrow - 5:mrow + 5, mcol - 5:mcol + 5] = 10
    data[1, mrow - 5:mrow + 5, mcol - 5:mcol + 5] = 200
    data[2, mrow - 5:mrow + 5, mcol - 5:mcol + 5] = 60
    tif = tmp_path / "marker.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=size, width=size,
                       count=3, dtype="uint8", crs="EPSG:3857",
                       transform=transform) as dst:
        dst.write(data)

    vrt = open_ortho(tif, zoom=z)
    try:
        img, _, _ = read_window(vrt, lon, lat, z, win)
    finally:
        vrt.close()

    # Le marqueur doit apparaître autour du pixel fenêtre (row=320+60, col=320+100)
    # = (380, 420) — tolérance ±2 px pour le rééchantillonnage sous-pixel.
    patch = img[378:383, 418:423]                          # (h,w,BGR)
    # au moins un pixel proche de BGR (60,200,10)
    close = (np.abs(patch[:, :, 0].astype(int) - 60) < 40) & \
            (np.abs(patch[:, :, 1].astype(int) - 200) < 40) & \
            (np.abs(patch[:, :, 2].astype(int) - 10) < 40)
    assert close.any(), "marqueur non trouvé au pixel attendu -> calage géo faux"
    # le centre de la fenêtre reste le fond noir (pas le marqueur)
    assert tuple(int(v) for v in img[320, 320]) == (0, 0, 0)


def test_close_releases_source_file(tmp_path):
    tif = tmp_path / "ortho.tif"
    _make_ortho(tif, 0.65, 47.33)
    vrt = open_ortho(tif, zoom=19)
    read_window(vrt, 0.65, 47.33, 19, 640)
    vrt.close()
    assert vrt.src_dataset.closed  # la source est bien fermée
