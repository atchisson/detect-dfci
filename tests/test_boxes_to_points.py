from detection_ortho.dataset import lonlat_to_global_px
from detection_ortho.infer import boxes_to_points


def test_box_center_maps_back_to_lonlat():
    lon, lat, z, win = 0.6531, 47.3305, 19, 640
    gx, gy = lonlat_to_global_px(lon, lat, z)
    origin_gx, origin_gy = gx - win / 2, gy - win / 2  # fenêtre centrée
    # une boîte au centre de la fenêtre (cx=cy=320) doit retomber sur (lon,lat)
    pts = boxes_to_points([(win / 2, win / 2, 0.9)], origin_gx, origin_gy, z)
    assert len(pts) == 1
    assert abs(pts[0]["lon"] - lon) < 1e-5
    assert abs(pts[0]["lat"] - lat) < 1e-5
    assert pts[0]["score"] == 0.9


def test_empty_boxes():
    assert boxes_to_points([], 0, 0, 19) == []
