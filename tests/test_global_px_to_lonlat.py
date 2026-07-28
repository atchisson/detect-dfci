from detection_ortho.dataset import lonlat_to_global_px, global_px_to_lonlat


def test_roundtrip_global_px():
    lon, lat, z = 0.6531, 47.3305, 19
    gx, gy = lonlat_to_global_px(lon, lat, z)
    lon2, lat2 = global_px_to_lonlat(gx, gy, z)
    assert abs(lon - lon2) < 1e-6
    assert abs(lat - lat2) < 1e-6


def test_known_tile_origin():
    # Le pixel global (x*256, y*256) doit retomber sur le coin NO de la tuile.
    import mercantile
    from detection_ortho.tiles import pixel_to_lonlat
    x, y, z = 264000, 180000, 19
    lon, lat = global_px_to_lonlat(x * 256, y * 256, z)
    lon0, lat0 = pixel_to_lonlat(x, y, z, 0, 0)
    assert abs(lon - lon0) < 1e-9 and abs(lat - lat0) < 1e-9
