import mercantile
from detection_ortho.tiles import (
    tile_for_lonlat,
    tile_url,
    pixel_to_lonlat,
    lonlat_to_pixel,
)


def test_tile_for_lonlat_matches_mercantile():
    lon, lat, z = 6.15, 43.42, 19  # secteur Var
    x, y = tile_for_lonlat(lon, lat, z)
    expected = mercantile.tile(lon, lat, z)
    assert (x, y) == (expected.x, expected.y)


def test_tile_url_contains_layer_and_indices():
    url = tile_url(42, 43, 19)
    assert "ORTHOIMAGERY.ORTHOPHOTOS" in url
    assert "TILEMATRIXSET=PM" in url
    assert "TILEMATRIX=19" in url
    assert "TILECOL=42" in url
    assert "TILEROW=43" in url


def test_pixel_lonlat_roundtrip():
    lon, lat, z = 6.15, 43.42, 19
    x, y, px, py = lonlat_to_pixel(lon, lat, z)
    lon2, lat2 = pixel_to_lonlat(x, y, z, px, py)
    assert abs(lon - lon2) < 1e-5
    assert abs(lat - lat2) < 1e-5


def test_pixel_center_is_inside_tile_bounds():
    lon, lat, z = 6.15, 43.42, 19
    x, y, px, py = lonlat_to_pixel(lon, lat, z)
    assert 0 <= px < 256
    assert 0 <= py < 256


def test_pixel_corners_match_tile_bounds():
    # Vérité-terrain indépendante : pixel (0,0) = coin (ouest, nord) de la tuile,
    # pixel (256,256) = coin (est, sud). Détecte une inversion d'axe.
    x, y, z = 42, 43, 19
    b = mercantile.bounds(mercantile.Tile(x, y, z))  # west, south, east, north
    lon_nw, lat_nw = pixel_to_lonlat(x, y, z, 0, 0)
    lon_se, lat_se = pixel_to_lonlat(x, y, z, 256, 256)
    assert abs(lon_nw - b.west) < 1e-6
    assert abs(lat_nw - b.north) < 1e-6
    assert abs(lon_se - b.east) < 1e-6
    assert abs(lat_se - b.south) < 1e-6
