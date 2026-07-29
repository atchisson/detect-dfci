def test_rasterio_importable():
    import rasterio
    from rasterio.vrt import WarpedVRT
    assert rasterio is not None and WarpedVRT is not None
