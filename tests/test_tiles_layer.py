from detection_ortho.tiles import tile_url, download_tile, LAYER, LAYER_IRC


def test_tile_url_layer():
    assert f"LAYER={LAYER_IRC}" in tile_url(1, 2, 19, layer=LAYER_IRC)
    assert "ORTHOIMAGERY.ORTHOPHOTOS.IRC" in tile_url(1, 2, 19, layer=LAYER_IRC)
    # défaut = RVB
    assert f"LAYER={LAYER}" in tile_url(1, 2, 19)


class _Resp:
    content = b"\xff\xd8\xff\xe0FAKE"

    def raise_for_status(self):
        pass


class _Session:
    def __init__(self):
        self.urls = []

    def get(self, url, headers=None, timeout=30):
        self.urls.append(url)
        return _Resp()


def test_cache_is_layer_specific(tmp_path):
    s = _Session()
    rgb = download_tile(5, 6, 19, tmp_path, session=s)            # RVB
    irc = download_tile(5, 6, 19, tmp_path, session=s, layer=LAYER_IRC)
    # deux fichiers de cache distincts (pas de collision)
    assert rgb != irc
    assert rgb.name == "19_5_6.jpg"          # RVB : nom inchangé
    assert "irc" in irc.name                  # IRC : suffixé
    assert LAYER_IRC in s.urls[1]             # 2e requête = couche IRC
