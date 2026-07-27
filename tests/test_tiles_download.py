from pathlib import Path
from detection_ortho.tiles import download_tile, tiles_in_bbox


class FakeResp:
    content = b"\xff\xd8\xff\xe0FAKEJPEG"  # entête JPEG bidon

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self):
        self.calls = 0

    def get(self, url, headers=None, timeout=30):
        self.calls += 1
        self.last_headers = headers
        return FakeResp()


def test_download_tile_writes_and_caches(tmp_path):
    sess = FakeSession()
    p1 = download_tile(42, 43, 19, tmp_path, session=sess)
    assert p1.exists()
    assert p1.read_bytes() == FakeResp.content
    # Deuxième appel : servi depuis le cache, pas de nouvel appel réseau.
    p2 = download_tile(42, 43, 19, tmp_path, session=sess)
    assert p2 == p1
    assert sess.calls == 1


def test_tiles_in_bbox_covers_area():
    # Petite bbox : au moins une tuile, toutes distinctes.
    tiles = tiles_in_bbox(6.14, 43.41, 6.16, 43.43, 17)
    assert len(tiles) >= 1
    assert len(set(tiles)) == len(tiles)
