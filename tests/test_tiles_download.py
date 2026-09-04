import pytest
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


class FlakySession:
    """Échoue `n_fail` fois avant de répondre correctement."""

    def __init__(self, n_fail):
        self.n_fail = n_fail
        self.calls = 0

    def get(self, url, headers=None, timeout=30):
        self.calls += 1
        if self.calls <= self.n_fail:
            raise RuntimeError("404 Client Error: Not Found")
        return FakeResp()


def test_download_tile_reessaie_une_erreur_passagere(tmp_path):
    sess = FlakySession(n_fail=2)
    p = download_tile(1, 2, 19, tmp_path, session=sess, pause=0)
    assert p.exists() and p.read_bytes() == FakeResp.content
    assert sess.calls == 3


def test_download_tile_abandonne_apres_les_essais(tmp_path):
    sess = FlakySession(n_fail=99)
    with pytest.raises(RuntimeError):
        download_tile(1, 2, 19, tmp_path, session=sess, tries=3, pause=0)
    assert sess.calls == 3
    assert not (tmp_path / "19_1_2.jpg").exists()  # pas de fichier tronqué


def test_download_tile_sans_reprise_si_tries_1(tmp_path):
    sess = FlakySession(n_fail=1)
    with pytest.raises(RuntimeError):
        download_tile(1, 2, 19, tmp_path, session=sess, tries=1, pause=0)
    assert sess.calls == 1
