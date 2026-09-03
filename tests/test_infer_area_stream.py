import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import infer_area  # noqa: E402


class FakeResp:
    content = b"\xff\xd8\xff\xe0FAKE"  # 8 octets

    def raise_for_status(self):
        pass


class FakeSession:
    """Session HTTP factice ; `fail_on` lève sur les tuiles listées."""

    def __init__(self, fail_on=()):
        self.calls = 0
        self.fail_on = set(fail_on)

    def get(self, url, headers=None, timeout=30):
        self.calls += 1
        for xy in self.fail_on:
            if f"TILECOL={xy[0]}" in url and f"TILEROW={xy[1]}" in url:
                raise RuntimeError("boom")
        return FakeResp()


class SyncFuture:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class SyncPool:
    """Pool factice exécutant les soumissions immédiatement, dans le thread."""

    def submit(self, fn, *args):
        return SyncFuture(fn(*args))


def _centers(n):
    return [(0.001 * i, 47.5) for i in range(n)]


def test_stream_rend_toutes_les_fenetres_dans_l_ordre(tmp_path):
    centers = _centers(40)
    out = list(infer_area.stream_windows(centers, tmp_path, 160, SyncPool(),
                                         FakeSession()))
    assert out == centers


def test_stream_borne_le_cache_et_le_vide_a_la_fin(tmp_path):
    centers = _centers(40)
    peak = 0
    for _ in infer_area.stream_windows(centers, tmp_path, 160, SyncPool(),
                                       FakeSession()):
        peak = max(peak, len(list(tmp_path.iterdir())))
    # budget 160 o / tuile de 8 o -> ~10 tuiles par tranche, deux tranches max
    assert peak <= 24, peak
    # la purge finale (tranche suivante vide) laisse le cache propre
    assert list(tmp_path.iterdir()) == []


def test_stream_telecharge_les_tuiles_une_seule_fois_par_tranche(tmp_path):
    centers = _centers(12)
    sess = FakeSession()
    tiles = set()
    for lon, lat in centers:
        t, _, _ = infer_area.window_tiles(lon, lat, infer_area.ZOOM,
                                          infer_area.WINDOW)
        tiles.update(t)
    list(infer_area.stream_windows(centers, tmp_path, 160, SyncPool(), sess))
    # Des re-téléchargements sont possibles (tuile purgée puis redemandée),
    # mais on reste dans le même ordre de grandeur que le nombre de tuiles.
    assert len(tiles) <= sess.calls <= 2 * len(tiles)


def test_stream_survit_a_une_tuile_en_echec(tmp_path, capsys):
    centers = _centers(12)
    t, _, _ = infer_area.window_tiles(centers[0][0], centers[0][1],
                                      infer_area.ZOOM, infer_area.WINDOW)
    out = list(infer_area.stream_windows(centers, tmp_path, 160, SyncPool(),
                                         FakeSession(fail_on=[t[0]])))
    assert out == centers  # l'inférence n'est pas interrompue
    assert "échec" in capsys.readouterr().err


def test_stream_sur_liste_vide(tmp_path):
    assert list(infer_area.stream_windows([], tmp_path, 160, SyncPool(),
                                          FakeSession())) == []
