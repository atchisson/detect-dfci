from detection_ortho.dataset import window_tiles
from detection_ortho.tilecache import (
    max_tiles_for_budget, next_chunk, purge_cache,
)

ZOOM = 19
WINDOW = 640


def _row_of_centers(n: int) -> list[tuple[float, float]]:
    """n fenêtres alignées d'ouest en est, comme les produit la grille."""
    return [(0.001 * i, 47.5) for i in range(n)]


def test_max_tiles_for_budget_reserve_deux_tranches():
    # 10 Go / (2 * 8 ko) : deux tranches doivent tenir dans le budget.
    assert max_tiles_for_budget(10e9, 8_000) == 625_000
    assert max_tiles_for_budget(0, 8_000) == 1  # jamais zéro


def test_next_chunk_borne_le_nombre_de_tuiles():
    centers = _row_of_centers(200)
    chunk, tiles, nxt = next_chunk(centers, 0, ZOOM, WINDOW, max_tiles=20)
    assert 0 < len(chunk) < len(centers)
    assert nxt == len(chunk)
    # La borne peut être dépassée d'au plus une fenêtre (la dernière ajoutée).
    per_window = len(window_tiles(*centers[0], ZOOM, WINDOW)[0])
    assert len(tiles) <= 20 + per_window
    # Les tuiles annoncées sont bien celles des fenêtres de la tranche.
    expected = set()
    for lon, lat in chunk:
        expected.update(window_tiles(lon, lat, ZOOM, WINDOW)[0])
    assert tiles == expected


def test_next_chunk_borne_le_nombre_de_fenetres():
    centers = _row_of_centers(50)
    chunk, _, nxt = next_chunk(centers, 0, ZOOM, WINDOW,
                               max_tiles=10**9, max_windows=7)
    assert len(chunk) == 7
    assert nxt == 7


def test_next_chunk_parcourt_toute_la_liste_sans_trou():
    centers = _row_of_centers(60)
    seen: list[tuple[float, float]] = []
    i = 0
    while i < len(centers):
        chunk, _, i = next_chunk(centers, i, ZOOM, WINDOW, max_tiles=15)
        assert chunk  # progresse toujours, sinon boucle infinie
        seen.extend(chunk)
    assert seen == centers


def test_next_chunk_en_fin_de_liste_est_vide():
    centers = _row_of_centers(3)
    chunk, tiles, nxt = next_chunk(centers, 3, ZOOM, WINDOW, max_tiles=100)
    assert chunk == [] and tiles == set() and nxt == 3


def test_purge_cache_supprime_hors_keep_et_compte_le_reste(tmp_path):
    for xy in [(10, 20), (11, 20), (12, 20)]:
        (tmp_path / f"{ZOOM}_{xy[0]}_{xy[1]}.jpg").write_bytes(b"x" * 100)
    deleted, kept, kept_bytes = purge_cache(tmp_path, {(11, 20)}, ZOOM)
    assert deleted == 2
    assert kept == 1
    assert kept_bytes == 100
    assert (tmp_path / f"{ZOOM}_11_20.jpg").exists()
    assert not (tmp_path / f"{ZOOM}_10_20.jpg").exists()


def test_purge_cache_epargne_les_autres_zooms_et_fichiers(tmp_path):
    (tmp_path / f"{ZOOM}_10_20.jpg").write_bytes(b"x")
    (tmp_path / "17_10_20.jpg").write_bytes(b"x")       # autre zoom
    (tmp_path / "notes.txt").write_bytes(b"x")          # pas une tuile
    (tmp_path / f"{ZOOM}_10_20_irc.jpg").write_bytes(b"x")  # couche IRC
    deleted, _, _ = purge_cache(tmp_path, set(), ZOOM)
    assert deleted == 2  # la tuile RVB et sa variante IRC du même zoom
    assert (tmp_path / "17_10_20.jpg").exists()
    assert (tmp_path / "notes.txt").exists()


def test_purge_cache_sur_repertoire_absent(tmp_path):
    assert purge_cache(tmp_path / "nexiste_pas", set(), ZOOM) == (0, 0, 0)
