import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import infer_area  # noqa: E402
from detection_ortho import checkpoint, tiles as tiles_mod  # noqa: E402
from detection_ortho.tiles import LAYER  # noqa: E402

L26 = "ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026"


class _Resp:
    content = b"\xff\xd8\xff\xe0FAKE"

    def raise_for_status(self):
        pass


class _Session:
    """Session factice : mémorise les URL demandées ; `fail` fait échouer chaque appel."""

    def __init__(self, fail=False):
        self.urls = []
        self.fail = fail

    def get(self, url, headers=None, timeout=30):
        self.urls.append(url)
        if self.fail:
            raise RuntimeError("404")
        return _Resp()


class _SyncFuture:
    def __init__(self, v):
        self.v = v

    def result(self):
        return self.v


class _SyncPool:
    def submit(self, fn, *args):
        return _SyncFuture(fn(*args))


def _centers(n):
    return [(0.001 * i, 47.5) for i in range(n)]


def test_stream_on_another_layer_fetches_it_and_purges_its_tagged_tiles(tmp_path):
    sess = _Session()
    seen_tagged = False
    for _ in infer_area.stream_windows(_centers(40), tmp_path, 160, _SyncPool(), sess,
                                       layer=L26):
        files = [p.name for p in tmp_path.glob("*.jpg")]
        seen_tagged = seen_tagged or any("rvb-express-2026" in n for n in files)
    assert seen_tagged, "les tuiles de la couche 2026 doivent être suffixées dans le cache"
    assert sess.urls and all(f"LAYER={L26}" in u for u in sess.urls)
    # Disque borné : tout est purgé à la fin, y compris les tuiles suffixées.
    assert list(tmp_path.glob("*.jpg")) == []


def test_download_ok_tries_once_on_another_layer_and_three_times_on_the_standard(monkeypatch, tmp_path):
    monkeypatch.setattr(tiles_mod.time, "sleep", lambda s: None)
    s26, s_std = _Session(fail=True), _Session(fail=True)
    assert infer_area.download_ok((1, 2), tmp_path, s26, L26) is not None
    assert len(s26.urls) == 1
    assert infer_area.download_ok((1, 2), tmp_path, s_std) is not None
    assert len(s_std.urls) == 3


def _args(**kw):
    base = dict(boundary="Cher", conf=0.25, overlap=0.2, weights="models/x.pt",
                ortho=None, insee="18", admin_level=None, layer=LAYER)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_fingerprint_is_unchanged_for_the_standard_layer():
    centers = _centers(10)
    legacy = checkpoint.fingerprint(centers, {
        "boundary": "Cher", "conf": 0.25, "overlap": 0.2, "zoom": 19, "window": 640,
        "weights": "x.pt", "ortho": "", "insee": "18", "admin_level": ""})
    assert infer_area.build_fingerprint(centers, _args()) == legacy


def test_fingerprint_changes_with_a_non_standard_layer():
    centers = _centers(10)
    assert infer_area.build_fingerprint(centers, _args(layer=L26)) != \
        infer_area.build_fingerprint(centers, _args())


def test_layer_with_local_ortho_is_rejected():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "infer_area.py"), "--boundary", "X",
         "--weights", "w.pt", "--layer", L26, "--ortho", "ortho.vrt"],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 2
    assert "incompatible" in r.stderr


def test_missing_known_false_file_fails_before_any_network_work(tmp_path):
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "infer_area.py"), "--boundary", "X",
         "--weights", "w.pt", "--known-false", str(tmp_path / "absent.csv")],
        capture_output=True, text=True, timeout=300)
    assert r.returncode != 0
    assert "absent.csv" in r.stderr


def test_help_mentions_the_new_options():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "infer_area.py"), "--help"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    for opt in ("--layer", "--known-false", "--known-false-m"):
        assert opt in r.stdout
