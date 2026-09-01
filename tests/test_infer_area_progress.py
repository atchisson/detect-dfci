import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import infer_area  # noqa: E402


def test_progress_passes_through_all_items():
    # min_interval=0 -> imprime à chaque item, mais surtout on vérifie le pass-through
    out = list(infer_area.progress(iter(range(5)), 5, "X", min_interval=0))
    assert out == [0, 1, 2, 3, 4]


def test_progress_prints_eta(capsys):
    list(infer_area.progress(iter(range(3)), 3, "Inférence", min_interval=0))
    printed = capsys.readouterr().out
    assert "ETA" in printed and "écoulé" in printed
    assert "3/3 (100%)" in printed  # dernière ligne toujours imprimée


def test_progress_appends_status(capsys):
    # status() est appelé à chaque impression et ajouté en fin de ligne
    seen = {"n": 0}

    def status():
        seen["n"] += 1
        return f"{seen['n']} détection(s)"

    list(infer_area.progress(iter(range(3)), 3, "Inférence", min_interval=0,
                             status=status))
    out = capsys.readouterr().out
    assert "détection(s)" in out
    assert out.rstrip().endswith("détection(s)")


def test_progress_quiet_when_interval_large(capsys):
    # intervalle géant -> seule la dernière ligne (i == total) s'imprime
    list(infer_area.progress(iter(range(4)), 4, "X", min_interval=10_000))
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len(lines) == 1 and "4/4 (100%)" in lines[0]
