from detection_ortho.compare import sweep_precision_recall


def test_sweep_basic():
    # 2 vrais (scores 0.9, 0.6), 2 faux (scores 0.7, 0.3)
    scored = [(0.9, True), (0.6, True), (0.7, False), (0.3, False)]
    rows = {r["conf"]: r for r in sweep_precision_recall(scored, [0.5, 0.8])}
    # seuil 0.5 : tp=2 (0.9,0.6), fp=1 (0.7) -> prec 2/3, rappel 2/2
    assert rows[0.5]["tp"] == 2 and rows[0.5]["fp"] == 1
    assert abs(rows[0.5]["precision"] - 2 / 3) < 1e-9
    assert rows[0.5]["recall"] == 1.0
    # seuil 0.8 : tp=1 (0.9), fp=0 -> prec 1.0, rappel 1/2
    assert rows[0.8]["tp"] == 1 and rows[0.8]["fp"] == 0
    assert rows[0.8]["precision"] == 1.0 and rows[0.8]["recall"] == 0.5


def test_sweep_no_positives_no_crash():
    rows = sweep_precision_recall([(0.2, False)], [0.5])
    assert rows[0]["precision"] == 0.0 and rows[0]["recall"] == 0.0
