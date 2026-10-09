import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import rattrapage  # noqa: E402
from detection_ortho import checkpoint  # noqa: E402

L26 = rattrapage.L26


def _args(tmp_path, **kw):
    base = dict(out_root=tmp_path / "root", weights=tmp_path / "w.pt", conf=0.25,
                device="cpu", workers=12, cache_gb=10.0, verdicts_dir=tmp_path / "v")
    base.update(kw)
    return types.SimpleNamespace(**base)


def _dep(insee, layer=None):
    return {"insee": insee, "nom": "Nom" + insee, "layer": layer}


def _finish(out):
    """Simule un run terminé : livrable présent, pas de point de reprise."""
    out.mkdir(parents=True, exist_ok=True)
    (out / rattrapage.DELIVERABLE).write_text('{"type": "FeatureCollection", "features": []}')
    checkpoint.clear(out)


def _pause(out):
    """Simule un run en pause : point de reprise présent."""
    checkpoint.save(out, "empreinte", 12, [])


def _candidates(out, points):
    out.mkdir(parents=True, exist_ok=True)
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
              "properties": ({} if score is None else {"score": score})}
             for lon, lat, score in points]
    (out / rattrapage.DELIVERABLE).write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}))


# --- table et commandes ------------------------------------------------------

def test_departements_order_2026_first_then_standard():
    codes = [d["insee"] for d in rattrapage.DEPARTEMENTS]
    assert codes == ["18", "28", "36", "37", "41", "45", "44", "49"]
    assert [bool(d["layer"]) for d in rattrapage.DEPARTEMENTS] == [True] * 6 + [False] * 2
    assert all(d["layer"] == L26 for d in rattrapage.DEPARTEMENTS if d["layer"])


def test_build_command_for_a_2026_department(tmp_path):
    a = _args(tmp_path)
    cmd = rattrapage.build_command(_dep("18", L26), a)
    assert cmd[1].endswith("infer_area.py")
    assert cmd[cmd.index("--boundary") + 1] == "Nom18"
    assert cmd[cmd.index("--insee") + 1] == "18"
    assert cmd[cmd.index("--layer") + 1] == L26
    assert cmd[cmd.index("--known-false") + 1] == str(tmp_path / "v" / "verdicts_18.csv")
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "root" / "18")
    assert cmd[cmd.index("--conf") + 1] == "0.25"
    assert cmd[cmd.index("--cache-gb") + 1] == "10.0"


def test_build_command_for_a_standard_department_has_no_layer(tmp_path):
    assert "--layer" not in rattrapage.build_command(_dep("49"), _args(tmp_path))


# --- état sur disque -----------------------------------------------------------

def test_state_detection(tmp_path):
    out = tmp_path / "d"
    assert not rattrapage.is_done(out) and not rattrapage.is_paused(out)
    _finish(out)
    assert rattrapage.is_done(out) and not rattrapage.is_paused(out)
    _pause(out)
    assert rattrapage.is_paused(out) and not rattrapage.is_done(out)


# --- vérifications au démarrage ----------------------------------------------

def test_preflight_reports_missing_weights_and_known_false_files(tmp_path):
    a = _args(tmp_path)
    errors = rattrapage.preflight([_dep("18"), _dep("28")], a)
    assert any("w.pt" in e for e in errors)
    assert any("verdicts_18.csv" in e for e in errors) and any("verdicts_28.csv" in e for e in errors)


def test_preflight_is_clean_when_everything_exists(tmp_path):
    a = _args(tmp_path)
    a.weights.write_text("x")
    a.verdicts_dir.mkdir()
    (a.verdicts_dir / "verdicts_18.csv").write_text("x")
    assert rattrapage.preflight([_dep("18")], a) == []


# --- boucle -----------------------------------------------------------------------

def test_run_all_skips_done_runs_the_rest_in_order(tmp_path):
    a = _args(tmp_path)
    deps = [_dep("18", L26), _dep("28", L26), _dep("36", L26)]
    _finish(rattrapage.dept_out(a.out_root, "18"))
    called = []

    def fake_run(cmd):
        insee = cmd[cmd.index("--insee") + 1]
        called.append(insee)
        _finish(Path(cmd[cmd.index("--out") + 1]))
        return 0

    status = rattrapage.run_all(deps, a, run=fake_run)
    assert called == ["28", "36"]
    assert status == {"18": "déjà terminé", "28": "terminé", "36": "terminé"}


def test_run_all_continues_after_a_failure(tmp_path):
    a = _args(tmp_path)
    deps = [_dep("18", L26), _dep("28", L26)]
    called = []

    def fake_run(cmd):
        insee = cmd[cmd.index("--insee") + 1]
        called.append(insee)
        if insee == "18":
            return 1
        _finish(Path(cmd[cmd.index("--out") + 1]))
        return 0

    status = rattrapage.run_all(deps, a, run=fake_run)
    assert called == ["18", "28"]
    assert status["18"].startswith("échec") and status["28"] == "terminé"


def test_run_all_stops_on_a_pause_without_starting_the_next_department(tmp_path):
    a = _args(tmp_path)
    deps = [_dep("18", L26), _dep("28", L26), _dep("36", L26)]
    called = []

    def fake_run(cmd):
        insee = cmd[cmd.index("--insee") + 1]
        called.append(insee)
        out = Path(cmd[cmd.index("--out") + 1])
        if insee == "28":
            out.mkdir(parents=True, exist_ok=True)
            _pause(out)
            return 0                       # infer_area s'arrête proprement : code 0
        _finish(out)
        return 0

    status = rattrapage.run_all(deps, a, run=fake_run)
    assert called == ["18", "28"]
    assert status == {"18": "terminé", "28": "pause", "36": "non traité"}


def test_run_all_resumes_a_paused_department(tmp_path):
    a = _args(tmp_path)
    out = rattrapage.dept_out(a.out_root, "18")
    out.mkdir(parents=True)
    _pause(out)
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        _finish(out)
        return 0

    status = rattrapage.run_all([_dep("18", L26)], a, run=fake_run)
    assert len(calls) == 1 and status == {"18": "terminé"}


def test_a_clean_exit_without_any_deliverable_is_a_failure(tmp_path):
    a = _args(tmp_path)
    status = rattrapage.run_all([_dep("18", L26)], a, run=lambda cmd: 0)
    assert status["18"].startswith("échec")


# --- fusion -------------------------------------------------------------------------

def test_merge_applies_the_threshold_of_each_layer(tmp_path):
    root = tmp_path / "root"
    _candidates(root / "18", [(1.0, 47.0, 0.40), (1.1, 47.0, 0.30)])       # couche 2026 : seuil 0,35
    _candidates(root / "49", [(0.0, 47.5, 0.65), (0.1, 47.5, 0.50)])       # standard : seuil 0,60
    res = rattrapage.merge_challenge(root, [_dep("18", L26), _dep("49")], 0.35, 0.60)
    scores = sorted(f["properties"]["score"] for f in res["fc"]["features"])
    assert scores == [0.40, 0.65]
    by_dept = {s["insee"]: (s["candidats"], s["retenus"]) for s in res["stats"]}
    assert by_dept == {"18": (2, 1), "49": (2, 1)}


def test_merge_dedups_across_departments_keeping_the_best_score(tmp_path):
    root = tmp_path / "root"
    _candidates(root / "18", [(1.00000, 47.0, 0.50)])
    _candidates(root / "36", [(1.00003, 47.0, 0.80)])                       # ~2 m plus loin
    res = rattrapage.merge_challenge(root, [_dep("18", L26), _dep("36", L26)], 0.35, 0.60)
    feats = res["fc"]["features"]
    assert len(feats) == 1
    assert feats[0]["properties"]["score"] == 0.80
    assert feats[0]["properties"]["dept"] == "36"
    assert res["doublons"] == 1


def test_merge_reports_missing_departments_and_still_merges_the_others(tmp_path):
    root = tmp_path / "root"
    _candidates(root / "18", [(1.0, 47.0, 0.9)])
    res = rattrapage.merge_challenge(root, [_dep("18", L26), _dep("28", L26)], 0.35, 0.60)
    assert res["manquants"] == ["28"]
    assert len(res["fc"]["features"]) == 1


def test_merge_ignores_a_paused_department(tmp_path):
    root = tmp_path / "root"
    _candidates(root / "18", [(1.0, 47.0, 0.9)])
    _pause(root / "18")
    res = rattrapage.merge_challenge(root, [_dep("18", L26)], 0.35, 0.60)
    assert res["manquants"] == ["18"] and res["fc"]["features"] == []


def test_merge_drops_candidates_without_a_score(tmp_path):
    root = tmp_path / "root"
    _candidates(root / "18", [(1.0, 47.0, None), (1.1, 47.0, 0.9)])
    res = rattrapage.merge_challenge(root, [_dep("18", L26)], 0.35, 0.60)
    assert len(res["fc"]["features"]) == 1


# --- main ---------------------------------------------------------------------------

def test_main_dry_run_prints_the_commands_and_runs_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(rattrapage, "run_infer",
                        lambda cmd: pytest.fail("aucun lancement en --dry-run"))
    code = rattrapage.main(["--dry-run", "--only", "18", "49", "--out-root", str(tmp_path / "r")])
    out = capsys.readouterr().out
    assert code == 0
    assert "--insee 18" in out and "--insee 49" in out and "--insee 28" not in out


def test_main_rejects_an_unknown_department_code(tmp_path):
    with pytest.raises(SystemExit) as exc:
        rattrapage.main(["--only", "99", "--out-root", str(tmp_path / "r")])
    assert exc.value.code == 2


def test_main_fails_fast_when_weights_or_false_files_are_missing(tmp_path, capsys):
    code = rattrapage.main(["--only", "18", "--out-root", str(tmp_path / "r"),
                            "--weights", str(tmp_path / "absent.pt"),
                            "--verdicts-dir", str(tmp_path / "absent")])
    assert code == 2
    err = capsys.readouterr().err
    assert "absent.pt" in err and "verdicts_18.csv" in err


def test_main_merge_only_writes_one_challenge_file(tmp_path, capsys):
    root = tmp_path / "root"
    _candidates(root / "18", [(1.0, 47.0, 0.9)])
    _candidates(root / "28", [(2.0, 48.0, 0.8)])
    code = rattrapage.main(["--merge-only", "--only", "18", "28", "--out-root", str(root)])
    assert code == 0
    fc = json.loads((root / "challenge_rattrapage.geojson").read_text(encoding="utf-8"))
    assert len(fc["features"]) == 2
    assert {f["properties"]["dept"] for f in fc["features"]} == {"18", "28"}
    assert (root / "recap_rattrapage.txt").exists()


def test_main_full_run_with_an_injected_runner_merges_at_the_end(tmp_path, monkeypatch):
    root = tmp_path / "root"
    weights = tmp_path / "w.pt"
    weights.write_text("x")
    vdir = tmp_path / "v"
    vdir.mkdir()
    (vdir / "verdicts_18.csv").write_text("x")

    def fake_run(cmd):
        out = Path(cmd[cmd.index("--out") + 1])
        _candidates(out, [(1.0, 47.0, 0.9)])
        checkpoint.clear(out)
        return 0

    monkeypatch.setattr(rattrapage, "run_infer", fake_run)
    code = rattrapage.main(["--only", "18", "--out-root", str(root), "--weights", str(weights),
                            "--verdicts-dir", str(vdir)])
    assert code == 0
    fc = json.loads((root / "challenge_rattrapage.geojson").read_text(encoding="utf-8"))
    assert len(fc["features"]) == 1


def test_main_returns_1_when_a_department_failed(tmp_path, monkeypatch):
    weights = tmp_path / "w.pt"
    weights.write_text("x")
    vdir = tmp_path / "v"
    vdir.mkdir()
    (vdir / "verdicts_18.csv").write_text("x")
    monkeypatch.setattr(rattrapage, "run_infer", lambda cmd: 1)
    code = rattrapage.main(["--only", "18", "--out-root", str(tmp_path / "r"),
                            "--weights", str(weights), "--verdicts-dir", str(vdir)])
    assert code == 1


def test_main_stops_without_merging_on_a_pause(tmp_path, monkeypatch):
    root = tmp_path / "root"
    weights = tmp_path / "w.pt"
    weights.write_text("x")
    vdir = tmp_path / "v"
    vdir.mkdir()
    (vdir / "verdicts_18.csv").write_text("x")

    def fake_run(cmd):
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        _pause(out)
        return 0

    monkeypatch.setattr(rattrapage, "run_infer", fake_run)
    code = rattrapage.main(["--only", "18", "--out-root", str(root), "--weights", str(weights),
                            "--verdicts-dir", str(vdir)])
    assert code == 0
    assert not (root / "challenge_rattrapage.geojson").exists()
