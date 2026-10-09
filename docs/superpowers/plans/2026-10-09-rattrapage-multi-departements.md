# Rattrapage multi-départements, un seul challenge — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un script qui enchaîne `infer_area.py` sur plusieurs départements (un run de plusieurs jours, reprenable, tournant le week-end) puis fusionne leurs candidats en **un seul** challenge MapRoulette.

**Architecture:** `scripts/rattrapage.py` lance `infer_area.py` en sous-processus, un département après l'autre, chacun dans son dossier `inference_rattrapage/<insee>` avec ses faux déjà rejetés. L'état de chaque département se lit sur le disque (livrable présent + point de reprise absent = terminé ; point de reprise présent = en pause), donc relancer la même commande reprend là où on s'est arrêté. La fusion filtre au seuil de chaque couche, dédoublonne aux frontières et écrit un seul GeoJSON.

**Tech Stack:** Python 3.12, pytest, subprocess (stdlib) ; modules existants `checkpoint`, `geo.dedup_points`, `geojson_io`, `maproulette`.

**Design validé avec l'utilisateur (2026-10-09).** Ordre : les 6 départements couverts en 2026 (18, 28, 36, 37, 41, 45) puis le 44 et le 49 sur l'ortho habituelle. Un département en échec n'arrête pas les suivants ; une pause arrête tout. Durée : de l'ordre de 15 à 19 h de CPU par département (extrapolée du 49), donc un week-end en traite 3 ou 4.

## Global Constraints

- Le script ne fait **aucun traitement d'image lui-même** : il lance `infer_area.py` (qui streame les tuiles en parallèle avec un cache disque borné et purgé). Il ne télécharge ni n'écrit aucune tuile.
- Reprise : relancer la **même commande** reprend. Un département terminé (livrable `detected_only.geojson` présent et pas de `checkpoint.json`) est sauté ; un département en pause reprend via le point de reprise d'`infer_area`.
- Une pause (code 0 avec un point de reprise présent) arrête le script **sans** enchaîner sur le département suivant et sans fusion. Un échec (code ≠ 0) passe au département suivant et est listé à la fin (code de sortie 1).
- Sur Ctrl+C, le sous-processus doit pouvoir enregistrer son point de reprise : ne pas le tuer tout de suite, l'attendre jusqu'à 300 s.
- Vérifications faites AU DÉMARRAGE, avant tout calcul : les poids existent et chaque `verdicts_<insee>.csv` existe (échec immédiat, code 2).
- Seuils d'export par couche : 0,35 pour les départements sur la couche 2026, 0,60 pour l'ortho habituelle. Dédoublonnage aux frontières : 10 m, on garde le meilleur score.
- Tests hors-ligne uniquement : aucun appel réel à `infer_area.py` (le lanceur est injectable).
- `README.md` est entièrement en CRLF : tout ajout doit rester en CRLF (vérifier : `.venv\Scripts\python -c "d=open('README.md','rb').read(); assert d.count(b'\r\n')==d.count(b'\n'); print('CRLF ok')"`).
- Chaque message de commit se termine par la ligne `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

- Un département déjà terminé n'est jamais relancé ; un département en pause reprend ; un dossier sans livrable ni point de reprise démarre à zéro (Task 1).
- Une pause arrête la boucle sans fusion, sans enchaîner ; un échec n'arrête pas la boucle (Task 1).
- Les poids ou un fichier de faux manquant font échouer le script avant le premier calcul (Task 1).
- Fusion : le seuil dépend de la couche du département ; un même point vu par deux départements voisins n'apparaît qu'une fois avec le meilleur score ; un département manquant est signalé sans empêcher la fusion partielle ; un candidat sans score est écarté (Task 1).
- Ctrl+C : le sous-processus n'est pas tué avant d'avoir pu écrire son point de reprise (Task 1).

---

### Task 1: `scripts/rattrapage.py`, tests et runbook

**Files:**
- Create: `scripts/rattrapage.py`
- Modify: `README.md` (sous-section à la fin de la section « Rattrapage des départements déjà traités », donc à la fin du fichier, CRLF)
- Test: `tests/test_rattrapage.py`

**Interfaces:**
- Consumes: `detection_ortho.checkpoint.load(out_dir)` (dict ou None), `detection_ortho.geo.dedup_points(points, radius_m)`, `detection_ortho.geojson_io.write_geojson(fc, path)`, `detection_ortho.maproulette.to_maproulette_tasks(points, instruction)`.
- Produces (dans `scripts/rattrapage.py`): `DEPARTEMENTS`, `dept_out(root, insee)`, `is_done(out)`, `is_paused(out)`, `build_command(dep, args)`, `preflight(deps, args) -> list[str]`, `run_infer(cmd) -> int`, `run_all(deps, args, run=run_infer) -> dict`, `load_candidates(out)`, `merge_challenge(root, deps, min_score_2026, min_score_standard, dedup_m=10.0, instruction=INSTRUCTION) -> dict`, `main(argv=None) -> int`.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_rattrapage.py` :

```python
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
```

- [ ] **Step 2: Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_rattrapage.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'rattrapage'`).

- [ ] **Step 3: Implémenter**

Créer `scripts/rattrapage.py` :

```python
"""Rattrapage multi-départements : un run de plusieurs jours, un seul challenge.

Enchaîne `infer_area.py` sur plusieurs départements (un sous-processus par
département, chacun dans `<out-root>/<insee>` avec ses faux déjà rejetés), puis
fusionne leurs candidats en UN challenge MapRoulette.

Reprise : relancer la MÊME commande. Un département terminé est sauté, un
département en pause reprend à son point de reprise. Une pause (Ctrl+C, ou un
fichier STOP déposé dans le dossier du département en cours) arrête aussi ce
script, sans enchaîner sur le suivant. Un échec n'empêche pas les départements
suivants ; il est listé à la fin.

Usage:
    python scripts/rattrapage.py
    python scripts/rattrapage.py --only 18 28 --dry-run
    python scripts/rattrapage.py --merge-only

Génération de FICHIERS uniquement : aucun upload MapRoulette.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_ortho import checkpoint
from detection_ortho.geo import dedup_points
from detection_ortho.geojson_io import write_geojson
from detection_ortho.maproulette import to_maproulette_tasks

L26 = "ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026"
INSTRUCTION = ("Une citerne semble présente ici sur l'ortho IGN. "
               "Vérifiez et ajoutez-la à OSM si confirmé.")
# Les départements couverts par l'ortho 2026 d'abord : ce qui se termine en
# premier est ce qui apporte l'imagerie fraîche. Le 44 et le 49 n'ont pas de 2026.
DEPARTEMENTS = [
    {"insee": "18", "nom": "Cher", "layer": L26},
    {"insee": "28", "nom": "Eure-et-Loir", "layer": L26},
    {"insee": "36", "nom": "Indre", "layer": L26},
    {"insee": "37", "nom": "Indre-et-Loire", "layer": L26},
    {"insee": "41", "nom": "Loir-et-Cher", "layer": L26},
    {"insee": "45", "nom": "Loiret", "layer": L26},
    {"insee": "44", "nom": "Loire-Atlantique", "layer": None},
    {"insee": "49", "nom": "Maine-et-Loire", "layer": None},
]
INFER_AREA = Path(__file__).resolve().parent / "infer_area.py"
DELIVERABLE = "detected_only.geojson"


def dept_out(root, insee) -> Path:
    return Path(root) / insee


def is_paused(out) -> bool:
    """Un point de reprise lisible existe : le run n'est pas terminé."""
    return checkpoint.load(out) is not None


def is_done(out) -> bool:
    """Livrable présent et plus de point de reprise (infer_area l'efface à la fin)."""
    return (Path(out) / DELIVERABLE).is_file() and not is_paused(out)


def build_command(dep, args) -> list[str]:
    cmd = [sys.executable, str(INFER_AREA),
           "--boundary", dep["nom"], "--insee", dep["insee"],
           "--weights", str(args.weights), "--conf", str(args.conf),
           "--known-false", str(Path(args.verdicts_dir) / f"verdicts_{dep['insee']}.csv"),
           "--device", args.device, "--workers", str(args.workers),
           "--cache-gb", str(args.cache_gb),
           "--out", str(dept_out(args.out_root, dep["insee"]))]
    if dep["layer"]:
        cmd += ["--layer", dep["layer"]]
    return cmd


def preflight(deps, args) -> list[str]:
    """Erreurs détectables avant tout calcul (poids, fichiers de faux)."""
    errors = []
    if not Path(args.weights).is_file():
        errors.append(f"poids introuvables : {args.weights}")
    for dep in deps:
        kf = Path(args.verdicts_dir) / f"verdicts_{dep['insee']}.csv"
        if not kf.is_file():
            errors.append(f"fichier de faux introuvable : {kf}")
    return errors


def run_infer(cmd) -> int:
    """Lance infer_area en direct (sa progression reste visible).

    Sur Ctrl+C, le sous-processus reçoit aussi le signal et enregistre son point
    de reprise : on l'attend (jusqu'à 300 s) au lieu de le tuer tout de suite.
    """
    proc = subprocess.Popen(cmd)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("\nInterruption : attente de l'arrêt propre d'infer_area "
              "(enregistrement du point de reprise)...", flush=True)
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise


def run_all(deps, args, run=run_infer) -> dict:
    """Lance les départements dans l'ordre ; retourne {insee: statut}.

    Statuts : « déjà terminé », « terminé », « pause », « échec (code N) »,
    « non traité » (après une pause). Une pause arrête la boucle ; un échec non.
    """
    status: dict[str, str] = {}
    for i, dep in enumerate(deps):
        out = dept_out(args.out_root, dep["insee"])
        label = f"{dep['insee']} {dep['nom']}"
        if is_done(out):
            status[dep["insee"]] = "déjà terminé"
            print(f"=== Département {label} : déjà terminé, sauté ===", flush=True)
            continue
        couche = "ortho 2026" if dep["layer"] else "ortho habituelle"
        print(f"\n=== Département {label} ({couche}) ===", flush=True)
        t0 = time.time()
        code = run(build_command(dep, args))
        heures = (time.time() - t0) / 3600
        if code == 0 and is_done(out):
            status[dep["insee"]] = "terminé"
            print(f"=== {label} terminé en {heures:.1f} h ===", flush=True)
        elif code == 0 and is_paused(out):
            status[dep["insee"]] = "pause"
            for rest in deps[i + 1:]:
                status[rest["insee"]] = "non traité"
            print(f"=== {label} en pause : on s'arrête ici ===", flush=True)
            break
        else:
            status[dep["insee"]] = f"échec (code {code})"
            print(f"=== {label} : ÉCHEC (code {code}), on passe au suivant ===",
                  file=sys.stderr, flush=True)
    return status


def load_candidates(out) -> list[dict]:
    fc = json.loads((Path(out) / DELIVERABLE).read_text(encoding="utf-8"))
    points = []
    for feat in fc.get("features", []):
        lon, lat = feat["geometry"]["coordinates"][:2]
        points.append({"lon": lon, "lat": lat,
                       "score": feat.get("properties", {}).get("score")})
    return points


def merge_challenge(root, deps, min_score_2026, min_score_standard,
                    dedup_m=10.0, instruction=INSTRUCTION) -> dict:
    """Fusionne les candidats des départements terminés en un seul challenge.

    Retourne {"fc", "stats", "manquants", "doublons"}. Le seuil dépend de la
    couche du département ; un candidat sans score est écarté ; les doublons
    entre départements voisins (<= dedup_m) gardent le meilleur score.
    """
    kept_all: list[dict] = []
    stats, manquants = [], []
    for dep in deps:
        out = dept_out(root, dep["insee"])
        if not is_done(out):
            manquants.append(dep["insee"])
            continue
        seuil = min_score_2026 if dep["layer"] else min_score_standard
        cands = load_candidates(out)
        kept = [dict(p, dept=dep["insee"]) for p in cands
                if p["score"] is not None and p["score"] >= seuil]
        stats.append({"insee": dep["insee"], "nom": dep["nom"], "seuil": seuil,
                      "candidats": len(cands), "retenus": len(kept)})
        kept_all += kept
    merged = dedup_points(kept_all, dedup_m)
    fc = to_maproulette_tasks(merged, instruction)
    for feat, p in zip(fc["features"], merged):
        feat["properties"]["dept"] = p["dept"]
    return {"fc": fc, "stats": stats, "manquants": manquants,
            "doublons": len(kept_all) - len(merged)}


def format_recap(res, status=None) -> str:
    lines = ["Récapitulatif du rattrapage", ""]
    for s in res["stats"]:
        lines.append(f"  {s['insee']} {s['nom']:<18} seuil {s['seuil']:.2f} : "
                     f"{s['retenus']} retenue(s) sur {s['candidats']} candidate(s)")
    lines.append("")
    lines.append(f"  Doublons entre départements fusionnés : {res['doublons']}")
    lines.append(f"  Tâches du challenge : {len(res['fc']['features'])}")
    if res["manquants"]:
        lines.append(f"  ⚠ Départements non terminés (absents du challenge) : "
                     f"{', '.join(res['manquants'])}")
    if status:
        echecs = [f"{k} ({v})" for k, v in status.items() if v.startswith("échec")]
        if echecs:
            lines.append(f"  ⚠ Échecs : {', '.join(echecs)}")
    return "\n".join(lines)


def _write_merge(deps, args, status=None) -> None:
    res = merge_challenge(args.out_root, deps, args.min_score_2026,
                          args.min_score_standard, args.dedup_m)
    out_path = Path(args.out_root) / "challenge_rattrapage.geojson"
    write_geojson(res["fc"], out_path)
    recap = format_recap(res, status)
    (Path(args.out_root) / "recap_rattrapage.txt").write_text(recap + "\n", encoding="utf-8")
    print("\n" + recap)
    print(f"\nChallenge écrit dans {out_path}. À importer manuellement dans MapRoulette.")


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-root", type=Path, default=Path("inference_rattrapage"),
                    help="dossier racine : un sous-dossier par département")
    ap.add_argument("--weights", type=Path, default=Path("models/citernes-yolov8n.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cache-gb", type=float, default=10.0,
                    help="plafond disque du cache de tuiles de chaque run")
    ap.add_argument("--verdicts-dir", type=Path, default=Path("verdicts_maproulette"),
                    help="dossier des verdicts_<insee>.csv (faux déjà rejetés)")
    ap.add_argument("--only", nargs="+", metavar="INSEE", default=None,
                    help="ne traiter que ces départements (ex. 18 28)")
    ap.add_argument("--min-score-2026", type=float, default=0.35,
                    help="seuil du challenge pour les départements sur l'ortho 2026")
    ap.add_argument("--min-score-standard", type=float, default=0.60,
                    help="seuil du challenge pour les départements sur l'ortho habituelle")
    ap.add_argument("--dedup-m", type=float, default=10.0,
                    help="rayon (m) de fusion des doublons entre départements")
    ap.add_argument("--merge-only", action="store_true",
                    help="ne rien lancer : fusionner les départements déjà terminés")
    ap.add_argument("--dry-run", action="store_true",
                    help="afficher les commandes prévues sans rien lancer")
    args = ap.parse_args(argv)

    deps = DEPARTEMENTS
    if args.only:
        connus = {d["insee"] for d in DEPARTEMENTS}
        inconnus = [c for c in args.only if c not in connus]
        if inconnus:
            ap.error(f"département(s) inconnu(s) : {', '.join(inconnus)} "
                     f"(connus : {', '.join(sorted(connus))})")
        deps = [d for d in DEPARTEMENTS if d["insee"] in args.only]

    if args.merge_only:
        _write_merge(deps, args)
        return 0

    if args.dry_run:
        for dep in deps:
            if is_done(dept_out(args.out_root, dep["insee"])):
                print(f"# {dep['insee']} {dep['nom']} : déjà terminé, serait sauté")
                continue
            print(" ".join(f'"{c}"' if " " in c else c for c in build_command(dep, args)))
        return 0

    errors = preflight(deps, args)
    if errors:
        for e in errors:
            print(f"Erreur : {e}", file=sys.stderr)
        return 2

    try:
        # `run=run_infer` est résolu ICI, à l'appel : le défaut de run_all est figé
        # à la définition et ne verrait pas un remplacement par les tests.
        status = run_all(deps, args, run=run_infer)
    except KeyboardInterrupt:
        print("\nArrêt demandé : relancez la même commande pour reprendre.", file=sys.stderr)
        return 130

    print("\n=== Bilan ===")
    for dep in deps:
        print(f"  {dep['insee']} {dep['nom']:<18} {status.get(dep['insee'], 'non traité')}")
    if "pause" in status.values():
        print("\nPause : relancez la même commande pour reprendre. "
              "(Aucune fusion tant que la pause n'est pas terminée ; --merge-only "
              "pour fusionner les départements déjà terminés.)")
        return 0
    _write_merge(deps, args, status)
    return 1 if any(v.startswith("échec") for v in status.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Vérifier la réussite**

Run: `.venv\Scripts\python -m pytest tests/test_rattrapage.py -q`
Expected: PASS (23 tests). Si un test du plan échoue pour une raison qui semble un défaut du test lui-même, le signaler plutôt que le modifier en silence.

- [ ] **Step 5: Lecture du script sur de vrais arguments (sans rien lancer)**

Run: `.venv\Scripts\python scripts\rattrapage.py --dry-run --only 18 49`
Expected: deux lignes de commande `infer_area.py` (la première avec `--layer ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026`, la seconde sans `--layer`), code de sortie 0.

- [ ] **Step 6: Runbook dans le README**

À la fin de `README.md`, en CRLF, ajouter la sous-section suivante (à la suite de la section « Rattrapage des départements déjà traités ») :

````markdown

### Un seul challenge pour tous les départements (run de plusieurs jours, week-end)

`scripts/rattrapage.py` enchaîne `infer_area.py` sur les 8 départements déjà traités, un à la fois, puis
fabrique **un seul** challenge. Ordre : les départements couverts par l'ortho 2026 (18, 28, 36, 37, 41, 45),
puis le 44 et le 49 sur l'ortho habituelle. Chaque département écrit dans `inference_rattrapage/<insee>`, avec
ses faux déjà rejetés (`verdicts_maproulette/verdicts_<insee>.csv`, vérifiés avant tout calcul).

    .venv\Scripts\python scripts\rattrapage.py

- **Durée** : de l'ordre de 15 à 19 h de CPU par département (extrapolé du 49, non mesuré sur la 2026) :
  un week-end en traite 3 ou 4, le reste continue à la relance suivante.
- **Reprise** : relancer la **même commande**. Un département terminé est sauté, un département en pause
  reprend à son point de reprise.
- **Pause** : Ctrl+C, ou déposer un fichier `STOP` dans le dossier du département en cours
  (`New-Item inference_rattrapage\18\STOP -ItemType File`). Le script s'arrête sans enchaîner sur le
  département suivant ; aucun travail n'est perdu.
- **Échec** : un département en échec (panne réseau, erreur d'emprise) n'empêche pas les suivants ; il est
  listé dans le bilan et le code de sortie vaut 1. Relancer reprend les départements non terminés.
- **Un département à refaire de zéro** : supprimer son dossier `inference_rattrapage/<insee>`.
- **Résultat** : `inference_rattrapage/challenge_rattrapage.geojson` (un seul fichier, chaque tâche porte son
  département) et `recap_rattrapage.txt`. Seuils : 0,35 en ortho 2026, 0,60 en ortho habituelle ; les doublons
  entre départements voisins (à moins de 10 m) sont fusionnés en gardant le meilleur score. Import manuel
  dans MapRoulette.

Options utiles : `--only 18 28` (quelques départements), `--dry-run` (afficher les commandes sans rien
lancer), `--merge-only` (fusionner les départements déjà terminés sans rien relancer, par exemple pour
publier un premier challenge en cours de route), `--min-score-2026` et `--min-score-standard`.
````

Puis vérifier le CRLF (commande dans les contraintes globales).

- [ ] **Step 7: Suite complète puis commit**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS (tous les tests).

```bash
git add scripts/rattrapage.py tests/test_rattrapage.py README.md
git commit -m "feat: rattrapage multi-départements — un run reprenable et un seul challenge

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```
