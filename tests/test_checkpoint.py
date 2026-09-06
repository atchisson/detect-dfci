import json

from detection_ortho import checkpoint

CENTERS = [(0.001 * i, 47.5) for i in range(50)]
PARAMS = {"boundary": "Maine-et-Loire", "conf": 0.4, "overlap": 0.2}


def test_fingerprint_stable_et_sensible_aux_parametres():
    a = checkpoint.fingerprint(CENTERS, PARAMS)
    assert a == checkpoint.fingerprint(CENTERS, dict(PARAMS))  # déterministe
    assert a != checkpoint.fingerprint(CENTERS, {**PARAMS, "conf": 0.25})
    assert a != checkpoint.fingerprint(CENTERS[:-1], PARAMS)   # grille décalée
    assert a != checkpoint.fingerprint([(9.9, 9.9)] + CENTERS[1:], PARAMS)


def test_fingerprint_sur_liste_vide():
    assert checkpoint.fingerprint([], PARAMS)  # ne lève pas


def test_save_puis_load_restitue_l_etat(tmp_path):
    det = [{"lon": 1.0, "lat": 47.0, "score": 0.8}]
    fp = checkpoint.fingerprint(CENTERS, PARAMS)
    checkpoint.save(tmp_path, fp, 1234, det)
    ck = checkpoint.load(tmp_path)
    assert ck["fingerprint"] == fp
    assert ck["done"] == 1234
    assert ck["detections"] == det


def test_save_est_atomique_sans_residu(tmp_path):
    checkpoint.save(tmp_path, "abc", 1, [])
    assert (tmp_path / checkpoint.NAME).exists()
    assert not (tmp_path / (checkpoint.NAME + ".tmp")).exists()


def test_load_absent_ou_corrompu_rend_none(tmp_path):
    assert checkpoint.load(tmp_path) is None
    (tmp_path / checkpoint.NAME).write_text("{tronqu", encoding="utf-8")
    assert checkpoint.load(tmp_path) is None


def test_load_refuse_une_version_inconnue(tmp_path):
    (tmp_path / checkpoint.NAME).write_text(
        json.dumps({"version": 999, "fingerprint": "x", "done": 1,
                    "detections": []}), encoding="utf-8")
    assert checkpoint.load(tmp_path) is None


def test_load_refuse_un_done_aberrant(tmp_path):
    (tmp_path / checkpoint.NAME).write_text(
        json.dumps({"version": checkpoint.VERSION, "fingerprint": "x",
                    "done": -5, "detections": []}), encoding="utf-8")
    assert checkpoint.load(tmp_path) is None


def test_clear_supprime_tout(tmp_path):
    checkpoint.save(tmp_path, "abc", 1, [])
    (tmp_path / (checkpoint.NAME + ".tmp")).write_text("x", encoding="utf-8")
    checkpoint.clear(tmp_path)
    assert checkpoint.load(tmp_path) is None
    assert not (tmp_path / (checkpoint.NAME + ".tmp")).exists()
    checkpoint.clear(tmp_path)  # idempotent


def test_stop_requested_et_clear_stop(tmp_path):
    assert not checkpoint.stop_requested(tmp_path)
    (tmp_path / "STOP").write_text("", encoding="utf-8")
    assert checkpoint.stop_requested(tmp_path)
    checkpoint.clear_stop(tmp_path)
    assert not checkpoint.stop_requested(tmp_path)
    checkpoint.clear_stop(tmp_path)  # idempotent
