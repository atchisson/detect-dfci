"""Appariement spatial entre détections et citernes OSM connues."""
from __future__ import annotations

from detection_ortho.geo import haversine_m


def match_detections(
    detections: list[dict], osm_points: list[dict], radius_m: float
) -> dict:
    """Classe détections et points OSM en trois catégories.

    - matched        : détections appariées à un point OSM (< radius_m).
    - detected_only  : détections sans OSM proche -> candidats MapRoulette.
    - osm_only        : points OSM non détectés -> faux négatifs / disparus.

    Chaque point OSM ne peut être apparié qu'une fois (au plus proche voisin).
    Les détections sont traitées par score décroissant pour la stabilité.
    """
    used_osm: set[int] = set()
    matched: list[dict] = []
    detected_only: list[dict] = []

    ordered = sorted(
        detections, key=lambda d: d.get("score", 0.0), reverse=True
    )
    for det in ordered:
        best_i, best_d = None, radius_m
        for i, osm in enumerate(osm_points):
            if i in used_osm:
                continue
            dist = haversine_m(det["lon"], det["lat"], osm["lon"], osm["lat"])
            if dist <= best_d:
                best_i, best_d = i, dist
        if best_i is None:
            detected_only.append(det)
        else:
            used_osm.add(best_i)
            matched.append({"detection": det, "osm": osm_points[best_i]})

    osm_only = [osm for i, osm in enumerate(osm_points) if i not in used_osm]
    return {"matched": matched, "detected_only": detected_only, "osm_only": osm_only}


def compare_to_verdicts(
    detections: list[dict], verdicts: list[dict], radius_m: float
) -> dict:
    """Croise de nouvelles détections avec des verdicts connus (par proximité).

    Mesure le gain avant/après : combien de faux positifs connus ne sont plus
    détectés (fp_suppressed) et combien de vrais positifs restent détectés
    (tp_kept).
    """
    def detected(pt: dict) -> bool:
        return any(
            haversine_m(pt["lon"], pt["lat"], d["lon"], d["lat"]) <= radius_m
            for d in detections
        )

    faux = [v for v in verdicts if v.get("verdict") == "faux"]
    vrai = [v for v in verdicts if v.get("verdict") == "vrai"]
    fp_still = sum(1 for v in faux if detected(v))
    tp_kept = sum(1 for v in vrai if detected(v))
    return {
        "fp_total": len(faux),
        "fp_still_detected": fp_still,
        "fp_suppressed": len(faux) - fp_still,
        "tp_total": len(vrai),
        "tp_kept": tp_kept,
        "n_candidates_new": len(detections),
    }


def sweep_precision_recall(scored, thresholds) -> list:
    """Précision/rappel par seuil. `scored` = liste de (best_score, is_true).

    Un point « tire » à un seuil si best_score >= seuil.
    """
    n_true = sum(1 for _s, t in scored if t)
    rows = []
    for th in thresholds:
        tp = sum(1 for s, t in scored if t and s >= th)
        fp = sum(1 for s, t in scored if (not t) and s >= th)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / n_true if n_true else 0.0
        rows.append({"conf": th, "precision": prec, "recall": rec, "tp": tp, "fp": fp})
    return rows
