"""Itération 2 — Compare de nouvelles détections aux verdicts de revue connus.

Usage:
    python scripts/compare_to_verdicts.py \
        --detections inference_out/detected_only.geojson \
        --verdicts verdicts.csv --radius 25

Rapport avant/après : faux positifs supprimés, vrais positifs conservés,
nombre de candidats. Génération/lecture de fichiers uniquement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection_ortho.dataset import parse_verdicts
from detection_ortho.compare import compare_to_verdicts


def _load_points(geojson_path: Path) -> list[dict]:
    fc = json.loads(geojson_path.read_text(encoding="utf-8"))
    pts = []
    for feat in fc.get("features", []):
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        pts.append({"lon": coords[0], "lat": coords[1]})
    return pts


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", type=Path, required=True,
                    help="GeoJSON des nouvelles détections (detected_only.geojson)")
    ap.add_argument("--verdicts", type=Path, required=True,
                    help="CSV de revue (verdicts.csv)")
    ap.add_argument("--radius", type=float, default=25.0)
    args = ap.parse_args()

    detections = _load_points(args.detections)
    verdicts = parse_verdicts(args.verdicts.read_text(encoding="utf-8").splitlines())
    r = compare_to_verdicts(detections, verdicts, radius_m=args.radius)

    print("\n=== Comparaison aux verdicts connus ===")
    print(f"  Candidats (nouveau run)        : {r['n_candidates_new']}")
    print(f"  Faux positifs supprimés        : {r['fp_suppressed']}/{r['fp_total']}")
    print(f"  Faux positifs encore détectés  : {r['fp_still_detected']}/{r['fp_total']}")
    print(f"  Vrais positifs conservés       : {r['tp_kept']}/{r['tp_total']}")


if __name__ == "__main__":
    main()
