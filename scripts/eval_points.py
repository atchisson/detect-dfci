"""Test terrain NIR — évalue un modèle YOLO aux points labellisés de verdicts.

Pour chaque point de verdicts.csv, assemble la fenêtre (RVB, ou [R,G,NIR] avec
--nir), lance le modèle, et note si le modèle « tire » à cet emplacement. Les
points tirés alimentent compare_to_verdicts (FP supprimés / vrais conservés).
Aucun upload, lecture/impression seulement.

Usage:
    python scripts/eval_points.py --weights runs/citernes/weights/best.pt \
        --verdicts verdicts.csv --conf 0.55
    python scripts/eval_points.py --weights runs/citernes_nir/weights/best.pt \
        --verdicts verdicts.csv --conf 0.55 --nir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from detection_ortho.dataset import assemble_window, compose_rgn, parse_verdicts
from detection_ortho.tiles import LAYER_IRC
from detection_ortho.infer import result_to_boxes, boxes_to_points, is_detected_near
from detection_ortho.compare import compare_to_verdicts


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--nir", action="store_true",
                    help="composer [R,G,NIR] (bleu remplacé par le NIR de l'IRC)")
    ap.add_argument("--radius", type=float, default=25.0)
    ap.add_argument("--window", type=int, default=640)
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--cache", type=Path, default=Path("tiles_cache/eval"))
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    verdicts = parse_verdicts(args.verdicts.read_text(encoding="utf-8").splitlines())
    print(f"{len(verdicts)} point(s) à évaluer "
          f"({'[R,G,NIR]' if args.nir else 'RVB'}, conf {args.conf}).")

    model = YOLO(str(args.weights))
    detections: list[dict] = []
    for i, v in enumerate(verdicts, 1):
        lon, lat = v["lon"], v["lat"]
        try:
            img, ogx, ogy = assemble_window(lon, lat, args.zoom, args.window, args.cache)
            if args.nir:
                irc, _, _ = assemble_window(lon, lat, args.zoom, args.window,
                                            args.cache, layer=LAYER_IRC)
                img = compose_rgn(img, irc)
        except Exception as exc:  # noqa: BLE001
            print(f"  point {i}: échec fenêtre ({exc})", file=sys.stderr)
            continue
        res = model.predict(img, conf=args.conf, device=args.device, verbose=False)
        pts = boxes_to_points(result_to_boxes(res[0].boxes), ogx, ogy, args.zoom)
        if is_detected_near(pts, lon, lat, args.radius):
            detections.append({"lon": lon, "lat": lat})
        print(f"  {i}/{len(verdicts)}", end="\r", file=sys.stderr, flush=True)

    r = compare_to_verdicts(detections, verdicts, args.radius)
    print("\n=== Évaluation aux points de verdict ===")
    print(f"  Points où le modèle tire       : {r['n_candidates_new']}")
    print(f"  Faux positifs supprimés        : {r['fp_suppressed']}/{r['fp_total']}")
    print(f"  Faux positifs encore détectés  : {r['fp_still_detected']}/{r['fp_total']}")
    print(f"  Vrais positifs conservés       : {r['tp_kept']}/{r['tp_total']}")


if __name__ == "__main__":
    main()
