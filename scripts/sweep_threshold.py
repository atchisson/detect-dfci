"""Calibration de seuil — précision/rappel vs confiance sur points labellisés.

Infère UNE fois par point (à --conf-min), retient le meilleur score près du
centre, puis balaye le seuil en mémoire. Lecture/impression seulement.

Usage:
    python scripts/sweep_threshold.py --weights runs/citernes/weights/best.pt \
        --verdicts verdicts.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from detection_ortho.dataset import assemble_window, compose_rgn, parse_verdicts
from detection_ortho.tiles import LAYER_IRC
from detection_ortho.infer import result_to_boxes, boxes_to_points, max_score_near
from detection_ortho.compare import sweep_precision_recall


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--nir", action="store_true")
    ap.add_argument("--conf-min", type=float, default=0.05)
    ap.add_argument("--conf-max", type=float, default=0.9)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--radius", type=float, default=25.0)
    ap.add_argument("--window", type=int, default=640)
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--cache", type=Path, default=Path("tiles_cache/eval"))
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    verdicts = parse_verdicts(args.verdicts.read_text(encoding="utf-8").splitlines())
    print(f"{len(verdicts)} point(s), inférence à conf {args.conf_min} "
          f"({'[R,G,NIR]' if args.nir else 'RVB'})...")

    model = YOLO(str(args.weights))
    scored: list = []
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
        res = model.predict(img, conf=args.conf_min, device=args.device, verbose=False)
        pts = boxes_to_points(result_to_boxes(res[0].boxes), ogx, ogy, args.zoom)
        scored.append((max_score_near(pts, lon, lat, args.radius),
                       v.get("verdict") == "vrai"))
        print(f"  {i}/{len(verdicts)}", end="\r", file=sys.stderr, flush=True)

    n = int(round((args.conf_max - args.conf_min) / args.step)) + 1
    thresholds = [round(args.conf_min + k * args.step, 4) for k in range(max(n, 1))]
    rows = sweep_precision_recall(scored, thresholds)
    print("\n=== Précision/rappel vs seuil ===")
    print("  seuil  précision  rappel   tp   fp")
    for r in rows:
        print(f"  {r['conf']:.2f}    {r['precision']:.3f}     {r['recall']:.3f}   "
              f"{r['tp']:>3}  {r['fp']:>3}")


if __name__ == "__main__":
    main()
