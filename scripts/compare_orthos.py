"""Deux orthos pour un même département : l'union ou l'accord valent-ils le coup ?

Infère le même modèle, aux mêmes points labellisés, sur deux couches WMTS, puis
compare à rappel égal : chaque couche seule, union (max des scores) et accord
(min des scores). Seuls les points disponibles sur les deux couches comptent.
Lecture/impression seulement ; les scores sont écrits en CSV pour être rejoués.

Usage:
    python scripts/compare_orthos.py --weights models/citernes-yolov8n.pt \
        --verdicts verdicts_maproulette/verdicts_36.csv --device 0
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from detection_ortho.compare import sweep_precision_recall
from detection_ortho.dataset import assemble_window, parse_verdicts, prefetch_points
from detection_ortho.infer import boxes_to_points, max_score_near, result_to_boxes
from detection_ortho.tiles import LAYER

LAYER_2026 = "ORTHOIMAGERY.ORTHOPHOTOS.RVB-EXPRESS.2026"


def scores_for_layer(model, verdicts, layer, args) -> dict[int, float]:
    """Meilleur score près du point, par index de verdict ; absent = fenêtre en échec."""
    errors = prefetch_points(verdicts, args.cache, [layer], workers=args.workers,
                             zoom=args.zoom, window_px=args.window)
    print(f"[{layer}] tuiles préchargées : {len(errors)} en échec.", flush=True)
    out: dict[int, float] = {}
    for i, v in enumerate(verdicts):
        try:
            img, ogx, ogy = assemble_window(v["lon"], v["lat"], args.zoom, args.window,
                                            args.cache, layer=layer)
        except Exception:  # noqa: BLE001 — fenêtre sans imagerie : point écarté
            continue
        if img.mean() > 250:  # fenêtre blanche (hors couverture)
            continue
        res = model.predict(img, conf=args.conf_min, device=args.device, verbose=False)
        pts = boxes_to_points(result_to_boxes(res[0].boxes), ogx, ogy, args.zoom)
        out[i] = max_score_near(pts, v["lon"], v["lat"], args.radius)
        print(f"  {len(out)} points", end="\r", file=sys.stderr, flush=True)
    return out


def table(titre: str, scored: list, thresholds: list[float]) -> None:
    print(f"\n--- {titre} ---")
    print("  seuil  précision  rappel   tp   fp")
    for r in sweep_precision_recall(scored, thresholds):
        print(f"  {r['conf']:.2f}    {r['precision']:.3f}     {r['recall']:.3f}   "
              f"{r['tp']:>3}  {r['fp']:>3}")


def fp_at_recall(scored: list, targets: list[float]) -> dict[float, tuple]:
    """Pour chaque rappel cible : (faux positifs, seuil) minimaux qui l'atteignent."""
    rows = sweep_precision_recall(scored, [round(0.05 + 0.01 * k, 2) for k in range(86)])
    out = {}
    for tgt in targets:
        ok = [r for r in rows if r["recall"] >= tgt]
        out[tgt] = (min((r["fp"], r["conf"]) for r in ok) if ok else None)
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--layer-a", default=LAYER, help="ortho habituelle")
    ap.add_argument("--layer-b", default=LAYER_2026, help="ortho 2026")
    ap.add_argument("--conf-min", type=float, default=0.05)
    ap.add_argument("--radius", type=float, default=25.0)
    ap.add_argument("--window", type=int, default=640)
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--cache", type=Path, default=Path("tiles_cache/eval"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("out/compare_orthos.csv"))
    args = ap.parse_args()

    verdicts = parse_verdicts(args.verdicts.read_text(encoding="utf-8").splitlines())
    print(f"{len(verdicts)} point(s) labellisé(s)", flush=True)
    model = YOLO(str(args.weights))
    sa = scores_for_layer(model, verdicts, args.layer_a, args)
    sb = scores_for_layer(model, verdicts, args.layer_b, args)

    ids = sorted(set(sa) & set(sb))
    vrai = {i: verdicts[i].get("verdict") == "vrai" for i in ids}
    print(f"\nPoints disponibles sur les deux couches : {len(ids)} sur {len(verdicts)} "
          f"({sum(vrai.values())} vrais, {len(ids) - sum(vrai.values())} faux)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "lat", "lon", "vrai", "score_a", "score_b"])
        for i in ids:
            w.writerow([i, verdicts[i]["lat"], verdicts[i]["lon"], int(vrai[i]), sa[i], sb[i]])
    print(f"Scores écrits dans {args.out}")

    ths = [round(0.05 * k, 2) for k in range(1, 18)]
    combos = {
        "A seule (ortho habituelle)": [(sa[i], vrai[i]) for i in ids],
        "B seule (ortho 2026)": [(sb[i], vrai[i]) for i in ids],
        "Union (max des scores)": [(max(sa[i], sb[i]), vrai[i]) for i in ids],
        "Accord (min des scores)": [(min(sa[i], sb[i]), vrai[i]) for i in ids],
    }
    for titre, scored in combos.items():
        table(titre, scored, ths)

    print("\n=== Faux positifs minimaux à rappel égal (FP, seuil) ===")
    targets = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    res = {t: fp_at_recall(s, targets) for t, s in combos.items()}
    print(f"  {'rappel >=':<10}" + "".join(f"{t[:22]:<26}" for t in combos))
    for tgt in targets:
        cells = []
        for t in combos:
            v = res[t][tgt]
            cells.append("inatteignable" if v is None else f"{v[0]} FP (seuil {v[1]:.2f})")
        print(f"  {tgt:<10}" + "".join(f"{c:<26}" for c in cells))


if __name__ == "__main__":
    main()
