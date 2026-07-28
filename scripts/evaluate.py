"""Jalon 2 — Évalue le détecteur sur le jeu de test tenu à l'écart.

Usage:
    python scripts/evaluate.py --weights runs/citernes/weights/best.pt \
        --data dataset/data.yaml --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _test_images_dir(data_yaml) -> Path:
    """Répertoire des images de test à partir du data.yaml (path + test)."""
    import yaml  # ultralytics embarque pyyaml

    d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    return Path(d["path"]) / d["test"]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data", type=str, default="dataset/data.yaml")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=Path, default=Path("runs/eval"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split="test", device=args.device)
    b = metrics.box
    print("\n=== Évaluation (jeu de test) ===")
    print(f"  Précision  : {b.mp:.3f}")
    print(f"  Rappel     : {b.mr:.3f}")
    print(f"  mAP@50     : {b.map50:.3f}")
    print(f"  mAP@50-95  : {b.map:.3f}")
    seuil = 0.60
    verdict = "ATTEINT" if b.map50 >= seuil else "NON ATTEINT"
    print(f"  Critère mAP@50 >= {seuil} : {verdict}")

    # Prédictions annotées sur le test (inspection visuelle, ex. rejet piscines).
    imgs = _test_images_dir(args.data)
    if imgs.exists():
        out = args.out.resolve()  # absolu : évite l'imbrication sous runs/detect
        model.predict(source=str(imgs), device=args.device, save=True,
                      project=str(out), name="test_preds", exist_ok=True)
        print(f"\nPrédictions annotées : {out / 'test_preds'}")


if __name__ == "__main__":
    main()
