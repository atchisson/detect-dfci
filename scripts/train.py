"""Jalon 2 — Entraîne un YOLOv8n sur le dataset de citernes (local CPU par défaut).

Usage:
    python scripts/train.py --data dataset/data.yaml --epochs 100 --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset/data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu",
                    help="cpu (défaut) ou 0 pour le premier GPU CUDA")
    ap.add_argument("--model", type=str, default="yolov8n.pt")
    ap.add_argument("--project", type=str, default="runs")
    ap.add_argument("--name", type=str, default="citernes")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)  # poids COCO pré-entraînés
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device,
        project=args.project, name=args.name, pretrained=True,
    )
    print(f"Entraînement terminé. Poids : {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
