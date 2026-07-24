"""Baseline de détection sans deep learning : seuillage HSV + filtrage forme.

Fragile par nature (ombres, bâches, toits de même teinte) : sert de référence
pédagogique et de premier point de comparaison OSM, pas de solution finale.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectionParams:
    hsv_low: tuple[int, int, int]
    hsv_high: tuple[int, int, int]
    min_area: float
    max_area: float
    min_aspect: float
    max_aspect: float


def default_params() -> DetectionParams:
    """Valeurs de départ, à affiner en observant les vraies imagettes (Jalon 1)."""
    return DetectionParams(
        hsv_low=(35, 80, 60),
        hsv_high=(85, 255, 255),
        min_area=300,
        max_area=30000,
        min_aspect=0.25,
        max_aspect=4.0,
    )


def detect_in_image(img_bgr: np.ndarray, params: DetectionParams) -> list[dict]:
    """Détecte les objets correspondant aux critères couleur/forme.

    Retourne un centroïde par objet retenu : {px, py, area, score}.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(params.hsv_low), np.array(params.hsv_high))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets: list[dict] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params.min_area or area > params.max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h else 0
        if aspect < params.min_aspect or aspect > params.max_aspect:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        dets.append({"px": cx, "py": cy, "area": area, "score": 1.0})
    return dets
