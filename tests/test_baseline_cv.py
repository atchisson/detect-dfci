import cv2
import numpy as np
from detection_ortho.baseline_cv import (
    detect_in_image,
    default_params,
    DetectionParams,
)


def _canvas():
    # Fond vert « végétation » sombre.
    # Cette couleur doit rester HORS du seuil HSV [hsv_low..hsv_high] utilisé par les
    # tests (sa saturation est sous la borne basse) pour ne pas être segmentée comme
    # premier plan.
    return np.full((256, 256, 3), (60, 90, 60), dtype=np.uint8)  # BGR


def test_detects_bright_green_rectangle():
    img = _canvas()
    # Citerne factice : rectangle vert vif bien distinct.
    cv2.rectangle(img, (100, 110), (140, 150), (60, 200, 60), thickness=-1)
    params = DetectionParams(
        hsv_low=(40, 120, 80), hsv_high=(80, 255, 255),
        min_area=200, max_area=20000, min_aspect=0.3, max_aspect=3.0,
    )
    dets = detect_in_image(img, params)
    assert len(dets) == 1
    d = dets[0]
    assert 100 <= d["px"] <= 140
    assert 110 <= d["py"] <= 150


def test_ignores_too_small_blob():
    img = _canvas()
    cv2.rectangle(img, (10, 10), (14, 14), (60, 200, 60), thickness=-1)  # minuscule
    params = DetectionParams(
        hsv_low=(40, 120, 80), hsv_high=(80, 255, 255),
        min_area=200, max_area=20000, min_aspect=0.3, max_aspect=3.0,
    )
    assert detect_in_image(img, params) == []


def test_default_params_returns_dataclass():
    p = default_params()
    assert isinstance(p, DetectionParams)
    assert p.min_area < p.max_area
