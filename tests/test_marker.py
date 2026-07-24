import cv2
import numpy as np
from detection_ortho.tiles import save_tile_with_marker


def test_marker_modifies_image(tmp_path):
    # Tuile grise unie.
    src = tmp_path / "tile.jpg"
    cv2.imwrite(str(src), np.full((256, 256, 3), 128, dtype=np.uint8))
    out = tmp_path / "out.png"
    save_tile_with_marker(src, 128.0, 128.0, out)
    assert out.exists()
    img = cv2.imread(str(out))
    # Au centre, la croix a modifié des pixels (plus uniformément gris).
    center = img[120:136, 120:136]
    assert center.min() != center.max()
