import numpy as np

from detection_ortho.dataset import window_is_blank


def test_all_white_is_blank():
    assert window_is_blank(np.full((64, 64, 3), 255, np.uint8))


def test_gray_window_is_not_blank():
    assert not window_is_blank(np.full((64, 64, 3), 128, np.uint8))


def test_mostly_white_with_a_few_pixels_is_blank():
    img = np.full((100, 100, 3), 255, np.uint8)
    img[:1, :30] = 100          # 30 pixels sur 10 000 = 0,3 %
    assert window_is_blank(img)


def test_a_tenth_of_non_white_pixels_is_not_blank():
    img = np.full((100, 100, 3), 255, np.uint8)
    img[:10, :] = 100           # 10 %
    assert not window_is_blank(img)
