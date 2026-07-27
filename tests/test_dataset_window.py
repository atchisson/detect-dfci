from pathlib import Path

import cv2
import numpy as np

from detection_ortho.dataset import window_tiles, assemble_window, write_chip, lonlat_to_global_px


def test_window_tiles_cover_window():
    lon, lat, z, win = 0.653, 47.33, 19, 640
    tiles, ogx, ogy = window_tiles(lon, lat, z, win)
    gx, gy = lonlat_to_global_px(lon, lat, z)
    # origine = coin haut-gauche de la fenêtre centrée
    assert abs(ogx - (gx - win / 2)) < 1e-6
    assert abs(ogy - (gy - win / 2)) < 1e-6
    # assez de tuiles pour couvrir 640 px avec des tuiles de 256
    assert len(tiles) >= 9  # 3x3 minimum


def test_assemble_window_from_cached_tiles(tmp_path):
    lon, lat, z, win = 0.653, 47.33, 19, 640
    tiles, ogx, ogy = window_tiles(lon, lat, z, win)
    # pré-écrire des tuiles unies en cache pour éviter le réseau
    for (x, y) in tiles:
        img = np.full((256, 256, 3), 120, np.uint8)
        cv2.imwrite(str(tmp_path / f"{z}_{x}_{y}.jpg"), img)
    win_img, gx0, gy0 = assemble_window(lon, lat, z, win, tmp_path)
    assert win_img.shape == (win, win, 3)
    assert (gx0, gy0) == (ogx, ogy)


def test_write_chip_creates_image_and_label(tmp_path):
    img = np.zeros((640, 640, 3), np.uint8)
    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    write_chip(img, ["0 0.5 0.5 0.2 0.2"], imgs, lbls, "citerne_000")
    assert (imgs / "citerne_000.jpg").exists()
    assert (lbls / "citerne_000.txt").read_text().strip() == "0 0.5 0.5 0.2 0.2"
    # négatif : label vide
    write_chip(img, [], imgs, lbls, "neg_000")
    assert (lbls / "neg_000.txt").read_text() == ""
