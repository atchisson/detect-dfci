import numpy as np
from detection_ortho.dataset import compose_rgn


def test_blue_channel_becomes_nir():
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[..., 0] = 10   # B
    rgb[..., 1] = 20   # G
    rgb[..., 2] = 30   # R
    irc = np.zeros((4, 4, 3), np.uint8)
    irc[..., 2] = 200  # canal rouge IRC = NIR
    out = compose_rgn(rgb, irc)
    assert (out[..., 0] == 200).all()   # bleu <- NIR
    assert (out[..., 1] == 20).all()    # G conservé
    assert (out[..., 2] == 30).all()    # R conservé
    # n'altère pas l'entrée
    assert (rgb[..., 0] == 10).all()
