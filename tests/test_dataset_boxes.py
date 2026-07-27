from detection_ortho.dataset import element_to_box, split_indices, DEFAULT_BOX_M


def test_element_to_box_way_uses_polygon():
    el = {"type": "way", "tags": {"emergency": "water_tank"},
          "geometry": [{"lon": 0.60, "lat": 47.30}, {"lon": 0.61, "lat": 47.31}]}
    box = element_to_box(el)
    assert box["bbox_geo"] == (0.60, 47.30, 0.61, 47.31)
    assert abs(box["lon"] - 0.605) < 1e-9 and abs(box["lat"] - 47.305) < 1e-9


def test_element_to_box_node_uses_fixed_size():
    el = {"type": "node", "tags": {}, "lon": 0.653, "lat": 47.33}
    box = element_to_box(el, default_box_m=20.0)
    w, s, e, n = box["bbox_geo"]
    assert abs((n - s) * 111320 - 20.0) < 1.0
    assert box["lon"] == 0.653


def test_split_indices_deterministic_and_partition():
    a = split_indices(100, seed=0)
    b = split_indices(100, seed=0)
    assert a == b  # déterministe
    allidx = sorted(a["train"] + a["val"] + a["test"])
    assert allidx == list(range(100))  # partition complète, sans doublon
    assert len(a["train"]) == 70 and len(a["val"]) == 15 and len(a["test"]) == 15


def test_default_box_m_positive():
    assert DEFAULT_BOX_M > 0
