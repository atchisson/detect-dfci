from detection_ortho.infer import result_to_boxes


class _FakeBox:
    def __init__(self, cx, cy, score):
        self.xywh = [[cx, cy, 10.0, 10.0]]
        self.conf = [score]


def test_extracts_center_and_score():
    boxes = [_FakeBox(320.0, 200.0, 0.9), _FakeBox(100.0, 50.0, 0.5)]
    out = result_to_boxes(boxes)
    assert out == [(320.0, 200.0, 0.9), (100.0, 50.0, 0.5)]


def test_empty():
    assert result_to_boxes([]) == []
