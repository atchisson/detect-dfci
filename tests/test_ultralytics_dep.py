def test_yolo_importable():
    from ultralytics import YOLO
    assert YOLO is not None
