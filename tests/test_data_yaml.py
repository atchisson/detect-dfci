from detection_ortho.dataset import write_data_yaml


def test_write_data_yaml(tmp_path):
    out = tmp_path / "data.yaml"
    write_data_yaml(tmp_path / "dataset", out)
    text = out.read_text(encoding="utf-8")
    assert "train: images/train" in text
    assert "val: images/val" in text
    assert "test: images/test" in text
    assert "citerne" in text
    assert "names:" in text
