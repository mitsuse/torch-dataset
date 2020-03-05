from __future__ import annotations


def test_dataset_init_with_valid() -> None:
    from pathlib import Path

    from torch_dataset.detection import Dataset

    path = Path("tests/fixtures/dataset_detection-valid.json")
    dataset = Dataset(path)

    assert len(dataset) == 2
    assert len(dataset[0].seq_annotation) == 2
    assert len(dataset[1].seq_annotation) == 0
