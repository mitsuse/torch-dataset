from __future__ import annotations


def test_dataset_init_with_valid() -> None:
    from pathlib import Path

    from torch_dataset.classification import Dataset

    path = Path("tests/fixtures/dataset_classification-valid.json")
    dataset = Dataset(path)

    assert len(dataset) == 4
