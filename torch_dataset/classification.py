from __future__ import annotations

from typing import Any
from typing import Tuple

from dataclasses import dataclass
from pathlib import Path

from torch_dataset import DatasetMixin


@dataclass(frozen=True)
class Image:
    path: Path
    class_: int


class Dataset(DatasetMixin):
    def __init__(self, path: Path) -> None:
        import json

        super(Dataset, self).__init__()

        with path.open() as f:
            json_dataset = json.load(f)

        self.__seq_class = extract_seq_class(json_dataset)
        self.__seq_image = extract_seq_image(json_dataset, path.parent)

    def __getitem__(self, index: int) -> Image:
        return self.__seq_image[index]

    def __len__(self) -> int:
        return len(self.__seq_image)

    @property
    def seq_class(self) -> Tuple[str, ...]:
        return self.__seq_class


def extract_seq_image(json_dataset: Any, path_root: Path) -> Tuple[Image, ...]:
    return tuple(map(lambda x: convert_image(x, path_root), json_dataset["images"]))


def extract_seq_class(json_dataset: Any) -> Tuple[str, ...]:
    return tuple(map(str, json_dataset["classes"]))


def convert_image(json_image: Any, path_root: Path) -> Image:
    return Image(
        path=path_root.joinpath(json_image["path"]).resolve(),
        class_=int(json_image["class_"]),
    )
