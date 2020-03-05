from __future__ import annotations

from typing import Any
from typing import Tuple

from dataclasses import dataclass
from pathlib import Path

from torch_dataset import DatasetMixin


@dataclass(frozen=True)
class Image:
    path: Path
    seq_annotation: Tuple[Annotation, ...]


@dataclass(frozen=True)
class Annotation:
    bbox: Bbox
    class_: int


@dataclass(frozen=True)
class Bbox:
    top: float
    bottom: float
    left: float
    right: float


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
        seq_annotation=tuple(map(convert_annotation, json_image["annotations"])),
    )


def convert_annotation(json_annotation: Any) -> Annotation:
    return Annotation(
        bbox=convert_bbox(json_annotation["bbox"]),
        class_=int(json_annotation["class"]),
    )


def convert_bbox(json_bbox: Any) -> Bbox:
    return Bbox(
        top=float(json_bbox["top"]),
        bottom=float(json_bbox["bottom"]),
        left=float(json_bbox["left"]),
        right=float(json_bbox["right"]),
    )
