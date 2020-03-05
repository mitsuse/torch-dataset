from __future__ import annotations

from typing import Generic
from typing import Callable
from typing import TypeVar

from torch.utils.data import Dataset as _Dataset


A = TypeVar("A", covariant=True)
B = TypeVar("B", covariant=True)


class Dataset(Generic[A], _Dataset):
    def map(self, f: Callable[[A], B]) -> Dataset[B]:
        ...


class DatasetMixin(Dataset[A]):
    def map(self, f: Callable[[A], B]) -> Dataset[B]:
        get_item: Callable[[int], B] = lambda index: f(self[index])
        N = len(self)
        return MappedDataset(get_item, N)


class MappedDataset(DatasetMixin[A]):
    def __init__(self, get_item: Callable[[int], A], N: int) -> None:
        super(MappedDataset, self).__init__()

        self.__get_item = get_item
        self.__N = N

    def __getitem__(self, index: int) -> A:
        return self.__get_item(index)

    def __len__(self) -> int:
        return self.__N
