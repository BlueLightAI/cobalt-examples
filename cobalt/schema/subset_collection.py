# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import pandas as pd

from cobalt.schema.dataset import CobaltDataset, CobaltDataSubset

# SubsetCollection will mainly be an internal class. The main goal is to have
# something slightly more efficient than a list of CobaltDataSubset objects when
# there are lot of them. Also to allow an EvaluationMetric to be evaluated on a
# bunch of subsets at once nicely, e.g. for getting performance metrics on the
# nodes of a graph. It will probably also be a base class for GroupCollection,
# which will be user-facing and have a bunch of other useful stuff.

AggFn = Callable[[pd.Series], Any]
aggregation_functions: Dict[str, AggFn] = {
    "mean": lambda x: x.mean(),
    "sum": lambda x: x.sum(),
    "mode": lambda x: x.mode()[0],
}

T = TypeVar("T", bound="SubsetCollection")


class SubsetCollection:
    """A collection of subsets of a CobaltDataset."""

    def __init__(
        self,
        source_dataset: CobaltDataset,
        indices: Sequence[Sequence[int]],
        name: Optional[str] = None,
    ):
        self._indices = [np.asarray(idxs, dtype=np.int32) for idxs in indices]
        self.source_dataset = source_dataset
        # TODO: make this instantiation lazy
        self._subset_objects = [source_dataset.subset(idxs) for idxs in self._indices]
        self.name = name

    @classmethod
    def from_indices(
        cls: Type[T],
        source: Union[CobaltDataset, CobaltDataSubset],
        indices: Sequence[Sequence[int]],
        name: Optional[str] = None,
    ) -> T:
        if isinstance(source, CobaltDataSubset):
            indices = [source.indices[idxs] for idxs in indices]
            source_dataset = source.source_dataset
        else:
            source_dataset = source
        return cls(source_dataset, indices, name)

    @classmethod
    def from_subsets(
        cls: Type[T], subsets: Sequence[CobaltDataSubset], name: Optional[str] = None
    ) -> T:
        if len(subsets) == 0:
            raise ValueError("At least one subset must be provided")
        indices = [subset.indices for subset in subsets]
        inst = cls(subsets[0].source_dataset, indices, name)
        inst._subset_objects = list(subsets)
        return inst

    @overload
    def __getitem__(self, subset_index: int) -> CobaltDataSubset: ...

    @overload
    def __getitem__(
        self, subset_index: Union[slice, Sequence[int]]
    ) -> SubsetCollection: ...

    def __getitem__(self, subset_index):
        if isinstance(subset_index, int):
            if subset_index < 0:
                subset_index = subset_index + len(self)
            return self._subset_objects[subset_index]
        if isinstance(subset_index, slice):
            return self.from_subsets(self._subset_objects[subset_index])
        if isinstance(subset_index, Sequence) and type(subset_index) != tuple:
            return self.from_subsets([self[i] for i in subset_index])
        raise IndexError(f"Invalid index {subset_index}.")

    def __iter__(self) -> Iterator[CobaltDataSubset]:
        return (self[i] for i in range(len(self)))

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def n_subsets(self) -> int:
        return len(self._indices)

    def select_col(self, col: str) -> Sequence[pd.Series]:
        """Retrieve the values of a column on each subset."""
        series = self.source_dataset.select_col(col)
        return [series.iloc[idxs] for idxs in self._indices]

    def aggregate_col(
        self,
        col: str,
        method: Optional[Union[Literal["mean", "sum", "mode"], AggFn]] = None,
    ) -> Sequence[float]:
        """Aggregate the values of a column within each subset using the specified method."""
        series = self.source_dataset.select_col(col)
        if method is None:
            method = (
                "mean" if pd.api.types.is_any_real_numeric_dtype(series) else "mode"
            )
        if isinstance(method, str):
            try:
                agg_fn = aggregation_functions[method]
            except KeyError:
                raise ValueError(f"Unsupported aggregation method '{method}'") from None
        elif callable(method):
            agg_fn = method
        else:
            raise ValueError("method must be a string or callable.")

        return [agg_fn(series.iloc[idxs]) for idxs in self._indices]

    def get_array(self, key: str) -> Sequence[np.ndarray]:
        """Retrieve the slice of an array for each subset."""
        arr = self.source_dataset.get_array(key)
        return [arr[idxs] for idxs in self._indices]

    # TODO: other set operations? intersection?
    def concatenate(self) -> CobaltDataSubset:
        """Concatenate all subsets in the collection."""
        if len(self._indices) > 0:
            indices = np.concatenate(self._indices)
        else:
            indices = np.empty(0, dtype=np.int32)
        return self.source_dataset.subset(indices)

    def is_pairwise_disjoint(self):
        """Return True if there are no overlaps between subsets, False otherwise."""
        membership_counts = np.zeros(len(self.source_dataset), dtype=np.int32)
        for idxs in self._indices:
            membership_counts[idxs] += 1
        return np.all(membership_counts < 2)

    def __repr__(self) -> str:
        return (
            f"SubsetCollection(source_dataset={self.source_dataset._brief_repr()}, "
            f"n_subsets={len(self)})"
        )

    def to_dict(self) -> Dict:
        return {
            "__class__": self.__class__.__name__,
            "source_dataset": self.source_dataset,
            "indices": self._indices,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls: Type[T], d) -> T:
        cls_name = d.pop("__class__")
        if cls_name != cls.__name__:
            raise ValueError(
                f"Cannot instantiate {cls.__name__} "
                f"from dictionary containing serialized object of type {cls_name}."
            )
        return cls(**d)
