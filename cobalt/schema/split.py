# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from cobalt.schema.dataset import CobaltDataset, CobaltDataSubset

SubsetDescriptor = Union[CobaltDataSubset, List[int], np.ndarray]
SplitDescriptor = Union[
    Sequence[int], Sequence[SubsetDescriptor], Dict[str, SubsetDescriptor]
]


class DatasetSplit(dict):
    """The DatasetSplit object can contain any number of user-defined subsets of data.

    This can be used to separate out training data from production data, or a
    baseline dataset from a comparison set, or labeled from unlabeled data, or
    any number of divisions. These subsets are stored as a dictionary of
    CobaltDataSubsets, each with a name. When an object that is not a
    CobaltDataSubset is added to the dictionary, it is automatically converted
    to a subset by calling dataset.subset(). This means that the split can be
    created or updated by simply adding lists of data point indices.

    There are a few special subset names that will be given extra meaning by
    Cobalt: "train", "test", and "prod". The "train" subset is meant to include
    data that was used to train the model under consideration, the "test" subset
    data that was originally used to evaluate that model, and "prod" data
    collected later, e.g. when the model is in production. If specified, these
    subsets will be used in automated failure mode and problem analyses.
    """

    def __init__(  # noqa: D417: train, test, prod described under "split"
        self,
        dataset: CobaltDataset,
        split: Optional[SplitDescriptor] = None,
        train: Optional[SubsetDescriptor] = None,
        test: Optional[SubsetDescriptor] = None,
        prod: Optional[SubsetDescriptor] = None,
    ):
        """Construct a `DatasetSplit` object.

        Args:
            dataset: The `CobaltDataset` that this separates into subsets.
            split: A collection of subsets. Can be given as any of the
                following:

                - a sequence of integers indicating how many data points fall in each split
                - a sequence of subsets
                - a dict mapping subset names to subsets.

                Subsets can be provided either as `CobaltDataSubset` objects or as
                arrays of indices into `dataset`. If none is provided, a single
                subset named "all" will be created, containing all data points.

                There are three special names for subsets, "train", "test", and
                "prod", which are used to inform the automatic model analysis.
                These can also be passed as keyword parameters for convenience, e.g.
                ``DatasetSplit(dataset, train=np.arange(1000), prod=np.arange(1000,2000))``.
        """
        self.dataset = dataset

        super().__init__()

        if isinstance(split, Sequence) and len(split) > 0:
            if isinstance(split[0], int):
                total_split_size = sum(split)
                if total_split_size != len(dataset):
                    raise ValueError(
                        f"Sum of split sizes ({total_split_size}) "
                        f"must equal the size of the dataset ({len(dataset)})."
                    )
                split = self._subsets_from_sizes(split)
            split = self._name_subsets(split)

        if split is not None:
            for name, subset in split.items():
                self[name] = subset

        self["all"] = self.dataset.as_subset()

        if train is not None:
            if "train" in self:
                raise ValueError(
                    "train subset provided as both keyword argument and in the `split` parameter"
                )
            self["train"] = train

        if test is not None:
            if "test" in self:
                raise ValueError(
                    "test subset provided as both keyword argument and in the `split` parameter"
                )
            self["test"] = test

        if prod is not None:
            if "prod" in self:
                raise ValueError(
                    "prod subset provided as both keyword argument and in the `split` parameter"
                )
            self["prod"] = prod

    @staticmethod
    def _subsets_from_sizes(subset_sizes: Sequence[int]) -> List[np.ndarray]:
        subsets = []
        i = 0
        for size in subset_sizes:
            subsets.append(np.arange(i, i + size))
            i += size
        return subsets

    @staticmethod
    def _name_subsets(
        subsets: Sequence[SubsetDescriptor],
    ) -> Dict[str, SubsetDescriptor]:
        return {f"split_{i}": subset for i, subset in enumerate(subsets)}

    def __setitem__(self, key, value):
        if isinstance(value, CobaltDataSubset):
            if value.source_dataset is not self.dataset:
                raise ValueError(
                    "To add a `CobaltDataSubset`, it must be a subset of `self.dataset`."
                )
            super().__setitem__(key, value)
        else:
            try:
                super().__setitem__(key, self.dataset.subset(value))
            except ValueError as e:
                raise ValueError(
                    "To add a subset, it must either be a `CobaltDataSubset` "
                    "or a valid argument to `CobaltDataset.subset()`."
                ) from e

    @property
    def has_multiple_subsets(self) -> bool:
        """Whether this split has multiple disjoint subsets that can be compared."""
        return len(self.comparable_subset_pairs) > 0

    @property
    def comparable_subset_pairs(
        self,
    ) -> List[Tuple[Tuple[str, CobaltDataSubset], Tuple[str, CobaltDataSubset]]]:
        """Returns a list of pairs of disjoint subsets in this split, with names.

        Each pair is returned in both orders.
        """
        # TODO: subset vs rest?
        pairs = []
        for split_name_1, split_1 in self.items():
            for split_name_2, split_2 in self.items():
                if len(split_1.intersect(split_2)) == 0:
                    pairs.append(((split_name_1, split_1), (split_name_2, split_2)))

        return pairs

    @property
    def names(self) -> List[str]:
        """Names of subsets in this split."""
        return list(self.keys())

    @property
    def train(self) -> Optional[CobaltDataSubset]:
        """The training subset, if it exists."""
        return self.get("train", None)

    @property
    def test(self) -> Optional[CobaltDataSubset]:
        """The testing subset, if it exists."""
        return self.get("test", None)

    @property
    def prod(self) -> Optional[CobaltDataSubset]:
        """The production subset, if it exists."""
        return self.get("prod", None)
