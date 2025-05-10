# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Union
from uuid import UUID

import pandas as pd

from cobalt.cobalt_types import GroupType
from cobalt.schema import CobaltDataSubset

# these are dicts of the form
# `{"bucket_sizes": [1,5,3], "bucket_names": ["one", "two", "three"]}`
# they can be passed to the Histogram constructor by spreading:
# `Histogram(**true_class_distribution)`
CategoricalHistogram = Dict[str, Union[List[int], List[str]]]

# TODO: refactor schema for better modularity and compatibility


@dataclass
class GroupDisplayInfo:
    histograms: Dict[str, CategoricalHistogram] = field(default_factory=dict)
    """A collection of named histograms to display."""

    feature_descriptions: Dict[str, str] = field(default_factory=dict)
    """A collection of feature name => statistical summary pairs."""

    feature_stat_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    """A set of named tables containing feature statistics on this group."""

    # TODO: should these be more specific?
    textual_descriptions: Dict[str, str] = field(default_factory=dict)
    """Named textual descriptions for the group content."""


class Group(Protocol):
    name: str
    subset: CobaltDataSubset
    summary: str
    metrics: Dict[str, float]
    group_details: GroupDisplayInfo
    visible: bool = True
    run_id: Optional[UUID] = None


@dataclass
class ProblemGroup(Group):
    """A group representing a problem with a model."""

    name: str
    """A name for this group."""

    subset: CobaltDataSubset
    """The subset of data where the problem is located."""

    problem_description: str
    """A brief description of the problem."""

    metrics: Dict[str, float]
    """A dictionary of relevant model performance metrics on the subset."""

    summary: str = ""
    """A brief summary of the group attributes."""

    severity: float = 1.0
    """A score representing the degree of seriousness of the problem.

    Used to sort a collection of groups. Typically corresponds to the value of a
    performance metric on the group, and in general is only comparable within
    the result set of a single algorithm run.
    """

    group_details: GroupDisplayInfo = field(default_factory=GroupDisplayInfo)
    """Details about the group to be displayed."""

    group_type: GroupType = GroupType.failure
    """The type of group represented, e.g. drifted or high-error."""

    visible: bool = True
    run_id: Optional[UUID] = None

    # This is not a strict __repr__() method since it can't be used to reinstantiate the object
    # but that wouldn't be easily possible anyway
    # We use __repr__() instead of __str__() so that it controls the default
    # rendering when a ProblemGroup is in the output of a Jupyter cell.
    def __repr__(self):
        return (
            f"ProblemGroup(name={self.name!r}, "
            f"subset={self.subset.__class__.__name__}(n_points={len(self.subset)}), "
            f"problem_description={self.problem_description!r}, "
            f"metrics={self.metrics})"
        )


@dataclass
class Cluster(Group):
    name: str
    subset: CobaltDataSubset
    problem_description: str = ""
    summary: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    group_details: GroupDisplayInfo = field(default_factory=GroupDisplayInfo)
    group_type: GroupType = GroupType.cluster
    visible: bool = True
    run_id: Optional[UUID] = None

    def __repr__(self):
        return f"Cluster(subset={self.subset.__class__.__name__}(n_points={len(self.subset)}))"
