# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from enum import Enum


class SelectionState(Enum):
    initial = "initial"
    selected = "selected"


class GroupType(Enum):
    user = "Saved Group"
    failure = "Failure Group"
    drift = "Drifted Group"
    cluster = "Cluster"
    node = "Node"
    any = "Group"


class RunType(Enum):
    """Type for GroupResultCollection."""

    MANUAL = "manual"
    AUTOMATIC = "automatic"


class ColumnDataType(Enum):

    numerical = "numerical"
    text = "text"
    datetime = "datetime"
    other = "other"
