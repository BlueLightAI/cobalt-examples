# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from __future__ import annotations

import html
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Union
from uuid import uuid4

import pandas as pd

from cobalt.cobalt_types import ColumnDataType, GroupType
from cobalt.config import get_logger, handle_cb_exceptions
from cobalt.event_bus import EventBusController
from cobalt.repositories.graph_repository import GraphRepository
from cobalt.repositories.groups_repository import GroupRepository
from cobalt.repositories.run_repository import (
    GroupResultsCollection,
    GroupResultsCollectionRepository,
)
from cobalt.schema import CobaltDataset, CobaltDataSubset, DatasetSplit
from cobalt.schema.metadata import DatasetColumnMetadata, collect_columns_metadata

logger = get_logger()


class State:
    def __init__(self, dataset: CobaltDataset, split: DatasetSplit, workspace_id=None):
        self.dataset = dataset
        self.split = split
        self.workspace_id = workspace_id
        self.group_repo: GroupRepository = GroupRepository()
        self.graph_repo: GraphRepository = GraphRepository()
        self.run_repo: GroupResultsCollectionRepository = (
            GroupResultsCollectionRepository()
        )
        self.graph_event_bus = EventBusController.get_graph_event_bus(self.workspace_id)
        self.group_event_bus = EventBusController.get_group_event_bus(self.workspace_id)
        self.run_event_bus = EventBusController.get_run_event_bus(self.workspace_id)
        self.dataset_event_bus = EventBusController.get_dataset_event_bus(
            self.workspace_id
        )
        # for compatibility, we'll get rid of this once we've replaced the drift
        # plot and drift score coloring
        self.workspace_id = workspace_id if workspace_id else uuid4()

        self.metrics_data: Dict = {
            "all": {},
            "graph": {},
            "groups": {},
            "failure_groups": {},
        }
        self.data_filters = DataExplorerFilters()
        self.coloring_menu: Dict = {
            "selector_dropdown_value": None,
            "selector_cmap_value": None,
        }

    # GroupsRepository
    def get_groups_info(self):
        return self.group_repo.get_groups_info()

    def add_group(self, group, group_name, notify=True):
        self.group_repo.add_group(group, group_name)
        if notify:
            self.group_event_bus.run_all()

    def add_groups(self, groups):
        self.group_repo.add_groups(groups)
        self.group_event_bus.run_all()

    def rename_group(self, new_name, old_name, notify=True):
        self.group_repo.rename_group(new_name=new_name, old_name=old_name)
        if notify:
            self.group_event_bus.run_all()

    def get_group(self, group_name):
        return self.group_repo.get_group(group_name)

    def get_groups(self):
        return self.group_repo.get_groups()

    def delete_group(self, group_name, notify=True):
        self.group_repo.delete_group(group_name)
        if notify:
            self.group_event_bus.run_all()

    def is_group_name_unique(self, group_name):
        return self.group_repo.is_group_name_unique(group_name)

    def has_name_among_failure_groups(self, group_name):
        """Check problem group unique name in the specific run."""
        # TODO: remove group_type argument when we can show multiple group types
        visible_runs = self.run_repo.get_runs(
            group_type=GroupType.failure, run_visible=True
        )
        result = False
        for run_name in visible_runs:
            result = result and not self.run_repo.is_name_unique_problem_groups(
                run_name, group_name
            )
        return result

    def add_graph(self, graph, graph_name):
        """Updates an entry in the graph repository and triggers listeners."""
        self.graph_repo.add_graph(graph=graph, graph_name=graph_name)
        self.graph_event_bus.run_all()

    def get_graphs(self):
        return self.graph_repo.get_graphs()

    def get_graph(self, graph_name):
        return self.graph_repo.get_graph(graph_name=graph_name)

    def update_graph(self, graph, graph_name):
        """Updates an entry in the graph repository without triggering listeners."""
        self.graph_repo.update_graph(graph, graph_name)

    def add_run(self, run):
        self.run_repo.add_run(run)

    def collect_metrics_for_current_run(self, run: GroupResultsCollection):
        for fg in run.groups:
            self.collect_metrics(
                field="failure_groups", group_id=fg.name, dataset=fg.subset
            )

    @handle_cb_exceptions
    def collect_metrics(
        self,
        dataset: Union[CobaltDataset, CobaltDataSubset],
        field: Literal["all", "graph", "groups", "failure_groups"],
        group_id: Optional[str] = None,
    ):
        """Calculate metrics for given group or whole dataset."""
        # Check that dataset.model is setup
        model = dataset.model
        # Work around B018 linter error
        if not model:
            return

        if (
            dataset.model.outcome_column is None
            or dataset.model.prediction_column is None
        ):
            self.metrics_data[field] = {}
            return

        result = dataset.overall_model_performance_scores(model_index=0)
        if group_id:
            self.metrics_data[field][group_id] = result
        else:
            self.metrics_data[field] = result

        return result

    @property
    def metadata(self):
        return self.dataset.metadata

    def get_subset(self, subset: Union[str, CobaltDataSubset]) -> CobaltDataSubset:
        if isinstance(subset, str):
            return self.split.get(subset, None) or self.get_group(subset)
        else:
            return subset

    # DataExplorerFilters
    def clear_filters(self):
        self.data_filters.clear()

    def remove_filter(self, data_filter: DataFilter):
        self.data_filters.remove_filter(data_filter)

    def add_filter(self, column, op, value):
        columns_metadata = self.dataset.metadata.data_types
        column_metadata = columns_metadata.get(column)
        if not column_metadata:
            # Warnings? Collect again? Exception ?
            return
        data_filter = DataFilter(column=column_metadata, op=op, value=value)
        self.data_filters.add_filter(data_filter=data_filter)
        return data_filter

    def update_column_metadata(self, categorical_max_unique_count=10):
        self.dataset.metadata.data_types = collect_columns_metadata(
            self.dataset.df, categorical_max_unique_count
        )


numerical_filter_predicates: Dict[str, Callable[[pd.Series, Any], pd.Series]] = {
    "eq": lambda col, val: col == val,
    "gt": lambda col, val: col > val,
    "lt": lambda col, val: col < val,
    "gte": lambda col, val: col >= val,
    "lte": lambda col, val: col <= val,
}
text_filter_predicates: Dict[str, Callable[[pd.Series, str], pd.Series]] = {
    "is_case_sensitive_on": lambda col, val: col == val,
    "is_case_sensitive_off": lambda col, val: col.str.lower() == val.lower(),
    "contains_sensitive_off": lambda col, val: col.str.contains(
        val, case=False, na=False, regex=False
    ),
    "contains_sensitive_on": lambda col, val: col.str.contains(
        val, case=True, na=False, regex=False
    ),
}


class DataExplorerFilters:
    # map filter operator names to predicates on pd.Series
    filter_ops_map: ClassVar[
        Dict[ColumnDataType, Dict[str, Callable[[pd.Series, Any], pd.Series]]]
    ] = {
        ColumnDataType.numerical: numerical_filter_predicates,
        ColumnDataType.text: text_filter_predicates,
    }

    def __init__(self):
        self.filters: List[DataFilter] = []

    def add_filter(self, data_filter: DataFilter):
        self.filters.append(data_filter)

    def remove_filter(self, data_filter: DataFilter):
        for dfilter in self.filters:
            if dfilter == data_filter:
                self.filters.remove(dfilter)

    def clear(self):
        self.filters = []


@dataclass
class DataFilter:
    """Individual filter for data explorer column."""

    column: DatasetColumnMetadata
    op: str
    value: str

    def __post_init__(self):
        legal_ops = DataExplorerFilters.filter_ops_map[self.column.col_type]
        if self.op not in legal_ops:
            raise ValueError(f"op must be one of {legal_ops}")

    def __eq__(self, other):
        if not isinstance(other, DataFilter):
            return False
        return (
            (self.column == other.column)
            and (self.op == other.op)
            and (self.value == other.value)
        )


def apply_filters_df(df: pd.DataFrame, data_filters: DataExplorerFilters):
    def apply_filter(data: pd.DataFrame, col: DatasetColumnMetadata, op: str, val):
        filter_ops_map = data_filters.filter_ops_map.get(col.col_type)
        if not filter_ops_map:
            return data
        if val == "nan" and op in [
            "is_case_sensitive_on",
            "eq",
            "gt",
            "lt",
            "gte",
            "lte",
        ]:
            filter_mask = data[col.name].isna()
            return data.loc[filter_mask]
        filter_op = filter_ops_map[op]
        column = df[col.name]
        if op in text_filter_predicates:
            column = normalize_apostrophes(column)
            val = normalize_apostrophes(val)

        filter_mask = filter_op(column, val)
        return data.loc[filter_mask]

    # applies pandas filters to df one by one
    for data_filter in data_filters.filters:
        df = apply_filter(df, data_filter.column, data_filter.op, data_filter.value)
    return df


def normalize_apostrophes_text(text):
    """Normalize apostrophes and decode HTML entities."""
    if isinstance(text, str):
        text = html.unescape(text)  # Convert HTML entities (e.g., &#x27; → ')
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("ʼ", "'")  # noqa: RUF001
        text = text.replace("’", "'")  # noqa: RUF001
        text = text.replace("ʹ", "'")  # noqa: RUF001
    return text


def normalize_apostrophes_series(column: pd.Series):
    """Normalize apostrophes and decode HTML entities for pd.Series.

    It works 10x faster than normalize_apostrophes + apply.
    """
    if isinstance(column.dtype, pd.CategoricalDtype):
        column = column.cat.add_categories([""]).fillna("")
    else:
        column = column.fillna("")  # html.unescape fails if there are NaNs
    series = column.apply(lambda x: html.unescape(x) if isinstance(x, str) else x)
    apostrophe_variants = r"[’ʼ`´ʹ]"  # noqa: RUF001
    return series.str.normalize("NFKC").str.replace(
        apostrophe_variants, "'", regex=True
    )


def normalize_apostrophes(obj):
    if isinstance(obj, str):
        return normalize_apostrophes_text(obj)
    elif isinstance(obj, pd.Series):
        return normalize_apostrophes_series(obj)
