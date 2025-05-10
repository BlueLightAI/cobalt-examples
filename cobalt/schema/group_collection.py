# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import (
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import numpy as np
import pandas as pd
import scipy.stats

from cobalt.cobalt_types import GroupType
from cobalt.problem_group.schema import GroupDisplayInfo
from cobalt.schema.dataset import CobaltDataset, CobaltDataSubset
from cobalt.schema.model_metadata import ModelMetadata
from cobalt.schema.subset_collection import SubsetCollection
from cobalt.text.ctfidf import CTFIDFKeywordAnalysis


def wilcoxon_wrapper(x, y, alternative="two-sided"):
    if np.all(x == y):
        return SimpleNamespace(statistic=0.0, pvalue=np.nan)
    try:
        res = scipy.stats.wilcoxon(x, y, alternative=alternative)
        return res
    except Exception:
        try:
            return scipy.stats.wilcoxon(
                x, y, alternative=alternative, zero_method="zsplit"
            )
        except Exception:
            return SimpleNamespace(statistic=0.0, pvalue=np.nan)


def ttest_wrapper(x, y, alternative="two-sided"):
    return scipy.stats.ttest_rel(x, y, alternative=alternative)


paired_test_fns = {
    "t-test": scipy.stats.ttest_rel,
    "wilcoxon": wilcoxon_wrapper,
}


@dataclass
class GroupKeywords:
    col: str
    terms: Tuple[str, ...]
    scores: Tuple[float, ...]
    match_rate: Tuple[float, ...]
    params: Dict = field(default_factory=dict)

    def get_keyword_string(
        self, n_keywords: int = 3, min_match_rate: float = 0, delimiter: str = ", "
    ) -> str:
        # TODO: filter keywords for redundancy to make better names
        return delimiter.join(
            itertools.islice(
                (
                    t
                    for t, mr in zip(self.terms, self.match_rate)
                    if t and mr >= min_match_rate
                ),
                n_keywords,
            )
        )

    def __len__(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:
        return self.get_keyword_string(n_keywords=len(self.terms))


@dataclass
class GroupMetadata:
    subset: CobaltDataSubset
    """The data points included in this group."""

    name: Optional[str] = None
    """The group's name. Should be unique within a SubsetCollection."""

    metrics: Dict[str, float] = field(default_factory=dict)
    """Relevant numeric metrics for this group."""

    description: Optional[str] = None
    """A short description of the contents of the group."""

    display_info: GroupDisplayInfo = field(default_factory=GroupDisplayInfo)
    """Information to be displayed in the group explorer in the UI."""

    keywords: Dict[str, GroupKeywords] = field(default_factory=dict)
    """Distinctive keywords found in text columns in the group."""

    group_type: GroupType = GroupType.any
    """Describes the semantic meaning of the group in context."""
    # TODO: other fields
    # should metrics be grouped by model, or at least have the option?

    def __repr__(self) -> str:
        return f"GroupMetadata(name={self.name!r}, keywords={self.keywords!r})"


# TODO: methods to reorder/filter groups based on metadata fields
# TODO: move problem_group.processing functions here
class GroupCollection(SubsetCollection):
    """A collection of groups from a source CobaltDataset.

    A group consists of a subset of data points together with some metadata
    about the subset. This metadata can include things like:

    - A name for the group
    - Distinctive keywords for the group
    - Model performance metrics on the group
    - Distinctive features for the group

    The schema for metadata is defined in the ``GroupMetadata`` class.

    The groups in a collection are stored in a specific order, and can be
    accessed by indexing, e.g. ``collection[0]`` to get the first group. If a
    group has been assigned a name, it can also be accessed by name, e.g.
    ``collection["group name"]``. This will return the CobaltDataSubset
    containing the data points in the group. To access the metadata for a group,
    index into ``collection.metadata`` in the same way.

    It should not usually be necessary to manually instantiate GroupCollection
    objects, but they will be returned by various Cobalt methods and functions.

    The GroupCollection interface is under development and changes may be made
    in the near future.
    """

    def __init__(
        self,
        source_dataset: CobaltDataset,
        indices: Sequence[Sequence[int]],
        name: Optional[str] = None,
        group_type: GroupType = GroupType.any,
    ):
        super().__init__(source_dataset, indices, name)

        self.group_type = group_type

        self._group_objects = [GroupMetadata(subset) for subset in self._subset_objects]
        self._group_name_to_id: Dict[str, int] = {}

    @classmethod
    def from_groups(cls, groups: Sequence[GroupMetadata]):
        """Create a GroupCollection from a list of GroupMetadata objects."""
        if len(groups) == 0:
            raise ValueError("At least one group must be provided")
        indices = [group.subset.indices for group in groups]
        inst = cls(groups[0].subset.source_dataset, indices)
        inst._subset_objects = [g.subset for g in groups]
        inst._group_objects = list(groups)
        return inst

    @classmethod
    def from_subset_collection(cls, subsets: SubsetCollection):
        """Promote a SubsetCollection to a GroupCollection.

        This allows adding metadata to each subset.
        """
        return cls.from_subsets(subsets._subset_objects, name=subsets.name)

    def _get_group_id_by_name(self, name: str) -> int:
        # TODO: handle changing names of groups better
        # this may need listeners on the GroupMetadata objects?
        group_id = self._group_name_to_id.get(name)
        if group_id:
            return group_id
        for group_id, group in enumerate(self._group_objects):
            if group.name == name:
                self._group_name_to_id[name] = group_id
                return group_id
        raise KeyError(f"No group with name {name!r} exists.")

    @overload
    def __getitem__(self, key: Union[int, str]) -> CobaltDataSubset: ...

    @overload
    def __getitem__(self, key: Union[slice, Sequence[int]]) -> GroupCollection: ...

    def __getitem__(self, key):
        if isinstance(key, str):
            group_id = self._get_group_id_by_name(key)
            return self[group_id]
        if isinstance(key, int):
            return super().__getitem__(key)
        # TODO: tuples for metadata indexing?
        # TODO: improve performance of slicing?
        if isinstance(key, slice):
            indices = list(range(len(self)))[key]
            groups = [self.metadata[group_id] for group_id in indices]
            new_groups = self.from_groups(groups)
            return new_groups
        if isinstance(key, Sequence) and type(key) != tuple:
            groups = [self.metadata[group_id] for group_id in key]
            new_groups = self.from_groups(groups)
            return new_groups

        raise IndexError(f"Invalid group identifier {key!r}.")

    @property
    def metadata(self) -> GroupMetadataIndexer:
        """Get a group together with its metadata."""
        return GroupMetadataIndexer(self)

    def compute_group_keywords(  # noqa: D417
        self,
        col: Optional[str] = None,
        n_keywords: int = 10,
        set_names: bool = False,
        **kwargs,
    ):
        """Find distinctive keywords for each group and store them in the group metadata.

        Args:
            col: The column containing text from which to extract keywords.
            n_keywords: The number of keywords to find for each group.
            set_names: If True, will set each group's name based on the
                discovered keywords, using the default parameters to
                ``set_names_from_keywords()``.
        """
        if not col:
            col = self.source_dataset.metadata.default_topic_column
        if col is None:
            warnings.warn(
                "No keyword column specified, and no default has been set. "
                "No keywords were generated.",
                stacklevel=2,
            )
            return

        # TODO: get this from saved spot
        keyword_analyzer = CTFIDFKeywordAnalysis(self.source_dataset, col, **kwargs)
        keywords, scores, match_rates = keyword_analyzer.get_keywords(
            list(self), n_keywords=n_keywords
        )
        for m, kws, scs, mrs in zip(self.metadata, keywords, scores, match_rates):
            m.keywords[col] = GroupKeywords(
                col, kws, scs, mrs, params={"n_keywords": 10, **kwargs}
            )

        if set_names:
            self.set_names_from_keywords(col)

    def set_names_from_keywords(
        self,
        col: str,
        n_keywords: int = 3,
        delimiter: str = ", ",
        min_match_rate: float = 0.0,
    ):
        """Set names for each group based on already-computed keywords.

        Names groups with a string containing a number of the top keywords found
        for that group.

        If two groups would end up with the same name, groups after the first
        will be named with a number to ensure names are unique.

        Args:
            col: The column whose keywords should be used to create the group names.
            n_keywords: The number of keywords to use to form each name.
            delimiter: The character(s) that should separate keywords from each
                other in the group names.
            min_match_rate: The minimum fraction of data points in the group
                that should contain a keyword in order for it to be used in the
                group name.
        """
        # TODO: default for col?
        name_counts: Dict[str, int] = {}
        for idx, m in enumerate(self.metadata):
            if col not in m.keywords:
                raise ValueError(
                    f"No keywords for column {col} available for group {m}. "
                    "Run compute_group_keywords()."
                ) from None
            candidate_name = m.keywords[col].get_keyword_string(
                n_keywords, min_match_rate=min_match_rate, delimiter=delimiter
            )
            if candidate_name in name_counts:
                name_counts[candidate_name] += 1
                candidate_name = f"{candidate_name} ({name_counts[candidate_name]})"
            else:
                name_counts[candidate_name] = 1
            if m.name in self._group_name_to_id:
                del self._group_name_to_id[m.name]
            m.name = candidate_name
            self._group_name_to_id[candidate_name] = idx

    def set_names_sequential(
        self,
        prefix: Optional[str] = None,
        prefix_source: Literal["group_type", "collection_name"] = "group_type",
        sep: str = " ",
    ):
        """Set names for each group sequentially with a prefix string."""
        if prefix is None:
            prefix = (
                self.group_type.value
                if prefix_source == "group_type"
                else (self.name if self.name else "Group")
            )
        for idx, m in enumerate(self.metadata):
            m.name = f"{prefix}{sep}{idx}"
            self._group_name_to_id[m.name] = idx

    def summary_table(
        self, keyword_col: Optional[str] = None, set_index_to_name: bool = True
    ) -> pd.DataFrame:
        columns: Dict[str, List] = {}
        columns["name"] = [m.name for m in self.metadata]
        columns["size"] = [len(g) for g in self]
        if keyword_col is None:
            keyword_col = self.source_dataset.metadata.default_topic_column
        if keyword_col is not None:
            columns["keywords"] = [
                (
                    m.keywords[keyword_col].get_keyword_string(
                        n_keywords=len(m.keywords[keyword_col])
                    )
                    if m.keywords
                    else None
                )
                for m in self.metadata
            ]
        # TODO: other metadata fields
        df = pd.DataFrame(columns)
        missing_names = df["name"].isna()
        if set_index_to_name and not missing_names.any() and df["name"].is_unique:
            df = df.set_index("name", verify_integrity=False)
        elif missing_names.all():
            df = df.drop("name", axis=1)
        return df

    def evaluate_model(
        self,
        model: Union[ModelMetadata, str],
        metrics: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Produce a dataframe containing model performance metrics for each group.

        Args:
            model: Name of the model to evaluate, or a ModelMetadata object to evaluate.
            metrics: Names of the metrics to evaluate on the model. By default,
                will use all metrics defined for the model.
        """
        if isinstance(model, str):
            model = self.source_dataset._get_model_by_name(model)

        eval_df = self.summary_table()
        if metrics is None:
            metrics = list(model.evaluation_metrics.keys())

        for metric_name in metrics:
            metric = model.evaluation_metrics[metric_name]
            metric_scores = [
                metric.overall_score(subset)[metric.get_key()] for subset in self
            ]
            # is this the best column name?
            eval_df[f"{model.name}_{metric_name}"] = metric_scores

        return eval_df

    def compare_models(
        self,
        models: Sequence[Union[ModelMetadata, str]],
        metrics: List[str],
        select_best_model: bool = True,
        statistical_test: Optional[Literal["t-test", "wilcoxon"]] = None,
    ) -> pd.DataFrame:
        """Produce a dataframe comparing two or more models on each group.

        Evaluates each specified metric for each model on each group, and puts
        these values in a column called "model_name_metric_name". If
        select_best_model is True, will also include a column indicating the
        best model for each group with respect to each metric, as well as the
        change in performance compared to the next-best model. If
        statistical_test is specified, will also run a test that the performance
        difference is significantly different between the two models on each
        group. The resulting p-values are not currently adjusted for multiple
        comparisons.
        """
        # TODO: allow specifying a baseline model?
        if len(models) < 2:
            warnings.warn(
                "compare_models() called with only one model. "
                "Use evaluate_model() to get performance metrics for a single model.",
                stacklevel=2,
            )
            select_best_model = False

        models = [
            self.source_dataset._get_model_by_name(m) if isinstance(m, str) else m
            for m in models
        ]

        comparison_df = self.summary_table()
        summary_cols = list(comparison_df.columns)
        model_metric_dfs = [
            self.evaluate_model(model, metrics)[
                [f"{model.name}_{metric_name}" for metric_name in metrics]
            ]
            for model in models
        ]
        comparison_df = pd.concat([comparison_df, *model_metric_dfs], axis=1)
        results_df = pd.DataFrame(comparison_df)

        if select_best_model:
            for metric_name in metrics:
                col_set = [f"{model.name}_{metric_name}" for model in models]
                scores_arr = results_df[col_set].to_numpy()
                sort_ix = np.argsort(scores_arr, axis=1)
                if models[0].evaluation_metrics[metric_name].lower_values_are_better:
                    top_two_model_ix = sort_ix[:, :2]
                else:
                    top_two_model_ix = sort_ix[:, -2:][:, ::-1]
                top_scores = np.take_along_axis(scores_arr, top_two_model_ix, axis=1)
                score_diffs = np.abs(top_scores[:, 0] - top_scores[:, 1])
                best_model_names = [
                    models[ix].name if (score_diff > 0) else "tie"
                    for score_diff, ix in zip(score_diffs, top_two_model_ix[:, 0])
                ]
                results_df[f"best_model_{metric_name}"] = best_model_names
                results_df[f"{metric_name}_margin"] = score_diffs
                if statistical_test:
                    warnings.warn(
                        "Statistical tests for model comparisons are experimental. "
                        "Results may be inaccurate and are subject to change.",
                        stacklevel=2,
                    )

                    test_fn = paired_test_fns[statistical_test]
                    alternative = (
                        "less"
                        if models[0]
                        .evaluation_metrics[metric_name]
                        .lower_values_are_better
                        else "greater"
                    )
                    pvalues = [
                        test_fn(
                            models[best_ix]
                            .evaluation_metrics[metric_name]
                            .calculate(subset)[metric_name],
                            models[second_best_ix]
                            .evaluation_metrics[metric_name]
                            .calculate(subset)[metric_name],
                            alternative=alternative,
                        ).pvalue
                        for subset, best_ix, second_best_ix in zip(
                            self,
                            top_two_model_ix[:, 0],
                            top_two_model_ix[:, 1],
                        )
                    ]
                    # TODO: multiple comparisons correction
                    results_df[f"{metric_name}_pval"] = pvalues

        column_blocks = [
            (
                [f"{model.name}_{metric_name}" for model in models]
                + (
                    [f"best_model_{metric_name}", f"{metric_name}_margin"]
                    if select_best_model
                    else []
                )
                + (
                    [f"{metric_name}_pval"]
                    if (select_best_model and statistical_test)
                    else []
                )
            )
            for metric_name in metrics
        ]
        new_column_order = summary_cols + list(
            itertools.chain.from_iterable(column_blocks)
        )

        # TODO: other aggregation methods applied to the metrics, e.g. variance?
        # TODO: add results to metrics to display?

        return results_df[new_column_order]

    def __repr__(self) -> str:
        if self.name:
            return f"GroupCollection(n_subsets={len(self)}, name='{self.name}')"
        return f"GroupCollection(n_subsets={len(self)})"

    def _repr_html_(self):
        """Defines representation in an IPython output cell."""
        table_html = self.summary_table()._repr_html_()
        n_groups = len(self)
        n_groups_str = (
            f"{n_groups} {self.group_type.value}s"
            if n_groups != 1
            else f"1 {self.group_type.name}"
        )
        if self.name:
            n_groups_str = f"{self.name}: {n_groups_str}"
        return f"<h4>{n_groups_str}</h4> {table_html}"


class GroupMetadataIndexer:
    """Indexes into the metadata objects stored in a GroupCollection."""

    def __init__(self, obj: GroupCollection):
        self.obj = obj

    def __getitem__(self, key) -> GroupMetadata:
        if isinstance(key, str):
            group_id = self.obj._get_group_id_by_name(key)
        elif isinstance(key, int):
            group_id = key
        else:
            raise KeyError("Invalid group identifier {key!r}")
        return self.obj._group_objects[group_id]

    def __iter__(self) -> Iterator[GroupMetadata]:
        return (self[i] for i in range(len(self.obj)))
