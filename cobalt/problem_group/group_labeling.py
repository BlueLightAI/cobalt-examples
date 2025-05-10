# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from cobalt.feature_compare import (
    feature_descriptions_from_tables,
    subset_description_tables,
)
from cobalt.problem_group.schema import (
    CategoricalHistogram,
    Group,
    ProblemGroup,
)
from cobalt.schema import CobaltDataSubset, ModelMetadata
from cobalt.schema.model_metadata import ModelTask
from cobalt.text import ctfidf


# TODO: add color map for histogram.
def add_model_data_histograms(
    groups: Sequence[Group],
    model: ModelMetadata,
    max_n_classes: int = 6,
):
    if model.task != ModelTask.classification:
        # TODO: handle other model tasks
        return
    if not model.outcome_column or not model.prediction_column:
        # TODO: handle the case where there's only one
        return

    for group in groups:
        histograms = [
            get_column_distribution_categorical(group.subset, model.outcome_column),
            get_column_distribution_categorical(group.subset, model.prediction_column),
        ]
        collate_categorical_histograms(histograms, max_n_cats=max_n_classes)
        group.group_details.histograms = {
            f"Label Distribution ({model.outcome_column})": histograms[0],
            f"Prediction Distribution ({model.prediction_column})": histograms[1],
        }


def add_text_keyword_group_descriptions(
    groups: Sequence[Group],
    text_columns: Optional[Sequence[str]] = None,
    use_all_text_columns: bool = True,
    n_keywords: int = 10,
    add_summary: bool = True,
    summary_column: Optional[str] = None,
):
    """Adds keyword-based descriptions to the provided groups.

    Args:
        groups: The groups to add descriptions to.
        text_columns: The columns in the dataset that contain text for
            topic/keyword analysis. If None, will choose all long text columns or
            the default text column, depending on the value of use_all_text_columns.
        use_all_text_columns: Only used if text_columns=None. If True, will use
            all long text columns; otherwise will use only the dataset's default
            topic column.
        n_keywords: The number of keywords to include in the keyword list.
        add_summary: Whether to add a short text summary of the keywords for
            display in the group list.
        summary_column: The column to use for the summary description. If not
            specified, will use the dataset's default topic column, if it is in
            text_columns.
    """
    if len(groups) == 0:
        return

    source_dataset = groups[0].subset.source_dataset

    if text_columns is None:
        if summary_column is None:
            summary_column = source_dataset.metadata.default_topic_column
        if use_all_text_columns:
            text_columns = source_dataset.metadata.long_text_columns
        else:
            text_columns = [summary_column]
    else:
        if summary_column is None:
            default_column = source_dataset.metadata.default_topic_column
            if default_column in text_columns or len(text_columns) == 0:
                summary_column = default_column
            else:
                summary_column = text_columns[0]

    if len(text_columns) == 0:
        return

    use_column_name = len(text_columns) > 1

    all_group_indices = np.concatenate([g.subset.indices for g in groups])
    remainder_group = source_dataset.subset(all_group_indices).complement()
    ctfidf_groups = [group.subset for group in groups]
    ctfidf_groups.append(remainder_group)

    for col in text_columns:
        keyword_analysis = ctfidf.CTFIDFKeywordAnalysis(
            source_dataset, col, "up_to_bigrams"
        )
        keywords, _, _ = keyword_analysis.get_keywords(ctfidf_groups, n_keywords)

        for i, g in enumerate(groups):
            keyword_list = keywords[i]
            keyword_key = f"Keywords ({col})" if use_column_name else "Keywords"
            g.group_details.textual_descriptions[keyword_key] = keyword_list
            if add_summary and col == summary_column:
                keyword_str = " | ".join(keyword_list[:3])
                g.summary = f"{keyword_key}: {keyword_str}"


def short_feature_description_string(
    num_stats: pd.DataFrame, cat_stats: pd.DataFrame
) -> str:
    # TODO: pick two features in order based on p-value or something
    if len(num_stats) > 0:
        num_feature_name = num_stats.iloc[0]["feature"]
        feature_mean = num_stats.iloc[0]["mean"]
        complement_mean = num_stats.iloc[0]["complement mean"]
        dir_str = "↑" if feature_mean > complement_mean else "↓"
        num_str = f"{num_feature_name} mean={feature_mean:.2g} ({dir_str})"
    else:
        num_str = ""
    cat_str = ""
    if len(cat_stats) > 0:
        cat_feature_name = cat_stats.iloc[0]["feature"]
        feature_mode = cat_stats.iloc[0]["mode"]
        mode_freq = cat_stats.iloc[0]["frequency (%)"]
        if mode_freq > 50:
            cat_str = f"{cat_feature_name}={feature_mode} ({mode_freq:.1f}%)"
    if len(num_str) > 0:
        summary_str = f"{num_str} | {cat_str}" if len(cat_str) > 0 else num_str
    else:
        summary_str = cat_str

    return summary_str


def add_feature_group_descriptions(
    groups: Sequence[Group],
    omit_columns: Optional[List[str]] = None,
    set_summary: bool = True,
):
    # TODO: improve efficiency of statistical analysis
    for group in groups:
        num_stats, cat_stats = subset_description_tables(
            group.subset,
            omit_columns=omit_columns,
            max_categorical_features=100,
            max_numerical_features=100,
        )
        group.group_details.feature_stat_tables = {
            "numerical_comparison_stats": num_stats,
            "categorical_comparison_stats": cat_stats,
        }
        group.group_details.feature_descriptions = feature_descriptions_from_tables(
            num_stats.iloc[: min(3, len(num_stats)), :],
            cat_stats.iloc[: min(3, len(cat_stats)), :],
        )
        if set_summary:
            group.summary = short_feature_description_string(num_stats, cat_stats)


def create_failure_group_metadata(
    fg: CobaltDataSubset,
    model_metadata: ModelMetadata,
    failure_metric_name: str,
    group_name: str = "",
    severity_is_inverse: bool = False,
) -> ProblemGroup:
    """Create a ProblemGroup containing information about a raw failure group."""
    # ugh, don't love this
    failure_metric_average = (
        model_metadata.get_performance_metric_for(failure_metric_name)
        .calculate(fg)[failure_metric_name]
        .mean()
    )

    severity = failure_metric_average
    if severity_is_inverse:
        severity = -severity

    problem_group = ProblemGroup(
        name=group_name,
        subset=fg,
        problem_description=(
            f"{len(fg)} points | "
            f"{failure_metric_name}: {failure_metric_average:.3g}"
        ),
        metrics={failure_metric_name: failure_metric_average},
        severity=severity,
    )
    return problem_group


def get_column_distribution_categorical(
    subset: CobaltDataSubset, col: str
) -> CategoricalHistogram:
    """Calculates the distribution of a categorical column.

    Output is in the format required by the FailureGroupDetails object, to
    support display as a histogram.
    """
    value_counts = subset.select_col(col).value_counts()
    return {
        "bucket_sizes": value_counts.to_list(),
        "bucket_names": value_counts.index.to_list(),
    }


# TODO: include an other category?
def collate_categorical_histograms(
    histograms: List[CategoricalHistogram], max_n_cats: int = 8
):
    """Combines the set of classes for the two histograms in a FailureGroupDetails object."""
    if max_n_cats < 2:
        raise ValueError(
            "max_n_cats must be at least 2 to produce meaningful histograms."
        )
    # TODO: there has to be a nicer way to do this
    # first, get the collection of all bucket names from all histograms
    unique_buckets: set[str] = set()
    for hist in histograms:
        try:
            unique_buckets.update(hist["bucket_names"])
        except KeyError:
            pass
        except AttributeError:
            pass
    new_buckets = list(unique_buckets)

    # add the empty buckets to each histogram, and make sure they're in the same order
    for hist in histograms:
        relabel_buckets(hist, new_buckets)

    # pick the max_n_cats most common categories
    total_counts = np.sum([hist["bucket_sizes"] for hist in histograms], axis=0)
    sorted_indices = np.argsort(total_counts)
    if max_n_cats < len(new_buckets):
        # so that "other" can be added to bring it back up to max_n_cats
        max_n_cats -= 1
    top_indices = np.sort(sorted_indices[-max_n_cats:])
    remaining_indices = sorted_indices[:-max_n_cats]
    new_bucket_names = [new_buckets[i] for i in top_indices]
    # sort by length of the name for display purposes
    # this means the leftmost class names don't stick out as far to the left
    top_indices, new_bucket_names = zip(
        *sorted(zip(top_indices, new_bucket_names), key=lambda x: len(str(x[1])))
    )
    new_bucket_names = list(new_bucket_names)
    if len(remaining_indices) > 0:
        new_bucket_names.append("other")

    for hist in histograms:
        hist["bucket_names"] = list(new_bucket_names)
        new_bucket_sizes = [hist["bucket_sizes"][i] for i in top_indices]
        if len(remaining_indices) > 0:
            other_count = sum(hist["bucket_sizes"][i] for i in remaining_indices)
            new_bucket_sizes.append(other_count)
        hist["bucket_sizes"] = new_bucket_sizes


def relabel_buckets(histogram: Optional[Dict], new_buckets: List):
    """Replaces the set of buckets in a histogram with new labels.

    The bucket size for any label not in the original set will be zero, and
    bucket sizes for previous labels are kept.
    """
    try:
        old_buckets = histogram["bucket_names"]
        old_counts = histogram["bucket_sizes"]
        histogram["bucket_names"] = new_buckets
        new_counts = []
        for bucket in new_buckets:
            if bucket in old_buckets:
                i = old_buckets.index(bucket)
                new_counts.append(old_counts[i])
            else:
                new_counts.append(0)
        histogram["bucket_sizes"] = new_counts
    except KeyError:
        pass
    except TypeError:
        pass
