# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from cobalt.schema import CobaltDataSubset
from cobalt.schema.dataset import DatasetBase
from cobalt.stats.stats_tests import (
    pd_is_any_real_numeric_dtype,
    power_divergence_of_series,
    safe_ttest,
)


def get_numerical_features(df: pd.DataFrame) -> List[str]:
    numerical_features = [c for c in df.columns if pd_is_any_real_numeric_dtype(df[c])]
    return numerical_features


def get_categorical_columns(dataset: DatasetBase) -> List[str]:
    """List of the columns of this dataset that are categorical."""
    hidable_columns = set(dataset.metadata.hidable_columns)
    return [
        col for col in dataset.get_categorical_columns() if col not in hidable_columns
    ]


def feature_compare(
    group_1: CobaltDataSubset,
    group_2: CobaltDataSubset,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    numerical_test: Literal["t-test"] = "t-test",
    categorical_test: Literal["G-test"] = "G-test",
    include_nan: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not numerical_features:
        numerical_features = get_numerical_features(group_1.source_dataset.df)
    if not categorical_features:
        categorical_features = get_categorical_columns(group_1.source_dataset)

    return numerical_feature_compare(
        group_1, group_2, numerical_features, numerical_test, include_nan=include_nan
    ), categorical_feature_compare(
        group_1, group_2, categorical_features, categorical_test
    )


def numerical_feature_compare(
    group_1: CobaltDataSubset,
    group_2: CobaltDataSubset,
    features: List[str],
    test: Literal["t-test"] = "t-test",
    include_nan: bool = False,
) -> pd.DataFrame:
    if test != "t-test":
        raise ValueError(f"Unsupported test type {test}.")
    # TODO: consider factoring into one function per test when more test types are added

    # TODO: can this be parallelized?
    scores = [
        (c, *safe_ttest(group_1.select_col(c), group_2.select_col(c), "two-sided"))
        for c in features
    ]
    stats_df = pd.DataFrame(
        scores,
        columns=["feature", "t-score", "p-value"],
    )

    means_1 = [group_1.select_col(col).mean() for col in features]
    means_2 = [group_2.select_col(col).mean() for col in features]
    stats_df["mean A"] = means_1
    stats_df["mean B"] = means_2
    stats_df = stats_df[["feature", "mean A", "mean B", "p-value"]]

    if not include_nan:
        stats_df = stats_df.dropna(axis=0, how="any")

    stats_df["abs_mean_diff"] = (stats_df["mean A"] - stats_df["mean B"]).abs()

    stats_df = stats_df.sort_values(
        by=["p-value", "abs_mean_diff"],
        ascending=[True, False],
    ).drop("abs_mean_diff", axis=1)
    return stats_df


def categorical_feature_compare(
    group_1: CobaltDataSubset,
    group_2: CobaltDataSubset,
    features: List[str],
    test: Literal["G-test"] = "G-test",
) -> pd.DataFrame:
    if test != "G-test":
        raise ValueError(f"Unsupported test type {test}.")
    modes = [group_1.select_col(col).mode() for col in features]
    full_modes = [list(m) if len(m) > 0 else [np.nan] for m in modes]

    # TODO: can this be parallelized?
    results = [
        power_divergence_of_series(group_1.select_col(col), group_2.select_col(col))
        for col in features
    ]
    p_values = [result.pvalue for result in results]

    modes_formatted = [
        (
            mode[0]
            if len(mode) == 1
            else (
                f"TIE: {', '.join([str(m) for m in mode])}"
                if len(mode) < 5
                else "TIE: multiple values"
            )
        )
        for mode in full_modes
    ]

    frequency_of_mode = [
        (
            100 * (group_1.select_col(col) == m[0]).mean() if not pd.isna(m[0]) else 100
        )  # nan can only be a mode if it is all the values
        for col, m in zip(features, full_modes)
    ]
    c_stats_df = pd.DataFrame(
        {
            "feature": features,
            "mode": modes_formatted,
            "frequency (%)": frequency_of_mode,
            "p-value": p_values,
        }
    )

    c_stats_df = c_stats_df.sort_values(
        by=["p-value", "frequency (%)"], ascending=[True, False]
    )
    return c_stats_df


def subset_description_tables(
    subset: CobaltDataSubset,
    p_val_threshold: float = 0.001,
    omit_columns: Optional[List[str]] = None,
    max_numerical_features: int = 3,
    max_categorical_features: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Produce a pair of statistics tables with highly distinguishing features for a subset.

    Compares the given subset with the rest of the dataset using the
    `feature_compare()` function, and chooses up to `max_numerical_features`
    numerical features and `max_categorical_features` categorical features to
    display.

    Any features whose significance level for the test does not reach
    `p_val_threshold` will be discarded. Columns can be excluded from the
    description by adding them to the `omit_columns` argument.
    """
    complement = subset.complement()
    if omit_columns is not None:
        numerical_features = get_numerical_features(subset.source_dataset.df)
        numerical_features = list(set(numerical_features).difference(omit_columns))
        num_stats = numerical_feature_compare(subset, complement, numerical_features)
        categorical_features = get_categorical_columns(subset.source_dataset)
        categorical_features = list(set(categorical_features).difference(omit_columns))
        cat_stats = categorical_feature_compare(
            subset, complement, categorical_features
        )
    else:
        num_stats, cat_stats = feature_compare(subset, complement)

    # use up to 3 categorical features if the p-value is small enough and the mode is common enough
    cat_stats = cat_stats[
        (cat_stats["p-value"] <= p_val_threshold) & (cat_stats["frequency (%)"] >= 50)
    ].head(max_categorical_features)

    # get the first few numerical features. use up to 3 if the p-value is small enough
    num_stats = num_stats[
        (num_stats["p-value"] <= p_val_threshold)
        & ~(num_stats["feature"].str.startswith(tuple(cat_stats["feature"])))
    ].head(max_numerical_features)

    cat_stats["complement frequency (%)"] = [
        (complement.select_col(row["feature"]) == row["mode"]).mean() * 100
        for _, row in cat_stats.iterrows()
    ]
    num_stats["mean"] = num_stats["mean A"]
    num_stats["complement mean"] = num_stats["mean B"]

    return (
        num_stats[["feature", "mean", "complement mean"]],
        cat_stats[["feature", "mode", "frequency (%)", "complement frequency (%)"]],
    )


def describe_subset_features(
    subset: CobaltDataSubset,
    p_val_threshold: float = 0.001,
    omit_columns: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Produce a dictionary with feature: description pairs for a given subset.

    Compares the given subset with the rest of the dataset using the
    `feature_compare()` function, and chooses up to 3 distinguishing features of
    each numerical and categorical type. The result is of the form

    {"num_feature": "mean = {feature mean} (rest of data: {mean on complement})"}
    {"cat_feature": "mode = {mode} ({frequency}%, rest of data: {frequency in complement}%)"}

    Any features whose significance level for the test does not reach
    `p_val_threshold` will be discarded. Columns can be excluded from the
    description by adding them to the `omit_columns` argument.
    """
    num_stats, cat_stats = subset_description_tables(
        subset, p_val_threshold, omit_columns
    )
    return feature_descriptions_from_tables(num_stats, cat_stats)


def feature_descriptions_from_tables(num_stats: pd.DataFrame, cat_stats: pd.DataFrame):
    cat_descriptions = {
        row["feature"]: (
            f"mode = {row['mode']} ({row['frequency (%)']:.1f}%, "
            f"rest of data: {row['complement frequency (%)']:.1f}%)"
        )
        for _, row in cat_stats.iterrows()
    }

    num_descriptions = {
        row["feature"]: (
            f"mean = {row['mean']:.3g} " f"(rest of data: {row['complement mean']:.3g})"
        )
        for _, row in num_stats.iterrows()
    }
    return {**num_descriptions, **cat_descriptions}


def describe_subset(
    subset: CobaltDataSubset,
    p_val_threshold: float = 0.001,
    omit_columns: Optional[List[str]] = None,
) -> str:
    """Produce a string describing salient features for a subset.

    Compares the given subset with the rest of the dataset using the
    `feature_compare()` function, and chooses up to 3 distinguishing features of
    each numerical and categorical type. The result is of the form

    num_feature: mean = {feature mean} (rest of data: {mean on complement})
    [up to two more lines]
    cat_feature: mode = {mode} ({frequency}%, rest of data: {frequency in complement}%)
    [up to two more lines]

    Any features whose significance level for the test does not reach
    `p_val_threshold` will be discarded. Columns can be excluded from the
    description by adding them to the `omit_columns` argument.
    """
    stats = describe_subset_features(subset, p_val_threshold, omit_columns)

    descriptions = [f"{feature}: {desc}" for feature, desc in stats.items()]

    if len(descriptions) == 0:
        return "No distinguishing features"

    return "\n".join(descriptions)
