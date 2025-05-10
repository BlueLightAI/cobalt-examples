# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import random
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn import metrics


def get_split_numeric(feature, direction="up", target_size=None, target_subset=None):
    _, bins = pd.qcut(feature, 10, retbins=True, duplicates="drop")
    feature_test = feature[target_subset] if target_subset is not None else feature
    if direction == "up":  # select upwards interval
        if target_size is None:
            cut_val = random.choice(bins)
            subset_mask = feature > cut_val
        else:
            for cut_val in reversed(bins):
                if (feature_test > cut_val).sum() > target_size:
                    break
            subset_mask = feature > cut_val
    else:  # select downwards interval
        if target_size is None:
            cut_val = random.choice(bins)
            subset_mask = feature < cut_val
        else:
            for cut_val in bins:
                if (feature_test < cut_val).sum() > target_size:
                    break
            subset_mask = feature < cut_val
    return subset_mask, cut_val


def get_split_categorical(feature, target_size=None, target_subset=None, values_size=2):
    feature_ = feature.astype("category")
    feature_test = feature_[target_subset] if target_subset is not None else feature_

    value_counts = feature_test.value_counts()
    values = set(value_counts.index)
    values_subset = []
    if target_size is None:
        values_subset = random.choices(
            list(values), weights=value_counts[list(values)], k=values_size
        )
    else:
        while value_counts[values_subset].sum() < target_size:
            choice = random.choices(
                list(values), weights=value_counts[list(values)], k=1
            )[0]
            values.remove(choice)
            values_subset.append(choice)
    subset_mask = feature.apply(lambda x: x in values_subset)
    return subset_mask, values_subset


def choose_failure_mode_features(
    features: pd.DataFrame,
    target_test_size: int,
    test_mask: Optional[pd.Series] = None,
    interaction_size: int = 1,
    reltol: float = 0.5,
    min_size: int = 50,
):
    """Constructs a subset of the dataset in `features` defined by an intersection of slices.

    Tries to get a subset whose intersection with a specified test subset is
    approximately `target_test_size`. This is measured by the `reltol`
    parameter.

    Chooses a subset defined by conditions on `interaction_size` different
    columns of the given data.

    Returns a mask for each condition as well as strings describing the
    conditions and the final size of the created subset in the test set.

    Does this by rejection sampling with a few tweaks to help get
    things that are roughly the right size.
    """
    # TODO: add option to disable per_feature_target_test_size
    # TODO: add option to always include certain features

    if interaction_size > len(features.columns):
        raise ValueError(
            "interaction size cannot be greater than the number of features"
        )
    if test_mask is None:
        test_mask = pd.Series(np.ones(len(features), dtype=np.bool_))
    failure_group_size = 0
    test_size = test_mask.sum()
    per_feature_target_test_size = min(
        target_test_size * interaction_size**2, test_size / 2
    )

    # we're doing rejection sampling to get a failure mode whose size in the
    # test set is close to the target and greater than the minimum
    while (
        abs(target_test_size - failure_group_size) > reltol * target_test_size
        or failure_group_size < min_size
    ):
        subset_masks = []
        descriptions = []
        available_columns = set(features.columns)
        for _ in range(interaction_size):
            c = random.choice(list(available_columns))
            available_columns.remove(c)
            # we bin numeric columns with too many different values
            if (
                pd.api.types.is_numeric_dtype(features[c].dtype)
                and len(features[c].unique()) > 10
            ):
                r = random.random()
                if r < 0.5:  # select upwards interval
                    subset_mask, cut_val = get_split_numeric(
                        features[c],
                        direction="up",
                        target_size=per_feature_target_test_size,
                        target_subset=test_mask,
                    )
                    descriptions.append(f"{c}>{cut_val}")
                else:  # select downwards interval
                    subset_mask, cut_val = get_split_numeric(
                        features[c],
                        direction="down",
                        target_size=per_feature_target_test_size,
                        target_subset=test_mask,
                    )
                    descriptions.append(f"{c}<{cut_val}")

            # otherwise just choose a subset of values
            else:
                subset_mask, values_subset = get_split_categorical(
                    features[c],
                    target_size=per_feature_target_test_size,
                    target_subset=test_mask,
                )
                if len(values_subset) > 1:
                    descriptions.append(f"{c} in {values_subset}")
                else:
                    descriptions.append(f"{c}=={values_subset[0]}")

            subset_masks.append(subset_mask)
        mask = pd.Series(np.ones(len(features), dtype=np.bool_))
        for subset_mask in subset_masks:
            mask &= subset_mask
        failure_group_size = mask[test_mask].sum()

    return mask, descriptions, failure_group_size


def perturb_training_labels(
    failure_mode: pd.Series,
    train_mask: pd.Series,
    y: pd.Series,
    method: Literal["least_common", "most_common", "uniform"] = "least_common",
):
    # assume y is categorical for now
    y_perturbed = y.copy()
    y_cat = y.astype("category")

    # should we look at the distribution on the whole dataset? if it differs
    # significantly between train and test we may already have a failure mode,
    # or this may not be a very good failure mode
    train_outcome_dist = y_cat[failure_mode & train_mask].value_counts()
    print(train_outcome_dist)
    if method == "least_common":
        new_label = train_outcome_dist.index[-1]
        y_perturbed[failure_mode & train_mask] = new_label
        print(f"changed all points in the group to {new_label}")
    elif method == "most_common":
        new_label = train_outcome_dist.index[0]
        y_perturbed[failure_mode & train_mask] = new_label
        print(f"changed all points in the group to {new_label}")
    elif method == "uniform":
        n_samples = (failure_mode & train_mask).sum()
        y_perturbed[failure_mode & train_mask] = [
            random.choice(train_outcome_dist.index) for i in range(n_samples)
        ]
    else:
        raise ValueError(f"unsupported method '{method}'")

    return y_perturbed


# need to check that failure modes are (pairwise) disjoint before training the model.
# really that they're reasonably well separated


def evaluate_failure_modes(
    discovered_modes: List[pd.Series],
    true_modes: List[pd.Series],
    precision_threshold_for_mapping=0.6,
):
    precision_scores = np.zeros((len(discovered_modes), len(true_modes)))
    for i in range(len(discovered_modes)):
        for j in range(len(true_modes)):
            precision_scores[i, j] = metrics.precision_score(
                true_modes[j], discovered_modes[i]
            )

    matched_modes = np.argmax(precision_scores, axis=1)
    best_precision_scores = precision_scores[
        range(len(discovered_modes)), matched_modes
    ]
    for i in range(len(discovered_modes)):
        if precision_scores[i, matched_modes[i]] < precision_threshold_for_mapping:
            matched_modes[i] = -1
            best_precision_scores[i] = 0

    recall_scores = np.zeros(len(true_modes))
    precision_cover_scores = np.zeros(len(true_modes))
    n_covering_modes = np.zeros(len(true_modes), dtype=np.int32)
    for i in range(len(true_modes)):
        mask = np.zeros(len(discovered_modes[0]), dtype=np.bool_)
        covering_modes = np.flatnonzero(matched_modes == i)
        n_covering_modes[i] = len(covering_modes)
        for j in covering_modes:
            mask |= discovered_modes[j]
        recall_scores[i] = metrics.recall_score(true_modes[i], mask)
        precision_cover_scores[i] = metrics.precision_score(true_modes[i], mask)

    return (
        matched_modes,
        best_precision_scores,
        recall_scores,
        precision_cover_scores,
        n_covering_modes,
    )
