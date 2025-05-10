# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

# TODO: Refactor this file


def transform_obj_to_cat_codes(data: pd.DataFrame):
    # Ideally, we don't do this copy
    # but otherwise it's modifying the df in place.
    data = data.copy()

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].astype("category").cat.codes

    return data


def get_labels_for_unsupervised_task(df: pd.DataFrame):
    LABEL = "label"
    df_c = df.copy()
    df_c[LABEL] = 0
    # subsample df copy to half its size
    df_c = df_c.sample(frac=0.5)
    # sample from the original df to match the size of the subsampled df

    # Create a new sampled dataframe of synthetic data where from each column we
    # interpret each column's values as a distribution and randomly sample from those distributions
    # independently.
    sampled_df = df.apply(lambda x: x.sample(n=df_c.shape[0], ignore_index=True))

    sampled_df[LABEL] = 1
    new_df = pd.concat([df_c, sampled_df])
    return new_df.drop(LABEL, axis=1), new_df[LABEL]


def get_rf_embeddings(
    df: pd.DataFrame,
    outcome_column: Optional[str] = None,
):
    embeddings = None
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    metric_choice = "hamming"  # default metric choice for RF embeddings

    df = transform_obj_to_cat_codes(df)

    clf = (
        RandomForestClassifier(
            n_estimators=50, max_depth=7, max_samples=0.25, n_jobs=-1
        )
        if not outcome_column or df[outcome_column].dtype == "object"
        else RandomForestRegressor(
            n_estimators=50, max_depth=7, max_samples=0.25, n_jobs=-1
        )
    )

    if outcome_column:
        X, y = df.drop(outcome_column, axis=1), df[outcome_column]
        clf.fit(X, y)
        embeddings = clf.apply(X)
    else:
        # Because there is no outcome column, we artificially label it so that
        # we can still run random forest on it.
        # THen we proceed as normal with a RandomForestClassifier trained
        # on this unsupervised task
        # and apply this model to the original dataframe.
        X, y = get_labels_for_unsupervised_task(df)
        clf.fit(X, y)
        embeddings = clf.apply(df)

    return embeddings, metric_choice


def get_tabular_embeddings(
    df: pd.DataFrame,
    model_name: Optional[Literal["rf"]] = None,
    outcome: Optional[str] = None,
) -> Tuple[np.ndarray, str, str]:
    """Create an embedding array based on the given df and embedding method.

    *Note that the design of this function is in flux.* Currently supports
    generating embeddings via a random forest model.

    Args:
        df: pandas.DataFrame containing the data.
        model_name: String indicating whether the model to be used is "rf".
        outcome: String name of the desired outcome column in `df`, for method
            == "model" embeddings.

    Returns:
        a tuple (embedding_array, metric, name).
    """
    if model_name == "rf":
        embeddings, metric_choice = get_rf_embeddings(df, outcome)
    else:
        raise ValueError(
            f"Invalid `model_name`: {model_name} specified for tabular embeddings"
        )

    return embeddings, metric_choice, f"{model_name} embedding"
