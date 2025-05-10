# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import numpy as np
import pandas as pd


def generate_clusters_dataframe(
    n_clusters, n_points, dim=16, prediction=None, seed=98293
):
    r = np.random.default_rng(seed=seed)
    cluster_centers = r.standard_normal((n_clusters, dim))
    baseline_cluster_ids = r.integers(0, n_clusters - 1, n_points)
    comparison_cluster_ids = r.integers(1, n_clusters, n_points)
    baseline_data = cluster_centers[baseline_cluster_ids, :]
    baseline_data += 0.9 * r.standard_normal(baseline_data.shape)
    comparison_data = cluster_centers[comparison_cluster_ids, :]
    comparison_data += 0.9 * r.standard_normal(comparison_data.shape)

    baseline_df = pd.DataFrame(baseline_data)
    baseline_df["cluster"] = baseline_cluster_ids
    baseline_df["timestamp"] = pd.Timestamp("2023-01-01")
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df["cluster"] = comparison_cluster_ids
    comparison_df["timestamp"] = pd.date_range(
        "2023-02-01", "2023-03-01", len(comparison_df)
    )
    # this avoids a bug in some recent versions of pandas
    baseline_df["timestamp"] = baseline_df["timestamp"].astype(
        comparison_df["timestamp"].dtype
    )
    df = pd.concat([baseline_df, comparison_df], axis=0, ignore_index=True)
    df["cluster"] = df["cluster"].astype("category")
    df["prediction"] = prediction if prediction else df["cluster"]
    for i in range(len(df)):
        if df.loc[i, "prediction"] == (n_clusters - 1) and r.uniform() < 0.3:
            df.loc[i, "prediction"] = r.integers(0, 4)
    df.columns = [f".dim_{c}" if isinstance(c, int) else c for c in df.columns]
    return df
