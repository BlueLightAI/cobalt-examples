import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

import cobalt


def feature_data_cross_correlations(
    data_matrix: np.ndarray,
    feature_hierarchical_graph,
    data_hierarchical_graph,
    feature_level: int,
    data_level: int,
) -> np.ndarray:
    """Return Data Group X Feature Group correlations.

    For each clusterat a given level i of feature_graph and given level j of data_graph.
    """
    out = np.zeros(
        (
            len(data_hierarchical_graph.levels[data_level].nodes),
            len(feature_hierarchical_graph.levels[feature_level].nodes),
        )
    )
    n_data_points = len(data_hierarchical_graph.levels[0].nodes)
    for i, dg in enumerate(data_hierarchical_graph.levels[data_level].nodes):
        mask = np.zeros(n_data_points)
        mask[dg] = 1
        for j, fg in enumerate(feature_hierarchical_graph.levels[feature_level].nodes):
            feature_selection = data_matrix[:, fg]
            feature_mean = feature_selection.mean(axis=1)
            corr_matrix = np.corrcoef(feature_mean, mask)
            correlation = corr_matrix[0, 1]
            out[i, j] = correlation
    return out


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def surface_insights(
    data_matrix,
    data_graph,
    feature_graph,
    data_level,
    feature_level,
    feature_names: np.ndarray,
    n_insights: int = 10,
):
    """Prints insights from data X features."""
    if feature_level >= len(feature_graph.levels):
        raise Exception(
            f"Feature Level too high {feature_level} >= {len(feature_graph.levels)}"
        )

    if data_level >= len(data_graph.levels):
        raise Exception(f"Data Level too high {data_level} >= {len(data_graph.levels)}")

    out = feature_data_cross_correlations(
        data_matrix, feature_graph, data_graph, feature_level, data_level
    )

    abs_r = np.abs(out)

    # Find Top n_insights Crosses in abs_r and return those Sets Cross One Another.
    x, y = largest_indices(abs_r, n_insights)

    for i, (data_group_index, feature_group_index) in enumerate(zip(x, y)):
        f_indices = feature_graph.levels[feature_level].nodes[feature_group_index]
        d_indices = data_graph.levels[data_level].nodes[data_group_index]
        print(
            f"Data Group {i} has interesting (elavated/depressed) values of \
              Features {feature_names[f_indices]}"
        )
        print(
            f"N = {len(d_indices)}, Global Average {data_matrix[:, f_indices].mean(axis = 0)},\
                  Group Average {data_matrix[:, f_indices][d_indices].mean(axis = 0)}"
        )


def main():

    data = load_iris()
    X = data.data
    target = data.target
    df = pd.DataFrame(X, columns=data.feature_names)
    df["target"] = target

    feature_df = pd.DataFrame(X, columns=data.feature_names)
    feature_df = feature_df.T
    feature_df.columns = [f"Feature {i}" for i in range(feature_df.shape[1])]

    ds = cobalt.CobaltDataset(df)
    ds.add_embedding_array(X)

    feature_ds = cobalt.CobaltDataset(feature_df)
    feature_ds.add_embedding_array(X.T)

    w_data = cobalt.Workspace(ds)
    w_features = cobalt.Workspace(feature_ds)

    ds_graph = w_data.new_graph()
    feature_graph = w_features.new_graph(metric="correlation")

    surface_insights(X, ds_graph, feature_graph, 5, 1, np.array(data.feature_names), 6)


if __name__ == "__main__":
    main()
