# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import numpy as np
import pynndescent
from scipy.stats import entropy


def generalization_score(
    production_embedding: np.ndarray,
    metric: str,
    train_embedding: np.ndarray,
    production_prediction: np.ndarray,
    production_target: np.ndarray,
    k: int = 5,
):
    """Scores the model performance k-nearest prod. data for each point in the train dataset."""
    index = pynndescent.NNDescent(production_embedding, metric)
    n, _ = index.query(train_embedding, k)
    generalization = np.zeros(len(train_embedding))
    for i, indices in enumerate(n):
        acc = (production_prediction[indices] == production_target[indices]).mean()
        generalization[i] = acc
    return generalization


def _neighborhood_entropy_score(neighbor_graph: np.ndarray, feature: np.ndarray):
    N = len(neighbor_graph)
    target_entropy = np.zeros(N)
    for i, indices in enumerate(neighbor_graph):
        # TODO: There is a way of parallelizing this.
        _, counts = np.unique(feature[indices], return_counts=True)
        normalized_counts = counts / len(indices)
        ent = entropy(normalized_counts)
        target_entropy[i] = ent
    return target_entropy


def neighborhood_entropy_score(
    embedding: np.ndarray,
    metric: str,
    categorical_feature: np.ndarray,
    k: int = 5,
):
    """Computes neighborhood entropy per data point given feature.

    Computes the entropy of the k-nearest neighbors
      for each point in the embedding given a corresponding
        categorical variable.

    You can interpret high entropy as high "confusion" points, indicating points on a boundary,
      and low entropy as low confusion "points", or points a distance away from a boundary.

    Notes:
    - You can pass in model predictions or targets for `categorical_feature`,
    or another feature entirely.

    """
    assert embedding.shape[0] == len(categorical_feature)

    # TODO: Implement version of idea of local variance,
    #  (some other metric?) for numerical features.

    index = pynndescent.NNDescent(embedding, metric, n_neighbors=k)
    n, _ = index.neighbor_graph
    return _neighborhood_entropy_score(n, categorical_feature)


def main():
    N = 30
    linear_embedding = np.arange(N)[:, np.newaxis] * np.ones((N, 4))
    prediction = np.zeros(N)
    prediction[(N // 2) :] = 1
    out = neighborhood_entropy_score(linear_embedding, "euclidean", prediction)
    print(out)


if __name__ == "__main__":
    main()
