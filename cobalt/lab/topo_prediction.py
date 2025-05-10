# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Union

import mapper
import numpy as np
import pandas as pd
import pynndescent


class NNIndexGraph:
    def __init__(
        self,
        graph: Union[mapper.HierarchicalPartitionGraph, mapper.DisjointPartitionGraph],
    ) -> None:
        self.graph = graph
        self._index = None

    def _get_graph_source_data(self):
        if isinstance(self.graph.source_dataset, mapper.MapperMatrix):
            source_data = self.graph.source_dataset.X
        else:
            source_data = self.graph.neighbor_graph.data_matrix.X
        return source_data

    def compute_index(self):
        nhbrs = self.graph.neighbor_graph.raw_graph._neighbors

        source_data = self._get_graph_source_data()

        metric = self.graph.neighbor_graph.metric
        self._index = pynndescent.NNDescent(
            source_data, metric=metric, init_graph=nhbrs
        )

    def get_index(self):
        return self._index

    def is_indexed(self):
        return self._index is not None

    def get_nearest_labels(
        self, X: np.ndarray, y: np.ndarray, n_neighbors: int = 5
    ) -> np.ndarray:
        """Returns labels of nearest neighbors for query X.

        Args:
            X (ndarray): array of query vectors of shape (n_queries, n_dimensions).
            y (ndarray): labels corresponding to the data in the underlying graph.
            n_neighbors (int): number of nearest neighbors to return.

        Returns:
            out (np.ndarray): nearest n_neighbors labels for each element in X.

        Notes:
        X must be a 2D array of shape (N, D), not a 1D vector of shape (D,).

        Finds the n_neighbors nearest neighbors in the index for each point in
        X, and returns the corresponding values in y.
        """
        # Consider conversion here.

        if len(X.shape) == 1:
            raise ValueError(
                "X should be 2-dimensional with shape (n_samples, n_dimensions)."
                f"X had shape {X.shape}."
            )

        if not isinstance(y, np.ndarray):
            y = np.array(y)

        source_data = self._get_graph_source_data()

        if len(y) != len(source_data):
            raise ValueError(
                f"len(y) ({len(y)}) should be equal to the number of "
                f"data points in the source graph ({len(source_data)})."
            )
        if not self.is_indexed():
            self.compute_index()
        descent = self.get_index()
        n, _ = descent.query(X, k=n_neighbors)
        return y[n]

    def make_prediction(
        self,
        X: np.ndarray,
        y: np.ndarray,
        is_classification: bool = True,
        n_neighbors: int = 5,
    ) -> np.ndarray:
        """Returns topological classification for input data X.

        Args:
            X (ndarray): Input array of shape (n_samples, n_dimensions).
            y (ndarray): Labels corresponding to training data (graph)
            is_classification (bool): Whether the task is classification or regression.
            n_neighbors (int): Number of neighbors to use when making predictions.

        Predictions are based on similarity to data points in the graph used to
        construct the NNIndexGraph object. Ground truth labels are provided in
        y, which must contain one entry for each data point in the original
        graph. X may have any number of data points but must have the same
        dimensionality as the data used to construct the original graph.

        If `is_classification`: return mode of nearest labels (i.e. a voting
        mechanism).  Otherwise, returns mean of nearest labels (regression
        problem).

        n_neighbors controls the number of nearest neighbors used to make the
        prediction.
        """
        nearest_labels = self.get_nearest_labels(X, y, n_neighbors)

        if is_classification:
            df = pd.DataFrame(nearest_labels)
            return df.mode(axis=1).to_numpy()[:, 0]
        else:
            return nearest_labels.mean(axis=1)
