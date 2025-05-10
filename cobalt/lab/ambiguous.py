# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Optional

import mapper
import numpy as np


def _get_depths(g, predictions):
    """Returns the smallest depth where predictions are pure.

    This is an experimental method that can be used as a scoring
    function for each data point in an embedding.
    It can be considered as a measure of how ambiguous or risky any given data point is.
    """
    g_levels = g.levels.copy()
    g_levels.reverse()

    depths = np.zeros_like(predictions) - 1

    prev_node_length = 0

    for i, graph in enumerate(g_levels):
        nodes = graph.nodes
        assert len(nodes) >= prev_node_length
        prev_node_length = len(nodes)

        count_uniques = [len(np.unique(predictions[dps])) for dps in nodes]
        pure_nodes = [i for i, length in enumerate(count_uniques) if length == 1]

        # TODO: This can be vectorized too. We could get a complete list of the
        # data points that are discovered at this step in the hierarchy and set
        # all of those at once.
        for j in pure_nodes:
            dps = nodes[j]

            # TODO Optimization: This can be vectorized.
            # Maybe with `depths[dp][depths[dp] == -1] = i`.
            for dp in dps:
                # This means that as we iterate through the levels,
                # it will not write it again.
                if depths[dp] == -1:
                    depths[dp] = i
    return depths


def get_n_levels(model_embedding: np.ndarray, metric: str):
    quick_graph = mapper.quick_graph(model_embedding, metric)
    return quick_graph.n_levels


def get_datapoint_ambiguity(
    model_embedding: np.ndarray, predictions: np.ndarray, metric: str
):
    """Return data point ambiguity based on cluster-based approach.

    data points with low scores (non -1) are highly ambiguous.
    """
    quick_graph = mapper.quick_graph(model_embedding, metric)
    return _get_depths(quick_graph, predictions)


def get_most_ambiguous_groups_as_mask(
    model_embedding: np.ndarray,
    predictions: np.ndarray,
    metric: str,
    top_levels: Optional[int] = 1,
):
    """Return scores per data point about the most ambiguous group using the cluster-based approach.

    Auto-masks it to return a mask of just the most ambiguous data points, tunable with

    top_levels: int, which returns clusters taken from the lowest `top_levels` number of levels
    """
    scores = get_datapoint_ambiguity(model_embedding, predictions, metric)

    if top_levels is None:
        return scores
    return scores > scores.max() - top_levels


def get_borders(
    embedding: np.ndarray, predictions: np.ndarray, metric: str
) -> np.ndarray:
    """Return ambiguity scores per data point.

    Internals: returns the highest level where the data point appears in a
    cluster by itself and is connected with a pure cluster of another prediction class.

    Low values (non -1) for this value can be interpreted as data points that the
      model is highly ambiguous on.

    Splits that occur at higher levels generally relay more abstract,
      conceptual going-ons of the model.

    """
    hierarchical_graph = mapper.quick_graph(embedding, metric)

    is_border_score = np.full_like(predictions, fill_value=-1)

    for level in range(len(hierarchical_graph.levels)):
        g = hierarchical_graph.levels[level]

        # edges_data consists of tuples of data points.
        edges_data = [(g.nodes[s], g.nodes[d]) for (s, d) in g.edge_list]

        predictions_data = [(predictions[s], predictions[d]) for s, d in edges_data]

        # Compute `is_unique` edge mask, each node the edge represents must contains predictions
        # of only one class.
        is_unique = np.array(
            [
                (len(np.unique(s)) == 1) and (len(np.unique(d)) == 1)
                for s, d in predictions_data
            ]
        )

        # For every edge check if the two nodes are both pure
        # and correspond to different predicted classes.
        is_mismatch = [
            is_unique and (x[0] != y[0])
            for (x, y), is_unique in zip(predictions_data, is_unique)
        ]

        # Filter data by `is_mismatch`.
        filtered_data = [
            (edges_data[i], i) for i in range(len(predictions_data)) if is_mismatch[i]
        ]

        # Write mismatched pure edges to their score.
        # Since edges become less pure as you ascend the hierarchy,
        # it will inherently stop at some point, so you can thus get
        # an ambiguity score for each data point.

        # TODO: Vectorize this.
        for (s, d), _ in filtered_data:
            is_border_score[s] = level
            is_border_score[d] = level

    return is_border_score
