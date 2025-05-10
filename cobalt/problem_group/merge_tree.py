# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from __future__ import annotations

from typing import List

import numpy as np
from mapper.graph import CSRGraph
from mapper.merge_tree import MergeTree


def graph_select_superlevel_components(
    graph: CSRGraph,
    node_values: np.ndarray,
    threshold: float,
) -> List[List[int]]:
    mt = MergeTree.from_csr_graph(graph, np.asarray(node_values, dtype=np.float64))
    segments = [
        segment
        for segment in mt.segments
        if segment.birth_level > threshold and segment.death_level <= threshold
    ]
    node_groups = [
        [n for n in segment.nodes if node_values[n] > threshold] for segment in segments
    ]
    return node_groups


def graph_select_merge_tree_components(
    graph: CSRGraph,
    node_values: np.ndarray,
    node_weights: np.ndarray,
    min_node_value: float,
    stability_threshold: float,
) -> List[List[int]]:
    """Select local superlevel components from a graph.

    Builds a merge tree for the provided node_values function, and selects
    branches from the merge tree to maximize the total weighted stability. Only
    returns groups of nodes whose stability is above stability_threshold.

    Args:
        graph: a mapper.CSRGraph representing the graph to be analyzed.
        node_values: an array of function values for the nodes in the graph.
        node_weights: weights for each node of the graph, used when calculating
            stability for a subset.
        min_node_value: the minimum function value for a node to be included in
            a returned group.
        stability_threshold: the minimum stability value for a returned group.
    """
    mt = MergeTree.from_csr_graph(graph, np.asarray(node_values, dtype=np.float64))
    mt.compute_stability_and_select_branches(
        node_weights=node_weights,
        min_birth_level=min_node_value,
    )
    stable_segments = [
        segment
        for segment in mt.segments
        if segment.selected and segment.stability >= stability_threshold
    ]
    node_groups = [
        [node for node in segment.nodes if node_values[node] > min_node_value]
        for segment in stable_segments
    ]
    node_groups = [nodes for nodes in node_groups if len(nodes) > 0]
    return node_groups
