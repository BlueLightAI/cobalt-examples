# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Any, Dict, List, Optional, Tuple

import mapper
import numpy as np
from mapper.cluster_tree import ClusterTree
from mapper.clustering_scores import (
    make_local_negative_modularity_score_fn,
    make_masked_negative_modularity_score_fn,
)
from mapper.flat_cluster import optimize_cluster_score_top_down

from cobalt.schema import CobaltDataSubset


# TODO: graph_data should just be part of graph
def autogroup_modularity(
    graph_data: CobaltDataSubset,
    graph: mapper.HierarchicalPartitionGraph,
    subset: Optional[CobaltDataSubset] = None,
    min_group_size: int = 1,
    max_group_size: int = np.inf,
    min_n_groups: int = 1,
    max_n_groups: int = 100,
    r: float = 0,
) -> Tuple[List[CobaltDataSubset], Dict[str, Any]]:
    """Find well-connected sets of nodes in a Mapper graph.

    Tries to maximize the modularity of the clustering for the graph, subject to
    constraints on the number of clusters and the cluster size.

    The modularity of a clustering, roughly speaking, measures how many edges in
    a graph stay inside of each cluster compared with what would happen in a
    completely random graph. Values closer to 1 indicate a better clustering,
    while values closer to 0 indicate that the clustering is not much different
    from random.

    Args:
        graph_data: A CobaltDataSubset containing the data the graph represents
        graph: A `mapper.HierarchicalPartitionGraph` built on `data`.
        subset: An optional CobaltDataSubset containing a subset of graph_data
            to run the clustering algorithm on. If None, runs the algorithm on
            all data points in the graph.
        min_group_size: The minimum size of a group to be returned.
        max_group_size: The maximum size of a group to be returned.
        min_n_groups: The minimum number of groups to return.
        max_n_groups: The maximum number of groups to return.
        r: A parameter controlling resolution. Higher values result in smaller clusters.

    Returns:
        A list of the discovered groups and a dictionary containing
        values of parameters that were used in the process.
    """
    if subset is None:
        subset = graph_data

    if subset == graph_data:
        # TODO: go down to a level with all nodes at most as large as min_group_size
        cluster_tree = ClusterTree.from_graph_with_constraints(
            graph, max_n_clusters=len(subset) / min_group_size
        )
        modularity_score = make_local_negative_modularity_score_fn(
            graph.neighbor_graph, modularity_r=r
        )
    else:
        try:
            subset_mask = subset.as_mask_on(graph_data)
        except ValueError:
            raise ValueError("subset must be a subset of graph_data.") from None
        subset_graph_indices = np.flatnonzero(subset_mask)

        cluster_tree = ClusterTree.from_subgraph_with_constraints(
            graph,
            subset_graph_indices,
            max_n_clusters=len(subset) / min_group_size,
        )
        modularity_score = make_masked_negative_modularity_score_fn(
            graph.neighbor_graph, subset_mask, r=r
        )
    modularity_clusters = optimize_cluster_score_top_down(
        cluster_tree,
        local_score_fns=[modularity_score],
        min_cluster_size=min_group_size,
        max_cluster_size=max_group_size,
        max_n_clusters=max_n_groups,
        min_n_clusters=min_n_groups,
    )

    groups = [graph_data.subset(indices) for indices in modularity_clusters]

    params = {
        "graph_data": graph_data,
        "graph": graph,
        "min_clusters": min_n_groups,
        "max_clusters": max_n_groups,
        "min_size": min_group_size,
    }

    return groups, params
