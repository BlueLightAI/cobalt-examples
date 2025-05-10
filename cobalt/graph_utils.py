# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import mapper
import networkx as nx
import numpy as np
import pandas as pd

from cobalt import CobaltDataset


def graph_to_subsets(graph: mapper.DisjointPartitionGraph, ds: CobaltDataset):
    return [ds.subset(indices) for indices in graph.nodes]


def aggregate_scores(
    subsets: List[List[int]],
    ds: CobaltDataset,
    columns: List[str],
    aggregation_method="mean",
) -> List[pd.Series]:
    if not isinstance(columns, list):
        raise ValueError(f"Columns is of type {type(columns)}, must be of type list")
    subset_columns = [ds.subset(indices).df[columns] for indices in subsets]
    if aggregation_method == "mean":
        return [series.mean() for series in subset_columns]
    else:
        raise NotImplementedError()


def aggregate_scores_multi(
    subsets: List[List[int]],
    ds: CobaltDataset,
    columns: List[str],
    aggregation_methods: List[Callable[[pd.Series], Any]],
    output_column_name_generator: Callable[[str, str], str],
) -> List[Dict[str, float]]:
    """Returns per subset summary aggregated scores.

    Args:
        subsets (List[List[int]]): groups of data points to aggregate over.
        ds (CobatDataset): dataset with columns that should be aggregated.
        columns (List[str]): columns upon which to aggregate.
        aggregation_methods (List[Callable]): Functions to aggregate series.
        output_column_name_generator: Method to translate column and agg method into new name.

    Returns:
        Per Subset List of summary scores. Each element contains the product of
        aggregation methods and the columns.

    """
    if not isinstance(columns, list):
        raise ValueError(f"Columns is of type {type(columns)}, must be of type list")
    subset_dataframes = [ds.subset(indices).df[columns] for indices in subsets]

    output = []
    # Per subset summary
    for df in subset_dataframes:
        outputs = {}
        for agg_method in aggregation_methods:
            for col in columns:
                series = df[col]
                column_name = output_column_name_generator(col, agg_method.__name__)
                score = agg_method(series)
                # Construct
                outputs[column_name] = score
        output.append(outputs)

    return output


def multi_level_graph_to_subsets(
    graph: mapper.HierarchicalPartitionGraph, ds: CobaltDataset
):
    """Returns Dictionary of Subsets per level of Hierarchical Graph."""
    # TODO: Combine this interface with aggregation?
    return {i: graph_to_subsets(g, ds) for i, g in enumerate(graph.levels)}


def multi_level_graph_to_subsets_with_aggregation(
    graph: mapper.HierarchicalPartitionGraph,
    ds: CobaltDataset,
    columns: Union[str, List[str]],
    min_level: int,
    max_level: int,
    aggregation_method: Literal["mean"] = "mean",
) -> List[List[pd.Series]]:
    """Returns Subsets per level of Hierarchical Graph."""
    if isinstance(columns, str):
        columns = [columns]

    return [
        aggregate_scores(graph.levels[i].nodes, ds, columns, aggregation_method)
        for i in range(min_level, max_level)
    ]


def node_ids_to_data_point_ids(
    g: mapper.DisjointPartitionGraph, nodes: np.ndarray
) -> np.ndarray:
    if len(nodes) > 0:
        return np.concatenate([g.nodes[i] for i in nodes])
    else:
        return np.array([], dtype=np.int32)


def get_edge_ranks(edge_list: List[Tuple[int, int]], n_nodes: int) -> np.ndarray:
    # assumes edges sorted in decreasing order of weight
    left_ranks = np.zeros(len(edge_list), dtype=np.int32)
    right_ranks = np.zeros(len(edge_list), dtype=np.int32)
    node_degree = np.zeros(n_nodes, dtype=np.int32)
    for e, (i, j) in enumerate(edge_list):
        left_ranks[e] = node_degree[i]
        right_ranks[e] = node_degree[j]
        node_degree[i] += 1
        node_degree[j] += 1

    return np.minimum(left_ranks, right_ranks)


def get_graph_edge_list(
    graph: mapper.DisjointPartitionGraph, n_edges: int, max_rank: Optional[int] = None
) -> Tuple[List[Tuple[int, int]], int]:
    edge_list = graph.edge_list[:n_edges]
    n_nodes = len(graph.nodes)
    if max_rank:
        ranks = get_edge_ranks(edge_list, n_nodes)
        edge_list = [e for e, rank in zip(edge_list, ranks) if rank <= max_rank]
    return edge_list, n_nodes


def graph_to_weighted_networkx(g: mapper.DisjointPartitionGraph) -> nx.Graph:
    """Convert a DisjointPartitionGraph into a weighted NetworkX Graph."""
    nxg = nx.Graph()
    nxg.add_nodes_from(range(len(g.nodes)))
    nxg.add_edges_from(
        [
            (u, v, {"weight": w, "idx": i})
            for i, ((u, v), w) in enumerate(zip(g.edge_list, g.edge_weights))
        ]
    )
    return nxg


def select_graph_degree(
    graph: mapper.DisjointPartitionGraph,
    max_avg_degree: float = 15,
    min_avg_degree: float = 6,
) -> float:
    """Choose an average degree for a graph.

    Tries to minimize this degree while maximizing connectivity of the graph,
    subject to the constraints given by max_avg_degree and min_avg_degree.
    """
    nxg = graph_to_weighted_networkx(graph)
    tr = nx.maximum_spanning_tree(nxg)
    n_nodes = len(graph.nodes)

    for e in sorted(tr.edges(data=True), key=lambda x: x[2]["weight"]):
        idx = e[2]["idx"]
        avg_deg = 2 * (idx + 1) / n_nodes
        if avg_deg <= max_avg_degree:
            break
    else:
        avg_deg = max_avg_degree

    return max(avg_deg, min_avg_degree)


def graph_superlevel_components(
    graph: mapper.DisjointPartitionGraph,
    node_values: np.ndarray,
    threshold: float,
    n_edges: int,
    max_edge_rank: Optional[int] = None,
) -> List[np.ndarray]:
    """Finds the superlevel components of a function defined on a MapperGraph.

    Returns the components as a list of numpy arrays containing node ids.
    """
    node_mask = node_values > threshold
    nodes = np.flatnonzero(node_mask)
    # TODO: do this without NetworkX
    edge_list, n_nodes = get_graph_edge_list(graph, n_edges, max_edge_rank)
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(n_nodes))
    nx_graph.add_edges_from(edge_list)
    induced_subgraph = nx.subgraph(nx_graph, nodes)
    components = [np.array(list(u)) for u in nx.connected_components(induced_subgraph)]
    return components
