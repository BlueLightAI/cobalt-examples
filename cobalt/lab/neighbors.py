import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from mapper import DisjointPartitionGraph, HierarchicalPartitionGraph

from cobalt import CobaltDataset, Workspace
from cobalt.schema.dataset import CobaltDataSubset, DatasetBase


def to_v2v(graph: DisjointPartitionGraph) -> Dict[int, List[int]]:
    """Computes a V2V Dictionary.

    Args:
        graph (DisjointPartitionGraph): Graph to convert.

    Returns:
        v2v (Dict[int, List[int]]): v2v structure keyed by vertex.

    """
    edges = graph.edge_list

    v2v = {}

    order = len(graph.nodes)
    for i in range(order):
        v2v[i] = []

    for i, j in edges:
        v2v[i].append(j)
        v2v[j].append(i)

    return v2v


def get_neighbors(
    graph: DisjointPartitionGraph,
    node_id: int,
    k_strongest_neighbors: Optional[int] = None,
) -> List[int]:
    """Returns list of neighbors.

    Args:
        graph (DisjointPartitionGraph): Graph to evaluate.
        node_id (int): Query vertex.
        k_strongest_neighbors (int): Number of adjacent nodes to return.

    Returns:
        neighbors (List[int]): Adjacent Node IDs
    """
    v2v = to_v2v(graph)
    adjacent_nodes = v2v[node_id]

    # Post process `adjacent_nodes` to extract k_nearest nodes if k_nearest is not None.

    if k_strongest_neighbors is None:
        return adjacent_nodes

    # Because we compute v2v from a sorted structure by weight, we can return
    # the neighbors that have the largest affinity with the query node.
    return adjacent_nodes[:k_strongest_neighbors]


def get_neighboring_labels(
    graph: DisjointPartitionGraph,
    node_id: int,
    label_source: List[str],
    k_strongest_neighbors: Optional[int] = None,
) -> List[str]:
    """Translates nearest neighbors to text labels given source."""
    nghbrs = get_neighbors(graph, node_id, k_strongest_neighbors)
    return [label_source[n] for n in nghbrs]


def get_neighboring_subsets(
    dataset: DatasetBase,
    graph: DisjointPartitionGraph,
    node_id: int,
    k_strongest: Optional[int] = None,
) -> List[CobaltDataSubset]:
    """Returns list of neighboring subsets.

    Args:
        dataset (DatasetBase): Dataset to pull subsets from.
        graph (DisjointPartitionGraph): Graph to evaluate.
        node_id (int): Query vertex.
        k_strongest (int): Number of adjacent nodes to return.

    Returns:
        neighbors (List[CobaltDataSubset]): Adjacent Groups

    """
    node_list = get_neighbors(graph, node_id, k_strongest)
    nodes = graph.nodes
    subsets = [dataset.subset(nodes[node_index]) for node_index in node_list]
    return subsets


def get_neighbors_all(
    graph: DisjointPartitionGraph,
    k_strongest_neighbors: Optional[int] = None,
) -> Dict[int, List[str]]:
    """Translates nearest neighbors to text labels given source."""
    v2v = to_v2v(graph)

    # TODO: Combine this interface with per node.

    adjacent_nodes = {node_id: v2v[node_id] for node_id in v2v}

    # Post process `adjacent_nodes` to extract k_nearest nodes if k_nearest is not None.

    if k_strongest_neighbors is None:
        return adjacent_nodes

    # Because we compute v2v from a sorted structure by weight, we can return
    # the neighbors that have the largest affinity with the query node.
    adjacency_sets = {node_id: v2v[node_id][:k_strongest_neighbors] for node_id in v2v}
    return adjacency_sets


def generate_cluster_df_at_coarseness_level(
    coarseness_lvl: int,
    g: HierarchicalPartitionGraph,
    keywords_per_level_per_node: Dict[int, List[str]],
    max_neighbors: int = 3,
    return_indices: bool = False,
) -> pd.DataFrame:
    """Returns summary of clusters data.

    Args:
        coarseness_lvl (int): Coarseness level, accessed via `g.levels[coarseness_lvl]`.
        g (HierarchicalPartitionGraph): the graph to extract a partition from.
        keywords_per_level_per_node (Dict[int, List[str]]): level wise, node wise labels.
        max_neighbors (int): max neighbors to include in data table.
        return_indices (bool): whether to return node indices or not.

    Returns:
        df (pd.DataFrame): table of information about each node at a single DisjointPartitionGraph.
        Elements in Neighbor ID columns are -1 if edges are not present.

    """
    labels_ = get_neighbors_all(g.levels[coarseness_lvl], max_neighbors)
    node_sizes = [len(n) for n in g.levels[coarseness_lvl].nodes]

    relevant_node_keywords = keywords_per_level_per_node[coarseness_lvl]

    def pad_with_none(array, target_length: int):
        return array + [None] * (target_length - len(array))

    labels_ = {key: pad_with_none(val, max_neighbors) for key, val in labels_.items()}
    df = pd.DataFrame(labels_).T

    df.columns = [f"Neighbor ID {i}" for i in range(max_neighbors)]
    for item in df.columns:
        df[item] = df[item].fillna(-1).astype(int)

    df["Label"] = df.index.map(lambda key: relevant_node_keywords[key])
    df["Node Size"] = df.index.map(lambda key: node_sizes[key])
    for i in range(max_neighbors):
        df[f"Neighbor Label {i}"] = df[f"Neighbor ID {i}"].map(
            lambda key: relevant_node_keywords[key] if key != -1 else None,
        )

    if return_indices:
        return df

    return (
        df.drop([f"Neighbor ID {i}" for i in range(max_neighbors)], axis=1)
        .reset_index()
        .drop("index", axis=1)
    )


def generate_cluster_df_at_levels(
    g: HierarchicalPartitionGraph,
    keywords_per_level_per_node: Dict[int, List[str]],
    selected_level_range: Tuple[int, int],
    max_neighbors: int = 3,
    return_indices: bool = False,
) -> pd.DataFrame:
    """Returns summary of clusters data.

    Args:
        g (HierarchicalPartitionGraph): the graph to extract a partition from.
        keywords_per_level_per_node (Dict[int, List[str]]): level wise, node wise labels.
        max_neighbors (int): max neighbors to include in data table.
        selected_level_range (Tuple[int, int]): start and end level of range.
        return_indices (bool): whether to return node indices or not.

    Returns:
        df (pd.DataFrame): table of information about each node at a single DisjointPartitionGraph.
        Elements in Neighbor ID columns are -1 if edges are not present.

    """
    dfs = []

    start = selected_level_range[0]
    end = selected_level_range[1]

    def pad_with_none(array, target_length: int):
        return array + [None] * (target_length - len(array))

    # it's erroring at the largest level.
    for coarseness_lvl in range(start, end):
        labels_ = get_neighbors_all(g.levels[coarseness_lvl], max_neighbors)
        node_sizes = [len(n) for n in g.levels[coarseness_lvl].nodes]

        relevant_node_keywords = keywords_per_level_per_node[coarseness_lvl]

        labels_ = {
            key: pad_with_none(val, max_neighbors) for key, val in labels_.items()
        }

        df = pd.DataFrame(labels_).T

        # This was previously a problem if we had singletons.
        if len(df.columns) != max_neighbors:
            raise Exception(
                f"Something went wrong. Got an unanticipated number of neighbors {len(df.columns)}"
            )

        df.columns = [f"Neighbor ID {i}" for i in range(max_neighbors)]
        for item in df.columns:
            df[item] = df[item].fillna(-1).astype(int)

        df["Label"] = [relevant_node_keywords[item] for item in df.index]
        df["Node Size"] = [node_sizes[item] for item in df.index]
        for i in range(max_neighbors):
            df[f"Neighbor Label {i}"] = [
                (relevant_node_keywords[key] if key != -1 else None)
                for key in df[f"Neighbor ID {i}"]
            ]
        df["level"] = coarseness_lvl
        dfs.append(df)
    df = pd.concat(dfs)

    if return_indices:
        return df

    return (
        df.drop([f"Neighbor ID {i}" for i in range(max_neighbors)], axis=1)
        .reset_index()
        .drop("index", axis=1)
    )


def get_all_subsets_with_label(
    coarseness: int,
    label: str,
    g: HierarchicalPartitionGraph,
    keywords_per_level: Dict[int, Dict[int, str]],
):
    """Returns list of subsets corresponding to a label at a given coarseness (resolution) level."""
    d = keywords_per_level[coarseness]
    topic_to_nodes = {}

    for _, val in d.items():
        topic_to_nodes[val] = []

    for key, val in d.items():
        topic_to_nodes[val].append(key)

    node_ids = topic_to_nodes[label]

    data_point_id_sets = [g.levels[coarseness].nodes[node_id] for node_id in node_ids]
    return data_point_id_sets


def is_label_plural_at_level(label: str, node_to_keyword: Dict[int, str]):
    """Returns True where label has count exceeding 1, False otherwise."""
    label_count = 0
    for _, val in node_to_keyword.items():
        if val == label:
            label_count += 1
            if label_count > 1:
                return True

    return False


def get_raw_subset_with_label(
    coarseness: int,
    label: str,
    g: HierarchicalPartitionGraph,
    ds: CobaltDataset,
    keywords_per_level: Dict[int, Dict[int, str]],
) -> CobaltDataSubset:
    """Views data given coarseness, label computation."""
    # TODO: Convert this whole structure into a class
    # so the notebook user does not need to keep track of intermediates.
    d = keywords_per_level[coarseness]

    # Validation Step.
    # Check if `label` provided is unique.
    if is_label_plural_at_level(label, d):
        warnings.warn(
            f"More than one subset has label '{label}'. Only one subset will be returned. "
            "Consider using get_all_subsets_with_label to find all subsets with this label.",
            stacklevel=1,
        )

    topic_to_node = {topic: i for i, topic in d.items()}
    node_id = topic_to_node[label]
    subset = g.levels[coarseness].nodes[node_id]
    return ds.subset(subset)


def get_raw_subsets_with_label(
    coarseness: int,
    label: str,
    g: HierarchicalPartitionGraph,
    ds: CobaltDataset,
    keywords_per_level: Dict[int, Dict[int, str]],
) -> List[CobaltDataSubset]:
    """Views data given coarseness, label computation."""
    # TODO: Convert this whole structure into a class
    # so the notebook user does not need to keep track of intermediates.
    subsets = get_all_subsets_with_label(coarseness, label, g, keywords_per_level)
    return [ds.subset(subset) for subset in subsets]


def main():
    embedding = np.arange(100)[:, np.newaxis] * np.ones(100)
    df = pd.DataFrame({"a": np.arange(100)})
    ds = CobaltDataset(df)
    ds.add_embedding_array(embedding)
    w = Workspace(ds)
    g = w.new_graph()

    graph = g.levels[4]

    subsets = get_neighboring_subsets(ds, graph, 2)

    print("Data points for Node ID = 2", graph.nodes[2])
    print("Neighboring Groups", [subset.indices for subset in subsets])

    label_test_simulation = [f"Node Label {i}" for i in range(len(graph.nodes))]

    labels = get_neighboring_labels(graph, 2, label_test_simulation, 3)
    print(f"Node Query Label: {label_test_simulation[2]}")
    print("Adjacent Labels:", labels)


if __name__ == "__main__":
    main()
