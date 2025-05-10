# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Dict, List, Literal, Tuple

import networkx as nx
import numpy as np
from mapper import DisjointPartitionGraph, MultiResolutionGraph, quick_graph
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from cobalt.schema import Classifier
from cobalt.workspace import Workspace


def disjoint_to_nx(graph: DisjointPartitionGraph) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(len(graph.nodes)))
    g.add_edges_from(graph.edge_list)
    return g


def get_pure_connected_components_subset(
    pg: DisjointPartitionGraph, subset: List[int]
) -> List:
    """Returns connected components of graph filtered by nodes.

    Return all nodes whos data points are a subset of `subset` indices.

    Use Case: You have a train and production set. You're looking for the pockets of production data
      that exist outside the training disribution. You can think of these as "gaps"
      in the training set. This function spots the gaps in these training data.

    """
    # Strong typing.

    assert isinstance(pg, DisjointPartitionGraph)
    subset = set(subset)

    intersections = [subset.intersection(dps) for dps in pg.nodes]

    selected_indices = [
        i for i in range(len(pg.nodes)) if len(pg.nodes[i]) == len(intersections[i])
    ]

    # List of nodes that are purely from the production set `subset`.

    g_nx = disjoint_to_nx(pg)
    subgraph = g_nx.subgraph(selected_indices)
    return list(nx.connected_components(subgraph))


def get_data_points_from_components(pg: DisjointPartitionGraph, components: List):
    """Returns data points from connected components.

    Note that we might want to do this one component at a time.
    """
    connected_nodes = list(set().union(*components))
    nodes = np.array(pg.nodes, dtype=object)
    return {dp for dps in nodes[connected_nodes] for dp in dps}


def is_level_pure_enough_for_subset(
    pg: DisjointPartitionGraph, subset: List[int], purity_threshold=0.9
):
    """Returns bool if the `pg`'s is pure enough for a certain `subset`.

    Of all of the nodes that contain the data points in the listed subset
    Computes the proportion of nodes that are pure.

    Returns boolean: proportion > purity_threshold

    """
    # This is really the generalization of the above method.
    subset = set(subset)
    dp_sets = [set(dps) for dps in pg.nodes]

    intersections = [subset.intersection(dps) for dps in dp_sets]

    has_val_of_interest = [len(intersection) > 0 for intersection in intersections]
    is_purely_val = np.array(
        [
            len(dp_set) == len(intersection)
            for dp_set, intersection in zip(dp_sets, intersections)
        ]
    )

    return is_purely_val[has_val_of_interest].mean() > purity_threshold


def autozoom_to_subset(
    mg: MultiResolutionGraph, subset: List[int], purity_threshold=0.9
) -> DisjointPartitionGraph:
    """Autozooms on the graph hierarchy so that `subset` is in focus."""
    for level in reversed(mg.levels):
        # Choose level where n_nodes is as small as possible,
        # and feature of interest is pure.
        if is_level_pure_enough_for_subset(level, subset, purity_threshold):
            return level

    return None


def get_drifted_components(
    hierarchical_graph: MultiResolutionGraph, subset: List[int], purity_threshold: float
) -> Tuple[List, DisjointPartitionGraph]:
    """Autozooms to correct graph, then extracts distinct production groups."""
    # Chooses the level where the production set is "in focus".
    production_group_in_focus: DisjointPartitionGraph = autozoom_to_subset(
        hierarchical_graph, subset, purity_threshold
    )

    components = get_pure_connected_components_subset(production_group_in_focus, subset)
    return components, production_group_in_focus


def get_classifiers_to_predict_components(
    components: List,
    graph: DisjointPartitionGraph,
    classifier_type: str,
    hyperparams: Dict,
) -> Tuple[List, List]:
    classifiers = []
    subsets = []
    for component in tqdm(components):
        dps = list(get_data_points_from_components(graph, [component]))
        if classifier_type == "knn":
            n_neighbors = hyperparams.get("n_neighbors", 1)
            classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif classifier_type == "Logistic Regression":
            C = hyperparams.get("C", 1)
            classifier = LogisticRegression(C=C)
        else:
            raise NotImplementedError("Classification Type Not Implemented.")
        classifiers.append(classifier)
        subsets.append(dps)
    return classifiers, subsets


def train_classifiers_to_predict_components(
    concatenated_embedding: np.ndarray,
    components: List,
    graph: DisjointPartitionGraph,
    classifier_type: str,
    hyperparams: Dict,
) -> List:
    classifiers, subsets = get_classifiers_to_predict_components(
        components, graph, classifier_type, hyperparams
    )

    for classifier, subset in zip(classifiers, subsets):
        y = np.zeros(len(concatenated_embedding))
        y[subset] = 1
        classifier.fit(concatenated_embedding, y)
    return classifiers


def train_classifiers_from_train_test_embeddings(
    train_embeddings: np.ndarray,
    production_embeddings: np.ndarray,
    metric: str,
    classifier_type: Literal["knn", "Logistic Regression"],
    hyperparams: Dict,
    purity_threshold: float = 0.9,
    min_component_size: int = 10,
) -> List[Classifier]:
    """Returns classifiers for gaps in training set.

    Picture a fabric of train and production.
    There are pockets within that fabric where only data from production lives.
    There is no data from the training set there. This function constructs the graph,
    chooses the right graph so that it's properly zoomed in on these region(s),
    and then identifies the pockets within the fabric and lastly trains simple
    classifiers to predict those pockets.
    """
    large_stack = np.concatenate([train_embeddings, production_embeddings], axis=0)
    production_subset = np.arange(len(train_embeddings), len(large_stack))

    # Build a graph of the concatenated embeddings.
    hg = quick_graph(large_stack, metric)

    # Choose a graph (really a clustering) that has as few nodes (low resolution) as possible.
    # Keep Moving Down the cluster tree from the top, iteratively check the degree to which data
    # within the production_subset
    # is clustered purely (no baseline subset in those clusters).

    # Question: What if this fails?
    # By definition, actually it will not fail because at coarseness = 0
    # the graph is perfectly pure.
    autoresolution_graph: DisjointPartitionGraph = autozoom_to_subset(
        hg, production_subset, purity_threshold
    )

    # From this autoresolution graph, remove the data points that don't come
    # from the production subset and return the remaining connected
    # components - these connected components are your "drifted regions."
    # Note that currently it is implemented too strictly currently. (TODO)
    components = get_pure_connected_components_subset(
        autoresolution_graph, production_subset
    )

    # Filter down your components to only include components that are greater than
    # size = `min_component_size`.
    components = [
        component
        for component in components
        if len(get_data_points_from_components(autoresolution_graph, [component]))
        >= min_component_size
    ]

    # Given components, train classifiers to predict them from the rest of the dataset.
    # These classifiers can now be thought of as features.
    classifiers = train_classifiers_to_predict_components(
        large_stack, components, autoresolution_graph, classifier_type, hyperparams
    )

    return classifiers


def identify_correct_training_data(
    train_embedding: np.ndarray,
    production_embedding: np.ndarray,
    data_lake_embedding: np.ndarray,
    metric: str,
    hyperparams: Dict,
    min_component_size: int = 10,
    as_mask: bool = False,
) -> List[int]:
    """Returns indices for which data should be concatenated to the original dataset.

    All three embeddings must be generated by the same model.
    This method is an all-in-one method that identifies the gaps in the train distribution.
    based on the production distribution and then trains classifiers to predict these gaps.
    And then uses those classifiers to select the subset of data in a data lake that
    corresponds to these regions.
    """
    classifiers = train_classifiers_from_train_test_embeddings(
        train_embedding,
        production_embedding,
        metric,
        classifier_type=hyperparams.get("classifier_type", "knn"),
        hyperparams=hyperparams,
        min_component_size=min_component_size,
    )

    # Two Options for Classifiers: LogisticRegression where we control
    # the probability to avoid false positives.
    # Must tune C in Logistic Regression.
    # KNN: Tune k: seems to have an issue with performance on `predict`

    # Use classifiers to select training data to concatenate with our original dataset.
    selected_mask = np.zeros(len(data_lake_embedding))
    for cl in classifiers:
        mask = cl.predict(data_lake_embedding).astype(np.bool_)
        selected_mask[mask] = 1

    if not as_mask:
        return np.nonzero(selected_mask)[0]

    return selected_mask


def get_drifted_regions(
    workspace: Workspace,
    new_region_name: str,
    embedding_index: int,
    metric: str,
    purity_threshold="all",
    graph_name: str = "drift",
    min_region_size=2,
    max_region_size=200,
):
    """Returns Drifted Subsets."""
    dataset = workspace.state.dataset.as_subset()

    results = []

    embedding = workspace.state.dataset.embedding_metadata[embedding_index]

    assert metric in embedding.admissible_distance_metrics

    # TODO: Use Existing Graph If Possible.
    g = workspace.new_graph(graph_name, dataset, embedding_index, metric)

    if purity_threshold == "all":
        thresholds = np.linspace(0.01, 0.99, 10)

        # Experimentally, running this at many levels gives different interesting results.
        # This is not necessarily advisable.
        for threshold in thresholds:
            components, chosen_graph = get_drifted_components(
                g, workspace.state.split[new_region_name].indices, threshold
            )
            subsets = [
                dataset.subset(
                    list(get_data_points_from_components(chosen_graph, [component]))
                )
                for component in components
            ]

            subsets = [
                subset
                for subset in subsets
                if len(subset) >= min_region_size and len(subset) <= max_region_size
            ]
            for i, subset in enumerate(subsets):

                workspace.add_group(
                    f"Drifted Region_Purity = {threshold}, group {i}", subset
                )

            results.extend(subsets)
    return results
