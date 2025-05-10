# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import mapper
import numpy as np
from mapper import MultiResolutionGraph

from cobalt import schema
from cobalt.config import handle_cb_exceptions


@dataclass
class FilterSpec:
    """A set of parameters for a filter on a graph.

    Separates the dataset into n_bins bins, based on the values of f_vals for
    each data point. Data points within each bin are clustered to form nodes,
    and are linked together if they are in nearby bins.
    """

    f_vals: np.ndarray
    """An array of values, one for each data point."""
    n_bins: int = 10
    """The number of bins to separate the dataset into."""
    bin_method: Literal["rng", "uni"] = "rng"
    """Either "rng" or "uni". If "rng", the bins will have equal width;
    if "uni" they will have equal numbers of data points."""
    pruning_method: Literal["bin", "pct"] = "bin"
    """Either "bin" or "pct". If "bin", will only allow edges
    between nodes from nearby bins. If "pct", will only allow edges between
    nodes whose percentile difference for f_vals is within the given
    threshold.
    """
    pruning_threshold: Union[int, float] = 1
    """The maximum distance two nodes can be apart while still being connected."""


@dataclass
class GraphSpec:
    """A set of parameters for creating a graph."""

    X: np.ndarray
    """The source data. Should be an array of shape (n_points, n_dims)."""
    metric: str
    """The name of the distance metric to use to create the graph."""
    M: Optional[int] = None
    """The number of nearest neighbors to compute for each data point."""
    K: Optional[int] = None
    """The number of mutual nearest neighbors to keep for each data point."""
    min_nbrs: Optional[int] = None
    """The minimum number of neighbors to keep for each data point."""
    L_coarseness: int = 20
    """The number of neighbors to keep for each data point when
    clustering data points into graph nodes."""
    L_connectivity: int = 20
    """The number of neighbors to keep for each data point when
    connecting nodes in the graph."""
    filters: Sequence[FilterSpec] = ()
    """A (possibly empty) list of FilterSpec objects that describe
    filters to apply to the graph."""


class GraphBuilder:
    def __init__(self, state):
        self.state = state

    @staticmethod
    @handle_cb_exceptions
    def mapper_graph_from_spec(params: GraphSpec) -> mapper.HierarchicalPartitionGraph:
        """Construct a Mapper graph with the given parameters.

        Chooses some parameters automatically if not provided.
        """
        if params.M is None or params.K is None or params.min_nbrs is None:
            ng = mapper.NeighborGraph.with_automatic_params(params.X, params.metric)
        else:
            ng = mapper.NeighborGraph(
                params.X, params.M, params.K, params.metric, params.min_nbrs
            )

        filter_sources = [
            mapper.RawFunction(ng.data_matrix, spec.f_vals) for spec in params.filters
        ]
        filters = [
            mapper.Filter(
                fs,
                spec.n_bins,
                spec.bin_method,
                spec.pruning_method,
                spec.pruning_threshold,
            )
            for fs, spec in zip(filter_sources, params.filters)
        ]

        g = mapper.HierarchicalPartitionGraph(
            ng,
            filters=filters,
            L_coarseness=params.L_coarseness,
            L_connectivity=params.L_connectivity,
        )
        return g

    def new_graph(
        self,
        subset: Union[str, schema.CobaltDataSubset],
        embedding: Union[int, str, schema.Embedding],
        metric: Optional[str] = None,
        **kwargs,
    ) -> MultiResolutionGraph:
        """Create a new graph from a specified subset.

        The resulting graph will be returned and added to the Workspace.

        Args:
            subset: The subset of the dataset to include in the graph. If a string,
                will try to use a subset with that name from the dataset split or the
                saved groups (in that order). Otherwise, should be a `CobaltDataSubset`.
            embedding: The embedding to use to generate the graph. May be specified
                as an index into self.dataset.embeddings, the name of the embedding, or
                an `Embedding` object.
            metric: The distance metric to use when constructing the graph. If none
                is provided, will use the metric specified by the embedding.
            **kwargs: Any additional keyword parameters will be interpreted as parameters to
                construct a `GraphSpec` object.
        """
        subset_ = self.get_subset(subset)
        if len(subset_) < 2:
            raise ValueError(
                "Subset for graph construction must contain at least two elements."
            )
        X, embedding_metadata = self.get_embedding(subset_, embedding)

        if metric is None:
            metric = embedding_metadata.default_distance_metric

        g = self.mapper_graph_from_spec(GraphSpec(X, metric, **kwargs))
        return g

    def get_subset(
        self, subset: Union[str, schema.CobaltDataSubset]
    ) -> schema.CobaltDataSubset:
        return self.state.get_subset(subset)

    @staticmethod
    def get_embedding(
        subset: schema.CobaltDataSubset, embedding: Union[int, str, schema.Embedding]
    ) -> Tuple[np.ndarray, schema.Embedding]:
        X, embedding_metadata = subset.get_graph_inputs(embedding)
        return X, embedding_metadata
