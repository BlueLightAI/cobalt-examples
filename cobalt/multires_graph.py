# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import ipyvuetify as v
import landscape_widget
import mapper
import numpy as np
import pandas as pd
from mapper import MultiResolutionGraph

from cobalt import CobaltDataSubset
from cobalt.coloring.color_spec import (
    FunctionColoringSpec,
    GraphColoringSpec,
    NoColoringSpec,
)
from cobalt.config import settings
from cobalt.graph_utils import get_edge_ranks, select_graph_degree

LAYOUT_DEFAULT_WIDTH = "500px"
LAYOUT_DEFAULT_HEIGHT = "400px"
LAYOUT_EXPANDED_WIDTH = "100%"
LAYOUT_EXPANDED_HEIGHT = "700px"

DEFAULT_NODE_COLOR = "#888888"

NodeSizeFunction = Callable[[np.ndarray], List[float]]


def get_node_sizes_power_law(
    node_counts: np.ndarray,
    default_size: float = 4,
    min_scale: float = 0.75,
    max_scale: float = 6,
    max_power: float = 1,
) -> List[float]:
    """Compute radii for nodes in a graph based on the number of data points represented.

    Uses a power law scaling nodes between min_scale times the default size and
    max_scale times the default size.
    """
    min_count = np.quantile(node_counts, 0.01)
    max_count = np.quantile(node_counts, 0.99)
    if max_count == min_count:
        return [default_size] * len(node_counts)
    alpha = min(
        np.log(max_scale / min_scale) / np.log(max_count / min_count), max_power
    )
    node_counts = node_counts / min_count
    node_sizes = default_size * min_scale * (node_counts) ** (alpha / 2)
    node_sizes = np.minimum(node_sizes, default_size * max_scale)
    node_sizes = np.maximum(node_sizes, default_size * min_scale)
    return node_sizes.tolist()


def get_node_sizes_log(
    node_counts: np.ndarray,
    min_size: float = 4,
    base: float = 2,
    power: float = 0.5,
) -> List[float]:
    """Compute radii for nodes in a graph based on the number of data points represented.

    Node area will scale as 1 + log_base(n_points / min_points).
    """
    min_count = np.min(node_counts)
    max_count = np.max(node_counts)
    if max_count == min_count:
        return [min_size] * len(node_counts)
    node_counts = node_counts / min_count
    node_sizes = min_size * (1 + np.log(node_counts) / np.log(base)) ** power
    return node_sizes.tolist()


def select_graph_coarseness_values(
    nodes_per_level: List[int],
    init_max_nodes: int,
    global_max_nodes: int,
    global_min_nodes: int = 1,
) -> Tuple[int, ...]:
    """Choose maximum, minimum, and initial coarseness levels for the graph.

    The minimum and initial coarseness levels are chosen so that they are as
    fine as possible while not having more than global_max_nodes and
    init_max_nodes nodes, if possible.
    """
    # ensure global_max >= init_max >= global_min
    # global_max will always bind, since it's most important to ensure we don't have too many nodes
    init_max_nodes = min(init_max_nodes, global_max_nodes)
    global_min_nodes = min(global_min_nodes, init_max_nodes, global_max_nodes)

    max_value = len(nodes_per_level) - 1
    levels_less_than_global_max = [
        i for i, n_nodes in enumerate(nodes_per_level) if n_nodes <= global_max_nodes
    ]

    if not levels_less_than_global_max:
        # only one available level
        return max_value, max_value, max_value

    min_value = levels_less_than_global_max[0]

    levels_less_than_init_max = [
        i for i, n_nodes in enumerate(nodes_per_level) if n_nodes <= init_max_nodes
    ]
    if not levels_less_than_init_max:
        return min_value, max_value, max_value

    init_value = levels_less_than_init_max[0]

    levels_greater_than_global_min = [
        i for i, n_nodes in enumerate(nodes_per_level) if n_nodes >= global_min_nodes
    ]
    if levels_greater_than_global_min:
        max_value = levels_greater_than_global_min[-1]
    return min_value, init_value, max_value


class MultiResolutionLandscape(v.Flex):
    def __init__(
        self,
        graph: mapper.HierarchicalPartitionGraph,
        init_max_nodes: int = 500,
        global_max_nodes: int = 5000,
        init_max_degree: float = 15.0,
        global_min_nodes: int = 20,
        global_max_degree: float = 20.0,
        use_weights: bool = False,
        labels_per_level: Optional[Dict[int, List[str]]] = None,
        node_size_function: NodeSizeFunction = get_node_sizes_power_law,
        use_rich_labels: Optional[bool] = None,
        **kwargs,
    ):

        # TODO: selected data points really should be handled in this class
        self.graph = graph
        self.use_weights = use_weights
        self.node_size_function = node_size_function
        self.persistent_labels = False
        self.labels_per_level = labels_per_level
        if use_rich_labels is None:
            use_rich_labels = settings.USE_RICH_NODE_LABELS
        self.use_rich_labels = use_rich_labels

        layout = kwargs.get("layout", {})
        kwargs["layout"] = layout

        if "sigma_settings" in kwargs:
            kwargs["sigma_settings"]["showHtmlHoverDefault"] = use_rich_labels
        else:
            kwargs["sigma_settings"] = {"showHtmlHoverDefault": use_rich_labels}

        self.landscape_kwargs = kwargs

        self.coloring_config: GraphColoringSpec = NoColoringSpec()

        nodes_per_level = [len(level.nodes) for level in self.graph.levels]
        if nodes_per_level[-1] > global_max_nodes:
            warnings.warn(
                f"All graph levels have more than the limit of {global_max_nodes} nodes."
                "Visualization performance may be compromised.",
                stacklevel=2,
            )
        min_level, init_level, max_level = select_graph_coarseness_values(
            nodes_per_level, init_max_nodes, global_max_nodes, global_min_nodes
        )
        graph_dict = serialize_mapper_graph(
            graph, list(range(min_level, max_level + 1))
        )
        starting_graph = self.graph.levels[init_level]
        init_degree = select_graph_degree(
            starting_graph, init_max_degree, min_avg_degree=3
        )
        init_degree = round(math.ceil(init_degree * 10) / 10, 1)

        self._initialize_graph(graph_dict, init_level, init_degree)

        self.widgets = [self.graph_viewer]
        self.graph_viewer.observe(self.toggle_graph_layout, "is_expanded_layout")

        self.coloring_legend = None
        # max-height to prevent stretching this block bottom
        self.legend_box = v.Flex(
            style_=(
                "max-height: 400px; overflow-y: auto;"
                f"overflow-x: hidden, width: {LAYOUT_DEFAULT_WIDTH}"
            )
        )
        super().__init__(children=self.widgets, style_=f"width: {LAYOUT_DEFAULT_WIDTH}")

    def _initialize_graph(self, graph_dict: Dict, init_level: int, init_degree: float):
        self.current_graph = self.graph.levels[init_level]
        self.graph_dict = graph_dict

        self.graph_viewer = landscape_widget.Landscape(
            graph=graph_dict,
            selected_level=init_level,
            avg_degree=init_degree,
            **self.landscape_kwargs,
        )
        self.graph_viewer.observe(self._coarseness_updated, "selected_level")

    def reload_graph(
        self,
        order_edges_by_rank: bool = False,
        use_weights: bool = False,
        node_size_function: NodeSizeFunction = get_node_sizes_power_law,
    ):
        current_level = self.graph_viewer.selected_level
        current_connectivity = self.graph_viewer.avg_degree
        levels = list(self.graph_dict.keys())
        graph_dict = serialize_mapper_graph(
            self.graph,
            levels,
            use_weights=use_weights,
            order_by_rank=order_edges_by_rank,
            node_size_function=node_size_function,
        )
        # this should be discarded anyway, but just in case let's remove listeners
        self.graph_viewer.unobserve_all()
        self._initialize_graph(graph_dict, current_level, current_connectivity)
        self.graph_viewer.observe(self.toggle_graph_layout, "is_expanded_layout")
        self._apply_coloring()
        self.widgets[0] = self.graph_viewer
        self.children = self.widgets

    def get_selected_data_indices(self, graph: mapper.DisjointPartitionGraph):
        return np.array(self.graph_viewer.selected_points, dtype=np.int32)

    def set_coloring(
        self,
        f: pd.Series,
        color_map_name: str,
        aggregation_method: str = "mean",
        cmap_range=None,
        node_label_gen=None,
    ):
        self.set_coloring_spec(
            FunctionColoringSpec(
                f, color_map_name, aggregation_method, cmap_range, node_label_gen
            )
        )

    def set_coloring_spec(self, spec: GraphColoringSpec):
        self.coloring_config = spec
        self._apply_coloring()

    def clear_coloring(self):
        self.coloring_config = NoColoringSpec()
        self._apply_coloring()

    def _apply_coloring(self):
        colors = {}
        hover_labels = {}
        legends = {}
        for level in self.graph_dict:
            color, label, legend = self.coloring_config.apply_to_nodes(
                self.graph.levels[level].nodes, use_rich_labels=self.use_rich_labels
            )
            colors[level] = color
            hover_labels[level] = label
            legends[level] = legend
        self.legends = legends

        self.graph_viewer.colors = colors

        if self.persistent_labels:
            self.graph_viewer.node_labels = self.labels_per_level
        else:
            self.graph_viewer.node_labels = {}

        if self.use_rich_labels:
            self.graph_viewer.node_attrs = hover_labels
            self.graph_viewer.rich_node_hover_template = (
                self.coloring_config.rich_node_label_template
            )
        else:
            self.graph_viewer.node_hover_labels = hover_labels

        self.update_legend()

    def _coarseness_updated(self, *_):
        self.update_legend()
        self.current_graph = self.graph.levels[self.graph_viewer.selected_level]

    def update_legend(self, *_):
        self.coloring_legend = self.legends[self.graph_viewer.selected_level]()
        self.legend_box.children = [self.coloring_legend]

    def enable_persistent_labels(self):
        self.persistent_labels = True
        self._apply_coloring()

    def disable_persistent_labels(self):
        self.persistent_labels = False
        self._apply_coloring()

    def refresh_graph(self):
        self._apply_coloring()

    def toggle_graph_layout(self, *_):
        if self.graph_viewer.is_expanded_layout:
            self.graph_viewer.layout.width = LAYOUT_EXPANDED_WIDTH
            self.graph_viewer.layout.height = LAYOUT_EXPANDED_HEIGHT
            self.style_ = f"width: {LAYOUT_EXPANDED_WIDTH}"
        else:
            self.graph_viewer.layout.width = LAYOUT_DEFAULT_WIDTH
            self.graph_viewer.layout.height = LAYOUT_DEFAULT_HEIGHT
            self.style_ = f"width: {LAYOUT_DEFAULT_WIDTH}"
        self.refresh_graph()
        self.graph_viewer.reset_view()


def serialize_mapper_graph(
    g: mapper.HierarchicalPartitionGraph,
    levels: List[int],
    use_weights: bool = False,
    order_by_rank: bool = False,
    node_size_function: NodeSizeFunction = get_node_sizes_power_law,
) -> Dict[int, Dict]:
    """Convert g into a dict suitable as input for the Landscape widget."""
    out = {}
    for level in levels:
        pg = g.levels[level]
        node_sizes = node_size_function(np.array([len(u) for u in pg.nodes]))
        if order_by_rank:
            edge_ranks = get_edge_ranks(pg.edge_list, len(pg.nodes))
            edge_sort_ix = np.argsort(edge_ranks, kind="stable")
            edge_list = pg.edge_mtx[edge_sort_ix, :]
            weights = pg.edge_weights[edge_sort_ix]
        else:
            edge_list = pg.edge_mtx
            weights = pg.edge_weights

        if use_weights:
            edges = [
                {"s": int(s), "t": int(t), "w": float(w)}
                for ((s, t), w) in zip(edge_list, weights)
            ]
        else:
            edges = [{"s": int(s), "t": int(t)} for (s, t) in edge_list]

        nodes = [
            {
                "points": [int(i) for i in u],
                "size": float(size),
            }
            for u, size in zip(pg.nodes, node_sizes)
        ]
        out[level] = {"nodes": nodes, "edges": edges}

    return out


def get_selected_nodes_from_data_points(graph, selected_indices):
    selected_nodes = []
    for idx, node_indices in enumerate(graph.nodes):
        if any(data_point in selected_indices for data_point in node_indices):
            selected_nodes.append(idx)
    return selected_nodes


def create_multilandscape(
    graph: MultiResolutionGraph,
    subset: CobaltDataSubset,
    init_max_nodes: int = 500,
    init_max_degree: float = 15.0,
    labels_per_level: Optional[Dict[int, List[str]]] = None,
):
    if not isinstance(graph, MultiResolutionGraph):
        raise TypeError(
            f"You should add MultiResolutionGraph object as graph. "
            f"Got {type(graph)}"
        )
    # ensures a user can recover the source dataset from the graph object alone
    # this overwrites a (currently meaningless) MapperMatrix
    # TODO: adjust the implementation in mapper-core
    graph.cobalt_subset = subset
    graph_landscape = MultiResolutionLandscape(
        graph,
        init_max_nodes=init_max_nodes,
        init_max_degree=init_max_degree,
        labels_per_level=labels_per_level,
    )

    new_landscape_object = {"graph": graph_landscape, "subset": subset}
    return new_landscape_object
