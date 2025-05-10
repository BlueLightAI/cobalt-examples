# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from contextlib import contextmanager
from typing import Dict, Optional

from cobalt.cobalt_types import SelectionState
from cobalt.config import handle_cb_exceptions
from cobalt.schema import CobaltDataset, CobaltDataSubset
from cobalt.visualization import EmbeddingVisualization


class SelectionManager:
    def __init__(
        self,
        dataset: CobaltDataset,
        visualization: Optional[EmbeddingVisualization] = None,
    ):
        self._graph_selection_callbacks_enabled = True

        self.dataset = dataset
        self.visualization = visualization

        self.selection_state = SelectionState.initial

        self.graph_selection = dataset.subset([])
        self.group_selection = None
        self.comparison_group: Dict[str, str] = {}

        self.graph_selection_listeners = []
        self.group_selection_listeners = []
        self.comparison_group_listeners = []
        self.clear_selection_listeners = []

    def set_visualization(self, visualization):
        self.visualization = visualization

    @handle_cb_exceptions
    def set_graph_selection(self, selection: CobaltDataSubset):
        if self._graph_selection_callbacks_enabled:
            if len(selection) == 0:
                self.selection_state = SelectionState.initial
                self.graph_selection = selection.source_dataset.subset([])
            else:
                self.selection_state = SelectionState.selected
                self.graph_selection = selection

            for cb in self.graph_selection_listeners:
                cb(self.graph_selection)

    def on_graph_select(self, cb):
        self.graph_selection_listeners.append(cb)

    @handle_cb_exceptions
    def set_group_selection(self, selection):
        self.selection_state = SelectionState.selected
        self.group_selection = selection

        for cb in self.group_selection_listeners:
            cb(selection)

    @handle_cb_exceptions
    def set_comparison_group(self, group_type, group_id):
        self.comparison_group = {"group_type": group_type, "group_id": group_id}

        for cb in self.comparison_group_listeners:
            cb(self.comparison_group)

    @handle_cb_exceptions
    def trigger_clear_selection_listeners(self):
        for cb in self.clear_selection_listeners:
            cb()

    def on_clear_selection(self, cb):
        self.clear_selection_listeners.append(cb)

    def on_group_select(self, cb):
        self.group_selection_listeners.append(cb)

    def on_comparison_group_select(self, cb):
        self.comparison_group_listeners.append(cb)

    def get_node_selection(self, subset):
        if self.visualization:
            return self.visualization.get_nodes_from_subset(subset)
        else:
            return []

    def get_graph_point_selection(self, subset):
        if self.visualization:
            return self.visualization.get_points_from_subset(subset)
        else:
            return []

    @contextmanager
    def disable_graph_selection_callbacks(self):
        self._graph_selection_callbacks_enabled = False
        try:
            yield
        finally:
            self._graph_selection_callbacks_enabled = True

    @handle_cb_exceptions
    def update_graph_visualization(self, subset):
        with self.disable_graph_selection_callbacks():
            point_selection = self.get_graph_point_selection(subset)
            self.visualization.landscape.graph_viewer.selected_points = point_selection
