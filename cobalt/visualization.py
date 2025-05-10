# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, List, Optional

import ipyvuetify as v
import ipywidgets as w
import numpy as np

from cobalt.build_graph import GraphBuilder
from cobalt.config import handle_cb_exceptions
from cobalt.empty_visualization_block import EmptyVisualizationBlock
from cobalt.event_bus import EventBusController
from cobalt.multires_graph import MultiResolutionLandscape, create_multilandscape
from cobalt.schema import CobaltDataSubset
from cobalt.schema.dataset import fast_intersect_subset_with_indices
from cobalt.schema.embedding import INITIAL_EMBEDDING_INDEX, Embedding
from cobalt.state import State
from cobalt.ui_utils import with_tooltip
from cobalt.visualization_dialog import ReadModeVisualizationDialog, VisualizationDialog
from cobalt.widgets import (
    Button,
    NumericInput,
    ProgressBar,
    Select,
    TextField,
    VisualizationSelect,
)

metric_options = [
    "euclidean",
    "cosine",
    "hamming",
    "correlation",
]


class DataSourceStalenessManager:
    # _stale_sources[source] = True if that data source has changed but the
    # underlying data has not been requested.

    def __init__(self, sources) -> None:
        self._stale_sources = {source: True for source in sources}

    def set_all_stale(self):
        self._stale_sources = {source: True for source in self._stale_sources}

    def set_all_fresh(self):
        self._stale_sources = {source: False for source in self._stale_sources}

    def is_source_stale(self, source):
        return self._stale_sources[source]

    def set_source_stale(self, source):
        self._stale_sources[source] = True

    def set_source_fresh(self, source):
        self._stale_sources[source] = False


class DataSource(ABC):
    """Represents a CobaltDataSubset that may update in response to UI interactions."""

    def __init__(
        self,
        name: str,
    ):
        self.name = name
        self.update_callbacks = []

    @property
    @abstractmethod
    def data(self) -> CobaltDataSubset:
        pass

    @handle_cb_exceptions
    def on_update(self, cb: Callable):
        self.update_callbacks.append(cb)

    def update_data(self, *_):
        for cb in self.update_callbacks:
            cb()


class SubsetDataSource(DataSource):
    def __init__(self, subset: CobaltDataSubset, name: str):
        super().__init__(name)
        self.subset = subset

    @property
    def data(self):
        return self.subset


class LandscapeSelectionDataSource(DataSource):
    def __init__(self, dataselector):
        super().__init__("landscape selection")
        self.dataselector = dataselector
        self.dataselector.on_graph_select(self.update_data)

    @property
    def data(self) -> CobaltDataSubset:
        return self.dataselector.graph_selection


class DataSourceSampleSizeInput(NumericInput):
    def __init__(self, source: DataSource, init_value: int, **kwargs):
        self.source = source
        self.source.on_update(self.update_source_size)
        self.is_visible = False

        value = init_value
        self.data_source_sample_size_class = "my-2"

        super().__init__(
            label=f"Sample size ({self.source.name})",
            value=value,
            class_=self.data_source_sample_size_class,
            integer_only=True,
            allow_negative=False,
            allow_zero=False,
            minimum=1,
            maximum=len(self.source.data),
            **kwargs,
        )
        self.update_source_size()

    def update_source_size(self, *_):
        size = len(self.source.data)
        self.hint = f"{self.source.name} has {size} data points"

    def hide(self):
        self.class_ = f"{self.data_source_sample_size_class} d-none"
        self.is_visible = False

    def show(self):
        self.class_ = f"{self.data_source_sample_size_class} d-block"
        self.is_visible = True


class DataSourceSelector(v.Layout):
    """Provides a UI for selecting between a number of different data sources."""

    def __init__(
        self,
        data_source_options: List[DataSource],
        initial_selection: Optional[List[int]] = None,
        on_apply_callback: Optional[Callable] = None,
    ):
        self.data_source_options = data_source_options
        self.on_apply_callback = on_apply_callback

        self.apply_callbacks = [self.on_apply_callback]

        self.split_options = [
            ds for ds in self.data_source_options if not ds.name.startswith("Group")
        ]
        self.group_options = [
            ds for ds in self.data_source_options if ds.name.startswith("Group")
        ]

        self.data_source_selector = Select(
            items=self.create_items(),
            label="Data source",
            v_model=initial_selection,
            multiple=True,
            chips=True,
            small_chips=True,
            deletable_chips=True,
            item_text="title",
            item_value="value",
        )

        self.sample_size_inputs = [
            DataSourceSampleSizeInput(
                source,
                init_value=len(source.data),
            )
            for source in self.data_source_options
        ]

        for i, _ in enumerate(self.data_source_options):
            if i in initial_selection:
                self.sample_size_inputs[i].show()
            else:
                self.sample_size_inputs[i].hide()

        self.metric_selector = w.Box([])
        if self.data_source_options[0].data.embedding_metadata:

            initial_option, metric_options_data = self.get_metric_list_from_embeddor(
                self.data_source_options[0].data.embedding_metadata[
                    INITIAL_EMBEDDING_INDEX
                ]
            )

            self.metric_selector = Select(
                label="Distance metric",
                items=metric_options_data,
                v_model=initial_option,
            )

        source_dataset = self.data_source_options[0].data.source_dataset
        embedding_names = source_dataset.embedding_names()
        if len(embedding_names) > 1:
            self.embedding_drop_down = Select(
                label="Embedding",
                attach=True,
                items=source_dataset.embedding_names(),
                v_model=embedding_names[0],
                class_="mr-1",
            )

            def choose_embedding(*_):
                current_embedding = self.embedding_drop_down.v_model
                embeddor = source_dataset.get_embedding_metadata_by_name(
                    current_embedding
                )
                _, options = self.get_metric_list_from_embeddor(embeddor)
                self.metric_selector.items = options

            self.embedding_drop_down.on_event("change", choose_embedding)
        else:
            self.embedding_drop_down = w.Box([])

        self.current_data_sources = self.data_source_selector.v_model
        self.stale_source_manager = DataSourceStalenessManager(data_source_options)

        for source in self.current_data_sources:
            self.stale_source_manager.set_source_fresh(source)

        self.apply_button = Button(
            children=["Create Graph"], disabled=False, color="primary"
        )
        self.apply_button.on_event("click", self.apply)

        for i, source in enumerate(self.data_source_options):
            source.on_update(self.update_source_cb(i))

        super().__init__(
            column=True,
            children=[
                self.data_source_selector,
                self.embedding_drop_down,
                v.Flex(children=self.sample_size_inputs),
                self.metric_selector,
                self.apply_button,
            ],
        )

    def update_source_cb(self, source_idx: int):
        @handle_cb_exceptions
        def cb():
            self.stale_source_manager.set_source_stale(source_idx)

        return cb

    @staticmethod
    def get_metric_list_from_embeddor(embedding_metadata: Embedding):
        metric_options_data = [{"header": "Recommended metrics"}]
        recommended_metrics = embedding_metadata.admissible_distance_metrics
        if recommended_metrics:
            initial_option = recommended_metrics[0]
            for metric in recommended_metrics:
                metric_options_data.append({"text": metric})
        else:
            initial_option = metric_options[0]
        metric_options_data.append({"divider": True})
        metric_options_data.append({"header": "Other metrics"})
        for metric in metric_options:
            metric_options_data.append({"text": metric})
        return initial_option, metric_options_data

    @property
    def active_embedding(self):
        if self.current_data.embedding_metadata:
            embeddings = self.current_data.embedding_metadata
            if len(embeddings) == 1:
                return embeddings[0].name
            else:
                return self.embedding_drop_down.v_model
        else:
            return None

    @handle_cb_exceptions
    def source_update(self, *_):
        new_data_sources = self.data_source_selector.v_model

        for i in range(len(self.data_source_options)):
            if i in new_data_sources:
                self.sample_size_inputs[i].show()
            else:
                self.sample_size_inputs[i].hide()

    def get_selected_data_sources(self):
        data_source_idxs = self.data_source_selector.v_model
        data_sources = [self.data_source_options[i] for i in data_source_idxs]
        return data_sources

    def get_sample_sizes_per_data_source(self):
        data_source_idxs = self.data_source_selector.v_model
        sample_sizes = [int(self.sample_size_inputs[i].value) for i in data_source_idxs]
        return sample_sizes

    def get_data(self, random_state: Optional[int] = None) -> CobaltDataSubset:
        data_sources = self.get_selected_data_sources()
        sample_sizes = self.get_sample_sizes_per_data_source()
        subsets: List[CobaltDataSubset] = [
            source.data.sample(sample_size, random_state=random_state)
            for source, sample_size in zip(data_sources, sample_sizes)
        ]
        subset = self.concatenate(subsets)

        # TODO: Create another caching system
        self.current_data = subset

        return subset

    def concatenate(self, subsets: List[CobaltDataSubset]) -> CobaltDataSubset:
        # TODO move to CobaltDataSubset.
        if len(subsets) == 0:
            raise Exception("Data Subsets should not have length 0.")

        subset = subsets[0]
        for i in range(1, len(subsets)):
            subset = subset.concatenate(subsets[i])
        return subset

    def set_fresh_data_sources(self, data_sources):
        self.current_data_sources = data_sources
        for source in self.current_data_sources:
            self.stale_source_manager.set_source_fresh(source)

    def get_metric(self):
        # The embedding provided by the user currently goes unused.
        return self.metric_selector.v_model

    @handle_cb_exceptions
    def apply(self, *_):
        self.trigger_callbacks()

    def on_apply(self, cb):
        self.apply_callbacks.append(cb)

    def trigger_callbacks(self, *_):
        for cb in self.apply_callbacks:
            cb()

    def create_items(self):
        items = [{"header": "Splits"}]
        items += [
            {"title": ds.name, "value": i} for i, ds in enumerate(self.split_options)
        ]

        if self.group_options:
            items.append({"divider": True})
            items.append({"header": "Saved Groups"})
            items += [
                {"title": ds.name, "value": i + len(self.split_options)}
                for i, ds in enumerate(self.group_options)
            ]

        return items

    def update_options(self, new_data_source_options: List[DataSource]):
        self.data_source_options = new_data_source_options

        self.split_options = [
            ds for ds in self.data_source_options if not ds.name.startswith("Group")
        ]
        self.group_options = [
            ds for ds in self.data_source_options if ds.name.startswith("Group")
        ]

        self.data_source_selector.items = self.create_items()

        self.sample_size_inputs = [
            DataSourceSampleSizeInput(
                source,
                init_value=len(source.data),
            )
            for source in self.data_source_options
        ]

        self.source_update()
        self.refresh_layout()

    def refresh_layout(self):
        self.children = [
            self.data_source_selector,
            self.embedding_drop_down,
            v.Flex(children=self.sample_size_inputs),
            self.metric_selector,
            self.apply_button,
        ]


# TODO: cache graphs/SliderLandscapes generated from different data sources?


@contextmanager
def trivial_context(*args, **kwargs):
    yield


class EmbeddingVisualization(v.Flex):
    """Encapsulates an interactive Mapper graph and a data source selector for the UI.

    This class is not meant to be frequently reinstantiated in a layout.
    Instead, its configuration should be updated.
    """

    landscape: MultiResolutionLandscape

    def __init__(
        self,
        selection_manager,
        state: State,
        progress=None,
        max_nodes: int = 500,
        use_sliders: bool = False,
        width: str = "inherit",
        auto_graph=True,
        **kwargs,
    ):
        self.state = state
        self.graph_manager = GraphManager(visualization=self, state=self.state)
        self.selection_manager = selection_manager
        self.state = state
        if progress is None:
            progress = trivial_context
        self.progress = progress

        self.max_nodes = max_nodes

        self.use_sliders = use_sliders
        self.auto_graph = auto_graph
        self.kwargs = kwargs

        self.selected_data = None
        self.selection_listeners = []
        self.graph_update_listeners = []
        self.layout_update_listeners = []
        self.visualization_update_listeners = []

        self.data_source_options: List[DataSource] = self.create_data_source_options()

        initial_source_selection = self.get_initial_source_selection(state)

        self.data_source_widget = DataSourceSelector(
            self.data_source_options,
            initial_selection=initial_source_selection,
            on_apply_callback=self.close_visualization_dialog,
        )

        self.data_source_widget.on_apply(self.create_new_graph)

        self.width = width

        add_button = v.Btn(
            children=[v.Icon(children=["mdi-image-plus"], size=24)],
            color="primary",
            style_="min-width: 40px; min-height: 40px; padding: 0;",
            class_="d-flex justify-center align-center",
        )
        add_button.on_event("click", self.open_visualization_dialog)

        self.add_visualization_button = with_tooltip(
            add_button, tooltip_label="New Graph"
        )

        self.visualization_selector = VisualizationSelect()
        self.visualization_selector.on_event("change", self.select_graph)

        self.new_graph_name = TextField(
            label="Graph Name", v_model=GraphManager.generate_new_graph_name(self.state)
        )

        read_mode = not bool(self.state.dataset.embedding_metadata)
        if read_mode:
            empty_data_source_widget = DataSourceSelector(
                self.data_source_options,
                initial_selection=initial_source_selection,
                on_apply_callback=self.close_visualization_dialog,
            )
            self.visualization_dialog = ReadModeVisualizationDialog(
                v_model=False,
                on_close=self.close_visualization_dialog,
                children=[self.new_graph_name, empty_data_source_widget],
            )
        else:
            self.visualization_dialog = VisualizationDialog(
                v_model=False,
                on_close=self.close_visualization_dialog,
                children=[self.new_graph_name, self.data_source_widget],
            )

        self.empty_visualization_block = EmptyVisualizationBlock()

        self.spinner = ProgressBar(
            width=4, size=25, indeterminate=True, _style="margin: 0 auto;"
        )
        self.spinner.hide()

        self.set_and_display_default_graph()

        graph_event_bus = EventBusController.get_graph_event_bus(
            self.state.workspace_id
        )
        graph_event_bus.add_callback(self.update_graph_ui)

        group_event_bus = EventBusController.get_group_event_bus(
            self.state.workspace_id
        )
        group_event_bus.add_callback(self.update_data_source_options)

        self.form_controller = VisualizationFormController(
            data_source_widget=self.data_source_widget,
            graph_name_input=self.new_graph_name,
            source_update_callback=self.data_source_widget.source_update,
            graph_manager=self.graph_manager,
        )
        super().__init__(style_=f"width: {width}; flex-grow: 0")

    def get_initial_source_selection(self, state: State):
        data_source_indices = {
            s.name: i for i, s in enumerate(self.data_source_options)
        }
        if "baseline" in state.split and "comparison" in state.split:
            baseline_idx = data_source_indices["baseline"]
            comparison_idx = data_source_indices["comparison"]
            initial_source_selection = [baseline_idx, comparison_idx]
        elif state.split.test:
            test_idx = data_source_indices["test"]
            initial_source_selection = [test_idx]
        else:
            initial_source_selection = [0]
        return initial_source_selection

    def create_data_source_options(self):
        data_source_options = [
            SubsetDataSource(split, name) for name, split in self.state.split.items()
        ]

        saved_groups = self.state.get_groups()
        for group_name, group_data in saved_groups.items():
            data_source_options.append(
                SubsetDataSource(group_data, f"Group {group_name}")
            )

        return data_source_options

    @handle_cb_exceptions
    def update_data_source_options(self, *_):
        self.data_source_options = self.create_data_source_options()
        self.data_source_widget.update_options(self.data_source_options)
        self.form_controller.update_sample_size_observers()

    def open_visualization_dialog(self, widget, event, data):
        self.new_graph_name.v_model = GraphManager.generate_new_graph_name(self.state)
        self.new_graph_name.error_messages = []

        self.visualization_dialog.v_model = True

        # Reset the data source selection
        self.data_source_widget.data_source_selector.v_model = [0]
        self.data_source_widget.source_update()

        self.form_controller.is_graph_name_valid = True
        self.form_controller.update_apply_button_state()

    def close_visualization_dialog(self, widget=None, event=None, data=None):
        self.visualization_dialog.v_model = False
        self.reset_selection()

    @handle_cb_exceptions
    def update_selected_indices(self, *_):
        selected_data_indices = self.landscape.get_selected_data_indices(
            self.landscape.current_graph
        )

        self.selected_data = self.source_data.subset(selected_data_indices)

        for cb in self.selection_listeners:
            cb(self.selected_data)

    def reset_selection(self):
        self.selected_data = self.state.dataset.subset([])
        if hasattr(self, "landscape"):
            self.landscape.graph_viewer.clear_selection()

    def set_width(self, new_width: str):
        """Set a new width and update the visualization style."""
        self.width = new_width
        self.style_ = f"width: {self.width}; flex-grow: 0"

    def on_select(self, cb: Callable[[CobaltDataSubset], Any]):
        self.selection_listeners.append(cb)

    def get_points_from_subset(self, subset: CobaltDataSubset) -> List[int]:
        mask = subset.intersect(self.source_data).as_mask_on(self.source_data)
        points = np.flatnonzero(mask)
        return points.tolist()

    def get_nodes_from_subset(self, subset: CobaltDataSubset) -> List[int]:
        """Find the nodes in the current graph that intersect with the given subset."""
        # Could use source_data.subset(u) instead of reading the indices directly,
        # but when there are many nodes, constructing them takes too long.
        source_indices = self.source_data.indices
        node_set_indices = [
            source_indices[u] for u in self.landscape.current_graph.nodes
        ]
        return fast_intersect_subset_with_indices(subset, node_set_indices)

    def on_graph_update(self, cb: Callable):
        self.graph_update_listeners.append(cb)

    def graph_updated(self, *_):
        for cb in self.graph_update_listeners:
            cb()

    def on_layout_update(self, cb: Callable):
        self.layout_update_listeners.append(cb)

    def layout_updated(self, *_):
        for cb in self.layout_update_listeners:
            cb(self.landscape.graph_viewer.is_expanded_layout)
        self.toggle_layout_and_width()

    def on_visualization_update(self, cb: Callable):
        self.visualization_update_listeners.append(cb)

    def visualization_updated(self, selection):
        for cb in self.visualization_update_listeners:
            cb(selection)

    # this logic applies only to UIv1
    def toggle_layout_and_width(self, *_):
        if self.landscape.graph_viewer.is_expanded_layout:
            self.set_width("calc(100% - 175px)")
        else:
            self.set_width("inherit")

    def set_and_display_default_graph(self):
        """Sets the default graph to the first one added to the repo."""
        graphs = self.state.graph_repo.get_graphs()
        if len(graphs) > 0:
            first_graph = next(iter(graphs.keys()))
            self.graph_manager.active_graph = first_graph
            self.update_selected_graph()
        else:
            # just in case there are no graphs already created
            # the main situation this happens is in unit tests
            # could happen in the full app as well, though
            if self.auto_graph and self.state.dataset.embedding_metadata:
                self.create_new_graph()

    @handle_cb_exceptions
    def create_new_graph(self, *_):
        self.source_data = None  # graph_manager updates it
        self.spinner.show()

        graph_name, graph_dict = self.graph_manager.create_graph()
        # Landscape should be set before state runs callbacks
        self.landscape = graph_dict["graph"]
        self.state.update_graph(
            graph_name=graph_name,
            graph=graph_dict,
        )
        # update_graph() is called instead of add_graph() so that the listeners are not triggered
        # the only listener (FOR NOW) is self.update_graph_ui()
        # the call to self._update_landscape() does everything this would have done (and more)
        # and apparently it causes weird issues to run the callback earlier

        # TODO: make this play nicely with the event system

        self.spinner.hide()
        self._update_landscape()

    def update_selected_graph(self):
        if self.graph_manager.active_graph:
            self.landscape = self.graph_manager.get_current_graph()
        else:
            raise ValueError("There is no graph to show")
        self._update_landscape()

    def _update_landscape(self):
        self.multires_graph = self.landscape.graph
        self.landscape.graph_viewer.on_select_points(self.update_selected_indices)

        self.landscape.graph_viewer.observe(self.graph_updated, "graph")
        self.landscape.graph_viewer.observe(self.layout_updated, "is_expanded_layout")

        self.update_graph_selector_options()
        self.refresh_layout()
        self.graph_updated()
        self.visualization_updated(self.source_data)

    def update_graph_selector_options(self):
        graph_details = {}
        for graph_name, graph_entry in self.state.graph_repo.get_graphs().items():
            n_points = len(graph_entry["subset"])
            metric = graph_entry["graph"].graph.neighbor_graph.metric

            graph_details[graph_name] = {"n_points": n_points, "metric": metric}

        graphs_options_data = []

        for graph_name, details in graph_details.items():
            n_points = str(details.get("n_points", "Unknown"))
            metric = str(details.get("metric", "Unknown"))
            subtitle = f"{n_points} points, metric: {metric}"

            graphs_options_data.append({"title": graph_name, "subtitle": subtitle})

        initial_idx = 0
        for i, option in enumerate(graphs_options_data):
            if option.get("title") == self.graph_manager.active_graph:
                initial_idx = i
                break

        initial_option = graphs_options_data[initial_idx]
        v_model = initial_option.get("title", None)

        self.visualization_selector.items = graphs_options_data
        self.visualization_selector.v_model = v_model

    @handle_cb_exceptions
    def select_graph(self, *_):
        self.reset_selection()
        # Free up all the WEBGL context
        self.landscape.graph_viewer.dispose_graph()
        self.graph_manager.active_graph = self.visualization_selector.v_model
        self.update_selected_graph()

    def refresh_layout(self):
        top_layout = v.Layout(
            align_center=True,
            children=[
                self.spinner,
                self.add_visualization_button,
                self.visualization_selector,
                self.visualization_dialog,
            ],
            class_="mb-2",
        )

        no_graph = (
            not bool(self.state.dataset.embedding_metadata) or not self.auto_graph
        )

        if no_graph and not self.state.get_graphs():
            visualization_container_children = [
                v.Layout(
                    children=[self.empty_visualization_block],
                    style_="width: 640px;height: 425px;",
                )
            ]
        else:
            visualization_container_children = [
                self.landscape,
            ]

        visualization_container = v.Layout(
            column=True,
            children=visualization_container_children,
        )

        visualization_layout = v.Flex(
            children=[
                top_layout,
                visualization_container,
            ],
            class_="flex-grow-0",
            style_="min-width: 500px; min-height: 425px",
        )

        self.children = [visualization_layout]

    def update_graph_ui(self):
        self.update_graph_selector_options()
        self.refresh_layout()
        self.graph_updated()


class GraphManager:
    active_graph: str = None
    default_graph_name: str = "Default graph"
    default_graph_data_seed = 47207

    def __init__(self, visualization: EmbeddingVisualization, state: State):
        self.visualization = visualization
        self.state = state
        self.graph_builder = GraphBuilder(self.state)

    def get_current_graph(self):
        graph = self.state.get_graph(self.active_graph)
        self.visualization.source_data = graph["subset"]
        current_landscape = graph["graph"]
        return current_landscape

    def graph_name_exists(self, name: str) -> bool:
        return name in self.state.graph_repo.get_graphs()

    @handle_cb_exceptions
    def prepare_new_graph(self):
        subset = self.visualization.data_source_widget.get_data()
        data_sources = self.visualization.data_source_widget.get_selected_data_sources()

        self.visualization.source_data = subset

        self.visualization.data_source_widget.set_fresh_data_sources(data_sources)
        embedding = self.visualization.source_data.get_embedding_metadata_by_name(
            self.visualization.data_source_widget.active_embedding
        )
        return embedding, subset

    @handle_cb_exceptions
    def create_graph(self):
        embedding, subset = self.prepare_new_graph()

        metric = self.visualization.data_source_widget.get_metric()

        g = self.graph_builder.new_graph(
            subset=subset, embedding=embedding, metric=metric
        )
        subset_ = self.graph_builder.get_subset(subset)
        new_landscape = create_multilandscape(g, subset_)

        graph_name = (
            self.default_graph_name
            if not self.active_graph
            else self.visualization.new_graph_name.v_model
        )
        self.active_graph = graph_name

        return graph_name, new_landscape

    @staticmethod
    def generate_new_graph_name(state: State, prefix="New Graph"):
        """Generates a new graph name based on existing graph names and a prefix.

        The function first finds all graphs starting with the prefix,
        then identifies those ending with a number.
        It generates a new name using the next highest number in the sequence,
        or uses 1 if none of the graphs have a numerical ending.

        :param prefix: The prefix for new graph names
        :return: A new graph name with the appropriate number appended
        """
        existing_graphs = list(state.graph_repo.get_graphs().keys())
        highest_number = 0
        generate_name_with_number = False
        for name in existing_graphs:
            if name.startswith(prefix):
                generate_name_with_number = True
                potential_number = name[len(prefix) :].strip()
                if potential_number.isdigit():
                    highest_number = max(highest_number, int(potential_number))

        return f"{prefix} {highest_number + 1}" if generate_name_with_number else prefix


class VisualizationFormController:
    def __init__(
        self,
        data_source_widget,
        graph_name_input,
        source_update_callback,
        graph_manager,
    ):
        self.data_source_widget = data_source_widget
        self.graph_name_input = graph_name_input
        self.is_graph_name_valid = True
        self.are_data_sources_valid = True
        self.are_sample_sizes_valid = True
        self.source_update_callback = source_update_callback
        self.graph_manager = graph_manager

        self.graph_name_input.on_event("input", self.validate_graph_name)
        self.data_source_widget.data_source_selector.on_event(
            "change", self.validate_data_sources
        )

        for input_ in self.data_source_widget.sample_size_inputs:
            input_.observe(self.validate_sample_sizes, names=["valid"])

    def validate_graph_name(self, *_):
        graph_name = self.graph_name_input.v_model.strip()
        self.is_graph_name_valid = bool(
            graph_name
        ) and not self.graph_manager.graph_name_exists(graph_name)
        self.graph_name_input.error_messages = (
            [
                (
                    "Graph name cannot be empty."
                    if not graph_name
                    else "A graph with this name already exists."
                )
            ]
            if not self.is_graph_name_valid
            else []
        )
        self.update_apply_button_state()

    def validate_data_sources(self, *_):
        self.are_data_sources_valid = bool(
            self.data_source_widget.get_selected_data_sources()
        )
        self.data_source_widget.data_source_selector.hide_details = (
            self.are_data_sources_valid
        )
        self.data_source_widget.data_source_selector.error_messages = (
            ["At least one data source should be selected."]
            if not self.are_data_sources_valid
            else []
        )
        self.source_update_callback()
        self.validate_sample_sizes()
        self.update_apply_button_state()

    def validate_sample_sizes(self, *_):
        self.are_sample_sizes_valid = all(
            input_.valid
            for input_ in self.data_source_widget.sample_size_inputs
            if input_.is_visible
        )
        self.update_apply_button_state()

    def update_apply_button_state(self):
        all_conditions_met = (
            self.is_graph_name_valid
            and self.are_data_sources_valid
            and self.are_sample_sizes_valid
        )
        self.data_source_widget.apply_button.disabled = not all_conditions_met

    def update_sample_size_observers(self):
        for input_ in self.data_source_widget.sample_size_inputs:
            input_.unobserve_all()

        for input_ in self.data_source_widget.sample_size_inputs:
            input_.observe(self.validate_sample_sizes, names=["valid"])
