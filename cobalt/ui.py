# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional, Tuple

import ipyvuetify as v
import ipywidgets as w
import networkx as nx
from IPython.core.display import HTML
from IPython.display import display
from mapper import HierarchicalPartitionGraph, check_license_expiration_soon

from cobalt import config, schema, selection_manager
from cobalt.coloring.color_ui import ColoringConfig
from cobalt.config import handle_cb_exceptions
from cobalt.groups.groups_comparison import GroupsComparison
from cobalt.groups.user_groups import UserGroups
from cobalt.model_overview.model_overview_ui import ModelOverview
from cobalt.schema.dataset import CobaltDataset, DatasetBase
from cobalt.selection_details import SelectionDetails
from cobalt.styles import DEFAULT_GROUP_CONTAINER_STYLES, LIGHT_GREY_BORDER
from cobalt.table import table_select
from cobalt.table.table_constants import NUMBER_OF_DF_ROWS_TO_DISPLAY
from cobalt.visualization import EmbeddingVisualization
from cobalt.visualization_dialog import LicenseExpireSoonDialog
from cobalt.widgets import ExpansionPanel, ProgressBar, Tabs
from cobalt.workspace_overview.workspace_overview_ui import WorkspaceOverview

if TYPE_CHECKING:
    from cobalt.workspace import Workspace


# ignore jupyter UserWarning about non-JSON serializable out of range float values
warnings.filterwarnings("ignore", category=UserWarning, module="jupyter_client.session")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Tight layout not applied"
)


class UI:
    """An interactive UI visualizing the data in a Workspace.

    Args:
        workspace: the Workspace object that this UI will visualize
        dataset: the CobaltDataset being analyzed
        table_image_size: for datasets with images, the (height, width) size
            in pixels that these will be shown in the data table.
    """

    def __init__(
        self,
        workspace: Workspace,
        dataset: CobaltDataset,
        table_image_size: Tuple[int, int] = (80, 80),
    ):
        super().__init__()
        self.ws = workspace
        self.dataselector = selection_manager.SelectionManager(dataset)
        self.error_msg = w.HTML(value="")
        self.expansion_panel = ExpansionPanel("Table View", w.HTML(value=""))
        self.table_image_size = table_image_size

        self.selection_details = SelectionDetails(
            self.dataselector, state=self.ws.state
        )

        self._override_styles()
        if config.is_colab_environment():
            config.setup_colab_widget_environment()

        self._overview_ui = None
        self._discover_ui = None
        self.visualization = None

        self._previous_selection = None
        self._prepare_visualization_finished = False
        self._prepare_data_table_finished = False

    @property
    def overview(self):
        if self._overview_ui is None:
            self._overview_ui = self._init_overview_ui()

        return self._overview_ui

    @property
    def discover(self):
        if self._discover_ui is None:
            self._ensure_visualization_is_ready()
            self._prepare_data_table()

            self._discover_ui = self._init_discover_ui()

        return v.Layout(
            wrap=True,
            class_="mt-2",
            style_="gap: 8px",
            children=[self._discover_ui, self.expansion_panel],
        )

    def _ensure_visualization_is_ready(self):
        if self.visualization is not None:
            return

        self._prepare_visualization()

    def enable_persistent_labels(self):
        """Enable labels that persist (don't require hover over).

        Requires `ui.build()` to already have been run.
        Expected Behavior: as you zoom in and zoom out on the visual,
        labels will appear and disappear in a Google Maps like fashion.

        :meta private:
        """
        if not self.visualization or not self.visualization.landscape:
            return
        self.visualization.landscape.enable_persistent_labels()

    def disable_persistent_labels(self):
        """Disable labels that persist (don't require hover over).

        Requires `ui.build()` to already have been run.

        :meta private:
        """
        if not self.visualization or not self.visualization.landscape:
            return
        self.visualization.landscape.disable_persistent_labels()

    def refresh_coloring_and_table(self):
        empty_selection = self.dataselector.dataset.subset([])
        self._handle_table_update(empty_selection)
        new_coloring = self._generate_color_config(self.visualization)
        children_list = list(self._discover_ui.children)
        children_list[1] = new_coloring
        self._discover_ui.children = children_list

    def _show_expiration_soon_popup(self, text):
        self.license_expire_soon_popup.card_text.children = [text]
        self.license_expire_soon_popup.v_model = True

    @handle_cb_exceptions
    def _handle_table_update(self, selection):
        self.table = self._create_table(selection)
        self.expansion_panel.panel_content = self.table
        self.ws.state.collect_metrics(field="graph", dataset=selection)
        time_left_msg = check_license_expiration_soon()
        if time_left_msg:
            self._show_expiration_soon_popup(time_left_msg)

    @handle_cb_exceptions
    def _handle_graph_selection(self, _):
        current_selection = self.dataselector.graph_selection

        if current_selection == self._previous_selection:
            return

        self._handle_table_update(current_selection)
        self._previous_selection = current_selection

    @handle_cb_exceptions
    def _handle_group_selection(self, group):
        if group:
            subset = group.get("subset")

            self._handle_table_update(subset)
            if bool(self.ws.state.dataset.embedding_metadata):
                self.dataselector.update_graph_visualization(subset)

            if subset:
                self.groups_container.class_ = "d-none"
                self.groups_comparison_container.class_ = "d-block"
            else:
                self.groups_container.class_ = "d-block"
                self.groups_comparison_container.class_ = "d-none"
        else:
            self._handle_table_update(self.ws.state.dataset)

    @handle_cb_exceptions
    def _handle_clear_selection(self):
        self.groups_container.class_ = "d-block"
        self.groups_comparison_container.class_ = "d-none"

    @contextmanager
    def _show_loading(
        self, msg: Optional[str] = None, show_progress: bool = False, ui_box=None
    ):
        if ui_box is None:
            ui_box = self.ui_box

        old_msg = ui_box.status_text.value
        if msg is None:
            msg = old_msg
        ui_box.status_text.value = msg

        current_spinner_active = ui_box.spinner_active
        if show_progress:
            ui_box.start_progress(current_spinner_active)

            def _update_progress(completed: int, total: int, cb_msg: str = ""):
                ui_box.progress.value = completed / total * 100
                ui_box.status_text.value = f"{msg} {cb_msg} {completed}/{total}"

            try:
                yield _update_progress
            except Exception as e:
                ui_box.stop_loader()
                raise e
            ui_box.finish_progress(current_spinner_active)

        else:
            ui_box.start_without_progress()
            try:
                yield
            except Exception as e:
                ui_box.stop_loader()
                raise e
            ui_box.finish_without_progress(current_spinner_active)

        ui_box.status_text.value = old_msg

    def _generate_color_config(self, vis):
        metadata = self.ws.state.dataset.metadata
        metadata_columns = (
            metadata.media_columns
            + metadata.timestamp_columns
            + metadata.other_metadata_columns
        )

        for model_metadata in self.ws.state.dataset.models:
            metadata_columns += (
                model_metadata.error_columns
                + model_metadata.outcome_columns
                + model_metadata.prediction_columns
            )
        shown_data_columns = [
            c
            for c in self.ws.state.dataset.df.columns
            if c not in self.ws.state.dataset.metadata.hidable_columns
            and c not in metadata_columns
        ]

        color_config = ColoringConfig(
            visualization=vis,
            extra_data_columns=shown_data_columns,
            enable_drift_score=self.ws.state.split.has_multiple_subsets,
        )

        return color_config

    def _create_table(self, data: DatasetBase):
        if data.df.empty:
            data = self.ws.state.dataset.as_subset()
        columns = None
        if hasattr(self, "table") and self.table.columns is not None:
            columns = self.table.columns
        elif data.metadata.default_columns is not None:
            columns = data.metadata.default_columns

        columns_to_filter = None
        if hasattr(self, "table") and self.table.columns_to_filter is not None:
            columns_to_filter = self.table.columns_to_filter

        filter_criteria = None
        if hasattr(self, "table") and self.table.filter_criteria is not None:
            filter_criteria = self.table.filter_criteria
        if hasattr(self, "table"):
            self.table.close()
            # NOTE: widget.close() clears out rendered data but doesn't destroy
            # references to internal objects.
            # We should do it manually to let gc doing it's work
            self.table.df = None
            del self.table.df

        # WARNING: This is a problem and tells us we need to redesign things.
        # Workspace, and UI have a circular reference!
        # TODO: the TableSelector can get this metadata itself from the data(sub)set object
        media_columns = data.metadata.media_columns
        image_columns = [
            img_col.autoname_media_visualization_column() for img_col in media_columns
        ]
        html_columns = data.metadata.long_text_columns

        # Hard coding 2000 max rows to display.
        # TODO: This is often not ideal behavior.
        table_selector = table_select.TableSelector(
            data,
            self.ws.state,
            workspace_id=self.ws.workspace_id,
            columns=columns,
            image_columns=image_columns,
            html_columns=html_columns,
            columns_to_filter=columns_to_filter,
            filter_criteria=filter_criteria,
            image_size=self.table_image_size,
            max_rows_to_display=NUMBER_OF_DF_ROWS_TO_DISPLAY,
            run_server=self.ws.run_server,
        )

        return table_selector

    def get_graph_and_clusters(self) -> Tuple[nx.Graph, List[schema.CobaltDataSubset]]:
        """Return the current graph and the datapoints that belong to each node.

        Returns:
            A tuple(Graph, List[CobaltDataSubset]) representing the current graph as networkx,
            and a list of the datapoints that each node represents.

            Note that the graph has the same number of nodes
            as the number of elements in the list.
        """
        mapper_graph = self.get_current_graph()
        source = self.get_current_graph_source_data()
        clusters = mapper_graph.nodes
        cobalt_datasubsets = [source.subset(list(cluster)) for cluster in clusters]
        g = nx.Graph()
        g.add_nodes_from(range(len(mapper_graph.nodes)))
        g.add_edges_from(mapper_graph.edge_list)
        return g, cobalt_datasubsets

    def get_current_graph(self) -> HierarchicalPartitionGraph:
        """Return the currently shown graph."""
        return self.visualization.landscape.graph

    def get_graph_selection(self) -> schema.CobaltDataSubset:
        """Return the current subset selected in the graph."""
        return self.dataselector.graph_selection

    def get_current_graph_source_data(self) -> schema.CobaltDataSubset:
        """Return the current dataset being displayed in the current graph.

        Returns:
            A CobaltDataSubset of the data represented by the graph.
            Note that if sub-sampling is enabled, this may not be the entire dataset.
        """
        return self.visualization.source_data

    def build(self):
        """Construct the UI.

        This normally happens automatically when the UI object appears as an
        output in a notebook cell.
        """
        self.ui_box = UIBox(error_msg_html=self.error_msg)
        self._finish_build_sync()

        return self.ui_box

    def _finish_build_sync(self):
        # TODO: some kind of progress indicator?
        self._prepare_visualization()

        self._prepare_data_table()

        self._update_ui()

        # register callbacks to update the UI when underlying data changes
        self.ws.state.run_event_bus.add_callback(self.refresh_coloring_and_table)
        self.ws.state.dataset_event_bus.add_callback(self.refresh_coloring_and_table)

    def _prepare_visualization(self):
        if self._prepare_visualization_finished:
            return

        failure_groups_data = self.ws._get_or_create_displayed_groups()
        num_of_failure_groups = (
            len(failure_groups_data) if failure_groups_data is not None else 0
        )

        # Visualization must be initialized after failure groups so that the
        # default graph will show them if applicable
        self.visualization = EmbeddingVisualization(
            self.dataselector,
            self.ws.state,
            max_nodes=300,
            use_sliders=True,
            auto_graph=self.ws.auto_graph,
        )
        self.visualization.on_layout_update(self._handle_layout_update)

        self.user_groups = UserGroups(
            self.dataselector, self.visualization, self.ws, num_of_failure_groups
        )

        self.groups_comparison = GroupsComparison(
            self.dataselector, self.visualization, self.ws
        )

        self.groups_container = v.Flex(
            children=[self.user_groups],
            style_=DEFAULT_GROUP_CONTAINER_STYLES,
        )

        self.groups_comparison_container = v.Flex(
            children=[self.groups_comparison], style_=DEFAULT_GROUP_CONTAINER_STYLES
        )
        self.groups_comparison_container.class_ = "d-none"

        self.license_expire_soon_popup = LicenseExpireSoonDialog(v_model=False)

        self.visualization.on_select(self.dataselector.set_graph_selection)
        self.visualization.on_visualization_update(self._handle_table_update)

        self.dataselector.set_visualization(self.visualization)
        self.dataselector.on_graph_select(self._handle_graph_selection)
        self.dataselector.on_group_select(self._handle_group_selection)
        self.dataselector.on_clear_selection(self._handle_clear_selection)

        self.tabs = self._init_tabs_ui_component()

        # Hook to check if visualization finished properly
        self._prepare_visualization_finished = True
        if not bool(self.ws.state.dataset.embedding_metadata) or not self.ws.auto_graph:
            self.visualization.refresh_layout()

    def _handle_layout_update(self, is_expanded_layout: bool):
        if is_expanded_layout:
            self._discover_ui.class_ = "wrap"
        else:
            self._discover_ui.class_ = ""

    def _init_overview_ui(self):
        if self._overview_ui is None:
            self._overview_ui = v.Layout(
                column=True,
                children=[
                    WorkspaceOverview(
                        ws=self.ws,
                        column=True,
                        style_=LIGHT_GREY_BORDER,
                        class_="my-10",
                    ),
                    ModelOverview(
                        split=self.ws.state.split,
                        column=True,
                        style_=LIGHT_GREY_BORDER,
                    ),
                ],
            )

        return self._overview_ui

    def _init_discover_ui(self):
        if self._discover_ui is None:
            coloring = self._generate_color_config(self.visualization)
            self._discover_ui = v.Layout(
                children=[
                    self.visualization,
                    coloring,
                    self.groups_container,
                    self.groups_comparison_container,
                    self.license_expire_soon_popup,
                ],
                style_="gap: 8px; overflow-x: auto; padding-top: 8px",
            )

        return self._discover_ui

    def _init_tabs_ui_component(self) -> Tabs:
        tab_data = [
            {"tab_header": "Overview", "tab_content": self._init_overview_ui()},
            {"tab_header": "Discover", "tab_content": self._init_discover_ui()},
        ]
        if config.USE_EXPERIMENTAL:
            tab_data.append(
                {
                    "tab_header": "Experimental",
                    "tab_content": "Example of experimental widget/feature",
                }
            )
        return Tabs(tab_data, initial_tab=1)

    def _prepare_data_table(self):
        if self._prepare_data_table_finished:
            return

        self.table = self._create_table(self.ws.state.dataset)

        self.selection_details.update_selected_number_of_points(0)

        header_content_layout = v.Layout(
            children=[
                v.Flex(
                    tag="div",
                    children=[
                        v.Html(
                            tag="div",
                            class_="mb-2 font-weight-bold",
                            children=["Table View"],
                        ),
                        self.selection_details,
                    ],
                ),
            ],
        )

        self.expansion_panel = ExpansionPanel(header_content_layout, self.table)

        self._prepare_data_table_finished = True

    def _update_ui(self):
        self.ui_box.children = [
            v.Layout(
                wrap=True,
                class_="mb-2",
                style_="gap: 10px",
                children=[self.tabs, self.expansion_panel],
            )
        ]

    def _ipython_display_(self):
        # NOTE: override IPython.display for a possibility to render UI as Workspace().ui_v1
        if not hasattr(self, "ui_box"):
            self.build()
        if config.is_colab_environment():
            # workaround for https://github.com/googlecolab/colabtools/issues/3501
            # TODO: may want to add this to other components
            from ipyvue import Html

            display(Html(tag="div", style_="display: none"), self.ui_box)
        else:
            display(self.ui_box)

    # To adjust the width for better alignment of UI elements
    def _override_styles(self):
        css = """
        <style>
            #notebook-container {
                width: 85% !important;
            }

            body[data-notebook='notebooks'] .jp-WindowedPanel-outer {
                padding-left: 10% !important;
                padding-right: 10% !important;
            }

            #app {
                position: relative;
            }

            .v-dialog__content {
                position: absolute !important;
            }

            :root {
                --v-textBackground: #E3E4E7;
            }

            .theme--dark {
                --v-textBackground: #696969;
            }
        </style>
        """
        display(HTML(css))


class UIBox(w.VBox):
    """Main UI block that holds other widgets. Can show a loading spinner and progress bar."""

    def __init__(self, error_msg_html=None):
        self.status_text = w.HTML(value="")
        self.progress = ProgressBar(width=8, value=0, rotate=-90)
        self.spinner = ProgressBar(width=4, size=25, indeterminate=True)
        self.progress_active = False
        self.spinner_active = False
        self.progress.hide()
        self.spinner.hide()

        if error_msg_html is None:
            error_msg_html = w.HTML(value="")

        self.button_row = v.Layout(
            children=[
                self.status_text,
                self.progress,
                self.spinner,
                error_msg_html,
            ],
            style_="min-height: 35px;",
        )
        super().__init__(children=[self.button_row])

    def start_progress(self, current_spinner_active):
        if current_spinner_active:
            self.spinner.hide()
            self.spinner_active = False
        self.progress.show()
        self.progress_active = True

    def stop_loader(self):
        self.progress.hide()
        self.spinner.hide()
        self.progress_active = False
        self.spinner_active = False
        self.status_text.value = ""

    def finish_progress(self, current_spinner_active):
        self.progress.hide()
        self.progress_active = False
        if current_spinner_active:
            self.spinner.show()
            self.spinner_active = True
        self.progress.value = 0

    def start_without_progress(self):
        self.spinner.show()
        self.spinner_active = True

    def finish_without_progress(self, current_spinner_active):
        if not current_spinner_active:
            self.spinner.hide()
        self.spinner_active = current_spinner_active
