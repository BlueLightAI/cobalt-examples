# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Optional

import ipyvuetify as v

from cobalt.config import handle_cb_exceptions
from cobalt.event_bus import EventBusController
from cobalt.groups.failure_groups import FailureGroups
from cobalt.groups.groups_ui import GroupDisplay, GroupSaveButton
from cobalt.selection_details import SelectionDetails
from cobalt.selection_manager import SelectionManager
from cobalt.ui_utils import with_tooltip
from cobalt.visualization import EmbeddingVisualization


class UserGroups(v.Flex):
    def __init__(
        self,
        dataselector: SelectionManager,
        visualization: EmbeddingVisualization,
        workspace,
        num_of_failure_groups: Optional[int] = 0,
    ):
        self.workspace = workspace
        self.state = workspace.state
        self.dataselector = dataselector
        self.visualization = visualization
        self.num_of_failure_groups = num_of_failure_groups

        self.selection_details = SelectionDetails(self.dataselector, self.state)
        self.number_of_points = self.selection_details.number_of_points
        self.selection_details.update_details_from_graph(
            self.dataselector.graph_selection
        )

        self.bind_selection_details()

        self.number_of_groups = v.Text(children="")

        self.group_display = GroupDisplay(dataselector, visualization, self.state)
        group_event_bus = EventBusController.get_group_event_bus(
            self.state.workspace_id
        )

        group_event_bus.add_callback(self.update_empty_state)
        group_event_bus.add_callback(self.visualization.update_data_source_options)
        self.group_display.set_update_callback(self.update_empty_state)
        self.group_display.set_update_callback(
            self.visualization.update_data_source_options
        )

        self.save_button = GroupSaveButton(
            self.dataselector, self.state, group_display=self.group_display
        )

        self.save_button_with_tooltip = with_tooltip(self.save_button, "Create Group")
        self.save_button.on_event("click", self.save_button.update_data_points)

        self.groups_header = v.Flex(
            children=[
                v.Text(
                    children="Data groups",
                    class_="font-weight-bold",
                    style_="font-size: 16px",
                ),
                self.save_button_with_tooltip,
            ],
            class_="d-flex justify-space-between align-center pa-4",
            style_="width: 100%;",
        )

        self.failure_groups = FailureGroups(
            selection_manager=self.dataselector,
            workspace=self.workspace,
        )

        self.user_groups_title = v.Text(
            children=[
                self.number_of_groups,
                v.Text(
                    children="Saved Groups",
                ),
            ],
            class_="text-uppercase px-4",
            style_="font-size: 14px;",
        )

        self.list_box = v.List(
            dense=True, children=[self.group_display.list_item_group]
        )

        self.empty_state_groups_container = v.Flex(
            children=[
                v.Text(
                    children="No saved groups",
                    style_="font-size: 20px;",
                ),
                v.Text(
                    children="To create a new group, make a selection in the graph.",
                    class_="pt-4 text-wrap",
                ),
            ],
            class_="d-flex align-center flex-column",
        )

        self.update_empty_state()

        super().__init__(
            children=[
                self.groups_header,
                v.Divider(),
                self.create_groups_row([self.empty_state_groups_container]),
            ],
        )

    def on_selection_source_change(self, source):
        if source != "Graph":
            self.save_button.disabled = True
        else:
            self.save_button.disabled = False

    def bind_selection_details(self):
        self.selection_details = SelectionDetails(
            self.dataselector, self.state, self.on_selection_source_change
        )

    def create_groups_row(self, user_groups_content):
        return v.Html(
            tag="div",
            children=[
                v.Html(
                    tag="div",
                    children=[self.failure_groups],
                    class_="text-truncate",
                    style_="min-width: 250px; flex: 1;",
                ),
                v.Html(
                    tag="div",
                    children=user_groups_content,
                    class_="text-truncate",
                    style_="min-width: 250px; flex: 1;",
                ),
            ],
            class_="d-flex flex-wrap-reverse justify-space-between pt-2",
            style_="gap: 8px",
        )

    @handle_cb_exceptions
    def update_empty_state(self):
        groups = self.state.get_groups()
        num_saved_groups = len(groups)
        self.number_of_groups.children = str(num_saved_groups)

        groups_title = "Saved Group" if num_saved_groups == 1 else "Saved Groups"
        self.user_groups_title.children = [
            self.number_of_groups,
            v.Text(children=f" {groups_title}"),
        ]

        user_groups_content = (
            [self.user_groups_title, self.list_box]
            if num_saved_groups > 0
            else [self.empty_state_groups_container]
        )
        self.children = [
            self.groups_header,
            v.Divider(),
            self.create_groups_row(user_groups_content),
        ]
