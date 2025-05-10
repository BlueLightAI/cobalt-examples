# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v

from cobalt.config import handle_cb_exceptions
from cobalt.schema import CobaltDataSubset
from cobalt.selection_manager import SelectionState
from cobalt.state import State


class SelectionDetails(v.Html):
    def __init__(self, selection_manager, state, on_source_change=None):
        super().__init__()
        self.state: State = state
        self.selection_manager = selection_manager
        self.bind_selection_manager()
        self.on_source_change = on_source_change
        self.initialize_ui()

    def initialize_ui(self):
        self.tag = "div"
        self.selection_source_label = v.Html(
            tag="span",
            class_="mr-1 d-flex align-center",
            children=["No selection"],
        )
        self.selection_source_value = v.Html(
            tag="span",
            children=[""],
            style_="max-width: 500px; text-overflow: ellipsis; font-weight: bold;",
            class_="overflow-hidden",
        )

        self.number_of_points = v.Text(children=["0"])

        # widget children is triggered immediately after initiation.
        # Extending it in __init__ method do not show extra elements (Metrics)
        body = [
            v.Html(
                tag="div",
                class_="d-flex",
                style_="font-size: 14px",
                children=[
                    self.selection_source_label,
                    self.selection_source_value,
                ],
            ),
        ]

        self.children = body

    def bind_selection_manager(self):
        self.selection_manager.on_graph_select(self.update_details_from_graph)
        self.selection_manager.on_group_select(self.update_details_from_group)

    @handle_cb_exceptions
    def update_details_from_graph(self, selection: CobaltDataSubset):
        if self.selection_manager.selection_state == SelectionState.initial:
            self.update_selection_source("")
            self.update_selected_number_of_points(0)
        else:
            self.update_selection_source("Graph")
            self.update_selected_number_of_points(len(selection))

    @handle_cb_exceptions
    def update_details_from_group(self, selected_item):
        if selected_item:
            group_id = selected_item.get("group_id")
            subset = selected_item.get("subset")

            self.update_selection_source(group_id)

            number_of_points = len(subset) if subset else 0
            self.update_selected_number_of_points(number_of_points)

    def update_selection_source(self, source):
        self.selection_source_value.children = [source]
        if self.on_source_change:
            self.on_source_change(source)

    def update_selected_number_of_points(self, points: int):
        if points > 0:
            self.selection_source_label.children = [f"{points} points selected from"]
        else:
            self.selection_source_label.children = ["No selection"]
        self.number_of_points.children = [str(points)]
