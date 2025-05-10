# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from contextlib import contextmanager
from typing import List, Optional, Tuple

import ipyvuetify as v
import ipywidgets as w

from cobalt import CobaltDataSubset
from cobalt.cobalt_types import GroupType
from cobalt.config import handle_cb_exceptions
from cobalt.event_bus import EventBusController
from cobalt.selection_manager import SelectionManager
from cobalt.state import State
from cobalt.ui_utils import with_tooltip
from cobalt.visualization import EmbeddingVisualization
from cobalt.widgets import Button, Menu, MenuButton, TextField


def validate_group_name(state: State, group_name: str) -> Tuple[bool, List[str]]:
    is_valid = True
    error_list = []
    if not group_name.strip():
        is_valid = False
        error_list = ["Group name cannot be empty."]
    elif len(group_name) > 30:
        is_valid = False
        error_list = ["Group name cannot exceed 30 characters."]
    elif not state.is_group_name_unique(group_name):
        is_valid = False
        error_list = ["A group with this name already exists."]
    elif state.has_name_among_failure_groups(group_name):
        is_valid = False
        error_list = ["Group name is already taken by a failure group."]

    return is_valid, error_list


class GroupDisplay(w.Box):
    def __init__(
        self,
        dataselector: SelectionManager,
        visualization: EmbeddingVisualization,
        state: State,
    ):
        self.state: State = state
        self.dataselector = dataselector
        self.visualization = visualization
        group_event_bus = EventBusController.get_group_event_bus(
            self.state.workspace_id
        )
        group_event_bus.add_callback(self.update_displayed_groups)

        # This flag allows us to distinguish between a click on the edit button
        # for a group and a click on the group itself. It is set when the edit
        # button callbacks are run, and cleared on any subsequent interaction.
        # This is necessary because we cannot prevent click events from
        # filtering up the chain, so the click handler for the group label will
        # be called when the edit button is clicked. This approach works as long
        # as we can assume that the edit button click handler is always called
        # first, which should be the case.
        self.is_clicked_to_edit = False

        self.list_item_group = v.ListItemGroup(v_model=[], multiple=False, children=[])

        self.list_box = v.List(
            dense=True,
            children=[self.list_item_group],
            class_="overflow-y-auto rounded-md",
            style_="min-height: 300px;"
            "max-height: 300px;"
            "min-width: 300px;"
            "border: 1px solid grey;",
        )
        self._should_update_selected_group = True
        self.list_item_group.observe(self.update_selected_group, "v_model")
        self.list_item_group.observe(self.update_delete_button_state, "v_model")

        self.save_button = GroupSaveButton(
            self.dataselector, self.state, group_display=self
        )
        self.delete_button = v.Flex(
            children=[
                with_tooltip(
                    Button(
                        icon=True,
                        children=[v.Icon(children=["mdi mdi-delete"])],
                    ),
                    "Delete Group",
                ),
            ],
            class_="d-flex",
        )
        self.delete_button.on_event("click", self.delete_selected_groups)

        self.layout_with_list_box_and_delete = v.Layout(
            row=True,
            align_center=True,
            class_="mx-1",
            children=[self.list_box],
        )

        self.current_selected_group = None

        self.button_layout = v.Layout(
            column=True,
            children=[self.delete_button, self.save_button],
        )
        self.update_displayed_groups()
        super().__init__(
            children=[
                v.Layout(
                    children=[self.layout_with_list_box_and_delete, self.button_layout],
                )
            ],
            layout={"margin": "0 20px"},
        )

    @contextmanager
    def suppress_selection_update(self):
        try:
            self._should_update_selected_group = False
            yield
        finally:
            self._should_update_selected_group = True

    def update_selected_group(self, change):
        """Triggered when group is selected."""
        if not self._should_update_selected_group:
            return
        selected_group = change["new"]

        groups = self.state.get_groups()

        if selected_group is None or selected_group not in groups:
            self.dataselector.set_group_selection(None)
            return

        self.list_item_group.v_model = selected_group

        subset = groups[selected_group]

        user_selected_group = {"group_id": selected_group, "subset": subset}

        self.current_selected_group = selected_group

        if self.is_clicked_to_edit is False:
            self.dataselector.set_comparison_group(
                group_type=GroupType.user, group_id=selected_group
            )
            self.dataselector.set_group_selection(user_selected_group)
            with self.suppress_selection_update():
                self.list_item_group.v_model = None

    def set_update_callback(self, callback):
        self.update_callback = callback

    def _get_group_by_name(self, group_name):
        for group in self.list_item_group.children:
            if group.group_name == group_name:
                return group
        return None

    @handle_cb_exceptions
    def save_edited_group_name(self, *args):
        self.list_item_group.v_model = None
        old_group_name = self.current_selected_group
        new_group_name = None
        self.is_clicked_to_edit = False

        current_group = self._get_group_by_name(old_group_name)
        if current_group:
            new_group_name = current_group.edit_menu_name_input.v_model.strip()
            current_group.group_name = new_group_name

        if old_group_name and new_group_name and old_group_name != new_group_name:
            self.state.rename_group(new_name=new_group_name, old_name=old_group_name)
            self.update_displayed_groups()

            current_group.edit_menu_name_input.v_model = ""
            current_group.edit_menu.v_model = False

    @handle_cb_exceptions
    def update_displayed_groups(self):
        group_names = list(self.state.get_groups().keys())
        self.list_item_group.children = [
            GroupItem(group_name, self.state, self.delete_button, self)
            for group_name in group_names
        ]
        if hasattr(self, "update_callback") and callable(self.update_callback):
            self.update_callback()
        for list_item in self.list_item_group.children:
            list_item.on_event("mouseenter", list_item.show_edit_icon)
            list_item.on_event("mouseleave", list_item.hide_edit_icon)
            list_item.edit_menu.save_button.on_event(
                "click", self.save_edited_group_name
            )

    def update_delete_button_state(self, change):
        self.delete_button.disabled = change["new"] is None

    @handle_cb_exceptions
    def delete_selected_groups(self, *args):
        self.is_clicked_to_edit = False
        if self.current_selected_group:
            self.state.delete_group(self.current_selected_group)
            self.update_displayed_groups()
        self.current_selected_group = None


class GroupSaveButton(MenuButton):
    def __init__(
        self,
        dataselector: SelectionManager,
        state: State,
        group_display: Optional[GroupDisplay] = None,
    ):
        self.state = state
        self.dataselector = dataselector
        self.group_display = group_display

        self.group_name_input = TextField(label="Group name", v_model="", maxlength=30)
        self.group_name_input.on_event("input", self.perform_validation)

        self.data_points_count = v.Html(
            tag="div",
            children=[
                f"Number of data points: {len(self.dataselector.graph_selection)}"
            ],
            class_="d-flex",
            style_="font-size: 16px",
        )

        self.menu = Menu(
            menu_title="Create Group from Selection",
            menu_children=[self.group_name_input, self.data_points_count],
            save_button_label="Create",
        )
        self.menu.save_button.disabled = True
        self.menu.save_button.on_event("click", self.save_group)

        super().__init__(
            button_label=v.Icon(children=["mdi mdi-playlist-plus"]),
            menu=self.menu,
            color="secondary",
            flat=True,
        )

    def perform_validation(self, *args):
        group_name = self.group_name_input.v_model
        is_valid, error_list = validate_group_name(self.state, group_name)

        self.group_name_input.error_messages = error_list
        self.menu.save_button.disabled = not is_valid

    @handle_cb_exceptions
    def save_group(self, *args):
        self.perform_validation()
        if self.menu.save_button.disabled:
            return

        group_name = self.group_name_input.v_model.strip()
        new_group = self.dataselector.graph_selection

        if not new_group:
            self.group_name_input.error_messages = [
                "No data points selected. Select data points to create a group."
            ]
            return

        self.state.add_group(group=new_group, group_name=group_name, notify=True)

        self.menu.v_model = False
        self.menu.save_button.disabled = True
        self.group_name_input.v_model = ""

        if self.group_display is not None:
            self.group_display.update_displayed_groups()
            self.group_display.list_item_group.v_model = group_name

    def update_data_points(self, *_):
        self.data_points_count.children = [
            f"Number of data points: {len(self.dataselector.graph_selection)}"
        ]
        self.toggle_menu()


class GroupSaveModal(MenuButton):
    def __init__(
        self,
        state: State,
        data_points: CobaltDataSubset,
    ):
        self.state = state
        self.data_points = data_points

        self.group_name_input = TextField(label="Group name", v_model="", maxlength=30)
        self.group_name_input.on_event("input", self.perform_validation)

        self.data_points_count = v.Html(
            tag="div",
            children=[f"Number of data points: {len(self.data_points)}"],
            class_="d-flex",
            style_="font-size: 16px",
        )

        self.menu = Menu(
            menu_title="Create Group",
            menu_children=[self.group_name_input, self.data_points_count],
            save_button_label="Create",
        )
        self.menu.save_button.disabled = True
        self.menu.save_button.on_event("click", self.save_group)

        super().__init__(
            button_label=v.Icon(children=["mdi mdi-playlist-plus"]),
            menu=self.menu,
            color="secondary",
            flat=True,
        )

    def perform_validation(self, *args):
        group_name = self.group_name_input.v_model
        is_valid, error_list = validate_group_name(self.state, group_name)

        self.group_name_input.error_messages = error_list
        self.menu.save_button.disabled = not is_valid

    @handle_cb_exceptions
    def save_group(self, *args):
        self.perform_validation()
        if self.menu.save_button.disabled:
            return

        group_name = self.group_name_input.v_model.strip()
        new_group = self.data_points

        if not new_group:
            self.group_name_input.error_messages = [
                "No data points provided. Provide data points to create a group."
            ]
            return

        self.state.add_group(group=new_group, group_name=group_name, notify=True)

        self.menu.v_model = False
        self.menu.save_button.disabled = True
        self.group_name_input.v_model = ""

    def update_data_points(self, data_points: CobaltDataSubset):
        self.data_points = data_points
        self.data_points_count.children = [
            f"Number of data points: {len(self.data_points)}"
        ]


class GroupItem(v.ListItem):
    def __init__(
        self,
        group_name,
        state,
        delete_button,
        group_display: Optional[GroupDisplay] = None,
    ):
        self.state: State = state
        self.group_name = group_name
        self.delete_button = delete_button
        self.group_display = group_display

        self.edit_menu_name_input = TextField(
            label="Group Name",
            v_model="",
        )
        self.edit_menu_name_input.on_event("input", self.validate_edit_menu)
        self.edit_menu = Menu(
            menu_title="Edit Group",
            menu_children=[self.edit_menu_name_input],
            save_button_label="Save",
            delete_button=[self.delete_button],
            close_editing_menu_callback=self.close_editing_menu,
        )
        self.edit_menu.save_button.disabled = not bool(
            self.edit_menu_name_input.v_model.strip()
        )

        button_label = v.Icon(children=["mdi mdi-pencil"])
        self.edit_icon = MenuButton(
            icon=True, small=False, button_label=button_label, menu=self.edit_menu
        )
        self.edit_icon.on_event("click", self.show_prev_name)

        self.edit_icon_with_tooltip = with_tooltip(self.edit_icon, "Edit Group")

        self.edit_icon.style_ = "visibility: hidden;"
        style_ = "padding: 0 16px;"
        children = self._generate_children(group_name)

        super().__init__(style_=style_, value=group_name, children=children)

    def close_editing_menu(self, *args):
        self.group_display.is_clicked_to_edit = False
        self.edit_menu.v_model = None
        if self.group_display.list_item_group.v_model is not None:
            self.group_display.list_item_group.v_model = None

    def _generate_children(self, group_name):
        groups_info = self.state.get_groups_info()
        return [
            v.ListItemContent(
                children=[
                    v.ListItemTitle(
                        class_="font-weight-regular",
                        style_="font-size: 1rem;" "line-height: 1.2;",
                        children=[group_name],
                    ),
                    v.ListItemSubtitle(
                        children=[f" {groups_info.get(group_name)} points"]
                    ),
                ]
            ),
            self.edit_icon_with_tooltip,
        ]

    def show_edit_icon(self, *args):
        self.edit_icon.style_ = "visibility: visible;"

    def hide_edit_icon(self, *args):
        self.edit_icon.style_ = "visibility: hidden;"

    def show_prev_name(self, *args):
        self.group_display.is_clicked_to_edit = True
        self.edit_menu_name_input.v_model = self.group_name
        self.edit_icon.toggle_menu(*args)

    def validate_edit_menu(self, *args):
        self.group_display.is_clicked_to_edit = False
        new_group_name = self.edit_menu_name_input.v_model

        if not new_group_name.strip():
            self.edit_menu.save_button.disabled = True
            self.edit_menu_name_input.error_messages = [
                "New group name cannot be empty."
            ]
        elif not self.state.is_group_name_unique(new_group_name):
            self.edit_menu.save_button.disabled = True
            self.edit_menu_name_input.error_messages = [
                "A group with this name already exists."
            ]
        else:
            self.edit_menu.save_button.disabled = False
            self.edit_menu_name_input.error_messages = []
        self.group_display.list_item_group.v_model = None
