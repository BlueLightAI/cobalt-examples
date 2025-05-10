# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from threading import Timer
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import ipyvuetify as v
import ipywidgets as w
import pandas as pd

from cobalt.config import debug_logger, handle_cb_exceptions
from cobalt.groups.groups_ui import GroupSaveModal
from cobalt.schema.dataset import DatasetBase
from cobalt.state import DataFilter, State, apply_filters_df
from cobalt.table.filters_list import FiltersList
from cobalt.table.table_filter import TableFilter
from cobalt.table.table_view import TableView
from cobalt.ui_utils import with_tooltip
from cobalt.widgets import Autocomplete, Button


class TableSelector(w.VBox):
    """High-level container for table display and filtering.

    - Renders a table of data (via TableView).
    - Manages which columns are visible.
    - Integrates with TableFilter to apply filters, updating the underlying State.
    - Refreshes the UI (table display) whenever filters or columns change.
    """

    def __init__(
        self,
        source_data: DatasetBase,
        state: State,
        workspace_id: Optional[UUID] = None,
        columns: Optional[List[str]] = None,
        image_columns: Optional[List[dict]] = None,
        html_columns: Optional[List[str]] = None,
        columns_to_filter: Optional[List[str]] = None,
        filter_criteria: Optional[Dict] = None,
        image_size: Tuple[int, int] = (80, 80),
        max_rows_to_display: Optional[int] = None,
        run_server: bool = True,
    ):
        super().__init__()
        self.source_data = source_data
        self.state = state
        self.workspace_id = workspace_id
        self.columns = columns
        self.image_columns = image_columns or []
        self.html_columns = html_columns or []
        self.image_size = image_size
        self.columns_to_filter = columns_to_filter
        self.filter_criteria = filter_criteria or {}
        self.show_filters = False
        self.max_rows_to_display = max_rows_to_display
        self.run_server = run_server
        self._debounce_timer = None

        self.df = self._get_data_from_source()

        self._initialize_widgets()
        self._setup_widget_list()
        self._check_and_open_filters()

    def _get_data_from_source(self) -> pd.DataFrame:
        data = self.source_data
        table_df = data.create_rich_media_table(run_server=self.run_server)
        filtered_df = self._filter_df(table_df)
        return filtered_df

    def _initialize_widgets(self):
        self.table_view = self._generate_table(self.workspace_id)
        self.select_all_text = "Select All"

        self._init_column_selector()
        self._setup_filtering()
        self._init_save_group_modal()

        self.filter_view = self._get_filter_view()

    def _init_column_selector(self):
        self.column_selector = Autocomplete(
            items=[self.select_all_text, *self.df.columns],
            v_model=self.columns,
            multiple=True,
            clearable=True,
            placeholder="All columns displayed by default. Adjust as needed.",
        )
        self.column_selector.on_event("change", self._on_change_with_debounce)

    def _setup_filtering(self):
        filter_icon_button = Button(
            icon=True,
            children=[v.Icon(children=["mdi mdi-filter-variant"], color="primary")],
        )
        filter_icon_button.on_event("click", self._switch_filters_layout)

        self.filters_counter = v.Flex(
            children="",
            style_="position: absolute; right: 2px; bottom: 0px;",
        )

        filter_button_container = v.Flex(
            style_="max-width: 36px; height: 36px; position: relative;",
            class_="mx-4",
            children=[filter_icon_button, v.Flex(children=[self.filters_counter])],
        )

        self.filter_button_container_with_tooltip = with_tooltip(
            filter_button_container, "Show/Hide Filters"
        )

    def _init_save_group_modal(self):
        filtered_indices = self.state.dataset.df.index.get_indexer(self.df.index)
        filtered_data_points = self.state.dataset.subset(filtered_indices)
        self.group_save_modal = GroupSaveModal(
            state=self.state, data_points=filtered_data_points
        )
        self.group_save_modal_with_tooltip = with_tooltip(
            self.group_save_modal, "Create Group from Table Data"
        )

    def _setup_widget_list(self):
        self.main_area = v.Flex(class_="d-flex flex-column", children=[self.table_view])

        self.selector_controls = v.Html(
            tag="div",
            children=[
                self.column_selector,
                self.filter_button_container_with_tooltip,
                self.group_save_modal_with_tooltip,
            ],
            class_="d-flex align-center justify-center px-6",
        )

        self.children = [
            self.selector_controls,
            self.main_area,
        ]

    def _check_and_open_filters(self):
        if len(self.state.data_filters.filters) > 0:
            self.filters_counter.children = str(len(self.state.data_filters.filters))
            self._switch_filters_layout()

    def _on_change_with_debounce(self, widget, event, data, delay=1):
        self.update_columns(widget, event, data)

        if self._debounce_timer:
            self._debounce_timer.cancel()

        def trigger_update_table():
            self.update_table(widget, event, data)

        self._debounce_timer = Timer(delay, trigger_update_table)
        self._debounce_timer.start()

    def _switch_filters_layout(self, *_):
        self.show_filters = not self.show_filters
        self._set_filter_visibility(self.show_filters)

    @handle_cb_exceptions
    def _apply_filter(self, column, operator, value):
        try:
            self.state.add_filter(column=column, op=operator, value=value)
            self.filters_counter.children = str(len(self.state.data_filters.filters))
            self.update_table()
        except Exception as e:
            debug_logger.error(f"Error updating table: {e}")
            self._update_table_view()

    def _handle_chip_click(self, data_filter: DataFilter):
        self.state.remove_filter(data_filter)
        remaining_filters = len(self.state.data_filters.filters)

        self.df = self._get_data_from_source()
        if remaining_filters == 0:
            self.filters_counter.children = ""
        else:
            self.filters_counter.children = str(remaining_filters)

        self.update_table()

    def _clear_all_filters(self, *_):
        self.state.data_filters.clear()

        del self.df
        self.df = None
        self.df = self._get_data_from_source()

        self.update_table()
        self.filters_counter.children = ""

    def update_columns(self, widget, event, data):
        if self.select_all_text in self.column_selector.v_model:
            if self.select_all_text == "Select All":
                self.column_selector.v_model = list(self.df.columns)
                self.select_all_text = "Deselect All"
            else:
                self.column_selector.v_model = []
                self.select_all_text = "Select All"

            self.column_selector.items = [self.select_all_text, *self.df.columns]
        else:
            if (
                set(self.column_selector.v_model) != set(self.df.columns)
                and self.select_all_text == "Deselect All"
            ):
                self.select_all_text = "Select All"
                self.column_selector.items = [
                    self.select_all_text,
                    *self.df.columns,
                ]

    @handle_cb_exceptions
    def update_table(self, *args):
        self.columns = self.column_selector.v_model

        # Clear old dataframe references
        del self.df
        self.df = self._get_data_from_source()

        self._update_table_view()

        filtered_indices = self.state.dataset.df.index.get_indexer(self.df.index)
        filtered_data_points = self.state.dataset.subset(filtered_indices)
        self.group_save_modal.update_data_points(filtered_data_points)

    def _update_table_view(self):
        self.table_view = self._generate_table(self.workspace_id)
        self.filter_view = self._get_filter_view()
        self.main_area.children = [self.filter_view, self.table_view]

    def _set_filter_visibility(self, show: bool):
        if show:
            self.filter_view.children[1]._clear_fields()
            self.main_area.children = [self.filter_view, self.table_view]
        else:
            self.main_area.children = [self.table_view]

    def _filter_df(self, df) -> pd.DataFrame:
        return apply_filters_df(df, self.state.data_filters)

    def _generate_table(self, workspace_id: Optional[UUID] = None):
        df = self.df[self.columns] if self.columns else self.df
        return TableView(
            df,
            table_uuid=workspace_id,
            num_rows=self.max_rows_to_display,
            image_columns=self.image_columns,
            html_columns=self.html_columns,
            image_size=self.image_size,
        )

    def _get_filter_view(self):
        filters_list = FiltersList(
            state=self.state,
            handle_chip_click=self._handle_chip_click,
            handle_clear_filters=self._clear_all_filters,
        )

        table_filter = TableFilter(
            df=self.df,
            columns_metadata=self.source_data.metadata,
            on_apply_filter_callback=self._apply_filter,
        )

        return v.Flex(
            children=[filters_list, table_filter],
            class_="px-6",
        )
