from typing import List, Optional

import ipyvuetify as v
import pandas as pd

from cobalt.cobalt_types import ColumnDataType
from cobalt.config import handle_cb_exceptions
from cobalt.schema.metadata import DatasetMetadata, TextDataType, is_column_categorical
from cobalt.table.table_constants import (
    CATEGORICAL_TEXT_OPERATOR_SELECT_ITEMS,
    CONTAINS,
    DEFAULT_OPERATOR_SELECT_ITEMS,
    EQUALS,
    IS,
    MANUAL_NUMERIC_SELECT_ITEMS,
    MANUAL_TEXT_SELECT_ITEMS,
    NUMERICAL_OPERATOR_SELECT_ITEMS,
)
from cobalt.widgets import (
    Autocomplete,
    Button,
    Checkbox,
    SearchableSelect,
    Select,
)


class TableFilter(v.Layout):
    """Widget for building and applying filter criteria.

    - Shows a column selector, operator dropdown, and input field (or autocomplete).
    - Adapts UI based on column type (text, numeric, categorical).
    - Validates user input and, on success, invokes on_apply_filter_callback.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        columns_metadata: DatasetMetadata,
        on_apply_filter_callback,
        columns: Optional[List[str]] = None,
    ):
        super().__init__()
        self.df = df
        self.columns_metadata = columns_metadata
        self.columns = columns or list(df.columns)
        self.on_apply_filter_callback = on_apply_filter_callback

        self.columns_to_filter_selector = self._generate_columns_to_filter_selector()
        self.filter_select_operator = self._create_filter_select_operator()
        self.filter_input_value = self._create_filter_select_value()
        self.filter_autocomplete_value = self._create_filter_autocomplete_value()
        self.match_case_checkbox = self._create_match_case_checkbox()
        self.match_entire_cell_checkbox = self._create_match_entire_cell_checkbox()
        self.checkbox_wrapper = self._create_checkbox_wrapper()
        self.apply_button = self._create_apply_button()

        self._setup_filter_view()

    def _setup_filter_view(self):
        self.children = [
            v.Flex(
                children=[
                    v.Flex(
                        children=[
                            self.columns_to_filter_selector,
                            self.filter_select_operator,
                            self.filter_input_value,
                            self.filter_autocomplete_value,
                            self.checkbox_wrapper,
                            self.apply_button,
                        ],
                        style_="gap: 8px",
                        class_="d-flex justify-space-between",
                    ),
                ],
            ),
        ]

    @staticmethod
    def is_datetime(data):
        return pd.api.types.is_datetime64_any_dtype(data)

    @staticmethod
    def is_image(image_columns, column):
        return column in [i["column_name"] for i in image_columns]

    @staticmethod
    def is_filterable_dtype(data):
        return any(
            (
                pd.api.types.is_string_dtype(data),
                pd.api.types.is_integer_dtype(data),
                pd.api.types.is_float_dtype(data),
                pd.api.types.is_bool_dtype(data),
                is_column_categorical(data),
            )
        )

    def _generate_columns_to_filter_selector(self):
        image_columns = [
            img_col.autoname_media_visualization_column()
            for img_col in self.columns_metadata.media_columns
        ]
        available_filter_columns = [
            c
            for c in self.df.columns
            if not self.is_datetime(self.df[c])
            and not self.is_image(image_columns, c)
            and self.is_filterable_dtype(self.df[c])
        ]
        columns_to_filter = SearchableSelect(
            items=available_filter_columns,
            label="Select column to filter",
            attach=True,
            style_="max-width: 210px",
            hide_details=False,
            multiple=False,
            v_model=None,
        )
        columns_to_filter.on_event("change", self._on_column_select)
        return columns_to_filter

    def _create_filter_select_value(self):
        filter_value = v.TextField(
            label="Value",
            v_model="",
            class_="my-2",
            outlined=True,
            dense=True,
        )
        filter_value.on_event(
            "input", lambda *_: setattr(filter_value, "error_messages", [])
        )
        return filter_value

    def _create_filter_autocomplete_value(self):
        autocomplete_value = Autocomplete(
            items=[],
            v_model=None,
            clearable=True,
            hide_details=False,
            deletable_chips=True,
            placeholder="Value",
            class_="d-none",
        )
        autocomplete_value.on_event(
            "change", lambda widget, *_: setattr(widget, "error_messages", [])
        )
        return autocomplete_value

    def _create_filter_select_operator(self):
        filter_select_operator = Select(
            label="Operator",
            attach=True,
            items=DEFAULT_OPERATOR_SELECT_ITEMS,
            hide_details=False,
            v_model=None,
            multiple=False,
            style_="max-width: 140px",
        )
        filter_select_operator.on_event("change", self._on_operator_select)
        return filter_select_operator

    def _create_match_case_checkbox(self):
        return Checkbox(label="Match Case", v_model=None, class_="mt-3")

    def _create_match_entire_cell_checkbox(self):
        return Checkbox(label="Match entire cell", v_model=None, class_="mt-3")

    def _create_checkbox_wrapper(self):
        return v.Flex(
            children=[self.match_case_checkbox, self.match_entire_cell_checkbox],
            class_="d-none",
        )

    def _create_apply_button(self):
        apply_button = Button(
            children=["apply filter"],
            class_="my-2",
            color="primary",
        )
        apply_button.on_event("click", self._on_apply_button_click)
        return apply_button

    @handle_cb_exceptions
    def _on_column_select(self, widget, event, column_name):
        available_filter_columns = [
            c
            for c in self.df.columns
            if not (pd.api.types.is_datetime64_any_dtype(self.df[c]))
        ]

        if column_name not in available_filter_columns:
            self.columns_to_filter_selector.error_messages = [
                f"Selected column '{column_name}' is not valid."
            ]
            return

        self.columns_to_filter_selector.error_messages = []

        self._clear_fields(keep_column=True)
        columns_metadata = self.columns_metadata.data_types

        column_type = columns_metadata.get(column_name).col_type
        column_text_type = columns_metadata.get(column_name).text_type
        is_categorical = columns_metadata.get(column_name).is_categorical
        is_manual = columns_metadata.get(column_name).explicit_categorical

        column_values = columns_metadata.get(column_name).cat_values

        if is_categorical and column_text_type != TextDataType.long_text:
            self._show_autocomplete_value()
            self.filter_autocomplete_value.items = column_values

            # TODO (From Jakob)
            # "We may revisit this logic. I'm not sure that it makes sense
            # to treat manual categorical columns any differently here."
            if is_manual:
                if column_type == ColumnDataType.text:
                    self.filter_select_operator.items = MANUAL_TEXT_SELECT_ITEMS
                    self.filter_select_operator.v_model = IS
                if column_type == ColumnDataType.numerical:
                    self.filter_select_operator.items = MANUAL_NUMERIC_SELECT_ITEMS
                    self.filter_select_operator.v_model = EQUALS

            elif column_type == ColumnDataType.text:

                self.filter_select_operator.items = (
                    CATEGORICAL_TEXT_OPERATOR_SELECT_ITEMS
                )
                self.filter_select_operator.v_model = IS

            elif self.df[column_name].dtype == bool:
                self.filter_select_operator.items = MANUAL_NUMERIC_SELECT_ITEMS
                self.filter_select_operator.v_model = IS

            else:

                # condition for the categorical numerical columns
                self.filter_select_operator.items = NUMERICAL_OPERATOR_SELECT_ITEMS
        else:
            # condition for all non-categorical columns
            self._show_input_value()
            if column_type == ColumnDataType.text:

                self._show_checkbox_wrapper()
                self.filter_select_operator.items = DEFAULT_OPERATOR_SELECT_ITEMS
                self.filter_select_operator.v_model = CONTAINS
            else:

                # condition for the numerical columns
                self._hide_checkbox_wrapper()
                self.filter_select_operator.items = NUMERICAL_OPERATOR_SELECT_ITEMS

    @handle_cb_exceptions
    def _on_operator_select(self, widget, event, operator):
        widget.error_messages = []
        if self.columns_to_filter_selector.v_model:
            columns_metadata = self.columns_metadata.data_types
            column_type = columns_metadata.get(
                self.columns_to_filter_selector.v_model
            ).col_type

            is_categorical = columns_metadata.get(
                self.columns_to_filter_selector.v_model
            ).is_categorical

            if is_categorical and column_type == ColumnDataType.text:
                if operator == CONTAINS:
                    self._show_input_value()
                    self._show_checkbox_wrapper()
                else:
                    self._show_autocomplete_value()
                    self._hide_checkbox_wrapper()

    def _on_apply_button_click(self, *_):
        if self._validate_filters():
            column_value = self._get_column_value()
            operator = self._get_operator()
            self.on_apply_filter_callback(
                self.columns_to_filter_selector.v_model, operator, column_value
            )
            self._clear_fields()

    def _get_column_value(self, *_):
        operator = self.filter_select_operator.v_model
        columns_metadata = self.columns_metadata.data_types
        col_meta = columns_metadata.get(self.columns_to_filter_selector.v_model)
        is_categorical = col_meta.is_categorical
        column_type = col_meta.col_type

        # Categorical text
        if is_categorical and column_type == ColumnDataType.text:
            if operator in (IS, EQUALS):
                return self.filter_autocomplete_value.v_model
            elif operator == CONTAINS:
                return self.filter_input_value.v_model
            return self.filter_autocomplete_value.v_model

        # Categorical numeric
        elif is_categorical:
            return self.filter_autocomplete_value.v_model

        # Non-categorical text
        elif column_type == ColumnDataType.text:
            return self.filter_input_value.v_model

        # Numerical
        else:
            return float(self.filter_input_value.v_model)

    def _get_operator(self, *_):
        operator = self.filter_select_operator.v_model
        columns_metadata = self.columns_metadata.data_types
        col_meta = columns_metadata.get(self.columns_to_filter_selector.v_model)
        is_categorical = col_meta.is_categorical
        column_type = col_meta.col_type

        # Categorical text scenario
        if is_categorical and column_type == ColumnDataType.text:
            # If user selected "contains", incorporate checkboxes
            if operator == CONTAINS:
                return self._get_selected_checkboxes_value()
            else:
                # For "is"/"equals" on text-categorical, default to a case-sensitive 'is'
                return "is_case_sensitive_on"

        # Numerical columns
        elif column_type == ColumnDataType.numerical:
            return operator

        # Non-categorical text
        else:
            return self._get_selected_checkboxes_value()

    def _validate_filters(self, *_):
        def is_float(value):
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False

        column = self.columns_to_filter_selector.v_model
        operator = self.filter_select_operator.v_model
        input_value = self.filter_input_value.v_model or ""
        autocomplete_value = self.filter_autocomplete_value.v_model

        is_valid = True

        if not column:
            is_valid = False
            self.columns_to_filter_selector.error_messages = ["No column selected"]

        if not operator:
            is_valid = False
            self.filter_select_operator.error_messages = ["Operator should be selected"]

        if self._is_autocomplete_hidden():
            if not input_value.strip():
                is_valid = False
                self.filter_input_value.error_messages = ["Value should be filled in"]
            elif self._is_selected_numerical_col() and not is_float(input_value):
                is_valid = False
                self.filter_input_value.error_messages = ["Should be a numeric value"]
        else:
            if autocomplete_value is None:
                is_valid = False
                self.filter_autocomplete_value.error_messages = [
                    "Value should be selected"
                ]

        return is_valid

    def _is_autocomplete_hidden(self) -> bool:
        return "d-none" in self.filter_autocomplete_value.class_.split()

    def _is_selected_numerical_col(self) -> bool:
        column_name = self.columns_to_filter_selector.v_model
        if column_name:
            columns_metadata = self.columns_metadata.data_types
            col_meta = columns_metadata.get(column_name)
            if (
                col_meta.col_type == ColumnDataType.numerical
                and not col_meta.is_categorical
                and not col_meta.explicit_categorical
            ):
                return True
        return False

    def _show_input_value(self, *_):
        self.filter_autocomplete_value.class_ = "d-none"
        self.filter_input_value.class_ = "d-block my-2"

    def _hide_checkbox_wrapper(self, *_):
        self.checkbox_wrapper.class_ = "d-none"

    def _show_checkbox_wrapper(self, *_):
        self.checkbox_wrapper.class_ = "d-flex justify-space-around"

    def _show_autocomplete_value(self, *_):
        self.filter_autocomplete_value.class_ = "d-block my-2"
        self.filter_input_value.class_ = "d-none"

    def _clear_fields(self, keep_column=False):
        self._clear_field_errors()
        self.filter_input_value.v_model = ""
        self.filter_autocomplete_value.v_model = None
        self.match_case_checkbox.v_model = None
        self.match_entire_cell_checkbox.v_model = None
        self.filter_select_operator.v_model = None

        if not keep_column:
            self.columns_to_filter_selector.v_model = None

        self._hide_checkbox_wrapper()
        self._show_input_value()

    def _clear_field_errors(self, *_):
        self.columns_to_filter_selector.error_messages = []
        self.filter_select_operator.error_messages = []
        self.filter_input_value.error_messages = []
        self.filter_autocomplete_value.error_messages = []

    def _get_selected_checkboxes_value(self, *_):
        case_checked = self.match_case_checkbox.v_model
        entire_checked = self.match_entire_cell_checkbox.v_model

        if case_checked and entire_checked:
            return "is_case_sensitive_on"
        elif case_checked and not entire_checked:
            return "contains_sensitive_on"
        elif not case_checked and entire_checked:
            return "is_case_sensitive_off"
        else:
            return "contains_sensitive_off"
