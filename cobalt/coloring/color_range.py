# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import math
from typing import Callable, Optional, Tuple

import ipyvuetify as v


class ColorRangeInput(v.TextField):
    def __init__(self, value_changed_callback: Callable, **kwargs):
        super().__init__(**kwargs)
        self.value_changed_callback = value_changed_callback
        self.on_event("blur", self.vue_on_blur)

    def vue_on_blur(self, *_):
        if self.value_changed_callback:
            self.value_changed_callback()


class ColorRange(v.Row):
    def __init__(
        self,
        color_by,
        value_changed_callback: Callable,
        default_cmap_range: Optional[Tuple[float, float]] = None,
        boundaries_cmap_range: Optional[Tuple[float, float]] = None,
        is_auto: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__(
            class_="pa-0 ma-0 d-flex flex-column", no_gutter=True, **kwargs
        )

        self.color_by = color_by
        self.value_changed_callback = value_changed_callback

        self.default_cmap_range = (
            (
                self.process_value(default_cmap_range[0], default_cmap_range),
                self.process_value(default_cmap_range[1], default_cmap_range),
            )
            if default_cmap_range
            else None
        )
        self.boundaries_cmap_range = (
            (
                self.process_value(boundaries_cmap_range[0], boundaries_cmap_range),
                self.process_value(boundaries_cmap_range[1], boundaries_cmap_range),
            )
            if boundaries_cmap_range
            else None
        )

        self.default_min_value = (
            self.default_cmap_range[0] if self.default_cmap_range else None
        )
        self.default_max_value = (
            self.default_cmap_range[1] if self.default_cmap_range else None
        )

        self.min_limit = (
            self.boundaries_cmap_range[0] if self.boundaries_cmap_range else None
        )
        self.max_limit = (
            self.boundaries_cmap_range[1] if self.boundaries_cmap_range else None
        )

        self.is_auto = is_auto

        self.label = v.Label(children=["Color range"], class_="d-flex align-center")

        self.auto_checkbox = v.Checkbox(
            label="Auto",
            v_model=self.is_auto,
            dense=True,
            hide_details=True,
            class_="ma-0 pa-0",
        )
        self.auto_checkbox.on_event("change", self.on_range_change)

        self.label_checkbox_row = v.Row(
            no_gutters=True,
            class_="pa-0 ma-0 d-flex justify-space-between",
            children=[
                v.Col(class_="d-flex align-center", children=[self.label]),
                v.Col(
                    class_="d-flex justify-end align-center",
                    children=[self.auto_checkbox],
                ),
            ],
            style_="gap: 12px; height: 30px;",
        )

        self.inputs_block = v.Col(
            class_="d-flex pa-0 ma-0 justify-space-between",
            children=[],
            style_="gap: 12px;",
        )

        self.inputs_wrapper = v.Row(
            class_="ma-0 pa-0 d-none", no_gutters=True, children=[self.inputs_block]
        )

        self.create_inputs()

        self.children = [self.label_checkbox_row, self.inputs_wrapper]

    def reset(self, *_):
        self.default_cmap_range = None
        self.boundaries_cmap_range = None
        self.min_limit = None
        self.default_min_value = None
        self.max_limit = None
        self.default_max_value = None
        self.is_auto = True
        self.auto_checkbox.v_model = True
        self.handle_color_range_inputs_visibility()

    def on_range_change(self, widget, event, data):
        self.is_auto = data
        self.handle_color_range_inputs_visibility()

        if (
            self.is_auto
            and is_valid_range_value(self.default_cmap_range[0])
            and is_valid_range_value(self.default_cmap_range[1])
        ):
            self.default_min_value = self.default_cmap_range[0]
            self.default_max_value = self.default_cmap_range[1]

        self.create_inputs()

        if self.value_changed_callback:
            self.value_changed_callback()

    def handle_color_range_inputs_visibility(self, *_):
        if self.is_auto:
            self.inputs_wrapper.class_list.add("d-none")
        else:
            self.inputs_wrapper.class_list.remove("d-none")

    def hide(self, *_):
        self.class_list.remove("d-flex")
        self.class_list.add("d-none")

    def show(self, *_):
        self.class_list.remove("d-none")
        self.class_list.add("d-flex")

    def update_min_limit(self, value, *_):
        self.min_limit = float(value) if value is not None else None
        if self.min_limit is not None and (
            self.default_min_value is None or self.default_min_value == ""
        ):
            self.default_min_value = self.min_limit
        self.create_inputs()

    def update_max_limit(self, value: float, *_):
        self.max_limit = float(value) if value is not None else None
        if self.max_limit is not None and (
            self.default_max_value is None or self.default_max_value == ""
        ):
            self.default_max_value = self.max_limit
        self.create_inputs()

    def create_inputs(self, *_):
        min_limit_str_val = (
            format_float_to_str(self.min_limit)
            if isinstance(self.min_limit, float)
            else "None"
        )
        max_limit_str_val = (
            format_float_to_str(self.max_limit)
            if isinstance(self.max_limit, float)
            else "None"
        )

        self.min_input = ColorRangeInput(
            value_changed_callback=self.validate_and_callback,
            hint=f"Min (Limit: {min_limit_str_val})",
            outlined=False,
            persistent_hint=True,
            v_model=format_float_to_str(self.default_min_value),
        )

        self.max_input = ColorRangeInput(
            value_changed_callback=self.validate_and_callback,
            hint=f"Max (Limit: {max_limit_str_val})",
            outlined=False,
            persistent_hint=True,
            v_model=format_float_to_str(self.default_max_value),
        )

        self.inputs_block.children = [self.min_input, self.max_input]

    def validate_and_callback(self):
        min_value = self.min_input.v_model
        max_value = self.max_input.v_model
        min_error_message = ""
        max_error_message = ""

        if not min_value or not is_numeric(min_value):
            min_error_message = "Invalid input. Please enter a numeric value."

        if not max_value or not is_numeric(max_value):
            max_error_message = "Invalid input. Please enter a numeric value."

        if min_error_message or max_error_message:
            self.min_input.error_messages = (
                [min_error_message] if min_error_message else []
            )
            self.max_input.error_messages = (
                [max_error_message] if max_error_message else []
            )
            return

        min_value = float(min_value)
        max_value = float(max_value)

        if self.min_limit is not None and min_value < self.min_limit:
            min_error_message = f"Min value cannot be less than {self.min_limit}"
        if self.max_limit is not None and min_value > self.max_limit:
            min_error_message = f"Min value cannot be greater than {self.max_limit}"

        if self.min_limit is not None and max_value < self.min_limit:
            max_error_message = f"Max value cannot be less than {self.min_limit}"
        if self.max_limit is not None and max_value > self.max_limit:
            max_error_message = f"Max value cannot be greater than {self.max_limit}"

        if min_value > max_value:
            min_error_message = "Min value cannot be greater than Max value"
            max_error_message = "Max value cannot be less than Min value"

        self.min_input.error_messages = [min_error_message] if min_error_message else []
        self.max_input.error_messages = [max_error_message] if max_error_message else []

        if (
            self.value_changed_callback
            and not min_error_message
            and not max_error_message
        ):
            self.value_changed_callback()

    def set_default_range(self, range_value: Tuple[float, float], *_):
        self.default_cmap_range = (
            self.process_value(range_value[0], range_value),
            self.process_value(range_value[1], range_value),
        )
        self.default_min_value = None
        self.default_max_value = None
        self.min_input.v_model = None
        self.max_input.v_model = None
        self.update_min_limit(self.default_cmap_range[0])
        self.update_max_limit(self.default_cmap_range[1])

    def set_boundaries_range(self, range_value: Tuple[float, float], *_):
        self.boundaries_cmap_range = (
            self.process_value(range_value[0], range_value),
            self.process_value(range_value[1], range_value),
        )
        self.update_min_limit(self.boundaries_cmap_range[0])
        self.update_max_limit(self.boundaries_cmap_range[1])

    @staticmethod
    def process_value(value: float, value_range: Tuple[float, float], *_):
        range_min, range_max = map(float, value_range)
        range_size = range_max - range_min

        if range_size <= 0:
            return float(value)

        num_decimals = max(0, int(-math.log10(range_size))) + 3
        return round(float(value), num_decimals)

    def get_coloring_range(self, *_):
        if self.default_cmap_range is None or self.is_auto:
            return None
        min_value = (
            float(self.min_input.v_model)
            if is_valid_range_value(self.min_input.v_model)
            else self.default_cmap_range[0]
        )
        max_value = (
            float(self.max_input.v_model)
            if is_valid_range_value(self.max_input.v_model)
            else self.default_cmap_range[1]
        )
        return min_value, max_value


def format_float_to_str(value):
    return f"{value:.4g}" if value is not None else None


def is_valid_range_value(value):
    if value is None:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
