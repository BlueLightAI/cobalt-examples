from typing import Callable

import ipyvuetify as v

from cobalt.state import DataFilter, State
from cobalt.table.filter_chip import FilterChip
from cobalt.table.table_utils import get_text_operator
from cobalt.widgets import Button


class FiltersList(v.Flex):
    """Displays all active filters as chips and a `clear all filters` button."""

    def __init__(
        self,
        state: State,
        handle_chip_click: Callable[[DataFilter], None],
        handle_clear_filters: Callable[[], None],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.state = state
        self.handle_chip_click = handle_chip_click
        self.handle_clear_filters = handle_clear_filters

        self.filter_chip_items = self._build_filter_chips()

        self.clear_filters_button = self._create_clear_filters_button()

        self.children = [
            *self.filter_chip_items,
            self.clear_filters_button,
        ]
        self.class_ = "flex-wrap"

    def _build_filter_chips(self):
        chips = []
        for filter_item in self.state.data_filters.filters:
            chip_text = (
                f"{filter_item.column.name} "
                f"{get_text_operator(filter_item.op)} "
                f"{filter_item.value}"
            )

            def on_click_callback(item=filter_item):
                self.handle_chip_click(item)

            chip = FilterChip(
                chip_text=chip_text,
                on_click=on_click_callback,
            )
            chips.append(chip)
        return chips

    def _create_clear_filters_button(self):
        css_class = (
            "my-2 d-initial"
            if len(self.state.data_filters.filters) > 0
            else "my-2 d-none"
        )

        clear_button = Button(
            tile=True,
            text=True,
            children=[v.Icon(children=["mdi-close"]), "clear all"],
            class_=css_class,
        )
        clear_button.on_event("click", lambda *_: self.handle_clear_filters())
        return clear_button
