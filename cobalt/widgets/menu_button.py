# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Callable, Optional, Union

import ipyvuetify as v


class MenuButton(v.Btn):
    def __init__(
        self,
        button_label: Union[str, v.VuetifyWidget],
        menu: v.VuetifyWidget,
        validator: Optional[Callable] = None,
        **kwargs,
    ):
        self.menu = menu

        if "color" not in kwargs:
            kwargs["color"] = "info"
        if "small" not in kwargs:
            kwargs["small"] = True

        super().__init__(
            children=[
                button_label,
                self.menu,
            ],
            **kwargs,
        )
        self.validator = validator
        self.on_event("click", self.toggle_menu)
        self.validate_menu()

    def toggle_menu(self, *args):
        self.menu.v_model = not self.menu.v_model
        if self.menu.v_model:
            self.validate_menu()

    def validate_menu(self, *args):
        if self.validator is not None:
            self.validator()
