# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Callable, List, Optional

import ipyvuetify as v
import ipywidgets as w

from cobalt.widgets import Button, Dialog


class Menu(Dialog):
    def __init__(
        self,
        menu_title: str,
        menu_children: List[w.Widget],
        close_button_label: str = "Cancel",
        save_button_label: Optional[str] = None,
        delete_button: Optional[List[w.Widget]] = None,
        close_editing_menu_callback: Optional[Callable] = None,
    ):
        self.close_editing_menu_callback = close_editing_menu_callback
        self.close_button = Button(
            children=[close_button_label],
            density="compact",
            text=True,
            class_="mx-1",
        )
        self.close_button.on_event("click", self.close_menu)

        if save_button_label:
            self.save_button = Button(
                children=[save_button_label],
                density="compact",
                color="primary",
                text=True,
                class_="mx-1",
            )
            buttons = [self.close_button, self.save_button]
        else:
            buttons = [self.close_button]

        if delete_button is not None:
            buttons.insert(0, *delete_button)

        self.card = v.Card(
            class_="text-center pa-4",
            children=[
                v.CardTitle(children=[menu_title], class_="pa-0"),
                *menu_children,
                v.CardActions(children=buttons, class_="justify-end"),
            ],
        )

        super().__init__(
            v_model=False,
            width="300",
            activator="parent",
            children=[self.card],
        )

    def close_menu(self, *args):
        if self.close_editing_menu_callback:
            self.close_editing_menu_callback()
        self.v_model = False
