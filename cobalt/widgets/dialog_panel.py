# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v

from cobalt.widgets import Button, Dialog


class DialogPanel(v.Card):
    def __init__(self, header_content, content_widget, **kwargs):
        super().__init__(**kwargs)
        self.class_ = "ma-1"
        self.style_ = "width: 100%; background-color: rgb(232, 244, 250)"

        self.toggle_button_close = Button(
            icon=True, children=[v.Icon(children=["mdi-close"])]
        )

        self.content_widget_container = v.Col(children=[content_widget])

        self.dialog = Dialog(
            v_model=False,
            persistent=True,
            fullscreen=True,
            no_click_animation=True,
            children=[
                v.Card(
                    children=[
                        v.CardTitle(
                            class_="justify-space-between",
                            children=[
                                v.Html(
                                    tag="span", class_="h4", children=[header_content]
                                ),
                                self.toggle_button_close,
                            ],
                        ),
                        v.CardText(children=[self.content_widget_container]),
                    ],
                )
            ],
        )

        self.toggle_button_open = Button(
            icon=True,
            children=[v.Icon(children=["mdi-fullscreen"])],
            click=self.toggle_dialog,
        )

        self.card_title = v.CardTitle(
            class_="py-1 px-3 justify-space-between",
            children=[
                v.Html(tag="span", class_="body-1", children=[header_content]),
                self.toggle_button_open,
            ],
        )

        self.children = [self.card_title, self.dialog]

        self.toggle_button_open.on_event("click", self.toggle_dialog)
        self.toggle_button_close.on_event("click", self.toggle_dialog)

    def toggle_dialog(self, widget, event, data):
        self.dialog.v_model = not self.dialog.v_model

    def update_content_widget(self, new_content_widget):
        self.content_widget_container.children = [new_content_widget]
