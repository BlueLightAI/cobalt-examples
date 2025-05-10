# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v

from cobalt.config import handle_cb_exceptions
from cobalt.widgets import Button, Dialog


class HistogramDialog(Dialog):
    def __init__(self, v_model: bool, on_close, children, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.v_model = v_model
        self.on_close_callback = on_close

        self.close_button = Button(icon=True, children=[v.Icon(children=["mdi-close"])])
        self.close_button.on_event("click", self.on_close)

        self.children = children

    def on_close(self, widget, event, data):
        if self.on_close_callback:
            self.on_close_callback(widget, event, data)

    @handle_cb_exceptions
    def add_histogram(self, title, histogram):
        if not histogram:
            return

        display_title = title or "Histogram"

        self.children = [
            v.Card(
                children=[
                    v.CardTitle(
                        children=[
                            v.Flex(style_="max-width: 36px;"),
                            v.Spacer(),
                            display_title,
                            v.Spacer(),
                            self.close_button,
                        ],
                    ),
                    v.CardText(
                        children=[
                            v.Html(
                                tag="div",
                                class_="d-flex flex-column align-center",
                                children=[histogram],
                            )
                        ]
                    ),
                ]
            )
        ]
