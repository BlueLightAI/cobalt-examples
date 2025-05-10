# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v

from cobalt.widgets import Button, Dialog


class VisualizationDialog(Dialog):
    def __init__(self, v_model: bool, on_close, children, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.v_model = v_model
        self.on_close_callback = on_close

        self.width = 500
        self.title = "Add New Visualization"

        self.close_button = Button(icon=True, children=[v.Icon(children=["mdi-close"])])
        self.close_button.on_event("click", self.on_close)

        self.children = [
            v.Card(
                children=[
                    v.CardTitle(
                        children=[
                            self.title,
                            v.Spacer(),
                            self.close_button,
                        ]
                    ),
                    v.CardText(children=children),
                ]
            )
        ]

    def on_close(self, widget, event, data):
        if self.on_close_callback:
            self.on_close_callback(widget, event, data)


class ReadModeVisualizationDialog(VisualizationDialog):
    def __init__(self, v_model: bool, on_close, children, *args, **kwargs):
        super().__init__(v_model, on_close, children, *args, **kwargs)

        no_embs_text = v.Text(
            children="Graphs cannot be created because there are no embeddings available "
            "in the Workspace. Use Workspace.add_embedding_array() to add an "
            "embedding before creating a graph.",
            class_="pt-4",
        )

        children[0].disabled = True
        data_source_widget = children[1]

        ds_children = list(data_source_widget.children)
        ds_children.pop()

        for widget in ds_children:
            widget.disabled = True

        btn = Button(children=["Create Graph"], disabled=True, color="gray")

        ds_children.append(btn)
        data_source_widget.children = ds_children
        children.insert(0, no_embs_text)

        self.children = [
            v.Card(
                children=[
                    v.CardTitle(
                        children=[
                            self.title,
                            v.Spacer(),
                            self.close_button,
                        ]
                    ),
                    v.CardText(children=children),
                ]
            )
        ]


class LicenseExpireSoonDialog(Dialog):
    def __init__(self, v_model: bool, **kwargs):
        super().__init__(**kwargs)

        self.v_model = v_model
        self.width = 500

        self.close_button = Button(icon=True, children=[v.Icon(children=["mdi-close"])])
        self.close_button.on_event("click", self.on_close)

        self.card_text = v.CardText(children=["..."])

        self.children = [
            v.Card(
                children=[
                    v.CardTitle(
                        children=[
                            "License Expiring Soon",
                            v.Spacer(),
                            self.close_button,
                        ]
                    ),
                    self.card_text,
                ]
            )
        ]

    def on_close(self, *_):
        self.v_model = False
