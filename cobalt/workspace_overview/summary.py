# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v

from cobalt.schema import CobaltDataSubset, DatasetSplit

card_class = "ma-2"
card_style = (
    "height: 225px; "
    "overflow-y: auto; "
    "border: 1px solid #E0E0E0;"
    "box-shadow: none; "
    "border-radius: 0"
)


def get_metadata_placeholder():
    return [
        v.CardTitle(class_="subtitle-1 text--secondary pa-2", children=["Metadata"]),
        v.CardText(children=["Please select a model to view metadata."]),
    ]


def format_key_name(key):
    return " ".join(word.capitalize() for word in key.replace("_", " ").split())


def format_metadata_value(value):
    if isinstance(value, list):
        if not value:
            return "Not specified"
        return ", ".join(value)
    return value if value is not None else "Not specified"


class Summary(v.Layout):
    def __init__(self, ws, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = ws
        self.split: DatasetSplit = self.ws.state.split
        self.selected_split: CobaltDataSubset = next(iter(self.split.values()))
        self.dataset_name: str = self.ws.state.dataset.name

        self.summary = self.get_summary()
        self.selected_model = None

        splits_children = [
            v.Html(
                tag="div",
                class_="mb-2",
                children=[
                    v.Html(
                        tag="div", class_="body-1 text--primary", children=[split_name]
                    ),
                    v.Html(
                        tag="div",
                        class_="body-2",
                        children=[f"Number of points: {split_data['points_number']}"],
                    ),
                ],
            )
            for split_name, split_data in self.summary["splits"].items()
        ]
        self.splits_card = v.Card(
            style_=card_style,
            class_=card_class,
            children=[
                v.CardTitle(
                    class_="subtitle-1 text--secondary pa-2", children=["Splits"]
                ),
                v.CardText(children=splits_children),
            ],
        )

        self.models_card = v.Card(
            style_=card_style,
            class_=card_class,
            children=[
                v.CardTitle(
                    class_="subtitle-1 text--secondary pa-2", children=["Models"]
                ),
                v.CardText(
                    class_="pa-0",
                    children=[
                        v.List(
                            children=[
                                v.ListItemGroup(
                                    v_model="selected_model",
                                    children=[
                                        self.create_model_list(model)
                                        for model in self.summary["models"]
                                    ],
                                )
                            ],
                        )
                    ],
                ),
            ],
        )

        self.metadata_card = v.Card(
            style_=card_style, class_=card_class, children=get_metadata_placeholder()
        )

        self.children = [
            v.Layout(
                column=True,
                children=[
                    v.Html(
                        tag="div",
                        class_="subtitle-1",
                        children=[f"Dataset: {self.dataset_name}"],
                    ),
                    v.Layout(
                        children=[
                            v.Flex(xs12=True, md4=True, children=[self.splits_card]),
                            v.Flex(xs12=True, md4=True, children=[self.models_card]),
                            v.Flex(xs12=True, md4=True, children=[self.metadata_card]),
                        ]
                    ),
                ],
            )
        ]

    def get_summary(self):
        return self.ws._get_summary()

    def update_metadata_card(self, model_name):
        model = next(
            (m for m in self.summary["models"] if m["model_name"] == model_name),
            None,
        )
        if model is not None:
            metadata_items = [
                v.Html(
                    tag="div",
                    children=[
                        f"{format_key_name(key)}: {format_metadata_value(value)}"
                    ],
                )
                for key, value in model["model_metadata"].items()
            ]
            self.metadata_card.children = [
                v.CardTitle(
                    class_="subtitle-1 text--secondary pa-2",
                    children=[f"Metadata for {model_name} "],
                ),
                v.CardText(children=metadata_items),
            ]
        else:
            self.metadata_card.children = get_metadata_placeholder()

    def create_model_list(self, model):
        model_name = (
            model["model_name"]
            if model["model_name"] is not None
            else "No models available"
        )
        list_item = v.ListItem(
            key=model_name,
            value=model_name,
            children=[v.ListItemTitle(children=[model_name])],
            clickable=True,
            class_=(
                "secondary lighten-2"
                if model_name == self.selected_model and model["model_name"] is not None
                else ""
            ),
        )
        list_item.on_event(
            "click", lambda *args, **kwargs: self.select_model(model_name)
        )
        return list_item

    def select_model(self, model_name):
        if self.selected_model == model_name:
            self.selected_model = None
            self.metadata_card.children = get_metadata_placeholder()
        else:
            self.selected_model = model_name
            self.update_metadata_card(model_name)

        self.models_card.children[1].children[0].children = [
            self.create_model_list(model) for model in self.summary["models"]
        ]
