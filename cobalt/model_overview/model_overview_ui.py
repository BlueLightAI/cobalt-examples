# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from types import SimpleNamespace
from typing import List

import ipyvuetify as v
import pandas as pd

from cobalt.model_overview.confusion_matrix import ConfusionMatrix
from cobalt.model_overview.statistics_table import StatisticsTable
from cobalt.schema import CobaltDataSubset, DatasetSplit
from cobalt.widgets import EnhancedAutocomplete, Select, Tabs


def handle_model_errors(func):
    """Decorator to handle errors in model data fetching functions."""

    def wrapper(model, subset, selected_classes=None):
        try:
            return func(model, subset, selected_classes)
        except ValueError as e:
            str_err = str(e)
            empty_df = pd.DataFrame()
            empty_df.meta = SimpleNamespace(error=str_err)
            return empty_df

    return wrapper


@handle_model_errors
def fetch_confusion_matrix_data(model, subset, selected_classes=None):
    if selected_classes:
        return model.get_confusion_matrix(subset, selected_classes=selected_classes)
    return model.get_confusion_matrix(subset)


@handle_model_errors
def fetch_statistics_data(model, subset, selected_classes=None):
    if selected_classes is not None and len(selected_classes) == 0:
        selected_classes = None
    data = model.get_statistic_metrics(subset, selected_classes=selected_classes)
    return data.round(2)


class ModelOverview(v.Layout):
    def __init__(self, split: DatasetSplit, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.split: DatasetSplit = split
        self.selected_split: CobaltDataSubset = next(iter(self.split.values()))

        self.split_options: List[str] = self._get_split_options()
        self.selected_split_option: str = next(iter(self.split.keys()))

        self.model_options: List[str] = self._get_model_options()
        self.selected_model_option: str = (
            self.model_options[0] if self.model_options else None
        )

        if not self.model_options:
            self.children = [
                v.Layout(
                    column=True,
                    children=[
                        v.Text(class_="body-1 pa-4", children=["Model Overview"]),
                        v.Divider(),
                        v.Text(class_="body-2 pa-4", children=["No models available."]),
                    ],
                ),
            ]
            return

        self.selected_model = self._get_selected_model()
        self.confusion_matrix_data = fetch_confusion_matrix_data(
            self.selected_model, self.selected_split
        )
        self.statistics_data = fetch_statistics_data(
            self.selected_model, self.selected_split
        )

        self.selected_classes = list(self.confusion_matrix_data.columns)

        self.model_select = Select(
            label="Model",
            items=self.model_options,
            class_="mr-4",
            style_="max-width: 200px",
            v_model=self.selected_model_option,
        )
        self.model_select.on_event("change", self.on_model_select_change)

        self.classes_select = EnhancedAutocomplete(
            selected=self.selected_classes,
            items=self.selected_classes,
            multiple=True,
            label="Classes",
            placeholder="Select classes",
            class_="mr-4",
        )
        self.classes_select.on_event("change", self.on_classes_select_change)

        self.dataset_split_select = Select(
            label="Dataset Split",
            items=self.split_options,
            class_="mr-4",
            style_="max-width: 150px",
            v_model=self.selected_split_option,
        )
        self.dataset_split_select.on_event(
            "change", self.on_dataset_split_select_change
        )

        self.header = v.Layout(
            align_center=True,
            justify_space_between=True,
            class_="pa-4",
            style_="border-bottom: 1px solid #E0E0E0",
            children=[
                v.Flex(
                    xs12=True,
                    md3=True,
                    children=[v.Text(class_="body-1", children=["Model Overview"])],
                ),
                v.Flex(
                    xs12=True,
                    md9=True,
                    children=[
                        v.Layout(
                            align_center=True,
                            children=[
                                self.model_select,
                                self.classes_select,
                                self.dataset_split_select,
                            ],
                        )
                    ],
                ),
            ],
        )

        self.confusion_matrix = ConfusionMatrix(
            matrix_data=self.confusion_matrix_data,
            model_name=self.selected_model_option,
        )

        error = None
        if hasattr(self.confusion_matrix_data, "meta"):
            meta = self.confusion_matrix_data.meta
            if hasattr(meta, "error"):
                error = meta.error

        self.statistics_table = StatisticsTable(data=self.statistics_data, error=error)

        self.children = [
            self.header,
            Tabs(
                tab_data=[
                    {
                        "tab_header": "Confusion Matrix",
                        "tab_content": self.confusion_matrix,
                    },
                    {
                        "tab_header": "Statistics Table",
                        "tab_content": self.statistics_table,
                    },
                ],
                vertical=True,
            ),
        ]

    def _get_selected_model(self):
        selected_model_index = self.model_options.index(self.selected_model_option)
        return self.selected_split.models[selected_model_index]

    def _get_model_options(self):
        return [model.name for model in self.selected_split.models]

    def _get_split_options(self):
        return self.split.names

    def on_model_select_change(self, widget, event, data):
        self.selected_model_option = data
        self.selected_model = self._get_selected_model()
        self.load_confusion_matrix_data()
        self.load_statistics_data()

    def on_classes_select_change(self, data):
        self.confusion_matrix_data = fetch_confusion_matrix_data(
            self.selected_model, self.selected_split, selected_classes=data
        )
        self.statistics_data = fetch_statistics_data(
            self.selected_model, self.selected_split, selected_classes=data
        )
        self.confusion_matrix.update_matrix(
            self.confusion_matrix_data, self.selected_model_option
        )
        self.statistics_table.update_data(self.statistics_data)

    def on_dataset_split_select_change(self, widget, event, data):
        self.selected_split_option = data
        self.selected_split = self.split[self.selected_split_option]
        self.model_options = self._get_model_options()
        self.selected_model_option = (
            self.model_options[0] if self.model_options else None
        )
        self.selected_model = self._get_selected_model()
        self.load_confusion_matrix_data()
        self.load_statistics_data()
        self.update_classes_select()

    def load_confusion_matrix_data(self):
        self.confusion_matrix_data = fetch_confusion_matrix_data(
            self.selected_model, self.selected_split
        )
        self.confusion_matrix.update_matrix(
            self.confusion_matrix_data, self.selected_model_option
        )

    def load_statistics_data(self):
        self.statistics_data = fetch_statistics_data(
            self.selected_model, self.selected_split
        )
        self.statistics_table.update_data(self.statistics_data)

    def update_classes_select(self):
        self.selected_classes = list(self.confusion_matrix_data.columns)
        self.classes_select.items = self.selected_classes
        self.classes_select.v_model = self.selected_classes
