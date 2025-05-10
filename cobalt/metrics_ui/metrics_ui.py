# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v

import cobalt.selection_manager as sm
from cobalt import CobaltDataSubset
from cobalt.config import handle_cb_exceptions
from cobalt.metrics_ui.metrics_select import MetricsSelect


def format_metric_value(raw_value):
    if raw_value is None:
        return "None"
    try:
        formatted_value = f"{float(raw_value):.2g}"
    except ValueError:
        formatted_value = str(raw_value)
    return formatted_value


def construct_ui_metric_item(name, value):
    return {"id": name, "value": f"{format_metric_value(value)}"}


class MetricsDisplay(v.Flex):
    def __init__(
        self,
        visualization,
        selection_manager,
    ):
        self.metrics = None
        self.visualization = visualization
        self.selection_manager = selection_manager

        self.metrics_select = MetricsSelect()
        self.update_metrics_select_data()

        self.selection_manager.on_group_select(self.update_group_metrics)
        self.selection_manager.on_graph_select(self.update_graph_metrics)

        super().__init__(
            class_="d-flex align-center",
            style_="width: 260px; max-width: 260px; height: 40px;",
            children=self.get_ui_blocks_to_display(),
        )

    def get_ui_blocks_to_display(self):
        if self.metrics and len(self.metrics_data_list) > 0:
            return [self.metrics_select]
        return []

    def get_selected_item(self):
        if not self.metrics_select.selected_item:
            selected_key = (
                self.metrics_data_list[0]["id"]
                if len(self.metrics_data_list) > 0
                else None
            )
        else:
            selected_key = self.metrics_select.selected_item["id"]

        self.metrics_select.selected_item = next(
            (item for item in self.metrics_data_list if item["id"] == selected_key),
            None,
        )

    def update_metrics_select_data(self):
        self.metrics_data_list = self.convert_metrics_to_list(self.metrics)
        self.metrics_select.items = self.metrics_data_list
        self.get_selected_item()

    def get_model_performance_scores(self, subset: CobaltDataSubset):
        # TODO: Incorporate multiple models here.
        return (
            subset.overall_model_performance_scores(model_index=0)
            if len(subset.models) >= 1
            else {}
        )

    @handle_cb_exceptions
    def update_group_metrics(self, selected_item):
        if selected_item:
            subset = selected_item.get("subset", None)
            if subset and len(subset):
                self.metrics = self.get_model_performance_scores(subset)
            else:
                self.metrics = None
        else:
            self.metrics = None

        self.update_metrics_select_data()
        self.children = self.get_ui_blocks_to_display()

    @handle_cb_exceptions
    def update_graph_metrics(self, selection):
        if self.selection_manager.selection_state == sm.SelectionState.initial:
            self.metrics = None
        elif len(selection):
            self.metrics = self.get_model_performance_scores(selection)
        else:
            self.metrics = None

        self.update_metrics_select_data()
        self.children = self.get_ui_blocks_to_display()

    @staticmethod
    def convert_metrics_to_list(metrics):
        if not isinstance(metrics, dict) or len(metrics) == 0:
            return []
        return [
            construct_ui_metric_item(name, value) for name, value in metrics.items()
        ]
