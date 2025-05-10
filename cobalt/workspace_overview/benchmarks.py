# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import io
from typing import List

import ipyvuetify as v
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import traitlets

from cobalt.schema import CobaltDataSubset, DatasetSplit


def bar_plot(data: pd.DataFrame, x: str, y: str, hue: str) -> str:
    plt.figure(figsize=(10, 3))
    ax = sns.barplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
    )

    for p in ax.patches:
        height = p.get_height()
        annotation = f"{height:.2f}"
        ax.annotate(
            annotation,
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

    ax.legend(title=hue, loc="upper right", bbox_to_anchor=(1.05, 1.25))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Convert to SVG
    output = io.StringIO()
    plt.savefig(output, format="svg", bbox_inches="tight")
    plt.close()
    output.seek(0)
    return output.getvalue()


def transform_performance_metrics_data(performance_metrics):
    """Transforms performance metrics dictionary into a pandas DataFrame.

    Args:
        performance_metrics (dict): Dictionary of models with their metrics.

    Returns:
        pd.DataFrame: DataFrame with each metric and its value per model as rows.
    """
    benchmark_data = []

    for model_name, metrics in performance_metrics.items():
        for metric_name, value in metrics.items():
            benchmark_data.append(
                {"Model": model_name, "Metric": metric_name, "Value": value}
            )

    return pd.DataFrame(benchmark_data)


class Benchmarks(v.VuetifyTemplate):
    template = traitlets.Unicode(
        """
        <v-layout column>
            <v-layout class='mb-2'>
                <v-select
                    dense
                    outlined
                    hide-details
                    multiple
                    small-chips
                    deletable-chips
                    :items="model_options"
                    v-model="selected_models"
                    label="Models"
                    class='mr-2'
                    style="width: 200px"
                    @change="on_select_model">
                </v-select>
                <v-select
                    dense
                    outlined
                    hide-details
                    multiple
                    small-chips
                    deletable-chips
                    :items="performance_metric_options"
                    v-model="selected_performance_metrics"
                    label="Performance Metrics"
                    class='mr-2'
                    style="width: 200px"
                    @change="on_select_performance_metric">
                </v-select>
                <v-select
                    dense
                    outlined
                    hide-details
                    :items="split_options"
                    v-model="selected_split_option"
                    label="Dataset Split"
                    class='mr-2'
                    style="max-width: 150px"
                    @change="on_select_split">
                </v-select>
            </v-layout>
            <v-layout>
                <div v-html="plot_svg"></div>
            </v-layout>
        </v-layout>
    """
    ).tag(sync=True)

    model_options = traitlets.List().tag(sync=True)
    selected_models = traitlets.List(default_value=[]).tag(sync=True, allow_null=True)

    performance_metric_options = traitlets.List().tag(sync=True)
    selected_performance_metrics = traitlets.List(default_value=[]).tag(
        sync=True, allow_null=True
    )

    split_options = traitlets.List().tag(sync=True)
    selected_split_option = traitlets.Unicode("").tag(sync=True)

    plot_svg = traitlets.Unicode().tag(sync=True)

    def __init__(self, ws, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = ws
        self.split: DatasetSplit = self.ws.state.split
        self.selected_split: CobaltDataSubset = next(iter(self.split.values()))

        self.split_options: List[str] = self._get_split_options()
        self.selected_split_option: str = next(iter(self.split.keys()))

        self.model_options: List[str] = self._get_model_options()

        self.performance_metrics: dict = (
            self.selected_split.get_model_performance_metrics()
        )
        self.performance_metric_options: List[str] = (
            self._get_performance_metric_options(self.model_options)
        )

        self.update_model_and_metric_options_based_on_split(self.selected_split_option)

        self.plot_data = transform_performance_metrics_data(self.performance_metrics)
        self.update_plot(self.plot_data)

    def vue_on_select_model(self, new_models):
        self.selected_models = new_models
        self.refresh_plot_based_on_selections()

    def vue_on_select_performance_metric(self, new_metrics):
        self.selected_performance_metrics = new_metrics
        self.refresh_plot_based_on_selections()

    def vue_on_select_split(self, new_split):
        self.update_model_and_metric_options_based_on_split(new_split)
        self.refresh_plot_based_on_selections()

    def update_model_and_metric_options_based_on_split(self, split_key: str):
        self.selected_split = self.split[split_key]
        self.performance_metrics = self.selected_split.get_model_performance_metrics()
        self.model_options = self._get_model_options()
        self.selected_models = self.model_options

        self.performance_metric_options = self._get_performance_metric_options(
            self.selected_models
        )
        self.selected_performance_metrics = self.performance_metric_options

    def get_selected_metric_values(self):
        filtered_metrics = {}
        for model in self.selected_models:
            model_metrics = self.performance_metrics.get(model, {})
            filtered_metrics[model] = {
                metric: value
                for metric, value in model_metrics.items()
                if metric in self.selected_performance_metrics
            }
        return filtered_metrics

    def refresh_plot_based_on_selections(self):
        filtered_metrics = self.get_selected_metric_values()

        self.plot_data = transform_performance_metrics_data(filtered_metrics)
        self.update_plot(self.plot_data)

    def update_plot(self, plot_data):
        if plot_data.empty:
            self.plot_svg = """
                    <div
                        style='display: flex;
                               align-items: center;
                               justify-content: center;'
                    >
                        <span style='font-size: 18px' class='ml-2'>
                            No data available.
                            Please select at least one model and one performance metric.
                        </span>
                    </div>
                    """
        else:
            self.plot_svg = bar_plot(data=plot_data, x="Metric", y="Value", hue="Model")

    def _get_model_options(self):
        return [model.name for model in self.selected_split.models]

    def _get_performance_metric_options(self, selected_model_names: List[str]):
        """Retrieves a list of performance metrics that are common across the selected models.

        Args:
            selected_model_names (list of str): The names of the models.

        Returns:
            list: A list of common performance metrics across the selected models, or an
            empty list if no common metrics are found.
        """
        selected_metrics = {
            model_name: self.performance_metrics.get(model_name, {})
            for model_name in selected_model_names
        }

        metric_sets = [set(metrics.keys()) for metrics in selected_metrics.values()]
        common_metrics = set.intersection(*metric_sets) if metric_sets else set()
        return list(common_metrics)

    def _get_split_options(self):
        return self.split.names
