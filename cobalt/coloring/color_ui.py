# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import io
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Union

import ipyvuetify as v
import ipywidgets as w
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from cobalt.cobalt_types import ColumnDataType, GroupType
from cobalt.coloring.color_range import ColorRange
from cobalt.coloring.color_spec import (
    CategoricalColoringSpec,
    ColorMapOptions,
    GraphColoringSpec,
    NoColoringSpec,
    NumericColoringSpec,
    TimestampColoringSpec,
)
from cobalt.coloring.histogram_dialog import HistogramDialog
from cobalt.config import handle_cb_exceptions
from cobalt.repositories.run_repository import GroupResultsCollection
from cobalt.schema.dataset import CobaltDataset, CobaltDataSubset
from cobalt.state import State
from cobalt.ui_utils import with_tooltip
from cobalt.visualization import EmbeddingVisualization
from cobalt.widgets import Button, SearchableSelect, Select


def build_select_list(options: Sequence[str]):
    return [{"text": cmap_value} for cmap_value in options]


CATEGORICAL_COLORMAP_OPTIONS = build_select_list(ColorMapOptions.CATEGORICAL)
NUMERICAL_COLORMAP_OPTIONS = build_select_list(ColorMapOptions.NUMERICAL)
OTHER_COLORMAP_OPTIONS = build_select_list(ColorMapOptions.UNKNOWN)
COMBINED_COLORMAP_OPTIONS = [
    {"header": "Continuous"},
    *NUMERICAL_COLORMAP_OPTIONS,
    {"divider": True},
    {"header": "Categorical"},
    *CATEGORICAL_COLORMAP_OPTIONS,
]


class ColoringOption(ABC):

    @abstractmethod
    def get_color_spec(
        self, subset: CobaltDataSubset, config: dict
    ) -> GraphColoringSpec:
        pass

    @abstractmethod
    def get_histogram(self, subset: CobaltDataSubset) -> Optional[str]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name to display for this coloring option."""

    @property
    @abstractmethod
    def cmap_options(self):
        pass

    @property
    @abstractmethod
    def allow_color_range(self) -> bool:
        pass


class ColumnColoringOption(ColoringOption):
    def __init__(self, dataset: CobaltDataset, column: str):
        self.dataset = dataset
        self.column = column
        self.column_type = dataset.metadata.data_types[column]

    def get_color_spec(
        self, subset: CobaltDataSubset, config: dict
    ) -> GraphColoringSpec:
        # TODO: bettter way to handle config
        # TODO: type checking of color map name?
        f = subset.select_col(self.column)
        color_map_name = config.pop("color_map_name")
        if is_datetime64_any_dtype(f.dtype):
            config.pop("range_type", None)
            return TimestampColoringSpec(f, color_map_name, **config)
        if color_map_name in ColorMapOptions.CATEGORICAL:
            config.pop("range_type", None)
            return CategoricalColoringSpec(f, color_map_name, **config)
        else:
            return NumericColoringSpec(f, color_map_name, **config)

    def get_histogram(self, subset: CobaltDataSubset) -> Optional[str]:
        data = subset.select_col(self.column)
        if is_numeric_dtype(data.dtype):
            return create_histogram_svg(data.astype(float))
        return None

    @property
    def name(self) -> str:
        return self.column

    @property
    def cmap_options(self):
        # TODO: use precomputed dataset metadata
        data = self.dataset.select_col(self.column)
        if is_numeric_dtype(data.dtype):
            options_data = COMBINED_COLORMAP_OPTIONS
        elif is_datetime64_any_dtype(data.dtype):
            options_data = NUMERICAL_COLORMAP_OPTIONS
        elif isinstance(data.dtype, pd.api.types.CategoricalDtype):
            options_data = CATEGORICAL_COLORMAP_OPTIONS
        else:
            options_data = OTHER_COLORMAP_OPTIONS

        if options_data and "header" in options_data[0]:
            default_selection = options_data[1]["text"]
        elif options_data:
            default_selection = options_data[0]["text"]

        return options_data, default_selection

    @property
    def allow_color_range(self) -> bool:
        return True


class NoColoringOption(ColoringOption):
    def get_color_spec(
        self, subset: CobaltDataSubset, config: dict
    ) -> GraphColoringSpec:
        return NoColoringSpec()

    def get_histogram(self, subset: CobaltDataSubset) -> Optional[str]:
        return None

    @property
    def name(self) -> str:
        return NO_COLORING_OPTION

    @property
    def cmap_options(self):
        return [], None

    @property
    def allow_color_range(self) -> bool:
        return False


class DriftScoreColoringOption(ColoringOption):
    def __init__(
        self,
        baseline: CobaltDataSubset,
        comparison: CobaltDataSubset,
        baseline_name: str,
        comparison_name: str,
    ):
        self.baseline = baseline
        self.comparison = comparison
        self.baseline_name = baseline_name
        self.comparison_name = comparison_name

    def get_color_spec(
        self, subset: CobaltDataSubset, config: dict
    ) -> GraphColoringSpec:
        # TODO: restore graph smoothing? not sure it's worth the complexity
        f = np.zeros(len(subset))
        f -= subset.intersect(self.baseline).as_mask_on(subset)
        f += subset.intersect(self.comparison).as_mask_on(subset)
        f_series = pd.Series(f, name=self.name)
        config.pop("cmap_range", None)
        config.pop("range_type", None)

        spec = NumericColoringSpec(
            f_series,
            cmap_range=(-1, 1),
            node_label_gen=get_node_drift_score_labeler(f),
            **config,
        )
        return spec

    def get_histogram(self, subset: CobaltDataSubset) -> Optional[str]:
        return None

    @property
    def name(self) -> str:
        return f"{self.baseline_name} vs {self.comparison_name}"

    @property
    def cmap_options(self):
        options_data = NUMERICAL_COLORMAP_OPTIONS

        return options_data, ColorMapOptions.NUMERICAL[0]

    @property
    def allow_color_range(self) -> bool:
        return False


class GroupResultsCollectionColoringOption(ColoringOption):
    def __init__(self, results: GroupResultsCollection):
        self.results = results
        dataset = self.results.source_data.source_dataset
        f = np.full(len(dataset), "no group", dtype=object)
        for group in self.results.groups:
            f[group.subset.indices] = group.name
        self.f = f

    def get_color_spec(
        self, subset: CobaltDataSubset, config: dict
    ) -> GraphColoringSpec:
        f = self.f[subset.indices]
        f_series = pd.Series(f, name=self.name, dtype="category")
        config.pop("cmap_range", None)
        config.pop("range_type", None)
        color_map_name = config.pop("color_map_name")
        spec = CategoricalColoringSpec(
            f_series,
            color_map_name,
            **config,
        )
        return spec

    def get_histogram(self, subset: CobaltDataSubset) -> Optional[str]:
        return None

    @property
    def name(self) -> str:
        return f"{self.results.name}"

    @property
    def cmap_options(self):
        options_data = CATEGORICAL_COLORMAP_OPTIONS

        return options_data, ColorMapOptions.CATEGORICAL[0]

    @property
    def allow_color_range(self) -> bool:
        return True


class MetricColoringOption(ColoringOption):
    def __init__(self, metric, model_name: str):
        self.metric = metric
        self.model_name = model_name

    def get_color_spec(
        self, subset: CobaltDataSubset, config: dict
    ) -> GraphColoringSpec:
        f = self.metric.calculate(subset).get(self.metric.get_key())
        f.name = self.name
        color_map_name = config.pop("color_map_name", ColorMapOptions.NUMERICAL[0])
        config.pop("repeat_colors", None)

        spec = NumericColoringSpec(f, color_map_name=color_map_name, **config)
        return spec

    def get_histogram(self, subset: CobaltDataSubset) -> Optional[str]:
        return None

    @property
    def name(self) -> str:
        return f"{self.model_name}_{self.metric.get_key()}"

    @property
    def cmap_options(self):
        options_data = NUMERICAL_COLORMAP_OPTIONS

        return options_data, ColorMapOptions.NUMERICAL[0]

    @property
    def allow_color_range(self) -> bool:
        return True


NO_COLORING_OPTION = "No coloring"


def create_histogram_svg(column_data: pd.Series) -> str:
    fig, ax = plt.subplots()
    ax.hist(column_data, bins=20)

    # Convert to SVG
    output = io.BytesIO()
    plt.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)
    output.seek(0)
    svg_data = output.getvalue().decode("utf-8")

    return svg_data


def get_coloring_options(
    state: State,
    extra_metadata_columns: Optional[List[str]],
    extra_data_columns: Optional[List[str]],
    enable_drift_score: bool,
) -> Dict[str, List[ColoringOption]]:
    """Populate the collection of coloring options for this dataset.

    Returns a dict of lists by category.
    """
    coloring_options: Dict[str, List[ColoringOption]] = defaultdict(list)
    coloring_options["Coloring Options"] = [NoColoringOption()]
    if enable_drift_score:
        coloring_options["Drift score"] = [
            DriftScoreColoringOption(split_1, split_2, split_name_1, split_name_2)
            for (split_name_1, split_1), (
                split_name_2,
                split_2,
            ) in state.split.comparable_subset_pairs
        ]

    column_types = state.dataset.metadata.data_types
    for model in state.dataset.models:
        metrics = model.evaluation_metrics
        coloring_options["Performance metrics"].extend(
            [MetricColoringOption(metrics[metric], model.name) for metric in metrics]
        )

        coloring_options["Outcome"].extend(
            [ColumnColoringOption(state.dataset, col) for col in model.outcome_columns]
        )
        coloring_options["Prediction"].extend(
            [
                ColumnColoringOption(state.dataset, col)
                for col in model.prediction_columns
            ]
        )

    coloring_options["Timestamp"].extend(
        [
            ColumnColoringOption(state.dataset, col)
            for col in state.dataset.metadata.timestamp_columns
        ]
    )
    if extra_metadata_columns:
        coloring_options["Metadata"].extend(
            [
                ColumnColoringOption(state.dataset, col)
                for col in extra_metadata_columns
                if column_types[col].is_categorical
                or column_types[col].col_type == ColumnDataType.numerical
            ]
        )
    if extra_data_columns:
        coloring_options["Data columns"] = [
            ColumnColoringOption(state.dataset, col)
            for col in extra_data_columns
            if column_types[col].is_categorical
            or column_types[col].col_type == ColumnDataType.numerical
        ]
    clusters = state.run_repo.get_runs(group_type=GroupType.cluster, run_visible=True)
    if clusters:
        coloring_options["Clustering results"] = [
            GroupResultsCollectionColoringOption(run) for run in clusters.values()
        ]

    coloring_options = defaultdict(
        list,
        {cat: options for cat, options in coloring_options.items() if len(options) > 0},
    )

    return coloring_options


def get_name_to_options_dict(
    coloring_options: Dict[str, List[ColoringOption]]
) -> Dict[str, ColoringOption]:
    name_to_options_dict: Dict[str, ColoringOption] = {}
    for options in coloring_options.values():
        for option in options:
            name_to_options_dict[option.name] = option

    return name_to_options_dict


def build_vuetify_options_list(
    coloring_options: Dict[str, List[ColoringOption]]
) -> List[Dict[str, Union[str, bool]]]:
    coloring_options_data: List[Dict[str, Union[str, bool]]] = []
    # TODO: figure out how to avoid collisions between column names and
    # other coloring option names
    for header, options in coloring_options.items():
        if len(options) > 0:
            coloring_options_data.append({"header": header})
            for option in options:
                coloring_options_data.append({"text": option.name})
            coloring_options_data.append({"divider": True})
    return coloring_options_data


def get_initial_coloring_option(
    coloring_options: Dict[str, List[ColoringOption]]
) -> str:
    """Chooses the default coloring option for the graph."""
    # model performance metric
    if coloring_options["Performance metrics"]:
        # The first entry section will be an error metric
        return coloring_options["Performance metrics"][0].name
    # model predictions
    if coloring_options["Prediction"]:
        return coloring_options["Prediction"][0].name
    # model ground truth
    if coloring_options["Outcome"]:
        return coloring_options["Outcome"][0].name
    # drift score
    if coloring_options["Drift score"]:
        return coloring_options["Drift score"][0].name
    # autogroup clusters
    if coloring_options["Clustering results"]:
        return coloring_options["Clustering results"][0].name
    # first nontrivial coloring option
    for opts in coloring_options.values():
        if opts and opts[0].name != NO_COLORING_OPTION:
            return opts[0].name

    # if all else fails
    return NO_COLORING_OPTION


class ColoringConfig(v.Flex):
    def __init__(
        self,
        visualization: EmbeddingVisualization,
        extra_data_columns: Optional[List[str]] = None,
        extra_metadata_columns: Optional[List[str]] = None,
        enable_drift_score: bool = False,
        **kwargs,
    ):
        super().__init__(class_="d-flex", style_="max-width: 160px", **kwargs)

        self.connector = visualization.selection_manager
        self.dataset = visualization.state.dataset
        self.visualization = visualization
        state = visualization.state

        coloring_options = get_coloring_options(
            state, extra_metadata_columns, extra_data_columns, enable_drift_score
        )
        self.coloring_options_dict = get_name_to_options_dict(coloring_options)

        coloring_options_data = build_vuetify_options_list(coloring_options)

        initial_option = state.coloring_menu.get("selector_dropdown_value")
        if not initial_option:
            initial_option = get_initial_coloring_option(coloring_options)
            self.visualization.state.coloring_menu["selector_dropdown_value"] = (
                initial_option
            )

        self.selector_dropdown = SearchableSelect(
            label="Color by",
            items=coloring_options_data,
            v_model=initial_option,
            class_="mt-2",
        )
        self.selector_dropdown.on_event("change", self.on_selector_dropdown_change)

        cmap_options_data, v_model = self.get_cmap_options(
            self.selector_dropdown.v_model
        )
        self.selector_cmap = Select(
            label="Color map", items=cmap_options_data, v_model=v_model
        )
        self.selector_cmap.on_event("change", self.apply_coloring)

        self.color_range_ui = ColorRange(
            value_changed_callback=self.apply_coloring,
            color_by=self.selector_dropdown.v_model,
        )
        self.color_range_ui.hide()

        self.repeat_colors_checkbox = v.Checkbox(
            label="Repeat colors",
            v_model=False,
            dense=True,
            hide_details=True,
            class_="ma-0 pa-0",
        )

        self.repeat_colors_checkbox.hide()
        self.repeat_colors_checkbox.on_event("change", self.apply_coloring)

        color_by_btn = v.Html(
            tag="div",
            children=[
                with_tooltip(
                    Button(
                        class_="pa-0",
                        v_on="props.on",
                        height="40px",
                        min_width="40px",
                        children=[
                            v.Icon(
                                children=["mdi-palette"],
                                style_=":hover { color: rgba(41, 98, 255, 1); }",
                            )
                        ],
                    ),
                    "Graph Coloring",
                )
            ],
        )

        self.coloring_popup = v.Menu(
            top=False,
            left=True,
            attach=True,
            close_on_content_click=False,
            offset_y=True,
            v_slots=[
                {
                    "name": "activator",
                    "variable": "props",
                    "children": color_by_btn,
                },
            ],
            children=[
                v.Card(
                    class_="d-flex flex-column pa-3",
                    width="288",
                    children=[
                        self.selector_dropdown,
                        self.selector_cmap,
                        self.color_range_ui,
                        self.repeat_colors_checkbox,
                    ],
                    style_="gap: 12px;",
                )
            ],
            style_="z-index: 12;",
        )

        self.legend = v.Flex(
            class_="flex-grow-0 overflow-x-hidden",
            style_="max-height: 400px; z-index: 2;",
        )

        self.color_histogram_dialog = HistogramDialog(
            v_model=False,
            max_width="800px",
            children=[],
            on_close=self.close_color_histogram_dialog,
        )

        self.color_histogram_btn = Button(
            class_="pa-0 d-none",
            height="40px",
            min_width="40px",
            children=[v.Icon(children=["mdi-chart-histogram"])],
        )
        self.color_histogram_btn.on_event("click", self.open_color_histogram_dialog)
        self.color_histogram_btn_with_tooltip = with_tooltip(
            self.color_histogram_btn, "Color Histogram"
        )

        self.widgets = [
            v.Flex(
                class_="d-flex flex-column align-end justify-end",
                children=[
                    v.Html(
                        tag="div",
                        class_="d-flex justify-end",
                        children=[
                            self.color_histogram_btn_with_tooltip,
                            self.coloring_popup,
                        ],
                        style_="gap: 6px",
                    ),
                    self.legend,
                    self.color_histogram_dialog,
                ],
                style_="gap: 6px",
            ),
        ]

        self.children = self.widgets

        # this also runs self.apply_coloring()
        self.update_cmap_options()
        self.visualization.on_graph_update(self.apply_coloring)

    def on_selector_dropdown_change(self, *_):
        self.update_cmap_options()
        self.color_range_ui.reset()
        self.color_range_ui.is_auto = True
        self.color_range_ui.auto_checkbox.v_model = True
        self.apply_coloring()
        self.visualization.state.coloring_menu["selector_dropdown_value"] = (
            self.selector_dropdown.v_model
        )

    @handle_cb_exceptions
    def update_cmap_options(self, *_):
        new_options, v_model = self.get_cmap_options(self.selector_dropdown.v_model)
        self.selector_cmap.items = new_options
        if new_options:
            self.selector_cmap.v_model = v_model
            self.apply_coloring()

    def get_cmap_options(self, column: str):
        coloring_option = self.coloring_options_dict[column]
        options_data, default_selection = coloring_option.cmap_options

        previous_selected_column = self.visualization.state.coloring_menu.get(
            "selector_dropdown_value"
        )
        if previous_selected_column == column:
            v_model = self.visualization.state.coloring_menu.get("selector_cmap_value")
        else:
            v_model = default_selection

        return options_data, v_model

    @handle_cb_exceptions
    def apply_coloring(self, *_):
        if not hasattr(self.visualization, "source_data"):
            return

        option_name = self.selector_dropdown.v_model
        coloring_option = self.coloring_options_dict[option_name]

        source_data = self.visualization.source_data

        color_map_name = self.selector_cmap.v_model
        if not color_map_name:
            color_map_name = coloring_option.cmap_options[1]
            self.selector_cmap.v_model = color_map_name
        color_map_range = self.color_range_ui.get_coloring_range()
        range_type = "absolute" if color_map_range is not None else "auto"
        config = {
            "color_map_name": color_map_name,
            "cmap_range": color_map_range,
            "range_type": range_type,
        }
        if color_map_name in ColorMapOptions.CATEGORICAL:
            config["repeat_colors"] = self.repeat_colors_checkbox.v_model

        color_spec = coloring_option.get_color_spec(source_data, config)
        self.visualization.landscape.set_coloring_spec(color_spec)

        if not coloring_option.allow_color_range:
            self.color_range_ui.hide()
        else:
            self.update_color_range_ui(color_spec)

        self.legend.children = [self.visualization.landscape.coloring_legend]

        self.plot_feature_histogram(coloring_option, source_data)

        self.visualization.state.coloring_menu["selector_cmap_value"] = (
            self.selector_cmap.v_model
        )

    def plot_feature_histogram(
        self, coloring_option: ColoringOption, source_data: CobaltDataSubset
    ):
        hist = coloring_option.get_histogram(source_data)
        if not hist:
            self.color_histogram_dialog.children = []
            self.color_histogram_btn.class_list.add("d-none")
        else:
            image_widget = w.HTML(value=f"<div>{hist}</div>")

            self.color_histogram_dialog.add_histogram(
                title=f"Histogram for {coloring_option.name}", histogram=image_widget
            )
            self.color_histogram_btn.class_list.remove("d-none")

    # TODO: factor this into ColoringOption?
    def update_color_range_ui(self, coloring_config: GraphColoringSpec):
        if isinstance(coloring_config, NumericColoringSpec):
            if self.color_range_ui.default_cmap_range is None:
                self.color_range_ui.set_default_range(
                    (coloring_config.min_val, coloring_config.max_val),
                )
                self.color_range_ui.set_boundaries_range(
                    (coloring_config.lower_bound, coloring_config.upper_bound)
                )
            elif self.color_range_ui.color_by != self.selector_dropdown.v_model:
                self.color_range_ui.set_default_range(
                    (coloring_config.min_val, coloring_config.max_val),
                )
                self.color_range_ui.set_boundaries_range(
                    (coloring_config.lower_bound, coloring_config.upper_bound)
                )
                self.color_range_ui.color_by = self.selector_dropdown.v_model
            self.color_range_ui.show()
        else:
            self.color_range_ui.hide()

        if isinstance(coloring_config, CategoricalColoringSpec):
            self.repeat_colors_checkbox.show()
        else:
            self.repeat_colors_checkbox.hide()

    def open_color_histogram_dialog(self, *_):
        self.color_histogram_dialog.v_model = True

    def close_color_histogram_dialog(self, *_):
        self.color_histogram_dialog.v_model = False

    def refresh_layout(self, *_):
        widgets = [*self.widgets]
        self.children = widgets


def get_node_drift_score_labeler(raw_drift_labels: np.ndarray):
    def node_drift_score_label(u: np.ndarray, f_vals: pd.Series, node_val: float):
        f = raw_drift_labels[u]
        production = np.sum(f == 1)
        baseline = np.sum(f == -1)
        label = f"{len(u)} points, {baseline} baseline, {production} comparison"
        return label

    return node_drift_score_label
