# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import warnings
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import ipyvuetify as v
import ipywidgets as w
import numpy as np
import pandas as pd
from numba import njit
from pandas.api.types import CategoricalDtype, is_datetime64_any_dtype, is_numeric_dtype

from cobalt.coloring.color_map import ColorMap

# # ignore numpy percentile's interpolation deprecation warning
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="cobalt.coloring.color_spec"
)


class ColorMapOptions:
    DRIFT_SCORE = (
        "cividis",
        "viridis",
        "plasma",
        "inferno",
        "magma",
    )
    NUMERICAL = (
        "cividis",
        "viridis",
        "plasma",
        "inferno",
        "magma",
    )
    CATEGORICAL = (
        "tab10",
        "tab20",
        "tab20b",
        "tab20c",
    )
    UNKNOWN = (
        "tab20",
        "tab10",
    )

    @classmethod
    def top_categories_map(cls, cmap_name):
        categorical_cmaps = {
            "tab10": 10,
            "tab20": 20,
            "tab20b": 20,
            "tab20c": 20,
        }
        categorical_values = ColorMapOptions.CATEGORICAL + ColorMapOptions.UNKNOWN

        if cmap_name in categorical_values:
            return categorical_cmaps.get(cmap_name, 10)
        else:
            return 10


def make_spot_div(color: str):
    return v.Html(
        tag="div",
        children=[],
        style_="width: 14px; height:14px; border-radius: 50%;"
        f"border: 0px; flex-shrink: 0; background-color: {color};",
    )


def get_outlier_free_bounds(
    x: np.ndarray,
    max_iters: int = 4,
    initial_truncation_percentile: float = 2.0,
):
    """Finds a truncated maximum and minimum for the values in x.

    Makes sure tha the maximum and minimum are distinct if possible, but
    otherwise tries to eliminate the effect of outliers.
    """
    # tries to get a reasonable upper/lower bound by truncating outliers
    # as long as there is still some variation between them
    # TODO(Jakob): write a version of this that looks more closely at the distribution
    # TODO(Jakob): this may scale poorly with very large arrays

    truncation_size = initial_truncation_percentile
    for _ in range(max_iters):
        min_val, max_val = np.percentile(
            x,
            [truncation_size, 100 - truncation_size],
            method="nearest",
        )
        if min_val < max_val:
            break
        truncation_size = truncation_size / 2
    else:
        min_val = x.min()
        max_val = x.max()

    return min_val, max_val


@njit
def sum_on_subsets(f: np.ndarray, subset_membership: np.ndarray):
    n_subsets = np.max(subset_membership) + 1
    sums = np.zeros(n_subsets)
    counts = np.zeros(n_subsets, dtype=np.int32)
    for i in range(len(subset_membership)):
        if np.isnan(f[i]):
            continue
        subset = subset_membership[i]
        sums[subset] += f[i]
        counts[subset] += 1
    return sums, counts


def get_membership_vec(subsets: List[np.ndarray], n_points: int):
    membership_vec = np.zeros(n_points, dtype=np.int32)
    for i, u in enumerate(subsets):
        membership_vec[u] = i
    return membership_vec


@njit
def mode_on_subsets(f: np.ndarray, subset_membership: np.ndarray):
    n_subsets = np.max(subset_membership) + 1
    n_cats = np.max(f) + 1
    counts = np.zeros((n_subsets, n_cats), dtype=np.int32)
    for i in range(len(subset_membership)):
        subset = subset_membership[i]
        counts[subset, f[i]] += 1
    modes = np.argmax(counts, axis=1)
    mode_counts = np.empty_like(modes)
    for i in range(len(modes)):
        mode_counts[i] = counts[i, modes[i]]
    return modes, mode_counts


class GraphColoringSpec(ABC):
    """Contains all the data necessary to color the nodes of a graph at any resolution."""

    # TODO: maybe the right interface is to color any graph based on a given dataset
    # TODO: what is the right output? a dict of properties of the graph that can be set?
    # e.g. node colors, node labels, node borders, node shapes
    # also selection state?
    # TODO: is this really the right place to create the legend?

    rich_node_label_template: str

    @abstractmethod
    def apply_to_nodes(
        self, nodes: List[np.ndarray], use_rich_labels: bool = False
    ) -> Tuple[List[str], Union[Dict[int, str], Dict[int, dict]], w.Widget]:
        """Produce a coloring for a given list of nodes.

        The list should give the data point ids corresponding to each node.
        This produces the following:

            - a list of HTML color strings for each node
            - a dictionary {node_id: text} giving text to display when the user hovers over nodes
            - a coloring legend that can be rendered as an ipywidget
        """


# TODO: return legend information without creating the ipywidget here
# the UI should be separated from this coloring logic class


class NumericColoringSpec(GraphColoringSpec):
    aggregation_functions: ClassVar[Dict[str, Callable[[pd.Series], float]]] = {
        "mean": np.nanmean,
        "median": np.nanmedian,
    }

    coloring_type = "numeric"

    _default_rich_node_label_template = (
        "<div><div>${n_points} points</div>"
        "<div>${feat_name} mean = ${node_val}</div></div>"
    )

    def __init__(
        self,
        f: pd.Series,
        color_map_name: str,
        aggregation_method: str = "mean",
        cmap_range: Optional[Tuple[float, float]] = None,
        range_type: Literal["absolute", "percentile", "stdev", "auto"] = "absolute",
        node_label_gen: Optional[Callable[[np.ndarray, pd.Series, Any], str]] = None,
        rich_node_label_gen: Optional[
            Callable[[np.ndarray, pd.Series, Any], dict]
        ] = None,
        rich_node_label_template: Optional[str] = None,
    ):
        if not is_numeric_dtype(f.dtype):
            raise ValueError("Provided coloring function must have numeric type")
        self.f = f
        self.node_label_gen = node_label_gen or self._default_node_label_gen
        self.rich_node_label_gen = (
            rich_node_label_gen or self._default_rich_node_label_gen
        )
        self.rich_node_label_template = (
            rich_node_label_template or self._default_rich_node_label_template
        )
        self.color_map_name = color_map_name
        self.color_agg_method = aggregation_method
        self.cmap_range = cmap_range
        self.range_type = range_type
        self.transformed_f = f
        self.min_val = None
        self.max_val = None
        self.lower_bound = None
        self.upper_bound = None

    def get_node_values(self, nodes: List[np.ndarray]) -> np.ndarray:
        f = self.transformed_f
        agg_fn = self.aggregation_functions.get(self.color_agg_method)
        if agg_fn is None:
            raise NotImplementedError("unsupported node aggregation method")

        if self.color_agg_method == "mean":
            membership_vec = get_membership_vec(nodes, len(f))
            sums, counts = sum_on_subsets(f.to_numpy(), membership_vec)
            # Handle missing values
            node_values = np.divide(
                sums,
                counts,
                out=np.full_like(sums, np.nan),
                where=(counts != 0) & ~np.isnan(sums) & ~np.isnan(counts),
            )
        else:
            node_values = np.array([agg_fn(f.iloc[u]) for u in nodes])
        return node_values

    def set_min_max_val(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def set_column_boundaries(self):
        self.lower_bound = self.f.min()
        self.upper_bound = self.f.max()

    def get_min_max_val(self, node_values: np.ndarray) -> Tuple[float, float]:
        non_null_node_values = node_values[~np.isnan(node_values)]
        if len(non_null_node_values) == 0:
            return np.nan, np.nan

        if self.cmap_range is not None:
            if self.range_type == "absolute":
                min_val, max_val = map(np.float32, self.cmap_range)

            elif self.range_type == "percentile":
                min_val, max_val = np.percentile(non_null_node_values, self.cmap_range)

            elif self.range_type == "stdev":
                mean = np.mean(non_null_node_values)
                std = np.std(non_null_node_values)
                min_val = mean + self.cmap_range[0] * std
                max_val = mean + self.cmap_range[1] * std

            elif self.range_type != "auto":
                raise ValueError(f"invalid range_type={self.range_type}")

        elif self.range_type == "auto":
            min_val, max_val = get_outlier_free_bounds(non_null_node_values)

        else:
            min_val = non_null_node_values.min()
            max_val = non_null_node_values.max()

        return min_val, max_val

    def apply_to_nodes(
        self, nodes: List[np.ndarray], use_rich_labels: bool = False
    ) -> Tuple[List[str], Union[Dict[int, str], Dict[int, dict]], w.Widget]:
        f = self.transformed_f
        node_labels = {}
        node_values = self.get_node_values(nodes)

        if self.node_label_gen and not use_rich_labels:
            node_labels = {
                i: self.node_label_gen(u, f, node_values[i])
                for i, u in enumerate(nodes)
            }
        elif self.rich_node_label_gen and use_rich_labels:
            node_labels = {
                i: self.rich_node_label_gen(u, f, node_values[i])
                for i, u in enumerate(nodes)
            }

        min_val, max_val = self.get_min_max_val(node_values)

        self.set_min_max_val(min_val, max_val)
        self.set_column_boundaries()

        if min_val != max_val:

            def normalize(x: np.ndarray):
                x = x.astype(np.float32)
                return (x - min_val) / (max_val - min_val)

            cmap = ColorMap(self.color_map_name, normalize=normalize)
        else:
            cmap = ColorMap(self.color_map_name)

        node_colors = cmap(node_values)
        legend = self.construct_legend(min_val, max_val, cmap, f.name)

        return node_colors, node_labels, legend

    @staticmethod
    def _default_node_label_gen(
        u: np.ndarray, f_vals: pd.Series, node_val: float
    ) -> str:
        return f"{len(u)} points, {f_vals.name} mean: {node_val:.2f}"

    @staticmethod
    def _default_rich_node_label_gen(
        u: np.ndarray, f_vals: pd.Series, node_val: float
    ) -> dict:
        return {
            "n_points": len(u),
            "feat_name": f_vals.name,
            "node_val": f"{node_val:.2f}",
        }

    @staticmethod
    def construct_legend(
        min_val: float, max_val: float, cmap: ColorMap, name: str
    ) -> Callable[[], w.Widget]:
        color_vals = cmap(np.linspace(min_val, max_val, 20))
        css_color_string = f"linear-gradient(to top, {', '.join(color_vals)})"

        def make_widget():
            gradient_bar = v.Html(
                tag="div",
                class_="d-flex flex-shrink-0 flex-grow-0",
                style_=f"background: {css_color_string}; width: 8px; height: 155px",
            )
            bounds = v.Html(
                tag="div",
                class_="pa-0 d-flex flex-column justify-space-between",
                children=[
                    v.Row(
                        class_="d-flex flex-grow-0 flex-shrink-0",
                        no_gutters=True,
                        children=[f"{max_val:.4g}"],
                    ),
                    v.Row(
                        class_="d-flex flex-grow-0 flex-shrink-0",
                        no_gutters=True,
                        children=[f"{min_val:.4g}"],
                    ),
                ],
            )

            legend = v.Html(
                tag="div",
                class_="pb-0 pt-3 ma-0",
                children=[
                    v.Row(
                        class_="ma-3 d-flex flex-nowrap",
                        no_gutters=True,
                        children=[gradient_bar, bounds],
                        style_="gap: 12px; line-height: 15px;",
                    ),
                    wrap_text_with_tooltip(
                        truncate_str(name), f"Colored by: {name}", "--v-textBackground "
                    ),
                ],
                style_="border: 1px solid #E0E0E0; border-radius: 4px;" "width: 100px;",
            )
            return legend

        return make_widget


class TimestampColoringSpec(GraphColoringSpec):
    aggregation_functions: ClassVar[Dict[str, Callable[[pd.Series], float]]] = {
        "mean": np.mean,
        "median": np.median,
    }

    coloring_type = "timestamp"

    _default_rich_node_label_template = (
        "<div><div>${n_points} points</div>"
        "<div>${feat_name} mean = ${node_val}</div></div>"
    )

    def __init__(
        self,
        f: pd.Series,
        color_map_name: str,
        aggregation_method: str = "mean",
        cmap_range: Optional[Tuple[float, float]] = None,
        node_label_gen: Optional[Callable[[np.ndarray, pd.Series, Any], str]] = None,
        rich_node_label_gen: Optional[
            Callable[[np.ndarray, pd.Series, Any], dict]
        ] = None,
        rich_node_label_template: Optional[str] = None,
    ):
        if not is_datetime64_any_dtype(f):
            raise ValueError("Provided coloring function must have timestamp type")
        self.f = f
        self.node_label_gen = node_label_gen or self._default_node_label_gen
        self.rich_node_label_gen = (
            rich_node_label_gen or self._default_rich_node_label_gen
        )
        self.rich_node_label_template = (
            rich_node_label_template or self._default_rich_node_label_template
        )
        self.color_map_name = color_map_name
        self.color_agg_method = aggregation_method
        self.cmap_range = cmap_range

        self.transformed_f = f.astype("int64")

    def get_node_values(self, nodes: List[np.ndarray]) -> np.ndarray:
        f = self.transformed_f
        agg_fn = self.aggregation_functions.get(self.color_agg_method)
        if agg_fn is None:
            raise NotImplementedError("unsupported node aggregation method")
        node_values = np.array([agg_fn(f.iloc[u]) for u in nodes])
        return node_values

    def apply_to_nodes(
        self, nodes: List[np.ndarray], use_rich_labels: bool = False
    ) -> Tuple[List[str], Dict[int, str], w.Widget]:
        f = self.transformed_f
        min_timestamp, max_timestamp = pd.Timestamp(f.min()), pd.Timestamp(f.max())
        # TODO: handle range appropriately
        node_values = self.get_node_values(nodes)

        if self.node_label_gen and not use_rich_labels:
            node_labels = {
                i: self.node_label_gen(u, f, node_values[i])
                for i, u in enumerate(nodes)
            }
        elif self.rich_node_label_gen and use_rich_labels:
            node_labels = {
                i: self.rich_node_label_gen(u, f, node_values[i])
                for i, u in enumerate(nodes)
            }

        if min_timestamp != max_timestamp:

            def normalize(x):
                return (x - min_timestamp.value) / (
                    max_timestamp.value - min_timestamp.value
                )

            cmap = ColorMap(self.color_map_name, normalize=normalize)
        else:
            cmap = ColorMap(self.color_map_name)
        node_colors = cmap(node_values)
        legend = self.construct_legend(min_timestamp, max_timestamp, cmap, f.name)

        return node_colors, node_labels, legend

    @staticmethod
    def _default_node_label_gen(u: np.ndarray, f_vals: pd.Series, node_val):
        date_string = pd.Timestamp(node_val).strftime("%Y-%m-%d %H:%M:%S")
        return f"{len(u)} points, {f_vals.name} mean: {date_string}"

    @staticmethod
    def _default_rich_node_label_gen(u: np.ndarray, f_vals: pd.Series, node_val):
        date_string = pd.Timestamp(node_val).strftime("%Y-%m-%d %H:%M:%S")
        return {
            "n_points": len(u),
            "feat_name": f_vals.name,
            "node_val": date_string,
        }

    @staticmethod
    def construct_legend(
        min_date: pd.Timestamp, max_date: pd.Timestamp, cmap: ColorMap, name: str
    ) -> Callable[[], w.Widget]:
        num_range = np.linspace(min_date.value, max_date.value, 20)
        color_vals = cmap(num_range)
        css_color_string = f"linear-gradient(to top, {', '.join(color_vals)})"

        def make_widget():
            gradient_bar = v.Html(
                tag="div",
                class_="d-flex flex-shrink-0 flex-grow-0",
                style_=f"background: {css_color_string}; width: 8px; height: 155px",
            )

            max_date_label = v.Html(
                class_="ma-0 pa-0",
                tag="p",
                children=[
                    max_date.strftime("%Y-%m-%d"),
                    v.Html(tag="br"),
                    max_date.strftime("%H:%M:%S"),
                ],
            )

            min_date_label = v.Html(
                class_="ma-0 pa-0",
                tag="p",
                children=[
                    min_date.strftime("%Y-%m-%d"),
                    v.Html(tag="br"),
                    min_date.strftime("%H:%M:%S"),
                ],
            )

            bounds = v.Html(
                tag="div",
                class_="pa-0 d-flex flex-column justify-space-between",
                children=[max_date_label, min_date_label],
            )

            legend = v.Html(
                tag="div",
                class_="pb-0 pt-3",
                children=[
                    v.Row(
                        class_="ma-3 d-flex flex-nowrap",
                        no_gutters=True,
                        children=[gradient_bar, bounds],
                        style_="gap: 12px; line-height: 15px;",
                    ),
                    wrap_text_with_tooltip(
                        truncate_str(name), f"Colored by: {name}", "--v-textBackground"
                    ),
                ],
                style_="border: 1px solid #E0E0E0; border-radius: 4px;"
                "min-width: 100px;",
            )
            return legend

        return make_widget


class CategoricalColoringSpec(GraphColoringSpec):
    coloring_type = "categorical"

    aggregation_functions: ClassVar[Dict[str, Callable[[pd.Series], Any]]] = {
        "mode": lambda x: x.mode()[0]
    }

    _default_rich_node_label_template = (
        "<div><div>${feat_name}</div>"
        "<div>${node_val} (${value_count}/${n_points})</div></div>"
    )

    def __init__(
        self,
        f: pd.Series,
        color_map_name: str,
        aggregation_method: str = "mode",
        cmap_range: Optional[Tuple[float, float]] = None,
        node_label_gen: Optional[Callable[[np.ndarray, pd.Series, Any], str]] = None,
        repeat_colors: bool = False,
        rich_node_label_gen: Optional[
            Callable[[np.ndarray, pd.Series, Any], dict]
        ] = None,
        rich_node_label_template: Optional[str] = None,
    ):
        self.f = f
        self.node_label_gen = node_label_gen
        self.rich_node_label_gen = (
            rich_node_label_gen or self._default_rich_node_label_gen
        )
        self.rich_node_label_template = (
            rich_node_label_template or self._default_rich_node_label_template
        )
        # TODO: validate color map?
        self.color_map_name = color_map_name
        self.color_agg_method = aggregation_method
        self.cmap_range = cmap_range
        if isinstance(f.dtype, CategoricalDtype):
            if f.isna().any():
                f_ = (
                    f.cat.add_categories("no label")
                    if "no label" not in f.cat.categories
                    else f
                )
                f_ = f_.fillna("no label")
            else:
                f_ = f
        else:
            f_ = f.fillna("no label")
            f_ = f_.astype("category")
        self.transformed_f = f_
        if node_label_gen is None:
            self.node_label_gen = self._default_node_label_gen
        self.repeat_colors = repeat_colors
        # to avoid breaking the legend when repeat_colors is true
        self.max_n_classes = 1000

    def apply_to_nodes(
        self, nodes: List[np.ndarray], use_rich_labels: bool = False
    ) -> Tuple[List[str], Dict[int, str], w.Widget]:
        f_ = self.transformed_f
        color_map_name = self.color_map_name
        node_labels = {}
        # TODO: user-provided aggregation function

        total_n_cats = len(f_.cat.categories)
        num_top_cats = ColorMapOptions.top_categories_map(color_map_name)
        membership_vec = get_membership_vec(nodes, len(f_))
        node_codes, mode_counts = mode_on_subsets(
            f_.cat.codes.to_numpy(), membership_vec
        )
        if self.node_label_gen and not use_rich_labels:
            node_labels = {
                i: self.node_label_gen(
                    u, f_, mode_count, f_.cat.categories[node_codes[i]]
                )
                for i, (u, mode_count) in enumerate(zip(nodes, mode_counts))
            }
        if self.rich_node_label_gen and use_rich_labels:
            node_labels = {
                i: self.rich_node_label_gen(
                    u, f_, mode_count, f_.cat.categories[node_codes[i]]
                )
                for i, (u, mode_count) in enumerate(zip(nodes, mode_counts))
            }
        n_classes_to_truncate = (
            min(self.max_n_classes, total_n_cats)
            if self.repeat_colors
            else num_top_cats
        )
        (
            categories,
            cat_codes,
            new_node_codes,
            points_per_category,
        ) = self.construct_most_common_category_list(
            node_codes,
            f_,
            n=n_classes_to_truncate,
        )
        norm = (lambda x: x % num_top_cats) if self.repeat_colors else (lambda x: x)
        cmap = ColorMap(mapname=color_map_name, normalize=norm)
        node_colors = cmap(new_node_codes)
        legend = self.construct_legend(
            categories, cat_codes, cmap, points_per_category, f_.name
        )

        return node_colors, node_labels, legend

    def construct_most_common_category_list(
        self, int_node_values: List[int], f_: pd.Series, n=10
    ):
        top_counts = f_.value_counts().sort_values(ascending=False)
        # categories sorted by popularity
        categories = list(top_counts.index)
        # and their integer codes
        categories_codes = list(top_counts.index.codes)
        # if there are too many categories, keep the top n-1 and replace the last with "other"
        # (TODO: can you ever get 0 as a count?)
        points_per_category = top_counts.to_numpy()
        if len(top_counts) > n:
            (
                categories,
                categories_codes,
                points_per_category,
            ) = self.limit_num_of_top_categories(
                categories, categories_codes, points_per_category, n
            )

        # these codes might not be contiguous, so the color map might map some to the same color
        # we will replace them with values from 0 to n-1 to ensure uniqueness
        # (assuming the color map has at least n colors)
        # this has the side effect of ensuring that colors are assigned according to
        # popularity ranking of categories.
        codes_map = {value: i for i, value in enumerate(categories_codes)}
        new_node_values = [
            codes_map[value] if value in categories_codes else codes_map[-1]
            for value in int_node_values
        ]
        # the codes have been reassigned
        categories_codes = list(range(len(categories_codes)))
        return categories, categories_codes, new_node_values, points_per_category

    @staticmethod
    def limit_num_of_top_categories(
        categories: Sequence,
        categories_codes: Sequence,
        points_per_category: Sequence,
        n: int,
    ):
        categories = categories[: n - 1]
        categories_codes = categories_codes[: n - 1]
        categories.append("other")
        categories_codes.append(-1)

        top_per_category = points_per_category[: n - 1]
        other_per_category = points_per_category[n - 1 :]
        other_points_sum = sum(other_per_category)
        points_per_category = np.append(top_per_category, other_points_sum)
        return categories, categories_codes, points_per_category

    @staticmethod
    def _default_node_label_gen(
        u: np.ndarray, f_vals: pd.Series, value_count: int, node_val: str
    ) -> str:
        return f"{f_vals.name}: {node_val}" f" ({value_count}/{len(u)})"

    @staticmethod
    def _default_rich_node_label_gen(
        u: np.ndarray, f_vals: pd.Series, value_count: int, node_val: float
    ) -> dict:
        return {
            "n_points": len(u),
            "feat_name": f_vals.name,
            "node_val": node_val,
            "value_count": value_count,
        }

    @staticmethod
    def construct_legend(
        str_values: List[str],
        int_values: List[int],
        cmap: ColorMap,
        points_per_category: List[int],
        name: str,
    ) -> Callable[[], w.Widget]:
        colors = cmap(int_values)
        truncated_str_values = []
        for value in str_values:
            str_value = truncate_str(value)
            truncated_str_values.append(str_value)

        def make_widget():
            legend_entries = [
                v.Html(
                    tag="div",
                    class_="text-center ma-0 pa-0 d-flex flex-column align-center",
                    style_="gap: 0px;",
                    children=[
                        make_spot_div(color),
                        wrap_text_with_tooltip(
                            f"{truncated_value} ({points_per_cat})", f"{value}"
                        ),
                    ],
                )
                for color, truncated_value, value, points_per_cat in zip(
                    colors, truncated_str_values, str_values, points_per_category
                )
            ]

            legend = v.Html(
                tag="div",
                class_="pb-0 pt-3 ma-0 overflow-y-auto",
                children=[
                    v.Html(
                        tag="div",
                        class_="d-flex ma-0 pa-0 flex-column flex-wrap align-center",
                        children=legend_entries,
                        style_="gap: 8px;",
                    ),
                    wrap_text_with_tooltip(
                        truncate_str(name), f"Colored by: {name}", "--v-textBackground"
                    ),
                ],
                style_="position: relative; width: 100%; height: 100%;"
                "border: 1px solid #E0E0E0; border-radius: 4px;"
                "min-width: 110px; max-width: 150px;",
            )

            return legend

        return make_widget


class NoColoringSpec(GraphColoringSpec):

    rich_node_label_template = "<div>${n_points} points</div>"

    def __init__(self, color: str = "#999999"):
        self.default_color = color

    def apply_to_nodes(
        self,
        nodes: List[np.ndarray],
        use_rich_labels: bool = False,
    ) -> Tuple[List[str], Union[Dict[int, str], Dict[int, dict]], w.Widget]:
        node_colors = [self.default_color] * len(nodes)
        node_labels = (
            {i: {"n_points": len(u)} for i, u in enumerate(nodes)}
            if use_rich_labels
            else {i: f"{len(u)} points" for i, u in enumerate(nodes)}
        )
        legend = self.construct_legend(self.default_color)

        return node_colors, node_labels, legend

    @staticmethod
    def construct_legend(color: str) -> Callable[[], w.Widget]:
        def make_widget():
            spot = make_spot_div(color)
            legend = v.Flex(
                class_="d-flex align-center flex-column pt-3",
                children=[
                    spot,
                    wrap_text_with_tooltip("No coloring", "No coloring applied"),
                ],
                style_="border: 1px solid #E0E0E0; "
                "border-radius: 4px;"
                "width: 100px;",
            )
            return legend

        return make_widget


def truncate_str(val, max_len: int = 12, ellipsis_str: str = ".."):
    str_val = str(val)
    if len(str_val) <= max_len:
        return str_val
    return str_val[: (max_len - len(ellipsis_str))] + ellipsis_str


def wrap_text_with_tooltip(
    value: str, tooltip_label: Optional[str] = None, background: str = ""
) -> v.Tooltip:
    if tooltip_label is None:
        tooltip_text = value
        disp_value = truncate_str(value)
    else:
        tooltip_text = tooltip_label
        disp_value = value

    return v.Tooltip(
        bottom=True,
        children=[tooltip_text],
        v_slots=[
            {
                "name": "activator",
                "variable": "tooltipData",
                "children": v.Html(
                    tag="div",
                    class_="d-flex justify-center font-weight-medium pt-2 pb-3",
                    justify="center",
                    children=[disp_value],
                    style_="position: sticky; "
                    "bottom: 0; "
                    f"background-color: var({background}) !important;"
                    "border-radius: 0 0 4px 4px",
                    v_on="tooltipData.on",
                ),
            },
        ],
    )


def FunctionColoringSpec(
    f: pd.Series,
    color_map_name: str,
    aggregation_method: str = "mean",
    cmap_range: Optional[Tuple[float, float]] = None,
    node_label_gen: Optional[Callable[[np.ndarray, pd.Series, Any], str]] = None,
) -> GraphColoringSpec:
    """Create a GraphColoringSpec of appropriate type given the type of f and the color map."""
    if is_numeric_dtype(f.dtype) and color_map_name not in ColorMapOptions.CATEGORICAL:
        return NumericColoringSpec(
            f,
            color_map_name,
            aggregation_method,
            cmap_range,
            node_label_gen=node_label_gen,
        )
    elif is_datetime64_any_dtype(f.dtype):
        return TimestampColoringSpec(
            f, color_map_name, aggregation_method, cmap_range, node_label_gen
        )
    else:
        return CategoricalColoringSpec(
            f, color_map_name, aggregation_method, cmap_range, node_label_gen
        )
