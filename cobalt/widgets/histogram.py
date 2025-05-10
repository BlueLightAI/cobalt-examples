# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import io

import ipyvuetify as v
import ipywidgets as w
import matplotlib.pyplot as plt
from IPython.display import SVG, display


def plot_histogram(
    bucket_sizes, bucket_names=None, bounds=None, colors=None, integer_counts=True
):
    width = max(300, min(300, 15 * len(bucket_sizes)))
    height = max(100, min(100, 10 * len(bucket_sizes)))
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))

        indices = range(len(bucket_sizes))

        bar_colors = colors if colors is not None else "skyblue"
        ax.bar(indices, bucket_sizes, color=bar_colors)

        if bounds:
            ax.set_xticks([0, len(bucket_sizes) - 1])  # Set ticks only at the ends
            ax.set_xticklabels(
                [f"{bounds[0]:.2f}", f"{bounds[1]:.2f}"], rotation=45, ha="right"
            )
        elif bucket_names:
            plt.xticks(indices, bucket_names, rotation=45, ha="right")

        if integer_counts:
            ax.yaxis.get_major_locator().set_params(integer=True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        svg_io = io.StringIO()
        plt.savefig(svg_io, format="svg", bbox_inches="tight")
        plt.close(fig)
        svg_io.seek(0)
        svg_data = svg_io.getvalue()

    return svg_data


class HistogramPlotOutput(w.Output):
    def __init__(self, bucket_sizes, bucket_names=None, bounds=None, colors=None):
        super().__init__()
        with self:
            svg_data = plot_histogram(bucket_sizes, bucket_names, bounds, colors)
            display(SVG(svg_data))


def Histogram(bucket_sizes, bucket_names=None, bounds=None, colors=None):
    """Example.

    Histogram([5, 20, 10, 32, 6, 2], bucket_names=[
                "Rhodesian ridgeback",
                "Shih-Tzu",
                "Beagle",
                "Golden retriever",
                "Old English sheepdog",
                "Dingo",
            ], colors=["#FF5733", "#33FF57", "#3357FF", "#F733FF", "#33FFF7", "#F7FF33"])

    Histogram(
        [2, 5, 9, 15, 30, 15, 9, 5, 2],
        bounds=(-4, 4),
        colors=["red", "orange", "yellow", "green", "blue", "indigo", "violet", "purple", "pink"]
    )

    """
    return v.Layout(
        children=[HistogramPlotOutput(bucket_sizes, bucket_names, bounds, colors)]
    )
