# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import io
from typing import Optional

import ipyvuetify as v
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import traitlets

from cobalt.coloring.color_spec import ColorMapOptions


def confusion_matrix_plot(matrix_data: Optional[pd.DataFrame], cmap: str) -> str:
    if matrix_data.empty:
        return f"<p class='body-1'>{matrix_data.meta.error}</p>"

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        data=matrix_data,
        annot=True,
        cmap=cmap,
        fmt=".2f",
        ax=ax,
    )

    # Convert to SVG
    output = io.StringIO()
    plt.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)
    output.seek(0)
    return output.getvalue()


class ConfusionMatrix(v.VuetifyTemplate):
    template = traitlets.Unicode(
        """
        <v-layout column>
            <v-layout v-if="data_available" align-center class='mb-5'>
                <v-text class="subtitle-1 mr-5">Confusion Matrix for {{ model_name }}</v-text>
                <v-select
                    outlined
                    dense
                    hide-details
                    :items="color_map_options"
                    v-model="selected_color_map"
                    label="Color map"
                    style="max-width: 150px"
                    @change="on_select_color_map">
                </v-select>
            </v-layout>
            <v-layout>
                <div v-html="svg_content"></div>
            </v-layout>
        </v-layout>
        """
    ).tag(sync=True)

    data_available = traitlets.Bool(allow_none=True).tag(sync=True)
    color_map_options = traitlets.List(traitlets.Unicode()).tag(sync=True)
    selected_color_map = traitlets.Unicode().tag(sync=True)
    model_name = traitlets.Unicode().tag(sync=True)
    svg_content = traitlets.Unicode().tag(sync=True)

    def vue_on_select_color_map(self, new_color_map):
        self.selected_color_map = new_color_map
        self.svg_content = confusion_matrix_plot(
            self.matrix_data, self.selected_color_map
        )

    def __init__(
        self,
        matrix_data: Optional[pd.DataFrame] = None,
        model_name: Optional[str] = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.matrix_data = matrix_data
        self.color_map_options = ColorMapOptions.NUMERICAL
        self.selected_color_map = ColorMapOptions.NUMERICAL[0]
        self.model_name = model_name
        self.svg_content = confusion_matrix_plot(
            self.matrix_data, self.selected_color_map
        )
        self.update_matrix_data_availability()

    def update_matrix(self, new_matrix_data: pd.DataFrame, model_name: str):
        self.update_matrix_data_availability()
        self.matrix_data = new_matrix_data
        self.model_name = model_name
        self.svg_content = confusion_matrix_plot(
            self.matrix_data, self.selected_color_map
        )

    def update_matrix_data_availability(self):
        if self.matrix_data is None or self.matrix_data.empty:
            self.data_available = False
        else:
            self.data_available = True
