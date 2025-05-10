# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Callable, List, Optional

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def rgb_to_hex_array(colors: np.ndarray):
    colors = np.round(colors * 255).astype(np.int32)
    out_vals = [
        f"#{colors[i,0]:02x}{colors[i,1]:02x}{colors[i,2]:02x}"
        for i in range(colors.shape[0])
    ]
    return out_vals


class ColorMap:
    """Converts a list or array of numeric values into a list of HTML color strings."""

    def __init__(self, mapname: str = "viridis", normalize: Optional[Callable] = None):
        """Set up the ColorMap.

        Args:
            mapname: name of the matplotlib colormap that should be used
            normalize: a function that normalizes numpy arrays. If omitted, scales
                the array to range from 0 to 1.
        """
        registered_maps = ColorMap.registered_color_maps()

        if mapname in registered_maps:
            self.cmap = registered_maps[mapname]
        else:
            self.cmap = matplotlib.colormaps[mapname]
        self.normalize = normalize

    def __call__(self, f) -> List[str]:
        f = np.array(f)
        norm = self.normalize if self.normalize else matplotlib.colors.Normalize()
        node_colors = self.cmap(norm(f))
        hex_node_colors = rgb_to_hex_array(node_colors)
        return hex_node_colors

    @staticmethod
    def custom_gradient(color1: tuple, color2: tuple):
        colors = [color1, color2]
        cm = LinearSegmentedColormap.from_list("Custom", colors, N=20)
        return cm

    @staticmethod
    def registered_color_maps():
        return {
            "black-red-gradient": ColorMap.custom_gradient((0, 0, 0), (1, 0, 0)),
        }
