# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v

from cobalt.schema import DatasetSplit
from cobalt.widgets import Tabs
from cobalt.workspace_overview.benchmarks import Benchmarks
from cobalt.workspace_overview.summary import Summary


class WorkspaceOverview(v.Layout):
    def __init__(self, ws, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = ws
        self.split: DatasetSplit = self.ws.state.split

        self.summary = Summary(self.ws)
        self.benchmarks = Benchmarks(self.ws)

        self.children = [
            v.Layout(
                align_center=True,
                justify_space_between=True,
                class_="pa-4",
                style_="border-bottom: 1px solid #E0E0E0",
                children=[
                    v.Text(class_="body-1", children=["Workspace Overview"]),
                ],
            ),
            Tabs(
                tab_data=[
                    {"tab_header": "Summary", "tab_content": self.summary},
                    {"tab_header": "Benchmarks", "tab_content": self.benchmarks},
                ],
                vertical=True,
            ),
        ]
