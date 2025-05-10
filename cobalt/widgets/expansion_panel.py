# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v
from IPython.display import HTML, display


class ExpansionPanel(v.ExpansionPanels):
    def __init__(self, header_content, panel_content, **kwargs):
        self.header_content = header_content
        self._panel_content = panel_content
        super().__init__(**kwargs)
        self.children = self._generate_children()
        self._override_default_styles()

    @property
    def panel_content(self):
        return self._panel_content

    @panel_content.setter
    def panel_content(self, value):
        self._panel_content = value
        self.children[0].children[1].children = [self.panel_content]

    def _generate_children(self):
        inner_panel = v.ExpansionPanel(
            style_="border-radius: 0; border: 1px solid #E5E5E5"
        )
        inner_panel.children = [
            v.ExpansionPanelHeader(children=[self.header_content]),
            v.ExpansionPanelContent(children=[self.panel_content]),
        ]
        return [inner_panel]

    # The direct manipulation of pseudo-elements (::before)
    # without using CSS <style> or equivalent CSS injection
    # is outside the scope of what's possible with current web standards
    # and the capabilities of Jupyter Notebooks.

    def _override_default_styles(self):
        css = """
        <style>
            .v-expansion-panel::before {
                box-shadow: none !important;
            }
            .v-expansion-panel-content__wrap {
                padding: 0 !important;
            }
        </style>
        """
        display(HTML(css))
