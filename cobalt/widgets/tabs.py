# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import List

import ipyvuetify as v

INITIAL_TAB = 0


def Tabs(
    tab_data: List[dict],
    vertical: bool = False,
    initial_tab: int = INITIAL_TAB,
):
    tabs = [v.Tab(children=[item["tab_header"]]) for item in tab_data]
    tab_items = [v.TabItem(children=[item["tab_content"]]) for item in tab_data]

    tabs_component = v.Tabs(
        v_model=initial_tab,
        vertical=vertical,
        children=tabs,
        class_="flex-shrink-1" if vertical else "",
    )

    tab_items_style = "" if vertical else "border: 1px solid #E0E0E0;"

    tab_items_component = v.TabsItems(
        v_model=initial_tab,
        children=tab_items,
        style_=tab_items_style,
        class_="pa-5",
    )

    def on_tab_change(change):
        tab_items_component.v_model = change["new"]

    tabs_component.observe(on_tab_change, names="v_model")

    if vertical:
        layout = v.Layout(
            wrap=False,
            children=[
                v.Flex(
                    xs12=True,
                    md3=True,
                    class_="flex-shrink-1",
                    style_="border-right: 1px solid #E0E0E0",
                    children=[tabs_component],
                ),
                v.Flex(
                    xs12=True,
                    md9=True,
                    class_="flex-grow-1",
                    children=[tab_items_component],
                ),
            ],
        )
    else:
        layout = v.Flex(
            column=True,
            children=[tabs_component, tab_items_component],
        )

    return layout
