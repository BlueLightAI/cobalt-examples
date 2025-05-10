import ipyvuetify as v


def with_tooltip(component, tooltip_label: str, background: str = "") -> v.Tooltip:
    return v.Tooltip(
        top=True,
        v_slots=[
            {
                "name": "activator",
                "variable": "tooltipData",
                "children": v.Html(
                    tag="div",
                    style_=f"background-color: var({background}) !important;"
                    "border-radius: 4px;",
                    v_on="tooltipData.on",
                    children=[component],
                ),
            }
        ],
        children=[tooltip_label],
    )
