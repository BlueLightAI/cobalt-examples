import ipyvuetify as v


class EmptyVisualizationBlock(v.Layout):
    def __init__(self):
        super().__init__(
            children=[
                v.Html(
                    tag="div",
                    children=[
                        v.Html(
                            tag="span",
                            children=["There are no graphs in the Workspace."],
                        ),
                        v.Html(tag="br"),
                        v.Html(
                            tag="span",
                            children=[
                                "Click the 'New Graph' button or "
                                "run Workspace.new_graph() to create a graph to display."
                            ],
                        ),
                    ],
                    style_="""
                        font-size: 22px;
                        font-weight: bold;
                        color: white;
                        text-align: center;
                        padding: 20px
                    """,
                ),
            ],
            class_="justify-center align-center",
            style_="""
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.5);
                z-index: 2;
            """,
        )
