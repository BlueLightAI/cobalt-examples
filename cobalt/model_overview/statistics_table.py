# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v
import pandas as pd


class StatisticsTable(v.Layout):
    def __init__(self, data, error, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.error = error
        self.update_display()

    def _get_headers(self):
        if isinstance(self.data, pd.DataFrame):
            headers = [
                {"text": "", "align": "start", "sortable": False, "value": "metric"}
            ]
            headers.extend([{"text": col, "value": col} for col in self.data.columns])
            return headers
        return []

    def _get_items(self):
        if isinstance(self.data, pd.DataFrame):
            return [{"metric": index, **row} for index, row in self.data.iterrows()]
        return []

    def update_data(self, new_data):
        self.data = new_data
        self.update_display()

    def update_display(self):
        self.children = []
        if isinstance(self.data, pd.DataFrame) and not self.data.empty:
            self.headers = self._get_headers()
            self.items = self._get_items()
            self.data_table = v.DataTable(
                headers=self.headers,
                items=self.items,
                hide_default_footer=True,
            )
            self.children = [self.data_table]
        else:
            error_message = self.data if isinstance(self.data, str) else self.error
            self.children = [
                v.Text(
                    children=[error_message],
                    class_="body-1",
                )
            ]
