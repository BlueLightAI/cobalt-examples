# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import re
from typing import Dict, List

import ipyvuetify as v
import traitlets

from cobalt.cobalt_types import GroupType
from cobalt.config import handle_cb_exceptions
from cobalt.event_bus import EventBusController
from cobalt.metrics_ui.metrics_ui import format_metric_value
from cobalt.problem_group.schema import Group, ProblemGroup
from cobalt.selection_manager import SelectionManager
from cobalt.widgets import plot_histogram

# For complex objects like CobaltDataSubset,
# we need to convert these to a serialized form that Vue can understand.
# This means transforming the object into a dictionary
# with only the necessary information.


def get_title_and_feature_name(title: str) -> Dict[str, str]:
    match = re.search(r"(.*) \((.*)\)", title)
    if match:
        groups = match.groups()
        return {"title": groups[0], "feature": groups[1]}
    else:
        return {"title": title, "feature": ""}


def convert_group_to_dict(
    groups: List[Group],
) -> List[Dict]:
    out = []
    for g in groups:
        if g.group_details.feature_descriptions:
            feature_descriptions = g.group_details.feature_descriptions
        else:
            feature_descriptions = {}
        g_ui_dict = {
            "group_id": g.name,
            "group_size": len(g.subset),
            "metrics": {
                name: format_metric_value(value) for name, value in g.metrics.items()
            },
            "problem_description": (
                g.problem_description if hasattr(g, "problem_description") else ""
            ),
            "summary": g.summary,
            "histograms": [
                {**get_title_and_feature_name(title), "svg": plot_histogram(**hist)}
                for title, hist in g.group_details.histograms.items()
            ],
            "feature_descriptions": feature_descriptions,
            "text_descriptions": g.group_details.textual_descriptions,
        }
        out.append(g_ui_dict)
    return out


class FailureGroups(v.VuetifyTemplate):
    failure_groups_dict = traitlets.List([]).tag(sync=True)
    selected_group_id = traitlets.Union(
        [traitlets.Int(), traitlets.Unicode()], allow_none=True
    ).tag(sync=True)
    detail_view_item_id = traitlets.Union(
        [traitlets.Dict({}), traitlets.Unicode()], allow_none=True
    ).tag(sync=True)

    template = traitlets.Unicode(
        """
        <div>
            <template v-if="failure_groups_dict.length > 0">
                <v-text
                    class="px-4 text-uppercase "
                    style="font-size: 14px;"
                >
                    Autogroups
                </v-text>
            </template>
            <v-list>
                <v-list-item-group>
                    <template v-for="(item, index) in failure_groups_dict">
                            <v-list-item
                                :key="item.group_id"
                                :value="item.group_id"
                                @click="on_item_click_and_toggle_detail(item.group_id)"
                                class="px-4">
                                <v-list-item-content
                                    class="py-2"
                                >
                                    <v-list-item-title>{{ item.group_id }}</v-list-item-title>
                                    <v-list-item-subtitle
                                        v-if="item.problem_description">
                                        {{ item.problem_description }}
                                    </v-list-item-subtitle>
                                    <v-list-item-subtitle
                                        v-if="item.summary">
                                        {{ item.summary }}
                                    </v-list-item-subtitle>
                                </v-list-item-content>
                            </v-list-item>
                    </template>
                </v-list-item-group>
            </v-list>
        </div>
        <style scoped>
            .vuetify-styles .v-list-item--link:before {
                background-color: transparent !important;
            }
            .vuetify-styles .v-list-item--link:hover {
                background-color: rgba(100, 100, 100, 0.2) !important;
            }
        </style>
        """
    ).tag(sync=True)

    def __init__(
        self,
        selection_manager: SelectionManager,
        workspace,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.selection_manager = selection_manager
        self.workspace = workspace
        self.detail_view_item_id = None
        self.failure_groups = self.get_failure_groups()
        self.failure_groups_dict = convert_group_to_dict(self.failure_groups)
        self.id_to_subset_map = self.create_id_to_subset_map()

        run_event_bus = EventBusController.get_run_event_bus(
            self.workspace.state.workspace_id
        )
        run_event_bus.add_callback(self.update_failure_groups)

    def create_id_to_subset_map(self) -> Dict:
        return (
            {fg.name: fg.subset for fg in self.failure_groups}
            if self.failure_groups
            else {}
        )

    def get_failure_groups(self) -> List[ProblemGroup]:
        return self.workspace._get_displayed_groups()

    def update_failure_groups(self):
        self.failure_groups = self.get_failure_groups()
        self.failure_groups_dict = convert_group_to_dict(self.failure_groups)
        self.id_to_subset_map = self.create_id_to_subset_map()

    @handle_cb_exceptions
    def vue_on_item_click_and_toggle_detail(self, item_id):
        self.detail_view_item_id = (
            None if self.detail_view_item_id == item_id else item_id
        )

        self.selection_manager.set_comparison_group(
            group_type=GroupType.failure, group_id=item_id
        )

        self.selected_group_id = item_id
        selected_subset = self.id_to_subset_map.get(item_id)

        group = {"group_id": self.selected_group_id, "subset": selected_subset}

        self.selection_manager.set_group_selection(group)
