# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Any, Dict, List, Optional

import ipyvuetify as v
import traitlets

from cobalt.cobalt_types import GroupType
from cobalt.selection_manager import SelectionManager

GroupDetailDict = Dict[str, Any]


class GroupDetails(v.VuetifyTemplate):
    items = traitlets.List([]).tag(sync=True)
    item_id = traitlets.Union(
        [traitlets.Int(), traitlets.Unicode()], allow_none=True
    ).tag(sync=True)
    item_type = traitlets.Unicode(allow_none=True).tag(sync=True)
    groupTypeUser = traitlets.Unicode(allow_none=True).tag(sync=True)
    is_details_visible = traitlets.Union(
        [traitlets.Bool(), traitlets.Unicode()], allow_none=True
    ).tag(sync=True)

    template = traitlets.Unicode(
        """
        <div>
            <template v-if="is_details_visible">
                <template v-if="item_type == groupTypeUser">
                    <v-text
                        style="font-size: 14px;"
                    >
                        Please select a group for comparison.
                    </v-text>
                </template>
            <v-list style="max-height: 540px;" class="overflow-y-scroll">
                <v-list-item-group>
                    <template  v-if="items.length > 0" v-for="(item, index) in items">
                        <v-expand-transition>
                            <v-card flat v-if="item_id === item.group_id">
                                <v-chip
                                    small
                                    class="mb-1 mr-1"
                                    v-for="(value, name) in item.metrics"
                                >
                                        <b> {{ name }}: {{ value }} </b>
                                </v-chip>

                                <v-card
                                    v-if="item.histograms.length > 0"
                                    class="d-flex flex-wrap justify-center my-3 pt-5"
                                    style="gap: 4px"
                                >
                                    <v-card-text
                                        v-for="hist in item.histograms"
                                        class="d-flex flex-column pa-0"
                                        style="width: auto"
                                    >
                                        <h4 class="text-center">
                                        {{ hist.title }}
                                        (
                                            <v-chip
                                                x-small
                                                label
                                                class="px-1"
                                                style="margin-left: -2px; margin-right: -2px;"
                                            >
                                                {{ hist.feature }}
                                            </v-chip>
                                        )
                                        </h4>
                                        <div v-html="hist.svg" class="text-center"></div>
                                    </v-card-text>
                                </v-card>

                                <v-card
                                    v-if="Object.keys(item.text_descriptions).length > 0"
                                    class="my-3"
                                >
                                    <v-card
                                        flat
                                        v-for="(desc, feature) in item.text_descriptions"
                                    >
                                            <v-card-title>{{ feature }}</v-card-title>
                                            <v-card-text class="d-flex flex-wrap">
                                                <v-chip
                                                    v-for="keyword in desc"
                                                    class="mb-1 mr-1"
                                                    small
                                                >
                                                    {{ keyword }}
                                                </v-chip>
                                            </v-card-text>
                                    </v-card>
                                </v-card>

                                <v-card
                                    v-if="(Object.keys(item.feature_descriptions).length > 0)"
                                    class="my-3"
                                >
                                    <v-card-title>Distinguishing features:</v-card-title>
                                    <v-card-text>
                                        <v-sheet
                                        v-for="(desc, feature) in item.feature_descriptions"
                                        class="d-flex align-center my-1"
                                        >
                                            <v-tooltip bottom>
                                                <template v-slot:activator="{ on, attrs }">
                                                    <v-chip small label class="mr-2">
                                                        <span
                                                            class="text-truncate"
                                                            style="max-width: 250px"
                                                            v-bind="attrs"
                                                            v-on="on"
                                                        >
                                                            {{ feature }}
                                                        </span>
                                                    </v-chip>
                                                </template>
                                                <span>{{ feature }}</span>
                                            </v-tooltip>
                                            <div>{{ desc }}</div>
                                        </v-sheet>
                                    </v-card-text>
                                </v-card>

                            </v-card>
                        </v-expand-transition>
                    </template>
                </v-list-item-group>
            </v-list>
            </template>
        </div>
        """
    ).tag(sync=True)

    def __init__(
        self,
        dataselector: SelectionManager,
        items: Optional[List[GroupDetailDict]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataselector = dataselector
        self.items = items if items is not None else []
        self.item_id = None
        self.item_type = None
        self.is_details_visible = True
        self.groupTypeUser = GroupType.user.value

    def update_comparison_group(self, comparison_group):
        self.item_id = comparison_group["group_id"]
        self.item_type = comparison_group["group_type"].value

    def show_details(self):
        self.is_details_visible = True

    def hide_details(self):
        self.is_details_visible = False

    def update_items(self, items: List[GroupDetailDict]):
        self.items = items
