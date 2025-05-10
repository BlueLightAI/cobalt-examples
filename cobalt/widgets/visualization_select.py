# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v
import traitlets


class VisualizationSelect(v.VuetifyTemplate):
    items = traitlets.List(default_value=[]).tag(sync=True)
    v_model = traitlets.Unicode(None, allow_none=True).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.change_callback = None
        self.observe(self.on_selection_change, "v_model")

    def on_selection_change(self, change):
        if change["new"]:
            self.trigger_change_event()

    def trigger_change_event(self):
        if hasattr(self, "change_callback") and callable(self.change_callback):
            self.change_callback(self.v_model)

    def on_event(self, event_name, callback):
        if event_name == "change":
            self.change_callback = callback

    @traitlets.default("template")
    def _template(self):
        return """
                <template>
                  <v-select
                    v-model="v_model"
                    :items="items"
                    item-text="title"
                    item-value="title"
                    label="Graph"
                    attach
                    outlined
                    dense
                    hide-details
                    class="mx-2 elevation-1"
                    style="max-width: 280px"
                    ref="select"
                  >
                    <template v-slot:item="{ item, props }">
                      <v-list-item v-bind="props" @click="onSelectItem(item)">
                        <v-tooltip top content-class="py-1 caption">
                          <template v-slot:activator="{ on, attrs }">
                            <v-list-item-content
                              v-on="on"
                              v-bind="attrs"
                            >
                              <v-list-item-title
                                :class="{'primary--text': item.title == v_model}">
                                  {{ item.title }}
                              </v-list-item-title>
                              <v-list-item-subtitle>
                                  {{ item.subtitle }}
                              </v-list-item-subtitle>
                            </v-list-item-content>
                          </template>
                            {{ item.title }}
                        </v-tooltip>
                      </v-list-item>
                    </template>
                  </v-select>
                </template>
                <script>
                    export default {
                        props: ["items", "v_model"],
                        methods: {
                            onSelectItem(item) {
                                this.v_model = item.title;
                                this.closeMenu();
                            },
                            closeMenu() {
                                this.$refs.select.blur();
                            },
                        },
                    }
                </script>
            """
