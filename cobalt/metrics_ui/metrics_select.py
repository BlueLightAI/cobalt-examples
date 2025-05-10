# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v
from traitlets import Any, List, Unicode


class MetricsSelect(v.VuetifyTemplate):
    template = Unicode(
        """
          <v-select
            v-model="selected_item"
            v-show="items.length > 0"
            :items="items"
            item-value="id"
            attach
            solo
            background-color="transparent"
            hide-details
            flat
            dense
            placeholder="Select a metric to highlight"
            style="font-size: 13px;"
          >
            <template v-slot:selection="{ item }">
                <v-row
                  class="flex-nowrap justify-space-between" no-gutters align="center"
                  style="letter-spacing: 1px;"
                >
                    <v-tooltip bottom>
                      <template v-slot:activator="{ on, attrs }">
                        <span
                          class="d-inline-block text-uppercase text-truncate mr-1 metric-name"
                          v-bind="attrs"
                          v-on="on"
                        >
                          {{item.id}}
                        </span>
                      </template>
                      <span>{{item.id}}</span>
                    </v-tooltip>

                    <v-divider></v-divider>
                    <span class="d-inline-block font-weight-medium ml-2 metric-value">
                        {{item.value}}
                    </span>
                </v-row>
            </template>
            <template v-slot:item="{ active, item, attrs, on }">
              <v-list-item class="pa-0 pr-2" v-on="on" v-bind="attrs" #default="{ active }">
                <v-list-item-action class="ma-2">
                  <v-checkbox dense :input-value="active" style="height: 24px;"></v-checkbox>
                </v-list-item-action>
                <v-list-item-content>
                  <v-row class="flex-nowrap" no-gutters align="center"
                    style="letter-spacing: 2px; max-width: 100%;"
                  >
                      <span class="d-inline-block text-uppercase text-truncate metric-name"
                        :style="active ? 'font-weight: bold; opacity: 1;' : ''"
                      >
                        {{item.id}}
                      </span>
                      <v-spacer></v-spacer>
                      <span class="d-inline-block font-weight-medium metric-value">
                        {{item.value}}
                      </span>
                  </v-row>
                </v-list-item-content>
              </v-list-item>
            </template>
          </v-select>
          <style scoped>
            .metric-name {
              opacity: 0.6 !important;
            }

            .v-text-field.v-text-field--solo .v-input__control {
              min-height: 24px !important;
            }

            .metric-value {
              color: rgba(41, 98, 255) !important;
            }
          </style>
        """
    ).tag(sync=True)
    items = List([]).tag(sync=True)
    selected_item = Any(None).tag(sync=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
