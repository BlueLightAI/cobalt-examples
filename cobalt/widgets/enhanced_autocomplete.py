# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v
from traitlets import Bool, Dict, List, Unicode


class EnhancedAutocomplete(v.VuetifyTemplate):
    template = Unicode(
        """
          <template>
            <v-autocomplete
              v-model="selected"
              :items="items"
              :multiple="multiple"
              :label="label"
              :placeholder="placeholder"
              :class="class_"
              :style="style_"
              :dense="dense"
              outlined
              clearable
              hide-details
              @change="handleChange"
              @blur="handleBlur"
            >
              <template v-slot:selection="{ item, index }">
                <v-chip v-if="index < 3" small>
                  <span>{{ item }}</span>
                </v-chip>
                <span
                  v-if="index === 3"
                  class="grey--text text-caption"
                >
                  (+{{ selected.length - 3 }} {{ selected.length - 3 === 1 ? 'other' : 'others' }})
                </span>
              </template>
            </v-autocomplete>
          </template>
        """
    ).tag(sync=True)

    selected = List([]).tag(sync=True)
    items = List([]).tag(sync=True)
    multiple = Bool(True).tag(sync=True)
    label = Unicode("").tag(sync=True)
    placeholder = Unicode("").tag(sync=True)
    class_ = Unicode("").tag(sync=True)
    style_ = Unicode("").tag(sync=True)
    dense = Bool(True).tag(sync=True)
    event_handlers = Dict(default_value={}).tag(sync=False)

    def on_event(self, event_name, handler):
        if not callable(handler):
            raise ValueError("Handler must be callable")
        self.event_handlers[event_name] = handler

    def trigger_event(self, event_name, *args):
        if event_name in self.event_handlers and callable(
            self.event_handlers[event_name]
        ):
            self.event_handlers[event_name](*args)

    def vue_handleChange(self, value):
        self.trigger_event("change", value)

    def vue_handleBlur(self, value):
        self.trigger_event("blur", value)
