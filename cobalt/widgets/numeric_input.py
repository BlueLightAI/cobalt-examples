# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v
import traitlets

NUM_REGEX = r"^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$"
POS_NUM_REGEX = r"^([0-9]+([.][0-9]*)?|[.][0-9]+)$"

INT_REGEX = r"^[+-]?([0-9]\d*)$"
POS_INT_REGEX = r"^\+?([0-9]\d*)$"


class NumericInput(v.VuetifyTemplate):
    value = traitlets.Union(
        [traitlets.Unicode(), traitlets.Int(), traitlets.Float()],
        default_value=1,
        allow_none=True,
    ).tag(sync=True)

    label = traitlets.Unicode("Numeric Input").tag(sync=True)
    hint = traitlets.Unicode("").tag(sync=True)

    width = traitlets.Union(
        [traitlets.Int(), traitlets.Float()],
        default_value=None,
        allow_none=True,
    ).tag(sync=True)

    minimum = traitlets.Union(
        [traitlets.Int(), traitlets.Float()],
        default_value=None,
        allow_none=True,
    ).tag(sync=True)

    maximum = traitlets.Union(
        [traitlets.Int(), traitlets.Float()],
        default_value=None,
        allow_none=True,
    ).tag(sync=True)

    allow_zero = traitlets.Bool(
        default_value=True,
    ).tag(sync=True)

    allow_negative = traitlets.Bool(
        default_value=True,
    ).tag(sync=True)

    integer_only = traitlets.Bool(
        default_value=False,
    ).tag(sync=True)

    valid = traitlets.Bool(
        default_value=False,
    ).tag(sync=True)

    class_ = traitlets.Unicode("").tag(sync=True)

    @traitlets.default("template")
    def _template(self):
        self.REGEX, self.VAL_TYPE = (
            (INT_REGEX, "Integer") if self.integer_only else (NUM_REGEX, "Numeric")
        )

        value_required_rule = "v => !!v || 'Value required'"
        regex_test_rule = (
            f"v => /{self.REGEX}/.test(v) || '{self.VAL_TYPE} value required'"
        )
        nonzero_rule = (
            "v => v != 0 || 'Nonzero value required'" if not self.allow_zero else ""
        )
        nonneg_rule = (
            "v => v >= 0 || 'Nonnegative value required'"
            if not self.allow_negative
            else ""
        )
        minimum_rule = (
            f"v => v >= {self.minimum} || 'Minimum value: {self.minimum}'"
            if self.minimum
            else ""
        )
        maximum_rule = (
            f"v => v <= {self.maximum} || 'Maximum value: {self.maximum}'"
            if self.maximum
            else ""
        )
        greater_than_one_rule = "v => v > 1 || 'Value must be greater than 1'"

        RULE_LIST = [
            value_required_rule,
            regex_test_rule,
            nonzero_rule,
            nonneg_rule,
            minimum_rule,
            maximum_rule,
            greater_than_one_rule,
        ]

        STYLE = f"style='width: {self.width}px'" if self.width else ""

        return f"""
            <template>
              <div>
                <v-form v-model="valid">
                  <v-text-field
                    :class="class_"
                    outlined
                    dense
                    hide-details="auto"
                    :label="label"
                    :hint="hint"
                    {STYLE}
                    v-model="value"
                    :rules="[ {', '.join(filter(None, RULE_LIST))} ]"
                  />
                </v-form>
              </div>
            </template>
        """
