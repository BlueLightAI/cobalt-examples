# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import json
import re
from typing import List, Optional, Tuple
from uuid import UUID

import ipyvuetify as v
import pandas as pd
import traitlets

from cobalt.table.table_constants import DEFAULT_NUMERIC_FORMAT
from cobalt.table.table_utils import format_df_for_display


def _safe_column_name(col: str) -> str:
    return re.sub(r"\W+", "_", col)


class TableView(v.VuetifyTemplate):
    """A Vuetify-based table component for rendering a DataFrame.

    - Accepts a DataFrame and traitlets properties for columns/rows (headers, items).
    - Formats numerical columns, handles a limited row display, etc.
    - Supports additional templates for image or HTML columns.
    - Manages pagination in coordination with Vue/JavaScript logic.
    """

    table_uuid = traitlets.Unicode(allow_none=True).tag(sync=True)
    headers = traitlets.List([]).tag(sync=True, allow_null=True)
    items = traitlets.List([]).tag(sync=True, allow_null=True)
    footer_props = traitlets.Dict({}).tag(sync=True)
    style = traitlets.Unicode(allow_none=True).tag(sync=True)
    column_display_names = traitlets.Dict({}).tag(sync=True)

    def __init__(
        self,
        df: pd.DataFrame,
        table_uuid: Optional[UUID] = None,
        num_rows=None,
        image_columns: Optional[List[dict]] = None,
        html_columns: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (80, 80),
        numeric_format: str = DEFAULT_NUMERIC_FORMAT,
        style: Optional[str] = "",
        **kwargs,
    ):
        self.table_uuid = str(table_uuid) if table_uuid else None
        self.image_columns = image_columns or []
        self.html_columns = html_columns or []
        self.image_height, self.image_width = image_size

        displayed_df = format_df_for_display(
            df, num_rows=num_rows, numeric_format=numeric_format
        )

        self.safe_header_keys = {
            col: _safe_column_name(col) for col in displayed_df.columns
        }

        self.header_display_names = {
            self.safe_header_keys[col]: col for col in displayed_df.columns
        }

        _headers = [
            {
                "text": self.header_display_names[self.safe_header_keys[col]],
                "value": col,
                "class": "text-no-wrap",
            }
            for col in displayed_df.columns
        ]

        _items = displayed_df.to_dict(orient="records")

        _footer_props = {"itemsPerPageOptions": [5, 10, 25]}

        super().__init__(
            headers=_headers,
            items=_items,
            footer_props=_footer_props,
            style=style,
            column_display_names=self.column_display_names,
            **kwargs,
        )

    def _generate_column_templates(self) -> str:
        header_templates = "\n".join(
            [
                f"""
                <template v-slot:header.{safe_name}="props">
                  <span v-html="columnDisplayNames['{safe_name}'] || props.header.text"></span>
                </template>
                """
                for safe_name in self.safe_header_keys.values()
            ]
        )

        image_template = "\n".join(
            [
                f"""
                    <template v-slot:item.{img_col["column_name"]}="props">
                    <v-img
                        :src="props.item.{img_col["column_name"]}"
                        height="{self.image_height}"
                        width="{self.image_width}"
                    >
                    </v-img>
                    </template>"""
                for img_col in self.image_columns
            ]
        )

        html_template = "\n".join(
            [
                f"""
                <template v-slot:item.{_safe_column_name(text_col)}="props">
                  <div v-html="props.item['{_safe_column_name(text_col)}']"></div>
                </template>
                """
                for text_col in self.html_columns
            ]
        )

        return header_templates + "\n" + image_template + "\n" + html_template

    @traitlets.default("template")
    def _template(self):
        template = (
            """
            <template>
            <v-data-table
              :items="items"
              :headers="headers"
              :footer-props="footer_props"
              :data-table-id="tableId"
              :style="style"
              @update:items-per-page="onItemsPerPageChange"
            >
            """
            + self._generate_column_templates()
            + """
          </v-data-table>
          </template>

          <script>
          export default {
            data() {
              return {
                tableId: '',
                columnDisplayNames: """
            + json.dumps(self.column_display_names)
            + """
              };
            },
            mounted() {
              this.$nextTick(() => {
                if (this.table_uuid) {
                  this.tableId = 'table-' + this.table_uuid;

                  // Ensure DOM is fully updated
                  this.$nextTick(() => {
                    const tableElements = document.querySelectorAll(
                      `[data-table-id="${this.tableId}"]`
                    );

                    const storedItemsPerPage = sessionStorage.getItem(
                      `items_per_page_${this.tableId}`
                    ) || '10';

                    tableElements.forEach((tableElement) => {
                      const footerSelect = tableElement.querySelector(
                        '.v-data-footer__select'
                      );
                      if (footerSelect) {
                        const itemsPerPageInput = footerSelect.querySelector(
                          'input[aria-label="$vuetify.TableView.itemsPerPageText"]'
                        );
                        const itemsPerPageDisplay = footerSelect.querySelector(
                          '.v-select__selection--comma'
                        );

                        if (itemsPerPageInput) {
                          itemsPerPageInput.value = storedItemsPerPage;

                          // Trigger the change event to notify Vue
                          const event = new Event('input', { bubbles: true });
                          itemsPerPageInput.dispatchEvent(event);
                        }

                        if (itemsPerPageDisplay) {
                          itemsPerPageDisplay.innerText = storedItemsPerPage;
                        }
                      }
                    });
                  });
                }
              });
            },
            methods: {
              onItemsPerPageChange(itemsPerPage) {
                // Manually update the text because menu styles set the default
                // option to the first element. If you select this element,
                // the inner text won't change.

                if (this.table_uuid) {
                  this.$nextTick(() => {
                    const tableElements = document.querySelectorAll(
                      `[data-table-id="${this.tableId}"]`
                    );

                    sessionStorage.setItem(
                      `items_per_page_${this.tableId}`, itemsPerPage
                    );

                    tableElements.forEach((tableElement) => {
                      const footerSelect = tableElement.querySelector(
                        '.v-data-footer__select'
                      );
                      if (footerSelect) {
                        const itemsPerPageDisplay = footerSelect.querySelector(
                          '.v-select__selection--comma'
                        );
                        if (itemsPerPageDisplay) {
                          itemsPerPageDisplay.innerText = itemsPerPage;
                        }
                      }
                    });
                  });
                }
              }
            }
          };
          </script>
          """
        )
        return template
