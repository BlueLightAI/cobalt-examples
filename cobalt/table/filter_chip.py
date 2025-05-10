import ipyvuetify as v
import traitlets


class FilterChip(v.VuetifyTemplate):
    chip_text = traitlets.Unicode(allow_none=True).tag(sync=True)

    def __init__(
        self,
        chip_text: str,
        on_click: callable,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.chip_text = chip_text
        self.on_click = on_click

    @traitlets.default("template")
    def _template(self):
        return """
            <template>
                <v-chip
                  v-if="chip_text"
                  class="mr-2"
                  close
                  @click:close="on_handle_click"
                >
                 {{ chip_text }}
                </v-chip>
            </template>
        """

    def vue_on_handle_click(self, *_):
        if self.on_click:
            self.on_click()
