# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v

from cobalt.config import is_vscode

# The purpose of using `is_vscode()` function is to detect whether we are running
# in a VSCode environment. This is necessary because the dialog's
# attachment behavior (via the `attach` prop) differs between VSCode
# and Jupyter environments.
#
# In VSCode, we need to explicitly attach the dialog to `#app` to ensure
# it is properly displayed on the screen and not hidden under an overlay.
# In Jupyter Notebook and Jupyter Lab, however, explicitly attaching the
# dialog causes it to get hidden under the overlay, making the dialog
# inaccessible and preventing us from interacting with it.
# Therefore, we apply this behavior only when running in VSCode.


class Dialog(v.Dialog):
    def __init__(self, *args, **kwargs):
        attach_target = False

        if is_vscode():
            attach_target = "#app"

        defaults = {
            "attach": attach_target,
        }

        defaults.update(kwargs)

        super().__init__(*args, **defaults)
