# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v


def ProgressBar(*args, **kwargs):
    defaults = {"color": "blue"}

    defaults.update(kwargs)

    return v.ProgressCircular(*args, **defaults)
