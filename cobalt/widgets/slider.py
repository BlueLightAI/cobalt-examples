# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v


def Slider(*args, **kwargs):
    defaults = {"hide_details": True}

    defaults.update(kwargs)

    return v.Slider(*args, **defaults)
