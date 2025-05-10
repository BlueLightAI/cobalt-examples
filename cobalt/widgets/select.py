# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v


def Select(*args, **kwargs):
    defaults = {"dense": True, "outlined": True, "hide_details": True, "class_": "my-2"}

    if "class_" in kwargs:
        defaults["class_"] += " " + kwargs.pop("class_")

    defaults.update(kwargs)

    return v.Select(*args, **defaults)
