# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import ipyvuetify as v


def SearchableSelect(*args, **kwargs):
    defaults = {
        "dense": True,
        "outlined": True,
        "hide_details": True,
        "class_": "my-2",
        "return_object": False,  # to ensure that selected value is a string
    }

    if "class_" in kwargs:
        defaults["class_"] += " " + kwargs.pop("class_")

    defaults.update(kwargs)

    combobox = v.Combobox(*args, **defaults)

    return combobox
