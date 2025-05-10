# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import warnings


class WarningsHandler:
    @staticmethod
    def task_is_not_provided(category=DeprecationWarning, stacklevel=2):
        warnings.warn(
            """
            Task not provided: Assuming task == 'classification'.
            This will be an error in a future version.""",
            category=category,
            stacklevel=stacklevel,
        )

    @staticmethod
    def workplace_import(category=DeprecationWarning, stacklevel=2):
        warnings.warn(
            (
                "Import Workspace from cobalt.ui is deprecated. "
                "Please consider importing Workspace from top-level cobalt module"
            ),
            category=category,
            stacklevel=stacklevel,
        )

    @staticmethod
    def convert_numpy(embedding_type, category=UserWarning, stacklevel=1):
        warnings.warn(
            f"Attempting to convert {embedding_type} to numpy...",
            category=UserWarning,
            stacklevel=stacklevel,
        )

    @staticmethod
    def img_type_deprecated(category=UserWarning, stacklevel=1):
        warnings.warn(
            "Using img as file type is deprecated.",
            category=category,
            stacklevel=stacklevel,
        )
