import warnings

import numpy as np
import pandas as pd


def warn_if_index_not_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Raises user warning if index of dataframe is not unique."""
    if not df.index.is_unique:
        warnings.warn(
            "The range of the dataframe is not unique. Resetting its index and proceeding.",
            stacklevel=2,
        )
        df = df.reset_index(drop=False)
    return df


def raise_error_for_non_string_column_names(df: pd.DataFrame):
    """Validate that the column names in the dataset are strings."""
    non_valid_columns = [col for col in df.columns if not isinstance(col, str)]
    if non_valid_columns:
        raise ValueError(
            f"Column names must be strings. Invalid column ids are {non_valid_columns}"
        )


def indices_type_valid(indices: np.ndarray):
    """Check that indices is one dimensinal array and type of int or nd.int."""
    if indices.size == 0:
        return True
    return indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer)
