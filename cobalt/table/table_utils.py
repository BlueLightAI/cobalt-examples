from typing import Optional

import pandas as pd

from cobalt.table.table_constants import (
    DATETIME_FORMAT,
    DEFAULT_NUMERIC_FORMAT,
    OPERATOR_MAP,
)


def get_text_operator(op):
    return OPERATOR_MAP.get(op)


def is_datetime_col(series):
    return pd.api.types.is_datetime64_any_dtype(series)


def format_df_for_display(
    df: pd.DataFrame,
    num_rows: Optional[int] = None,
    numeric_format: str = DEFAULT_NUMERIC_FORMAT,
) -> pd.DataFrame:
    if num_rows is not None and len(df) > num_rows:
        df = df.head(num_rows)
    df = df.copy()

    for col in df.columns:
        if is_datetime_col(df[col]):
            df[col] = df[col].dt.strftime(DATETIME_FORMAT)

    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].apply(numeric_format.format)
    return df
