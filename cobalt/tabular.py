# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import re
import warnings
from enum import Enum
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cobalt import schema
from cobalt.generate_tab_embeddings import get_tabular_embeddings
from cobalt.warnings_handler import WarningsHandler

warnings.filterwarnings("always", category=DeprecationWarning, module="cobalt")


class EmbeddingColumnsMatcher(Enum):
    REGEX = 0
    EXPLICIT = 1


class EmbeddingColumnInformation:
    def __init__(self, matcher, data) -> None:
        self.matcher = matcher
        self.data = data

    def select_those_that_satisfy(self, strings):
        if self.matcher == EmbeddingColumnsMatcher.REGEX:
            return [col for col in strings if re.match(self.data, col)]
        elif self.matcher == EmbeddingColumnsMatcher.EXPLICIT:
            # TODO: This doesn't do anything yet.
            return self.data


def load_tabular_dataset(
    df: pd.DataFrame,
    embeddings: Optional[
        Union[pd.DataFrame, np.ndarray, List[str], Literal["numeric_cols", "rf"]]
    ] = None,
    rf_source_columns: Optional[List[str]] = None,
    metadata_df: Optional[pd.DataFrame] = None,
    timestamp_col: Optional[str] = None,
    outcome_col: Optional[str] = None,
    prediction_col: Optional[str] = None,
    other_metadata: Optional[List[str]] = None,
    hidden_cols: Optional[List[str]] = None,
    baseline_column: Optional[str] = None,
    baseline_end_time: Optional[pd.Timestamp] = None,
    baseline_indices: Optional[Union[List[int], np.ndarray]] = None,
    split_column: Optional[str] = None,
    embedding_metric: str = "euclidean",
    task: Optional[Literal["classification", "regression"]] = None,
    model_name: Optional[str] = None,
) -> Tuple[schema.CobaltDataset, schema.DatasetSplit]:
    """Loads tabular data from a pandas DataFrame into a CobaltDataset.

    Separate dataframes are used to specify the source data, embedding
    columns, and (optionally) metadata columns.

    Note: This function is deprecated. Users should transition to constructing a
    CobaltDataset directly from a DataFrame.

    Args:
        df: A pandas.DataFrame containing the tabular source data.
        embeddings: Specifies which data to use as embedding columns. May be of
            type pandas.DataFrame, np.ndarray, List[str], or may be
            "numeric_cols" or "rf", indicating that the numeric columns contained
            in `df` should be used as the embedding vectors, or that a random
            forest embedding will be generated, respectively.
        rf_source_columns: Columns to use in the random forest embedding.
            If embeddings == 'rf' then use rf_source_columns as the input
            columns for the RF embedding.
            If embeddings == 'rf' and rf_source_columns is None then the random
            forest embedding columns will default to all of the numerical
            columns within the dataframe (df).
        metadata_df: Optional pandas.DataFrame containing additional metadata
            columns. All specified columns may be in either `df` or `metadata_df`.
            All other (non-hidden) columns in this dataframe will be treated as if
            they were in `other_metadata`.
        timestamp_col: String name of the column containing datapoint timestamps.
        outcome_col: String name of the column containing a numeric or
            categorical outcome variable. (E.g., the variable 'y'.)
        prediction_col: String name of the column containing model predictions.
            (E.g., the variable 'ŷ'.)
        other_metadata: Optional list of strings indicating other metadata
            columns in `df`. The Workspace may use this information to decide what to
            display, e.g. as options for coloring a visualization.
        hidden_cols: Optional list of strings indicating columns that will
            not be displayed in Cobalt TableViews.
        baseline_column: Optional string name of an indicator (boolean) column
            marking datapoints as belonging to the baseline set. One of three
            options for specifying the baseline set, along with
            `baseline_end_time` and `baseline_indices`.
        baseline_end_time: An optional pd.Timestamp; datapoints with values in
            `timestamp_col` <= to this value will be marked as baseline.
            Ignored if `baseline_column` is specified.
        baseline_indices: Optional list or np.ndarray of row indices, indicating
            datapoints belonging to the baseline set. Ignored if `baseline_column`
            or `baseline_end_time` are set.
        split_column: The name of a categorical column containing labels of which
            split of the dataset each data point belongs to. These splits will be
            available as data sources in the UI.
        embedding_metric: String indicating the type of metric to be used with the
            specified data embedding. Default: "euclidean".
        task: The type of task performed by the model being debugged.
            Can currently be set to either "regression" or "classification".
        model_name: A string name to refer to the model being analyzed.

    Returns:
        A (CobaltDataset, DatasetSplit) tuple.

    Raises:
        ValueError: `timestamp_col` was not specified or was of
            incorrect type.
        ValueError: None of `baseline_column`, `baseline_end_time`, or
            `baseline_indices` was specified.
        ValueError: The number of embedding vectors does not exactly match
            the number of datapoints.
        ValueError: `outcome_col` or `prediction_col` dtypes do not match.
        ValueError: Mismatch between `outcome_type` and detected dtype of outcome
            or prediction columns.
    """
    warnings.warn(
        (
            "load_tabular_dataset() is deprecated and will be removed in Cobalt 0.4.0. "
            "Instead, instantiate a CobaltDataset directly from your DataFrame: CobaltDataset(df)."
        ),
        category=DeprecationWarning,
        stacklevel=2,
    )
    columns_schema = {
        "rf_source_columns": rf_source_columns,
        "timestamp_col": timestamp_col,
        "outcome_col": outcome_col,
        "prediction_col": prediction_col,
        "hidden_cols": hidden_cols,
        "other_metadata": other_metadata,
    }

    if task is None:
        WarningsHandler.task_is_not_provided()
        task = "classification"

    split_schema = {
        "baseline_column": baseline_column,
        "baseline_end_time": baseline_end_time,
        "baseline_indices": baseline_indices,
        "split_column": split_column,
    }

    loader = TabularDatasetLoader(
        df,
        embeddings,
        metadata_df,
        columns_schema,
        split_schema,
        embedding_metric,
        task,
        model_name,
    )

    dataset, subsets = loader.load()
    return dataset, subsets


class TabularDatasetLoader:
    def __init__(
        self,
        df,
        embeddings,
        metadata_df,
        columns_schema,
        split_schema,
        embedding_metric,
        task: str,
        model_name: Optional[str],
    ):
        self.df = df
        self.embeddings = embeddings
        self.metadata_df = metadata_df
        self.columns_schema = columns_schema
        self.split_schema = split_schema
        self.embedding_metric = embedding_metric
        self.task = schema.ModelTask(task)
        self.model_name = model_name
        self.error_cols = []
        self.arrays = {}

        self.validate_columns()

    def validate_columns(self):
        if not all(isinstance(c, str) for c in self.df.columns):
            print(
                "Warning: Some column names in df are not strings and will be converted to strings."
            )

            self.df.columns = [str(c) for c in self.df.columns]
        if self.metadata_df is not None and not all(
            isinstance(c, str) for c in self.metadata_df.columns
        ):
            print(
                "Warning: Some column names in metadata_df "
                "are not strings and will be converted to strings."
            )

            self.metadata_df.columns = [str(c) for c in self.metadata_df.columns]

    def update_other_metadata(self):
        other_metadata = self.columns_schema["other_metadata"]
        new_metadata_cols = set(self.metadata_df.columns)
        new_metadata_cols.discard(self.columns_schema["timestamp_col"])
        new_metadata_cols.discard(self.columns_schema["outcome_col"])
        new_metadata_cols.discard(self.columns_schema["prediction_col"])
        new_metadata_cols.difference_update(self.columns_schema["hidden_cols"])
        other_metadata.extend(new_metadata_cols)
        return other_metadata

    def _get_baseline_mask_and_indices(self):
        baseline_column = self.split_schema["baseline_column"]
        baseline_end_time = self.split_schema["baseline_end_time"]
        baseline_indices = self.split_schema["baseline_indices"]

        if baseline_column is not None:
            baseline_mask = self.df[baseline_column].astype("bool")
            baseline_indices = np.flatnonzero(baseline_mask)
        elif baseline_end_time is not None:
            baseline_mask = (
                self.df[self.columns_schema["timestamp_col"]] <= baseline_end_time
            )
            baseline_indices = np.flatnonzero(baseline_mask)
        elif baseline_indices is not None:
            baseline_mask = np.zeros(len(self.df), dtype=np.bool_)
            baseline_mask[baseline_indices] = 1
        else:
            raise ValueError(
                "One of baseline_column, baseline_end_time, or baseline_indices must be specified"
            )
        return baseline_mask, baseline_indices

    def _get_numeric_cols_embedding(self, original_df_cols):
        numeric_cols = [
            c for c in original_df_cols if pd.api.types.is_numeric_dtype(self.df[c])
        ]
        return schema.ColumnEmbedding(
            numeric_cols, self.embedding_metric, name="numeric columns"
        )

    def _get_rf_embedding(self, original_df_cols):
        rf_source_columns = self.columns_schema["rf_source_columns"]
        training_cols = (
            [c for c in original_df_cols if pd.api.types.is_numeric_dtype(self.df[c])]
            if rf_source_columns is None
            else rf_source_columns
        )
        print("Training random forest embeddings using columns:")
        print(training_cols)
        outcome_col = self.columns_schema["outcome_col"]
        if outcome_col is not None and outcome_col not in training_cols:
            training_cols.append(outcome_col)
        embedding_arr, metric, name = get_tabular_embeddings(
            self.df[training_cols],
            model_name="rf",
            outcome=outcome_col,
        )
        self.arrays[name] = embedding_arr
        embedding = schema.ArrayEmbedding(
            array_name=name, dimension=embedding_arr.shape[1], metric=metric, name=name
        )
        return embedding

    def get_embedding(self, original_df_cols):
        if isinstance(self.embeddings, str):
            if self.embeddings == "numeric_cols":
                embedding = self._get_numeric_cols_embedding(original_df_cols)
            elif self.embeddings == "rf":
                embedding = self._get_rf_embedding(original_df_cols)
            else:
                raise ValueError(f"Unsupported argument embeddings={self.embeddings}")
        elif isinstance(self.embeddings, list):
            if not all(
                pd.api.types.is_numeric_dtype(dtype)
                for dtype in self.df[self.embeddings].dtypes
            ):
                raise ValueError(
                    "Non-numeric columns detected with list of embedding column names"
                )
            embedding = schema.ColumnEmbedding(self.embeddings, self.embedding_metric)
        else:
            if self.embeddings.shape[0] != self.df.shape[0]:
                raise ValueError(
                    "There must be one embedding vector for each data point"
                )

            embedding_array = np.array(self.embeddings)

            if not pd.api.types.is_numeric_dtype(embedding_array.dtype):
                raise ValueError(
                    f"Unsupported dtype for embedding = {embedding_array.dtype}. "
                    "Embeddings should be of numeric dtype."
                )

            self.arrays["embedding"] = embedding_array
            embedding = schema.ArrayEmbedding(
                array_name="embedding",
                dimension=self.arrays["embedding"].shape[1],
                metric=self.embedding_metric,
            )
        return embedding

    def warn_type_mismatch_model(self, md: schema.model_metadata.ModelMetadata):
        outcome_col = md.outcome_column
        prediction_col = md.prediction_column

        out_col_type, pred_col_type = (
            self.df[outcome_col].dtype,
            self.df[prediction_col].dtype,
        )

        if out_col_type != pred_col_type:
            print(
                "Warning: Outcome and prediction column types do not match: "
                f"{out_col_type} vs. {pred_col_type}"
            )

    def get_outcome_and_prediction_columns(self):
        outcome_cols = (
            [self.columns_schema["outcome_col"]]
            if self.columns_schema["outcome_col"] is not None
            else []
        )
        prediction_cols = (
            [self.columns_schema["prediction_col"]]
            if self.columns_schema["prediction_col"] is not None
            else []
        )
        for col in outcome_cols + prediction_cols:
            if col not in self.df.columns:
                raise ValueError(f"Column {col} not found in dataset")

        return outcome_cols, prediction_cols

    def get_split_subsets(self):
        create_baseline_comparison = (
            self.split_schema["baseline_column"] is not None
            or self.split_schema["baseline_end_time"] is not None
            or self.split_schema["baseline_indices"] is not None
        )
        if create_baseline_comparison:
            baseline_mask, baseline_indices = self._get_baseline_mask_and_indices()
            comparison_mask = ~baseline_mask
            comparison_indices = np.flatnonzero(comparison_mask)

            splits = {"baseline": baseline_indices, "comparison": comparison_indices}
        else:
            splits = {}

        if self.split_schema["split_column"] is not None:
            split_column = self.split_schema["split_column"]
            for name in self.df[split_column].unique():
                mask = self.df[split_column] == name
                indices = np.flatnonzero(mask)
                splits[name] = indices

        return splits

    def get_model_name(self):
        if self.model_name:
            return self.model_name
        elif self.task == schema.ModelTask.classification:
            return "classifier"
        elif self.task == schema.ModelTask.regression:
            return "regressor"
        return None

    def load(self):
        original_df_cols = self.df.columns
        if self.columns_schema["other_metadata"] is None:
            self.columns_schema["other_metadata"] = []

        if self.metadata_df is not None:
            self.df = self.metadata_df.join(self.df)
            self.columns_schema["other_metadata"] = self.update_other_metadata()

        splits = self.get_split_subsets()

        # TODO: multiple outcome/pred cols
        outcome_cols, prediction_cols = self.get_outcome_and_prediction_columns()

        model = schema.model_metadata.ModelMetadata(
            outcome_cols, prediction_cols, task=self.task, name=self.get_model_name()
        )

        embedding = (
            self.get_embedding(original_df_cols)
            if self.embeddings is not None
            else None
        )
        metadata = schema.DatasetMetadata(
            timestamp_columns=(
                [self.columns_schema["timestamp_col"]]
                if self.columns_schema["timestamp_col"] is not None
                else None
            ),
            hidable_columns=self.columns_schema["hidden_cols"],
            other_metadata_columns=self.columns_schema["other_metadata"],
        )

        embeddings = [embedding] if embedding is not None else None
        dataset = schema.CobaltDataset(
            self.df, metadata, [model], embeddings=embeddings, arrays=self.arrays
        )

        if outcome_cols and prediction_cols:
            self.warn_type_mismatch_model(model)
            dataset.compute_model_performance_metrics()

        split = schema.DatasetSplit(dataset, splits)

        return dataset, split
